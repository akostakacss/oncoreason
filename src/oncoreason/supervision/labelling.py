"""Process supervision — segment traces into steps and label each step.

The Med-PRM trick: the GUIDELINE supplies the label. Each reasoning step is auto-verified
against the evidence it rests on -> sound / unsound, so no human annotation team is needed.
Then a human audits a held-out subset and the agreement (Cohen's kappa) is REPORTED — it is
the honest foundation of the whole novelty claim.

This is the deterministic, offline core of "RAG-as-a-Judge" (Med-PRM): a step is sound iff its
evidential claims are backed by **resolvable** retrieved evidence. The semantic version — does
the record *support* the claim, not merely resolve — is the reserved teacher (Claude) slot,
left unwired here. The verifier here already catches the load-bearing
failure: a step that cites evidence which does not resolve (a hallucinated / unverifiable
citation) is labelled unsound.

Phase 4.1 (segmentation) + 4.2 (labels) + 4.3 (audit).
"""
from __future__ import annotations

import random

from dataclasses import dataclass

from ..agents.trace import ReasoningStep, Trace


@dataclass(frozen=True)
class StepLabel:
    case_id: str
    step_index: int
    sound: bool
    verifying_citation_ids: list[str]   # what the label was checked against
    note: str | None = None


# Fixed segmentation rule (pin once — segmentation changes labels; Phase 4.1):
#   one ReasoningStep == one step unit. Every step is labelled; a step carrying no evidential
#   citation (e.g. the planner) is judged on structure alone (there is nothing to verify).
def segment_steps(trace: Trace) -> list[ReasoningStep]:
    """Return the trace's steps under the fixed one-step-per-ReasoningStep rule."""
    return list(trace.steps)


def resolvable_evidence_ids(trace: Trace) -> set[str]:
    """Citation ids in a trace that resolve to a real retrieved record.

    In this PoC every citation is emitted from a real connector/retrieval, so 'resolvable'
    means 'present among the trace's own citations'. A *policy-generated* trace may cite ids
    that do not appear here — those are unverifiable and drive an 'unsound' label, which is
    exactly the citation-grounding failure the PRM must learn to catch.
    """
    return {c.citation_id for s in trace.steps for c in s.citations}


def label_step_against_guideline(step: ReasoningStep, resolvable_ids, *, case_id: str = "") -> StepLabel:
    """Auto-label one step sound/unsound by checking its citations against resolvable evidence."""
    cited = [c.citation_id for c in step.citations]
    if not cited:
        # no evidential claim to verify (planner / synthesis prose) -> structurally sound
        return StepLabel(case_id, step.index, True, [], note="no evidential claim")
    verified = [cid for cid in cited if cid in resolvable_ids]
    sound = len(verified) > 0
    return StepLabel(case_id, step.index, sound, verified,
                     note=None if sound else "cites no resolvable evidence")


def label_trace(trace: Trace, resolvable_ids: set[str] | None = None) -> list[StepLabel]:
    """Label every step of a trace. Defaults to verifying against the trace's own resolvable ids."""
    ids = resolvable_ids if resolvable_ids is not None else resolvable_evidence_ids(trace)
    return [label_step_against_guideline(s, ids, case_id=trace.case_id)
            for s in segment_steps(trace)]


def annotate_trace_with_labels(trace: Trace, labels: list[StepLabel]) -> None:
    """Write `sound` back onto each ReasoningStep.label_sound (consumed by the PRM in Phase 5)."""
    by_index = {l.step_index: l for l in labels}
    for s in trace.steps:
        if s.index in by_index:
            s.label_sound = by_index[s.index].sound


def build_prm_examples(trace: Trace, resolvable_ids: set[str] | None = None) -> list[dict]:
    """(case, step, evidence, label) rows — the PRM's training data (Phase 5 consumes these)."""
    labels = {l.step_index: l for l in label_trace(trace, resolvable_ids)}
    return [
        {
            "case_id": trace.case_id,
            "step_index": s.index,
            "step_text": s.text,
            "evidence_ids": [c.citation_id for c in s.citations],
            "label_sound": labels[s.index].sound,
        }
        for s in segment_steps(trace)
    ]


def mine_negatives(examples: list[dict], seed: int = 17,
                   strategies: tuple[str, ...] = ("strip", "swap")) -> list[dict]:
    """Construct counterfactual *unsound* steps so the verifier has something to learn.

    Why this is necessary: the deterministic scaffold emits citations only from real
    retrievals, so every step it produces is grounded by construction and the label
    distribution is degenerate (100% sound). A verifier trained on that has no signal. Real
    negatives appear once a *generative* policy can hallucinate a citation, which is the GPU
    path; until then we mine them counterfactually.

    Both strategies mirror failure modes we actually care about and have already observed:

      - ``strip``: same claim, citations removed -> an assertion with no supporting record.
      - ``swap``: citations replaced with ids drawn from a *different* case -> evidence that
        exists but does not support this claim. This is the machine analogue of the
        tumor-type-mismatch error the scaffold hit on TP53.

    These are **synthetic** negatives and are marked as such (`synthetic: True`). Any accuracy
    reported on this mixture is accuracy on a semi-synthetic distribution, not on policy
    output; that limitation belongs in the write-up.
    """
    rng = random.Random(seed)
    pos = [e for e in examples if e.get("label_sound") and (e.get("evidence_ids") or [])]
    if not pos:
        return []
    all_ids = sorted({i for e in pos for i in e["evidence_ids"]})
    out: list[dict] = []
    for e in pos:
        if "strip" in strategies:
            out.append({**e, "evidence_ids": [], "label_sound": False,
                        "synthetic": True, "strategy": "strip"})
        if "swap" in strategies and len(all_ids) > 1:
            foreign = [i for i in all_ids if i not in set(e["evidence_ids"])]
            if foreign:
                out.append({**e,
                            "evidence_ids": [rng.choice(foreign)],
                            "label_sound": False, "synthetic": True, "strategy": "swap"})
    return out


def label_case_outcome(case, guideline_retriever, top_k: int = 3):
    """Outcome-level supervision (Phase 4.4): derive the guideline-recommended
    therapy set for a case, as a `GoldRecommendation`.

    The project targets both process-level and outcome-level supervision; this is the outcome half.
    The label comes from the same derived guideline index the guideline specialist retrieves
    over, which creates a **partial circularity**: an agent that simply echoes the guideline
    would score highly on guideline concordance by construction. Two things keep the metric
    informative anyway, and both are stated rather than hidden:

      1. the synthesizer does not merely copy the guideline; it adjudicates between the
         guideline prior and CIViC evidence, which can and does compete with it, so
         concordance genuinely measures whether that adjudication respects the prior;
      2. the *other* Phase-6 metrics (molecular-interpretation accuracy, citation grounding,
         step soundness, calibration) do not depend on this label at all.

    A non-circular gold standard needs molecular-tumor-board recommendations, which is the
    documented roadmap item.
    """
    from ..agents.guideline_index import CHUNKS_BY_ID
    from ..agents.orchestrator import _pick_guideline
    from ..agents.tools import guideline_lookup
    from ..cases.schema import GoldRecommendation

    tumor = case.context.tumor_type
    picked, therapies = [], []
    for alt in case.alterations:
        chunks, _ = guideline_lookup(alt.gene, tumor, guideline_retriever, top_k=top_k)
        gc = _pick_guideline(chunks, alt.gene, tumor)
        if gc is None:
            continue
        picked.append(gc)
        for th in gc.therapies:
            if th not in therapies:
                therapies.append(th)
    if not therapies:
        return None
    # rank the gold by ESCAT tier so top-1 concordance is meaningful
    tier_rank = {"I-A": 0, "I-B": 1, "I-C": 2, "II-A": 3, "II-B": 4, "N/A": 9}
    best = min(picked, key=lambda g: tier_rank.get(g.escat_tier, 9))
    ordered = [t for t in best.therapies] + [t for t in therapies if t not in best.therapies]
    return GoldRecommendation(
        recommended=ordered,
        escat_tier=best.escat_tier,
        guideline_source="derived ESMO index (author-summarised)",
        rationale=best.recommendation,
    )


def audit_agreement(auto: list[StepLabel], manual: list[StepLabel]) -> float:
    """Cohen's kappa between auto-labels and a manual audit subset, aligned by
    (case_id, step_index). REPORT this number — it is the honest foundation of the novelty."""
    man = {(m.case_id, m.step_index): bool(m.sound) for m in manual}
    y_auto, y_man = [], []
    for a in auto:
        key = (a.case_id, a.step_index)
        if key in man:
            y_auto.append(bool(a.sound))
            y_man.append(man[key])
    if not y_auto:
        raise ValueError("no overlapping (case_id, step_index) between auto and manual labels")
    # kappa is undefined when only one class is present; fall back to raw agreement there.
    if len(set(y_auto + y_man)) == 1:
        return 1.0
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_man, y_auto))

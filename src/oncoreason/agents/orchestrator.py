"""Multi-agent orchestration — planner -> specialists -> synthesizer.

Hierarchical context streams: a planner decomposes the case, dispatches to
specialist sub-agents each with their own context, and a synthesizer composes the final
recommendation with citations and an abstain path.

Design choice that makes this auditable and testable: the **decision** (ranked recommendation,
confidence, abstain) is derived *deterministically from retrieved evidence* — it does not
depend on an LLM's free text. An ``LLMClient`` (Claude teacher / Qwen policy) only narrates
each step, so the same run is reproducible offline with ``DeterministicLLM``. The evidence,
not the prose, carries the verdict. Phase 3.3.
"""
from __future__ import annotations

import re
from typing import Protocol

from ..cases.schema import Case
from ..datasources.base import Evidence, EvidenceKind
from ..retrieval.base import Retriever
from .guideline_index import CHUNKS_BY_ID
from .tools import evidence_citations, guideline_lookup, variant_lookup
from .trace import Citation, ReasoningStep, Trace

SPECIALISTS = ("variant", "guideline", "trial", "literature")

# Actionability weights (deliberately explicit, not learned — this is the auditable prior).
_GL_TIER_W = {"I-A": 0.95, "I-B": 0.88, "I-C": 0.80, "II-A": 0.75, "II-B": 0.70, "N/A": 0.50}
_CIVIC_LVL_W = {"A": 0.90, "B": 0.82, "C": 0.68, "D": 0.55, "E": 0.45}
_AGREEMENT_BONUS = 0.05
# Actionability is tumor-type-specific (risk register): CIViC evidence from a *different*
# tumor type is heavily discounted so it cannot outrank the tumor-matched guideline prior —
# e.g. a sarcoma-context TP53 response must not drive a lung recommendation.
_TUMOR_MISMATCH_PENALTY = 0.30


class LLMClient(Protocol):
    """Thin wrapper over a policy: Claude (teacher) or Qwen (trained policy)."""

    model: str

    def complete(self, prompt: str, **kw) -> str: ...


def _tokens(text: str | None) -> set[str]:
    return {t for t in re.split(r"\W+", (text or "").lower()) if len(t) > 2}


def _tumor_match(disease: str | None, tumor_type: str | None) -> bool:
    d, t = _tokens(disease), _tokens(tumor_type)
    return bool(d & t)


def _pick_guideline(chunks, gene, tumor_type):
    """Choose the guideline chunk for this alteration: a gene-specific recommendation if one
    exists, else the tumor-type 'no targetable driver' chunk (chemo/IO). Never fall back to an
    unrelated gene's chunk just because the tumor-type tokens matched — that would recommend
    the wrong targeted drug."""
    gene_match = next((CHUNKS_BY_ID[c.doc_id] for c in chunks
                       if CHUNKS_BY_ID.get(c.doc_id) and CHUNKS_BY_ID[c.doc_id].gene == gene), None)
    if gene_match is not None:
        return gene_match
    tt = _tokens(tumor_type)
    return next((gc for gc in CHUNKS_BY_ID.values()
                 if gc.gene == "" and (_tokens(gc.tumor_type) & tt)), None)


class Orchestrator:
    def __init__(
        self,
        llm: LLMClient | None = None,
        sources: dict | None = None,
        guideline_retriever: Retriever | None = None,
        abstain_threshold: float = 0.5,
    ) -> None:
        self.llm = llm
        self.sources = sources or {}
        self.guideline_retriever = guideline_retriever
        self.abstain_threshold = abstain_threshold

    # -- narration: LLM if present, else the composed summary verbatim -----------------
    def _narrate(self, role: str, summary: str) -> str:
        if self.llm is None:
            return summary
        return self.llm.complete(f"[{role}] Narrate this reasoning step.\n{summary}")

    def run(self, case: Case) -> Trace:
        trace = Trace(case_id=case.case_id,
                      model=(self.llm.model if self.llm else "deterministic-scaffold"))
        civic = self.sources.get("civic")
        clinvar = self.sources.get("clinvar")

        # -- planner ---------------------------------------------------------------
        alt_labels = [f"{a.gene} {a.variant}" for a in case.alterations]
        wired = [s for s in ("variant", "guideline")
                 if (s == "variant" and civic) or (s == "guideline" and self.guideline_retriever)]
        plan = (f"Plan: {len(case.alterations)} alteration(s) [{'; '.join(alt_labels)}] in "
                f"{case.context.tumor_type}. Consult specialists {wired}, then synthesize.")
        trace.steps.append(ReasoningStep(index=0, text=self._narrate("planner", plan)))
        trace.metadata["sub_questions"] = alt_labels
        trace.metadata["specialists_run"] = wired

        # collected across specialists, feeding the synthesizer
        all_predictive: list[Evidence] = []
        gl_by_gene: dict[str, object] = {}

        # -- specialists (one step per alteration per wired specialist) -------------
        idx = 1
        for a in case.alterations:
            if civic is not None:
                ev, calls = variant_lookup(a.gene, a.variant, case.context.tumor_type,
                                           civic, clinvar)
                best = min((e.evidence_level for e in ev if e.evidence_level), default="-")
                n_germ = sum(1 for e in ev if e.kind == EvidenceKind.PATHOGENICITY)
                summary = (f"Variant {a.gene} {a.variant}: {len(ev)} CIViC/ClinVar item(s), "
                           f"best CIViC level {best}, {n_germ} germline pathogenicity item(s).")
                trace.steps.append(ReasoningStep(
                    index=idx, text=self._narrate("variant", summary),
                    tool_calls=calls, citations=evidence_citations(ev)))
                idx += 1
                all_predictive += [e for e in ev if e.kind == EvidenceKind.PREDICTIVE]

            if self.guideline_retriever is not None:
                chunks, call = guideline_lookup(a.gene, case.context.tumor_type,
                                                self.guideline_retriever)
                gc = _pick_guideline(chunks, a.gene, case.context.tumor_type)
                if gc is not None:
                    gl_by_gene[a.gene] = gc
                    summary = (f"Guideline for {a.gene} {case.context.tumor_type}: "
                               f"{gc.recommendation} (ESCAT {gc.escat_tier}).")
                    cite = [Citation(citation_id=f"guideline:{gc.chunk_id}",
                                     source="guideline_index", claim=gc.recommendation)]
                    trace.steps.append(ReasoningStep(
                        index=idx, text=self._narrate("guideline", summary),
                        tool_calls=[call], citations=cite))
                    idx += 1

        # -- synthesizer -----------------------------------------------------------
        ranked, confidence = self._synthesize(all_predictive, gl_by_gene, case.context.tumor_type)
        # <= , not <: the "no targetable driver" fallback (_GL_TIER_W["N/A"]) lands exactly on
        # abstain_threshold by construction (both 0.50) — a strict < let that tie through as a
        # confident recommendation on every non-actionable case, so the system never abstained
        # (found by the Phase 6 evaluation harness: molecular_interpretation_accuracy 0.24, the
        # actionable base rate, with 0/50 abstentions).
        abstained = (not ranked) or confidence <= self.abstain_threshold
        rec_line = (f"Synthesis: recommend {ranked[:3] or '—'} at confidence {confidence:.2f}."
                    + (" Low confidence — abstain and defer to clinician." if abstained else ""))
        trace.steps.append(ReasoningStep(index=idx, text=self._narrate("synthesizer", rec_line)))

        trace.recommendation = ranked[:5]
        trace.confidence = round(confidence, 3)
        trace.abstained = abstained
        return trace

    # -- deterministic, auditable decision -------------------------------------------
    def _synthesize(self, predictive, gl_by_gene, tumor_type):
        """Merge guideline prior + CIViC predictive-sensitivity evidence into a ranked list.

        Guideline = the population prior; CIViC = the somatic-evidence signal; agreement
        between them raises confidence. Returns (ranked_therapies, confidence).
        """
        cand: dict[str, dict] = {}  # lower(therapy) -> {display, gl_w, civic_w}

        def bump(name, key, w):
            slot = cand.setdefault(name.lower(), {"display": name, "gl_w": 0.0, "civic_w": 0.0})
            slot[key] = max(slot[key], w)

        for gc in gl_by_gene.values():
            w = _GL_TIER_W.get(gc.escat_tier, 0.5)
            for th in gc.therapies:
                bump(th, "gl_w", w)

        for e in predictive:
            direction = e.payload.get("evidence_direction")
            sig = e.payload.get("significance") or ""
            if direction != "SUPPORTS" or "SENSITIVITY" not in sig.upper():
                continue
            w = _CIVIC_LVL_W.get(e.evidence_level or "E", 0.45)
            if not _tumor_match(e.payload.get("disease"), tumor_type):
                w = max(0.0, w - _TUMOR_MISMATCH_PENALTY)
            for th in e.payload.get("therapies", []):
                bump(th, "civic_w", w)

        scored = []
        for slot in cand.values():
            combined = max(slot["gl_w"], slot["civic_w"])
            if slot["gl_w"] > 0 and slot["civic_w"] > 0:
                combined = min(0.97, combined + _AGREEMENT_BONUS)
            scored.append((slot["display"], combined))
        scored.sort(key=lambda x: x[1], reverse=True)

        ranked = [name for name, _ in scored]
        confidence = scored[0][1] if scored else 0.15
        return ranked, confidence

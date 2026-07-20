#!/usr/bin/env python3
"""Stage I — the baseline arm, through the unmodified Phase 6 harness.

Phase 6's third honest limitation: *"No baseline comparison. There is no untrained-policy or
frontier-model arm yet, so none of these numbers can be called an improvement over anything."*

Stage E sampled 400 traces from an untrained Qwen2.5-3B-Instruct. That **is** the untrained
policy arm. Running it through the same eight metrics, against the same 50 cases and the same
guideline-derived gold, finally gives every other number in this project a reference point.

Three arms:

  scaffold        the deterministic multi-agent pipeline (what Phase 6 reported)
  policy          untrained Qwen2.5-3B, one sample per case — what a single call gets you
  policy + BoN    the same policy, with the stage F verifier choosing among 8 candidates

The harness is imported unchanged. Where a metric is degenerate for an arm it is reported as
degenerate rather than omitted — the sampled policy makes no tool calls, so tool reliability
describes the scaffold only, and saying so is more useful than a blank cell.

Usage:  python scripts/stage_i_baseline.py <candidates.jsonl>
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from oncoreason.agents import DeterministicLLM, Orchestrator            # noqa: E402
from oncoreason.agents.guideline_index import index_docs                # noqa: E402
from oncoreason.agents.trace import Citation, ReasoningStep, Trace      # noqa: E402
from oncoreason.cases.schema import Alteration, Case, ClinicalContext   # noqa: E402
from oncoreason.datasources import get_source                          # noqa: E402
from oncoreason.evaluation import (calibration, citation_grounding,     # noqa: E402
                                   deferral_curve, guideline_concordance,
                                   information_gathering,
                                   molecular_interpretation_accuracy,
                                   reasoning_step_accuracy, tool_use_reliability)
from oncoreason.retrieval.base import BM25Retriever                    # noqa: E402
from oncoreason.supervision import label_case_outcome                  # noqa: E402
from oncoreason.training import PRM, PRMConfig, best_of_n, split_by_case  # noqa: E402
from relabel_candidates import build_evidence_pools, normalise         # noqa: E402
from semantic_labels import build_record_index, label_step             # noqa: E402

SEED = 17
OUT = os.path.join(REPO, "results", "gpu_stage")

# The sampled policy writes its recommendation as free text, so the raw parse carries the
# trailing [CITE: ...] block and qualifiers like "first-line" into the therapy string. Scoring
# that against a therapy-name gold set measures my parser, not the policy: an early run scored
# the untrained policy at 0.020 top-1 while it was in fact naming the right drug. Cleaning is
# therefore part of reading the baseline honestly, not a thumb on the scale.
_CITE_RE = re.compile(r"\[\s*CITE:[^\]]*\]", re.I)
_QUALIFIER_RE = re.compile(
    r"^(?:first|second|third)[\s-]line\s+|^(?:consider|recommend(?:ed)?|prefer(?:red)?|"
    r"initiate|start|offer)\s+|\s*\(.*?\)\s*$", re.I)


def clean_recommendation(rec: list[str]) -> list[str]:
    """Strip citation blocks and leading qualifiers, and split residual lists."""
    out: list[str] = []
    for item in rec or []:
        item = _CITE_RE.sub(" ", item)
        for part in re.split(r"\s*(?:,|;|/| or | and | then )\s*", item):
            part = _QUALIFIER_RE.sub("", part.strip()).strip(" .;:")
            if part:
                out.append(part)
    return out




def build_cases(guideline) -> list[Case]:
    cases = []
    with open(os.path.join(REPO, "data", "cases", "cases.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            alts = [Alteration(a["gene"], a["variant"], a.get("kind", "mutation"),
                               a.get("is_somatic", True), a.get("escat_tier"))
                    for a in d["alterations"]]
            c = Case(d["case_id"], alts, ClinicalContext(**d["context"]),
                     provenance=d.get("provenance", {}))
            cases.append(Case(c.case_id, c.alterations, c.context,
                              gold=label_case_outcome(c, guideline),
                              split=c.split, provenance=c.provenance))
    return cases


def evaluate(traces, cases, resolvable) -> dict:
    return {
        "guideline_concordance": guideline_concordance(traces, cases),
        "molecular_interpretation_accuracy": molecular_interpretation_accuracy(traces, cases),
        "reasoning_step_accuracy": reasoning_step_accuracy(traces),
        "tool_use_reliability": tool_use_reliability(traces),
        "citation_grounding": citation_grounding(
            traces, resolver=lambda cid: cid in resolvable),
        "calibration": calibration(traces, cases),
        "deferral_curve": deferral_curve(traces, cases),
        "information_gathering": information_gathering(traces, cases),
        "n_abstained": sum(1 for t in traces if t.abstained),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    guideline = BM25Retriever(source="esmo_index")
    guideline.index(index_docs())
    cases = build_cases(guideline)
    print(f"{len(cases)} cases, {sum(1 for c in cases if c.gold)} with guideline gold")

    # -- arm 1: the deterministic scaffold ----------------------------------------
    orch = Orchestrator(llm=DeterministicLLM(),
                        sources={"civic": get_source("civic"), "clinvar": get_source("clinvar")},
                        guideline_retriever=guideline)
    scaffold = [orch.run(c) for c in cases]

    # -- arms 2 and 3: the sampled policy -----------------------------------------
    pools, global_ids = build_evidence_pools()
    claims = {cid: txt for p in pools.values() for cid, txt in p.items()}
    records, vocab, tumors = build_record_index()

    by_case: dict[str, list[Trace]] = {}
    examples: list[dict] = []
    for line in open(args.candidates):
        r = json.loads(line)
        cid, tumor = r["case_id"], tumors.get(r["case_id"], "lung")
        pool = pools.get(cid, {})
        steps = []
        for i, s in enumerate(r["steps"]):
            resolved, invented, _ = normalise(", ".join(s.get("cited") or []), global_ids)
            off = [c for c in resolved if c not in pool]
            sem = label_step(s["text"], resolved, records, tumor, vocab)
            sound = (bool(resolved) and not invented and not off
                     and sem["label_semantic_sound"] and not sem["disease_mismatch"])
            examples.append({"case_id": cid, "step_text": s["text"],
                             "evidence_ids": resolved, "label_sound": sound})
            steps.append(ReasoningStep(
                index=i, text=s["text"], label_sound=sound,
                citations=[Citation(citation_id=c, source=c.split(":")[0],
                                    claim=claims.get(c, "")) for c in resolved]))
        by_case.setdefault(cid, []).append(Trace(
            case_id=cid, steps=steps,
            recommendation=clean_recommendation(r.get("recommendation") or []),
            confidence=r.get("confidence"), abstained=bool(r.get("abstained")),
            model="Qwen/Qwen2.5-3B-Instruct"))

    # one sample per case is the honest single-call baseline
    policy = [by_case[c.case_id][0] for c in cases if c.case_id in by_case]

    # Best-of-N uses a verifier trained on train cases only; selection on every case would
    # let the verifier rank traces from cases it was fit on.
    train, _, test_cases = split_by_case(examples, test_frac=0.3, seed=SEED)
    prm = PRM(PRMConfig(seed=SEED)).fit(train)
    policy_bon = []
    for c in cases:
        cand = by_case.get(c.case_id)
        if not cand:
            continue
        policy_bon.append(best_of_n(prm, cand)[0] if len(cand) > 1 else cand[0])

    resolvable = set(global_ids) | {c.citation_id for t in scaffold for c in t.all_citations()}

    arms = {
        "scaffold_deterministic": evaluate(scaffold, cases, resolvable),
        "policy_untrained_1sample": evaluate(policy, cases, resolvable),
        "policy_untrained_best_of_8": evaluate(policy_bon, cases, resolvable),
    }
    result = {
        "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": SEED, "source": os.path.basename(args.candidates),
        "n_cases": len(cases),
        "note": ("Best-of-N verifier trained on train cases only; the selection arm is "
                 "therefore optimistic on train cases and honest on the 15 held out."),
        "arms": arms,
    }
    path = os.path.join(OUT, "stageI_baseline.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)

    def g(a, *keys):
        v = arms[a]
        for k in keys:
            v = v[k] if isinstance(v, dict) else v
        return v

    print(f"\n{'metric':38} {'scaffold':>12} {'policy x1':>12} {'policy BoN':>12}")
    rows = [("guideline concordance top-1", ("guideline_concordance", "top1", "rate")),
            ("guideline concordance any-match", ("guideline_concordance", "any_match", "rate")),
            ("molecular interp. agreement", ("molecular_interpretation_accuracy", "agreement", "rate")),
            ("step soundness", ("reasoning_step_accuracy", "step_soundness", "rate")),
            ("citation grounding", ("citation_grounding", "resolved", "rate")),
            ("calibration ECE", ("calibration", "ece")),
            ("abstained", ("n_abstained",))]
    for label, keys in rows:
        vals = []
        for a in arms:
            try:
                v = g(a, *keys)
                vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
            except Exception:
                vals.append("n/a")
        print(f"{label:38} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")
    print("\nwrote", path)


if __name__ == "__main__":
    main()

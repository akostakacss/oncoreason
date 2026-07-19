#!/usr/bin/env python3
"""End-to-end run: cases -> traces -> step+outcome labels -> PRM -> evaluation.

Usage:  python scripts/run_pipeline.py [--limit N]

This is the one command that exercises Phases 2 through 6 together and writes every number
the write-up reports. Deterministic (seed 17). Saves to results/<ts>-pipeline.{json,md}
plus results/INDEX.tsv, mirroring the runs/ and qc/ archive policy.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

from oncoreason.agents import DeterministicLLM, Orchestrator
from oncoreason.agents.guideline_index import index_docs
from oncoreason.cases.schema import Alteration, Case, ClinicalContext
from oncoreason.datasources import get_source
from oncoreason.evaluation import (
    calibration,
    citation_grounding,
    deferral_curve,
    guideline_concordance,
    information_gathering,
    molecular_interpretation_accuracy,
    reasoning_step_accuracy,
    tool_use_reliability,
)
from oncoreason.retrieval.base import BM25Retriever
from oncoreason.supervision import (
    annotate_trace_with_labels,
    build_prm_examples,
    label_case_outcome,
    label_trace,
    mine_negatives,
)
from oncoreason.training import (
    PRMConfig,
    best_of_n,
    build_dpo_pairs,
    reward_hacking_report,
    score_trace_with_prm,
    train_prm,
)

SEED = 17
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results")


def load_cases(limit: int) -> list[Case]:
    path = os.path.join(REPO, "data", "cases", "cases.jsonl")
    cases = []
    with open(path) as f:
        for line in f:
            if len(cases) >= limit:
                break
            d = json.loads(line)
            alts = [Alteration(a["gene"], a["variant"], a.get("kind", "mutation"),
                               a.get("is_somatic", True), a.get("escat_tier"))
                    for a in d["alterations"]]
            cases.append(Case(d["case_id"], alts, ClinicalContext(**d["context"]),
                              provenance=d.get("provenance", {})))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    guideline = BM25Retriever(source="esmo_index")
    guideline.index(index_docs())

    # -- Phase 2: cases, + Phase 4.4 outcome labels ------------------------------
    print("loading cases and deriving guideline outcome labels...")
    raw = load_cases(args.limit)
    cases = []
    for c in raw:
        gold = label_case_outcome(c, guideline)
        cases.append(Case(c.case_id, c.alterations, c.context, gold=gold,
                          split=c.split, provenance=c.provenance))
    n_gold = sum(1 for c in cases if c.gold)
    print(f"  {len(cases)} cases, {n_gold} with guideline-derived gold")

    # -- Phase 3: run the scaffold -----------------------------------------------
    print("running the agent scaffold...")
    orch = Orchestrator(llm=DeterministicLLM(),
                        sources={"civic": get_source("civic"), "clinvar": get_source("clinvar")},
                        guideline_retriever=guideline)
    traces = [orch.run(c) for c in cases]

    # -- Phase 4: step labels -----------------------------------------------------
    print("labelling reasoning steps (guideline-verified)...")
    prm_examples = []
    for t in traces:
        labels = label_trace(t)
        annotate_trace_with_labels(t, labels)
        prm_examples += build_prm_examples(t)
    n_sound = sum(1 for e in prm_examples if e["label_sound"])
    print(f"  {len(prm_examples)} step examples, {n_sound} sound "
          f"({n_sound / max(1, len(prm_examples)):.1%})")

    # The deterministic scaffold cannot emit an ungrounded step, so the label distribution is
    # degenerate and a verifier would have nothing to learn. Mine counterfactual negatives.
    negatives = mine_negatives(prm_examples, seed=SEED)
    prm_examples = prm_examples + negatives
    print(f"  + {len(negatives)} counterfactual negatives (synthetic) "
          f"-> {len(prm_examples)} total")

    # -- Phase 5: PRM + verifier-guided selection --------------------------------
    print("training the process reward model...")
    prm, report = train_prm(prm_examples, PRMConfig(seed=SEED),
                            out_dir=os.path.join(OUT, "prm"))
    print(f"  held-out accuracy {report.accuracy:.3f} on {report.n_test_cases} unseen cases "
          f"(T={report.temperature:.2f})")

    trace_scores = {t.case_id: score_trace_with_prm(prm, t) for t in traces}
    # Best-of-N needs >1 candidate per case; with one deterministic trace each I report the
    # machinery and the score distribution rather than a fabricated N-sample comparison.
    bo_best, bo_scores = best_of_n(prm, traces)
    pairs = build_dpo_pairs(prm, {"all": traces})
    hacking = reward_hacking_report(pairs)

    # -- Phase 6: evaluation ------------------------------------------------------
    print("running the clinical evaluation harness...")
    resolvable = {c.citation_id for t in traces for c in t.all_citations()}
    metrics = {
        "guideline_concordance": guideline_concordance(traces, cases),
        "molecular_interpretation_accuracy": molecular_interpretation_accuracy(traces, cases),
        "reasoning_step_accuracy": reasoning_step_accuracy(traces),
        "tool_use_reliability": tool_use_reliability(traces),
        "citation_grounding": citation_grounding(traces, resolver=lambda cid: cid in resolvable),
        "calibration": calibration(traces, cases),
        "deferral_curve": deferral_curve(traces, cases),
        "information_gathering": information_gathering(traces, cases),
    }

    result = {
        "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "n_cases": len(cases),
        "n_cases_with_gold": n_gold,
        "n_traces": len(traces),
        "n_abstained": sum(1 for t in traces if t.abstained),
        "step_labels": {"real_sound": n_sound, "synthetic_negatives": len(negatives),
                        "total": len(prm_examples)},
        "prm": report.summary(),
        "prm_trace_scores": {
            "min": round(min(trace_scores.values()), 4),
            "max": round(max(trace_scores.values()), 4),
            "mean": round(sum(trace_scores.values()) / len(trace_scores), 4),
        },
        "best_of_n_demo": {"selected": bo_best.case_id,
                           "n_candidates": len(bo_scores)},
        "dpo_pairs": len(pairs),
        "reward_hacking_check": hacking,
        "metrics": metrics,
    }

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    js = os.path.join(OUT, f"{ts}-pipeline.json")
    with open(js, "w") as f:
        json.dump(result, f, indent=1, default=str)

    index = os.path.join(OUT, "INDEX.tsv")
    new = not os.path.exists(index)
    with open(index, "a") as f:
        if new:
            f.write("timestamp\tn_cases\tprm_acc\tconcordance_top1\tstep_soundness\n")
        f.write("\t".join(map(str, [
            ts, len(cases), round(report.accuracy, 3),
            metrics["guideline_concordance"]["top1"]["rate"],
            metrics["reasoning_step_accuracy"]["step_soundness"]["rate"],
        ])) + "\n")

    print(json.dumps(result, indent=1, default=str))
    print(f"\nsaved: {js}\nindexed: {index}")


if __name__ == "__main__":
    main()

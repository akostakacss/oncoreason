#!/usr/bin/env python3
"""Stage C — does a verifier trained on counterfactual negatives transfer to real ones?

Phase 5 reports 0.914 held-out accuracy for a PRM trained on *constructed* negatives
(`supervision.mine_negatives`: citations stripped, or swapped in from another case), and
states plainly that its accuracy on real policy hallucinations is unmeasured. Stage B
produced those real hallucinations. This script measures the transfer.

Both arms are trained and tested on the **same case-level split**, so the only thing that
differs is the training distribution and the two numbers are directly comparable:

  arm A  train on SYNTHETIC negatives (train cases)  ->  test on REAL steps (test cases)
  arm B  train on REAL steps          (train cases)  ->  test on REAL steps (test cases)

Arm A is the number Phase 5 could not report. Arm B is the ceiling for this feature set.

Usage:  python scripts/stage_c_prm_real.py <candidates-relabelled.jsonl>
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from oncoreason.agents import DeterministicLLM, Orchestrator          # noqa: E402
from oncoreason.agents.guideline_index import index_docs              # noqa: E402
from oncoreason.cases.schema import Alteration, Case, ClinicalContext  # noqa: E402
from oncoreason.datasources import get_source                         # noqa: E402
from oncoreason.retrieval.base import BM25Retriever                   # noqa: E402
from oncoreason.supervision import (annotate_trace_with_labels,        # noqa: E402
                                    build_prm_examples, label_trace, mine_negatives)
from oncoreason.training import PRM, PRMConfig, split_by_case          # noqa: E402

SEED = 17
OUT = os.path.join(REPO, "results", "gpu_stage")


def confusion(model: PRM, examples: list[dict]) -> dict:
    probs = model.predict_proba(examples)
    y = [1 if e["label_sound"] else 0 for e in examples]
    tp = sum(1 for p, t in zip(probs, y) if p >= 0.5 and t == 1)
    fp = sum(1 for p, t in zip(probs, y) if p >= 0.5 and t == 0)
    tn = sum(1 for p, t in zip(probs, y) if p < 0.5 and t == 0)
    fn = sum(1 for p, t in zip(probs, y) if p < 0.5 and t == 1)
    n = len(examples)
    # Accuracy alone is misleading when the classes are skewed, so report balanced
    # accuracy alongside it: the mean of per-class recall.
    rec_pos = tp / (tp + fn) if (tp + fn) else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) else 0.0
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": round((tp + tn) / n, 3) if n else 0.0,
            "balanced_accuracy": round((rec_pos + rec_neg) / 2, 3),
            "recall_sound": round(rec_pos, 3), "recall_unsound": round(rec_neg, 3),
            "positive_rate": round(sum(y) / n, 3) if n else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("--with-evidence-text", action="store_true",
                    help="inline what each cited record says, not just its id")
    ap.add_argument("-o", "--out", default="stageC_prm_transfer.json")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from relabel_candidates import build_evidence_pools                # noqa: E402
    pools, _ = build_evidence_pools()
    claims = {cid: txt for p in pools.values() for cid, txt in p.items()}

    def with_text(e: dict) -> dict:
        """Attach the cited records' text so the model can judge support, not just presence."""
        if args.with_evidence_text:
            e["evidence_text"] = [claims[c] for c in e.get("evidence_ids") or [] if c in claims]
        return e

    # -- real examples, from the sampled policy traces ---------------------------
    real: list[dict] = []
    for line in open(args.candidates):
        r = json.loads(line)
        for s in r["steps"]:
            real.append(with_text({
                "case_id": r["case_id"], "step_text": s["text"],
                "evidence_ids": s["cited"], "label_sound": bool(s["label_sound"])}))

    # -- synthetic examples, exactly as the CPU pipeline builds them --------------
    guideline = BM25Retriever(source="esmo_index")
    guideline.index(index_docs())
    cases = []
    with open(os.path.join(REPO, "data", "cases", "cases.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            alts = [Alteration(a["gene"], a["variant"], a.get("kind", "mutation"),
                               a.get("is_somatic", True), a.get("escat_tier"))
                    for a in d["alterations"]]
            cases.append(Case(d["case_id"], alts, ClinicalContext(**d["context"])))

    orch = Orchestrator(llm=DeterministicLLM(),
                        sources={"civic": get_source("civic"), "clinvar": get_source("clinvar")},
                        guideline_retriever=guideline)
    synthetic: list[dict] = []
    for c in cases:
        t = orch.run(c)
        annotate_trace_with_labels(t, label_trace(t))
        synthetic += build_prm_examples(t)
    synthetic += mine_negatives(synthetic, seed=SEED)
    synthetic = [with_text(e) for e in synthetic]

    # -- one shared case-level split, so the two arms are comparable --------------
    _, _, test_cases = split_by_case(real, test_frac=0.3, seed=SEED)
    real_train = [e for e in real if e["case_id"] not in test_cases]
    real_test = [e for e in real if e["case_id"] in test_cases]
    syn_train = [e for e in synthetic if e["case_id"] not in test_cases]

    print(f"cases: {len(cases)} total, {len(test_cases)} held out")
    print(f"real:      {len(real_train)} train / {len(real_test)} test steps")
    print(f"synthetic: {len(syn_train)} train steps")

    arm_a = PRM(PRMConfig(seed=SEED)).fit(syn_train)
    arm_b = PRM(PRMConfig(seed=SEED)).fit(real_train)

    results = {
        "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "source": os.path.basename(args.candidates),
        "n_test_cases": len(test_cases),
        "n_real_train": len(real_train), "n_real_test": len(real_test),
        "n_synthetic_train": len(syn_train),
        "arm_a_synthetic_to_real": confusion(arm_a, real_test),
        "arm_b_real_to_real": confusion(arm_b, real_test),
        "arm_a_on_synthetic_holdout": confusion(
            arm_a, [e for e in synthetic if e["case_id"] in test_cases]),
    }

    results["features"] = "step + ids + evidence text" if args.with_evidence_text \
        else "step + ids only"
    path = os.path.join(OUT, args.out)
    with open(path, "w") as f:
        json.dump(results, f, indent=1)

    for k in ("arm_a_on_synthetic_holdout", "arm_a_synthetic_to_real", "arm_b_real_to_real"):
        m = results[k]
        print(f"\n{k}")
        print(f"  accuracy {m['accuracy']}  balanced {m['balanced_accuracy']}  "
              f"(TP {m['tp']} FP {m['fp']} TN {m['tn']} FN {m['fn']}, pos rate {m['positive_rate']})")
    print("\nwrote", path)


if __name__ == "__main__":
    main()

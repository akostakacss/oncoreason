#!/usr/bin/env python3
"""Stage H — does verifier-guided selection actually pick better traces?

Phase 5's second honest limitation: *"Best-of-N is implemented and tested but not meaningfully
demonstrated: the deterministic scaffold produces one trace per case, so there is nothing to
choose between."* Stage E produced 8 candidates per case, and stage F produced a verifier that
clears the decision gate. This closes that limitation with a measured number.

The design keeps the verifier's own score out of the outcome metric. Trace **quality** is the
fraction of steps the semantic labeller judges sound; the verifier only decides *which* trace
gets picked. Scoring selection with the same signal that drove selection would be circular, and
would report the verifier's confidence rather than its usefulness.

Three arms on held-out cases the verifier never saw:

  random     mean quality over all N candidates  — what one sample gets you in expectation
  best-of-N  quality of the candidate the verifier ranked highest
  oracle     quality of the best candidate available — the ceiling selection could reach

`best_of_n` and `score_trace_with_prm` are imported from the project rather than reimplemented,
so this exercises the shipped code path.

Usage:  python scripts/stage_h_best_of_n.py <candidates.jsonl>
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from oncoreason.agents.trace import Citation, ReasoningStep, Trace   # noqa: E402
from oncoreason.training import PRM, PRMConfig, best_of_n, split_by_case  # noqa: E402
from relabel_candidates import build_evidence_pools, normalise       # noqa: E402
from semantic_labels import build_record_index, label_step           # noqa: E402

SEED = 17
OUT = os.path.join(REPO, "results", "gpu_stage")


def load(path: str):
    """Return (examples for PRM training, traces grouped by case, quality per trace)."""
    pools, global_ids = build_evidence_pools()
    claims = {cid: txt for p in pools.values() for cid, txt in p.items()}
    records, vocab, tumors = build_record_index()

    examples: list[dict] = []
    by_case: dict[str, list] = {}
    quality: dict[tuple[str, int], float] = {}

    for line in open(path):
        r = json.loads(line)
        cid, tumor = r["case_id"], tumors.get(r["case_id"], "lung")
        pool = pools.get(cid, {})
        steps, sound_flags = [], []

        for i, s in enumerate(r["steps"]):
            resolved, invented, prose = normalise(", ".join(s.get("cited") or []), global_ids)
            off = [c for c in resolved if c not in pool]
            structural = bool(resolved) and not invented and not off
            sem = label_step(s["text"], resolved, records, tumor, vocab)
            sound = structural and sem["label_semantic_sound"] and not sem["disease_mismatch"]
            sound_flags.append(sound)

            examples.append({"case_id": cid, "step_text": s["text"],
                             "evidence_ids": resolved,
                             "evidence_text": [claims[c] for c in resolved if c in claims],
                             "label_sound": sound})
            steps.append(ReasoningStep(
                index=i, text=s["text"],
                citations=[Citation(citation_id=c, source=c.split(":")[0],
                                    claim=claims.get(c, "")) for c in resolved]))

        trace = Trace(case_id=cid, steps=steps,
                      recommendation=r.get("recommendation") or [],
                      confidence=r.get("confidence"), abstained=bool(r.get("abstained")))
        by_case.setdefault(cid, []).append(trace)
        quality[(cid, len(by_case[cid]) - 1)] = (
            statistics.fmean(sound_flags) if sound_flags else 0.0)

    return examples, by_case, quality


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    examples, by_case, quality = load(args.candidates)
    train, _, test_cases = split_by_case(examples, test_frac=0.3, seed=SEED)
    print(f"{len(by_case)} cases, {len(test_cases)} held out | "
          f"{len(train)} train steps")

    # TF-IDF backend: this runs on CPU, and stage F put it at 0.849 [0.768, 0.921] against
    # ModernBERT's 0.881 [0.798, 0.951] — statistically indistinguishable there, so the
    # selection quality measured here is not meaningfully limited by the choice.
    prm = PRM(PRMConfig(seed=SEED)).fit(train)

    rows = []
    for cid in sorted(test_cases):
        traces = by_case[cid]
        if len(traces) < 2:
            continue
        q = [quality[(cid, i)] for i in range(len(traces))]
        best, scores = best_of_n(prm, traces)
        picked = traces.index(best)
        rows.append({"case_id": cid, "n": len(traces),
                     "random": statistics.fmean(q),
                     "best_of_n": q[picked],
                     "oracle": max(q),
                     "worst": min(q),
                     "picked": picked, "prm_score": round(scores[picked], 3)})

    def boot(key_a: str, key_b: str, n_boot: int = 1000):
        """Bootstrap the paired delta over cases (the unit of independence)."""
        rng = random.Random(SEED)
        out = []
        for _ in range(n_boot):
            pick = [rng.choice(rows) for _ in rows]
            out.append(statistics.fmean(r[key_a] - r[key_b] for r in pick))
        out.sort()
        return round(out[int(0.025 * n_boot)], 3), round(out[int(0.975 * n_boot)], 3)

    arms = {k: round(statistics.fmean(r[k] for r in rows), 3)
            for k in ("worst", "random", "best_of_n", "oracle")}
    lift_lo, lift_hi = boot("best_of_n", "random")
    gap_lo, gap_hi = boot("oracle", "best_of_n")

    result = {
        "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "source": os.path.basename(args.candidates),
        "n_test_cases": len(rows),
        "n_candidates_per_case": rows[0]["n"] if rows else 0,
        "quality_metric": "fraction of steps judged sound by the semantic labeller",
        "backend": "tfidf",
        "arms": arms,
        "lift_best_of_n_over_random": round(arms["best_of_n"] - arms["random"], 3),
        "lift_ci95": [lift_lo, lift_hi],
        "remaining_gap_to_oracle": round(arms["oracle"] - arms["best_of_n"], 3),
        "gap_ci95": [gap_lo, gap_hi],
        "per_case": rows,
    }
    path = os.path.join(OUT, "stageH_best_of_n.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)

    print(f"\n  worst candidate   {arms['worst']:.3f}")
    print(f"  random (mean)     {arms['random']:.3f}")
    print(f"  best-of-N         {arms['best_of_n']:.3f}")
    print(f"  oracle (max)      {arms['oracle']:.3f}")
    print(f"\n  lift over random  {result['lift_best_of_n_over_random']:+.3f} "
          f"[{lift_lo:+.3f}, {lift_hi:+.3f}]")
    print(f"  gap to oracle     {result['remaining_gap_to_oracle']:+.3f} "
          f"[{gap_lo:+.3f}, {gap_hi:+.3f}]")
    print("\nwrote", path)


if __name__ == "__main__":
    main()

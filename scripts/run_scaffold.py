#!/usr/bin/env python3
"""Drive the Phase-3 scaffolding on cases and save auditable traces under traces/.

Usage:
  python scripts/run_scaffold.py                # 3 canonical demo cases (incl. an abstain)
  python scripts/run_scaffold.py --from-cases 5 # first 5 real cases from data/cases/cases.jsonl

Evidence lookups use the on-disk CIViC/ClinVar cache when present and degrade gracefully
(a failed tool call is logged, the run still completes). Writes traces/<ts>-scaffold.jsonl
(machine) + .md (readable) + INDEX.tsv, mirroring the runs/ and qc/ archive policy.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os

from oncoreason.agents import DeterministicLLM, Orchestrator
from oncoreason.agents.guideline_index import index_docs
from oncoreason.cases.schema import Alteration, Case, ClinicalContext
from oncoreason.datasources import get_source
from oncoreason.retrieval.base import BM25Retriever

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "traces")


def demo_cases() -> list[Case]:
    luad = ClinicalContext(tumor_type="lung adenocarcinoma", stage="IV")
    return [
        Case("demo:egfr-l858r", [Alteration("EGFR", "L858R")], luad),
        Case("demo:kras-g12c", [Alteration("KRAS", "G12C")], luad),
        # a driver with no targeted option -> should fall back to chemo/IO, not hallucinate a
        # targeted drug (the true abstain/defer path is exercised in tests/test_agents.py)
        Case("demo:tp53-nodriver", [Alteration("TP53", "R175H")], luad),
    ]


def load_cases(limit: int) -> list[Case]:
    path = os.path.join(REPO, "data", "cases", "cases.jsonl")
    if not os.path.exists(path):
        print(f"(no {path}; using demo cases)")
        return demo_cases()
    cases: list[Case] = []
    with open(path) as f:
        for line in f:
            if len(cases) >= limit:
                break
            d = json.loads(line)
            alts = [Alteration(a["gene"], a["variant"], a.get("kind", "mutation"),
                               a.get("is_somatic", True), a.get("escat_tier"))
                    for a in d["alterations"]]
            ctx = ClinicalContext(**{k: v for k, v in d["context"].items()})
            cases.append(Case(d["case_id"], alts, ctx, provenance=d.get("provenance", {})))
    return cases


def render_md(traces: list) -> str:
    out = ["# Scaffold run — traces\n"]
    for t in traces:
        out.append(f"\n## {t.case_id}  ·  model `{t.model}`")
        verdict = "ABSTAIN (defer)" if t.abstained else "recommend"
        out.append(f"- **{verdict}**: {t.recommendation or '—'}  ·  confidence {t.confidence}")
        n_calls = sum(len(s.tool_calls) for s in t.steps)
        n_fail = sum(1 for s in t.steps for c in s.tool_calls if not c.ok)
        out.append(f"- steps: {len(t.steps)}  ·  tool calls: {n_calls} ({n_fail} failed)  ·  "
                   f"citations: {len(t.all_citations())}")
        for s in t.steps:
            out.append(f"  {s.index}. {s.text}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cases", type=int, default=0,
                    help="run first N real cases from data/cases/cases.jsonl (0 = demo cases)")
    ap.add_argument("--abstain-threshold", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cases = load_cases(args.from_cases) if args.from_cases else demo_cases()

    guideline = BM25Retriever(source="esmo_index")
    guideline.index(index_docs())
    orch = Orchestrator(
        llm=DeterministicLLM(),
        sources={"civic": get_source("civic"), "clinvar": get_source("clinvar")},
        guideline_retriever=guideline,
        abstain_threshold=args.abstain_threshold,
    )

    traces = [orch.run(c) for c in cases]
    print(render_md(traces))

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    jl = os.path.join(OUT, f"{ts}-scaffold.jsonl")
    with open(jl, "w") as f:
        for t in traces:
            f.write(json.dumps(dataclasses.asdict(t)) + "\n")
    md = os.path.join(OUT, f"{ts}-scaffold.md")
    with open(md, "w") as f:
        f.write(render_md(traces))

    index = os.path.join(OUT, "INDEX.tsv")
    new = not os.path.exists(index)
    with open(index, "a") as f:
        if new:
            f.write("timestamp\tn_cases\tn_abstain\tsource\n")
        src = f"cases:{args.from_cases}" if args.from_cases else "demo"
        f.write(f"{ts}\t{len(traces)}\t{sum(t.abstained for t in traces)}\t{src}\n")

    print(f"saved: {jl}\nsaved: {md}\nindexed: {index}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Re-label sampled policy traces with a citation parser that does not over-count.

The in-notebook labeller matched cited strings against evidence ids exactly and split them
on commas only. Inspecting the first sampling run showed that inflates the hallucination
count roughly threefold, in three distinct ways:

  - **prefix dropped**  the policy writes `gl-ret` for `guideline:gl-ret`. Real evidence,
    scored as invented.
  - **delimiter**       `civic:EID2219 and civic:EID2232`, `a; b`, `a & b`. Both ids real,
    the pair scored as one invented string.
  - **prose**           `all guideline evidence`, `no specific evidence found`. A meta-
    statement, not a fabricated record — closer to citing nothing than to hallucinating.

Only what survives all three is a genuine hallucination: a well-formed id that resolves
against nothing. Usage:

    python scripts/relabel_candidates.py <candidates.jsonl> [-o <out.jsonl>]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

# a citation is a source-prefixed token; anything with whitespace is prose, not an id
ID_RE = re.compile(r"^[A-Za-z_]+:[A-Za-z0-9_.\-]+$")
BARE_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
SPLIT_RE = re.compile(r"\s*(?:,|;|&|\band\b)\s*", re.IGNORECASE)


def build_evidence_pools() -> tuple[dict[str, dict[str, str]], set[str]]:
    """Rebuild each case's evidence pool exactly as the sampling notebook did (cache-backed)."""
    from oncoreason.agents.guideline_index import index_docs
    from oncoreason.agents.tools import (evidence_citations, guideline_citations,
                                         guideline_lookup, variant_lookup)
    from oncoreason.datasources import get_source
    from oncoreason.retrieval.base import BM25Retriever

    guideline = BM25Retriever(source="esmo_index")
    guideline.index(index_docs())
    civic, clinvar = get_source("civic"), get_source("clinvar")

    pools: dict[str, dict[str, str]] = {}
    with open(os.path.join(REPO, "data", "cases", "cases.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            tumor = d["context"].get("tumor_type", "lung")
            pool: dict[str, str] = {}
            for a in d["alterations"]:
                ev, _ = variant_lookup(a["gene"], a["variant"], tumor, civic, clinvar)
                for c in evidence_citations(ev):
                    pool[c.citation_id] = c.claim
                chunks, _ = guideline_lookup(a["gene"], tumor, guideline)
                for c in guideline_citations(chunks):
                    pool[c.citation_id] = c.claim
            pools[d["case_id"]] = pool
    return pools, {cid for p in pools.values() for cid in p}


def normalise(raw: str, global_ids: set[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (resolved_ids, invented_ids, prose_fragments) for one raw CITE payload."""
    resolved, invented, prose = [], [], []
    # index bare ids (suffix after the prefix) so a dropped prefix can be recovered
    by_suffix: dict[str, str] = {}
    for gid in global_ids:
        by_suffix.setdefault(gid.split(":", 1)[1], gid)

    for tok in SPLIT_RE.split(raw):
        tok = tok.strip().strip(".")
        if not tok:
            continue
        if tok in global_ids:
            resolved.append(tok)
        elif BARE_RE.match(tok) and tok in by_suffix:
            resolved.append(by_suffix[tok])          # prefix dropped by the policy
        elif ID_RE.match(tok) or BARE_RE.match(tok):
            invented.append(tok)                     # well-formed but resolves to nothing
        else:
            prose.append(tok)                        # free text where an id belonged
    return resolved, invented, prose


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    out_path = args.out or args.candidates.replace(".jsonl", "-relabelled.jsonl")

    pools, global_ids = build_evidence_pools()
    rows = [json.loads(l) for l in open(args.candidates)]

    stats = collections.Counter()
    invented_strings: collections.Counter = collections.Counter()

    for r in rows:
        pool = pools.get(r["case_id"], {})
        for s in r["steps"]:
            raw = ", ".join(s.get("cited") or [])
            resolved, invented, prose = normalise(raw, global_ids)
            off_case = [c for c in resolved if c not in pool]
            s["cited"] = resolved
            s["invented"] = invented
            s["prose"] = prose
            s["off_case"] = off_case
            # A step is sound when every citation resolves, belongs to this case, and at
            # least one is present. Prose alone counts as uncited, not as fabrication.
            s["label_sound"] = bool(resolved) and not invented and not off_case
            s.pop("unresolvable", None)

            stats["steps"] += 1
            stats["sound"] += s["label_sound"]
            if invented:
                stats["steps_with_invented"] += 1
                invented_strings.update(invented)
            if prose:
                stats["steps_with_prose"] += 1
            if off_case:
                stats["steps_off_case"] += 1
            if not resolved and not invented:
                stats["steps_uncited"] += 1

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = stats["steps"]
    summary = {
        "n_samples": len(rows),
        "n_steps": n,
        "sound_rate": round(stats["sound"] / n, 3),
        "unsound_rate": round(1 - stats["sound"] / n, 3),
        "steps_uncited": stats["steps_uncited"],
        "steps_with_invented": stats["steps_with_invented"],
        "steps_with_prose": stats["steps_with_prose"],
        "steps_off_case": stats["steps_off_case"],
        "distinct_invented_ids": len(invented_strings),
        "top_invented": invented_strings.most_common(10),
    }
    print(json.dumps(summary, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main()

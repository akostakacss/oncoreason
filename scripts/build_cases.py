#!/usr/bin/env python3
"""Build the case set from cBioPortal (Phase 2, steps 4-6) and save it durably.

Usage:  python scripts/build_cases.py [--per-study N] [--no-annotate]

Writes data/cases/cases.jsonl (regenerable) + data/cases/manifest.json (summary/audit).
Deterministic given the seed, so the set is reproducible from this committed script.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
from collections import Counter

from oncoreason.cases.build import assemble_case, build_profiles, split_cases
from oncoreason.datasources import get_source
from oncoreason.datasources.cbioportal import CBioPortalSource

DRIVERS = {
    "EGFR": 1956, "KRAS": 3845, "ALK": 238, "ROS1": 6098, "BRAF": 673, "MET": 4233,
    "RET": 5979, "ERBB2": 2064, "NTRK1": 4914, "PIK3CA": 5290, "TP53": 7157,
    "STK11": 6794, "KEAP1": 9817, "NF1": 4763, "RB1": 5925, "NRAS": 4893,
}
SEED = 17
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "data", "cases")


def case_to_dict(c) -> dict:
    d = dataclasses.asdict(c)
    d["split"] = c.split.value if c.split else None
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-study", type=int, default=25)
    ap.add_argument("--no-annotate", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cbio = CBioPortalSource()
    civic = None if args.no_annotate else get_source("civic")
    clinvar = None if args.no_annotate else get_source("clinvar")

    print("building profiles (QC-filtered, patient-deduped)...")
    profiles = build_profiles(cbio, cbio.allowed_studies, list(DRIVERS.values()),
                              seed=SEED, per_study=args.per_study)
    print(f"  {len(profiles)} profiles")

    print("assembling + annotating cases (CIViC somatic / ClinVar germline)...")
    cases = [assemble_case(p, civic=civic, clinvar=clinvar) for p in profiles]
    cases = split_cases(cases, seed=SEED)

    # save
    jsonl = os.path.join(OUT, "cases.jsonl")
    with open(jsonl, "w") as f:
        for c in cases:
            f.write(json.dumps(case_to_dict(c)) + "\n")

    n_alt = sum(len(c.alterations) for c in cases)
    n_actionable = sum(1 for c in cases for e in c.provenance.get("evidence", {}).values()
                       if e.get("actionable"))
    manifest = {
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "studies": list(cbio.allowed_studies),
        "genes": list(DRIVERS),
        "annotated": not args.no_annotate,
        "n_cases": len(cases),
        "n_alterations": n_alt,
        "n_actionable_alterations": n_actionable,
        "by_split": dict(Counter(c.split.value for c in cases)),
        "by_study": dict(Counter(c.provenance["cbioportal_study"] for c in cases)),
        "cases_with_qc_flags": sum(1 for c in cases if c.provenance.get("qc_flags")),
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    print(json.dumps(manifest, indent=1))
    print(f"\nsaved: {jsonl}\nsaved: {os.path.join(OUT, 'manifest.json')}")


if __name__ == "__main__":
    main()

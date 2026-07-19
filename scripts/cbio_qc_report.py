#!/usr/bin/env python3
"""Run the cBioPortal QC battery on the allow-listed lung studies and save a durable,
searchable report under qc/ (mirrors the runs/ audit policy).

Usage:  python scripts/cbio_qc_report.py
"""
from __future__ import annotations

import datetime
import json
import os
from collections import Counter

from oncoreason.cases.qc import Severity, canonical_build, run_qc
from oncoreason.datasources.cbioportal import CBioPortalSource

# lung actionable + common co-drivers (Entrez ids) — the genes cases are built around
DRIVERS = {
    "EGFR": 1956, "KRAS": 3845, "ALK": 238, "ROS1": 6098, "BRAF": 673, "MET": 4233,
    "RET": 5979, "ERBB2": 2064, "NTRK1": 4914, "PIK3CA": 5290, "TP53": 7157,
    "STK11": 6794, "KEAP1": 9817, "NF1": 4763, "RB1": 5925, "NRAS": 4893,
}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QC = os.path.join(REPO, "qc")
os.makedirs(QC, exist_ok=True)


def main() -> None:
    src = CBioPortalSource()
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    reports, lines = [], []
    lines.append(f"# cBioPortal QC report — {ts}\n")
    lines.append(f"Genes: {', '.join(DRIVERS)}\n")

    for study in src.allowed_studies:
        recs = src.fetch_mutations(study, list(DRIVERS.values()))
        rep = run_qc(recs, study)
        reports.append(rep)
        builds = sorted({canonical_build(r["build"]) for r in recs} - {None})
        n_pat = len({r["patient_id"] for r in recs})
        sev = Counter(f.severity.value for f in rep.findings)

        lines.append(f"\n## {study}")
        lines.append(f"- records: {len(recs)}  ·  patients: {n_pat}  ·  builds: {builds}")
        lines.append(f"- findings: {len(rep.findings)}  "
                     f"(high {sev.get('high',0)}, medium {sev.get('medium',0)}, low {sev.get('low',0)})")
        by_check = Counter(f.check for f in rep.findings)
        for check, n in by_check.most_common():
            ex = next(f for f in rep.findings if f.check == check)
            lines.append(f"  - **{check}** ×{n} [{ex.severity.value}] e.g. {ex.detail}"
                         + (f" (sample {ex.sample_id})" if ex.sample_id else ""))

    md = os.path.join(QC, f"{ts}-cbio-qc.md")
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    js = os.path.join(QC, f"{ts}-cbio-qc.json")
    with open(js, "w") as f:
        json.dump([{**r.summary(),
                    "findings": [f.__dict__ | {"severity": f.severity.value} for f in r.findings]}
                   for r in reports], f, indent=1, default=str)

    index = os.path.join(QC, "INDEX.tsv")
    new = not os.path.exists(index)
    with open(index, "a") as f:
        if new:
            f.write("timestamp\tstudy\trecords\tfindings\thigh\tmedium\tlow\n")
        for r in reports:
            s = Counter(x.severity.value for x in r.findings)
            f.write("\t".join(map(str, [ts, r.study_id, r.n_records, len(r.findings),
                                        s.get("high", 0), s.get("medium", 0), s.get("low", 0)])) + "\n")

    print("\n".join(lines))
    print(f"\nsaved: {md}\nsaved: {js}\nindexed: {index}")


if __name__ == "__main__":
    main()

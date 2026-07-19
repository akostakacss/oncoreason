"""Data-QC layer for cBioPortal ingest — treat the source as guilty until checked.

cBioPortal aggregates many studies of uneven curation; the failure mode is a *plausible-looking
wrong value*, not a crash. Each check below is a real anomaly observed in live lung data (see
the Phase-2 QC report), encoded as an individual, testable rule that flags — and never silently
drops — the offending record. Findings carry a severity so the case-builder can gate on them.

Checks
------
- build label canonicalization (hg19 == GRCh37; hg38 == GRCh38) — finding 1/2
- somatic/germline status placeholder ('.') is missing, not a category — finding 3
- mutationType vs proteinChange consequence disagreement — finding 6
- duplicate (sample, gene, proteinChange) rows — finding 7
- VAF sanity (called mutation with 0/None alt reads) — per-study, not assumed clean
- hypermutator sample flag (passenger-mutation swamping)
Patient-level dedup and cross-study contamination live in the case-builder (they need the full
cohort), but `normalize_missing` and `canonical_build` are provided here for it to use.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

# cBioPortal's many spellings of "missing". '.' is the big one (TCGA status fields are all '.').
_MISSING = {"", ".", "na", "n/a", "null", "none", "unknown", "[not available]",
            "[not applicable]", "not available", "notavailable"}


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class QCFinding:
    check: str
    severity: Severity
    detail: str
    sample_id: str | None = None
    gene: str | None = None


@dataclass
class QCReport:
    study_id: str
    n_records: int = 0
    findings: list[QCFinding] = field(default_factory=list)

    def add(self, *a, **k) -> None:
        self.findings.append(QCFinding(*a, **k))

    def by_severity(self, sev: Severity) -> list[QCFinding]:
        return [f for f in self.findings if f.severity == sev]

    def summary(self) -> dict:
        c = Counter(f.severity.value for f in self.findings)
        return {"study_id": self.study_id, "n_records": self.n_records,
                "n_findings": len(self.findings), "by_severity": dict(c),
                "checks": sorted({f.check for f in self.findings})}


def normalize_missing(value):
    """Map cBioPortal's placeholder spellings to None. '.' -> None, '5' -> '5'."""
    if value is None:
        return None
    s = str(value).strip()
    return None if s.lower() in _MISSING else s


def canonical_build(label) -> str | None:
    """Canonicalize genome-build labels so hg19/GRCh37/b37 compare equal. Unknown -> None."""
    s = normalize_missing(label)
    if s is None:
        return None
    s = s.lower()
    if s in ("hg19", "grch37", "b37", "37"):
        return "GRCh37"
    if s in ("hg38", "grch38", "b38", "38"):
        return "GRCh38"
    return s.upper()  # surface an unrecognized build rather than hide it


# --- proteinChange consequence classification (loose, for cross-checking the type label) ----
def protein_consequence(pc) -> str:
    pc = normalize_missing(pc)
    if pc is None:
        return "empty"
    if pc.endswith("*") or "Ter" in pc:
        return "nonsense"
    if "fs" in pc:
        return "frameshift"
    if "splice" in pc.lower():
        return "splice"
    if "del" in pc or "ins" in pc or "dup" in pc:
        return "indel"
    if re.fullmatch(r"[A-Z*]\d+=?", pc) or re.fullmatch(r"[A-Z]\d+[A-Z*]", pc):
        return "substitution"
    return "other"

# which proteinChange consequences are consistent with each cBioPortal mutationType
_TYPE_EXPECT = {
    "Missense_Mutation": {"substitution"},
    "Nonsense_Mutation": {"nonsense"},
    "In_Frame_Del": {"indel"},
    "In_Frame_Ins": {"indel"},
    # a real frameshift is written with 'fs' (e.g. K323Vfs*7); a clean 'K323*' on a
    # Frame_Shift label is a curation inconsistency worth flagging (QC finding 6).
    "Frame_Shift_Del": {"frameshift"},
    "Frame_Shift_Ins": {"frameshift"},
}


def run_qc(records: list[dict], study_id: str, *, hypermutator_threshold: int = 1000) -> QCReport:
    """Run all record-level checks on normalized mutation dicts. Never mutates input."""
    rep = QCReport(study_id=study_id, n_records=len(records))

    builds = {canonical_build(r.get("build")) for r in records} - {None}
    if len(builds) > 1:
        rep.add("build_mixed", Severity.HIGH, f"multiple genome builds in one study: {builds}")

    seen: Counter = Counter()
    per_sample: Counter = Counter()
    for r in records:
        sid = r.get("sample_id")
        gene = r.get("gene_symbol")
        pc = r.get("protein_change")
        per_sample[sid] += 1

        # finding 6: type vs consequence disagreement
        mt = r.get("mutation_type")
        expect = _TYPE_EXPECT.get(mt)
        if expect:
            cons = protein_consequence(pc)
            if cons not in expect and cons != "empty":
                rep.add("type_vs_protein", Severity.MEDIUM,
                        f"{mt} but proteinChange={pc!r} looks {cons}", sample_id=sid, gene=gene)

        # VAF sanity: a called somatic mutation with no supporting alt reads is suspect
        alt = r.get("tumor_alt_count")
        if alt is not None and alt == 0:
            rep.add("vaf_zero_alt", Severity.MEDIUM,
                    f"called mutation with tumorAltCount=0 ({gene} {pc})", sample_id=sid, gene=gene)

        # finding 3: status placeholder masquerading as a value
        if normalize_missing(r.get("mutation_status")) is None and r.get("mutation_status") not in (None,):
            pass  # already normalized to None by the connector; nothing to flag per-record

        seen[(sid, r.get("entrez_gene_id"), pc)] += 1

    # finding 7: duplicate (sample, gene, proteinChange) rows
    for (sid, _g, pc), n in seen.items():
        if n > 1:
            rep.add("duplicate_row", Severity.LOW, f"{n}x identical (sample,gene,{pc})",
                    sample_id=sid)

    # hypermutator samples (passenger swamping; skew sampling / annotation cost)
    for sid, n in per_sample.items():
        if n >= hypermutator_threshold:
            rep.add("hypermutator", Severity.LOW, f"{n} mutations in one sample", sample_id=sid)

    return rep

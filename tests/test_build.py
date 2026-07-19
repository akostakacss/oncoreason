"""Case-builder: offline. Verifies QC-drop, patient dedup, assembly, and disjoint split."""
from __future__ import annotations

from oncoreason.cases.build import assemble_case, build_profiles, split_cases
from oncoreason.cases.schema import Split


class _FakeCBio:
    """Stands in for the connector: returns canned normalized records per study."""
    allowed_studies = ("studyA", "studyB")

    def __init__(self, recs):
        self._recs = recs

    def fetch_mutations(self, study_id, entrez):
        return [r for r in self._recs if r["study_id"] == study_id]


def _rec(study, sample, patient, gene, entrez, pc, mt="Missense_Mutation", alt=30):
    return dict(study_id=study, sample_id=sample, patient_id=patient, gene_symbol=gene,
                entrez_gene_id=entrez, protein_change=pc, mutation_type=mt, build="GRCh37",
                tumor_alt_count=alt, tumor_ref_count=70, mutation_status=None,
                validation_status=None, chrom="7", start=1, end=1, ref="T", alt_allele="G")


def test_patient_dedup_and_dropped_records():
    recs = [
        _rec("studyA", "S1", "P1", "EGFR", 1956, "L858R"),
        _rec("studyA", "S1", "P1", "TP53", 7157, "R175H"),
        # second sample, SAME patient -> must be deduped away
        _rec("studyA", "S2", "P1", "KRAS", 3845, "G12C"),
        # duplicate row for S3 -> the dup occurrence is dropped, one kept
        _rec("studyA", "S3", "P3", "TP53", 7157, "X126_splice"),
        _rec("studyA", "S3", "P3", "TP53", 7157, "X126_splice"),
        # zero-VAF record for S3 -> dropped
        _rec("studyA", "S3", "P3", "EGFR", 1956, "T790M", alt=0),
    ]
    profs = build_profiles(_FakeCBio(recs), ["studyA"], [1956, 7157, 3845],
                           seed=1, per_study=10)
    by_sample = {p["sample_id"]: p for p in profs}
    assert set(by_sample) == {"S1", "S3"}                 # S2 deduped (same patient as S1)
    s3 = by_sample["S3"]["alterations"]
    pcs = [a["protein_change"] for a in s3]
    assert pcs.count("X126_splice") == 1                  # duplicate collapsed
    assert "T790M" not in pcs                             # zero-VAF dropped
    assert "duplicate_row" in by_sample["S3"]["qc_flags"] # kept as a flag


def test_assemble_and_split_are_patient_disjoint():
    recs = [_rec("studyA", f"S{i}", f"P{i}", "EGFR", 1956, "L858R") for i in range(20)]
    profs = build_profiles(_FakeCBio(recs), ["studyA"], [1956], seed=2, per_study=100)
    cases = [assemble_case(p) for p in profs]             # no annotation (offline)
    assert all(c.gold is None for c in cases)             # unlabelled by design
    assert all(c.alterations[0].is_somatic for c in cases)
    split = split_cases(cases, seed=2)
    # each case is one patient, so splits cannot share a patient
    seen = {}
    for c in split:
        pat = c.provenance["patient_id"]
        assert pat not in seen, "patient leaked across cases"
        seen[pat] = c.split
    assert {s for s in seen.values()} <= set(Split)
    assert any(c.split == Split.GENERALIZATION for c in split)

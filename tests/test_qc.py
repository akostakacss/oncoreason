"""QC layer — pure offline. Each check is exercised on a record that reproduces a real
cBioPortal anomaly from the Phase-2 QC report."""
from __future__ import annotations

from oncoreason.cases.qc import (
    Severity,
    canonical_build,
    normalize_missing,
    protein_consequence,
    run_qc,
)


def test_normalize_missing_handles_cbio_placeholders():
    for junk in [".", "", "NA", "N/A", "[Not Available]", "Unknown", None, "null"]:
        assert normalize_missing(junk) is None
    assert normalize_missing("5") == "5"
    assert normalize_missing("L858R") == "L858R"


def test_canonical_build_unifies_labels():
    assert canonical_build("hg19") == canonical_build("GRCh37") == "GRCh37"
    assert canonical_build("hg38") == canonical_build("GRCh38") == "GRCh38"
    assert canonical_build(".") is None
    assert canonical_build("hs37d5") == "HS37D5"  # unknown surfaced, not hidden


def test_protein_consequence_classes():
    assert protein_consequence("L858R") == "substitution"
    assert protein_consequence("K323*") == "nonsense"
    assert protein_consequence("K323Vfs*7") == "frameshift"
    assert protein_consequence("E746_A750del") == "indel"
    assert protein_consequence("X126_splice") == "splice"
    assert protein_consequence(".") == "empty"


def _rec(**kw):
    base = dict(study_id="s", sample_id="S1", patient_id="P1", gene_symbol="EGFR",
                entrez_gene_id=1956, protein_change="L858R", mutation_type="Missense_Mutation",
                build="GRCh37", tumor_alt_count=30, tumor_ref_count=70,
                mutation_status=None, validation_status=None)
    base.update(kw)
    return base


def test_build_mixed_flagged():
    rep = run_qc([_rec(build="GRCh37"), _rec(build="hg38", sample_id="S2")], "s")
    assert any(f.check == "build_mixed" and f.severity == Severity.HIGH for f in rep.findings)


def test_type_vs_protein_mismatch_flagged():
    # Frame_Shift label but a clean stop-gain proteinChange (the curation error)
    rep = run_qc([_rec(mutation_type="Frame_Shift_Del", protein_change="K323*")], "s")
    hits = [f for f in rep.findings if f.check == "type_vs_protein"]
    assert hits and hits[0].severity == Severity.MEDIUM
    # a legit frameshift notation is NOT flagged
    rep2 = run_qc([_rec(mutation_type="Frame_Shift_Del", protein_change="K323Vfs*7")], "s")
    assert not [f for f in rep2.findings if f.check == "type_vs_protein"]


def test_zero_vaf_flagged():
    rep = run_qc([_rec(tumor_alt_count=0)], "s")
    assert any(f.check == "vaf_zero_alt" for f in rep.findings)


def test_duplicate_rows_flagged():
    dup = _rec(protein_change="X126_splice")
    rep = run_qc([dict(dup), dict(dup)], "s")
    hits = [f for f in rep.findings if f.check == "duplicate_row"]
    assert hits and hits[0].severity == Severity.LOW


def test_hypermutator_flagged():
    recs = [_rec(protein_change=f"A{i}V") for i in range(12)]
    rep = run_qc(recs, "s", hypermutator_threshold=10)
    assert any(f.check == "hypermutator" for f in rep.findings)


def test_clean_records_no_findings():
    rep = run_qc([_rec(), _rec(sample_id="S2", protein_change="T790M")], "s")
    assert rep.findings == []
    assert rep.summary()["n_records"] == 2

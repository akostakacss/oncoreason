"""cBioPortal connector: offline normalization + allow-list guard, plus a live round-trip."""
from __future__ import annotations

import urllib.error

import pytest

from oncoreason.datasources import get_source
from oncoreason.datasources.cbioportal import CBioPortalSource

# a raw cBioPortal mutation record (real shape), including '.' status placeholders
_RAW = {
    "studyId": "luad_tcga_pan_can_atlas_2018", "sampleId": "TCGA-01-01", "patientId": "TCGA-01",
    "gene": {"hugoGeneSymbol": "EGFR"}, "entrezGeneId": 1956,
    "proteinChange": "L858R", "mutationType": "Missense_Mutation",
    "chr": "7", "startPosition": 55259515, "endPosition": 55259515,
    "referenceAllele": "T", "variantAllele": "G", "ncbiBuild": "GRCh37",
    "tumorAltCount": 40, "tumorRefCount": 60,
    "mutationStatus": ".", "validationStatus": ".",  # placeholders -> None
}


def test_normalize_shape_and_placeholders():
    r = CBioPortalSource._normalize(_RAW)
    assert r["gene_symbol"] == "EGFR" and r["entrez_gene_id"] == 1956  # gene join key
    assert r["protein_change"] == "L858R"
    assert r["mutation_status"] is None and r["validation_status"] is None  # '.' -> None
    assert r["tumor_alt_count"] == 40


def test_allow_list_and_restricted_guard():
    src = CBioPortalSource(use_cache=False)
    with pytest.raises(ValueError):
        src.fetch_mutations("some_genie_study", [1956])       # restricted -> refused
    with pytest.raises(ValueError):
        src.fetch_mutations("luad_random_unvetted", [1956])   # not on allow-list -> refused


def test_retrieve_is_not_an_evidence_source():
    from oncoreason.datasources import EvidenceQuery
    with pytest.raises(NotImplementedError):
        CBioPortalSource().retrieve(EvidenceQuery(gene="EGFR"))


def test_live_fetch_and_qc():
    """Real fetch of EGFR mutations from a curated open study, then run QC. Skips offline."""
    from oncoreason.cases.qc import run_qc

    src = get_source("cbioportal")
    try:
        recs = src.fetch_mutations("luad_tcga_pan_can_atlas_2018", [1956])
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"no network for live cBioPortal test: {exc}")
    assert recs and all(r["entrez_gene_id"] == 1956 for r in recs)
    assert any(r["protein_change"] == "L858R" for r in recs)
    # TCGA status fields are all '.', so normalization must have nulled them
    assert all(r["mutation_status"] is None for r in recs)
    rep = run_qc(recs, "luad_tcga_pan_can_atlas_2018")
    assert rep.n_records == len(recs)  # report is produced without error

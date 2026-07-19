"""ClinVar connector tests: offline mapping (always) + live round-trip (network-gated)."""
from __future__ import annotations

import urllib.error

import pytest

from oncoreason.datasources import EvidenceKind, EvidenceQuery, get_source
from oncoreason.datasources.clinvar import ClinVarSource

# trimmed real esummary record for ClinVar Variation 16609 (EGFR L858R)
_REC_16609 = {
    "uid": "16609",
    "accession": "VCV000016609",
    "title": "NM_005228.5(EGFR):c.2573T>G (p.Leu858Arg)",
    "protein_change": "L858R",
    "obj_type": "single nucleotide variant",
    "genes": [{"symbol": "EGFR", "geneid": "1956"}],
    "germline_classification": {
        "description": "drug response",
        "review_status": "reviewed by expert panel",
        "last_evaluated": "2021/03/24 00:00",
    },
    "oncogenicity_classification": {"description": "Oncogenic"},
    "somatic_clinical_impact": None,
}
# a record with no germline classification -> must be skipped, not crash
_REC_NOGERM = {"uid": "999", "accession": "VCV000000999", "genes": [],
               "germline_classification": {}}


@pytest.fixture
def offline_clinvar(monkeypatch):
    src = ClinVarSource(use_cache=False)
    monkeypatch.setattr(src, "_esummary",
                        lambda ids: {i: {"16609": _REC_16609, "999": _REC_NOGERM}[i] for i in ids})
    return src


def test_by_ids_maps_germline_axis(offline_clinvar):
    ev = offline_clinvar.by_ids(["16609"])
    assert len(ev) == 1
    e = ev[0]
    assert e.source == "clinvar" and e.kind == EvidenceKind.PATHOGENICITY
    assert e.citation_id == "clinvar:VCV000016609"
    assert e.is_somatic is False                      # germline axis, distinct from CIViC
    assert e.payload["entrez_gene_id"] == "1956"      # gene-level join key
    assert e.payload["germline_description"] == "drug response"
    assert e.evidence_level == "reviewed by expert panel"
    assert e.payload["oncogenicity_classification"]["description"] == "Oncogenic"


def test_records_without_germline_are_skipped(offline_clinvar):
    assert offline_clinvar.by_ids(["999"]) == []
    assert offline_clinvar.by_ids([]) == []


def test_retrieve_searches_then_summarizes(monkeypatch, offline_clinvar):
    # verify retrieve() queries ClinVar with the three-letter HGVS form
    seen = {}
    monkeypatch.setattr(offline_clinvar, "_esearch",
                        lambda term, retmax: seen.setdefault("term", term) and [] or ["16609"])
    ev = offline_clinvar.retrieve(EvidenceQuery(gene="EGFR", variant="p.L858R", top_k=5))
    assert "p.Leu858Arg" in seen["term"] and "EGFR[gene]" in seen["term"]
    assert ev and ev[0].citation_id == "clinvar:VCV000016609"


def test_requires_gene_and_variant(offline_clinvar):
    with pytest.raises(ValueError):
        offline_clinvar.retrieve(EvidenceQuery(gene="EGFR"))


def test_live_by_id():
    """Real ClinVar round-trip on the EGFR L858R variation id. Skips when offline."""
    clinvar = get_source("clinvar")
    try:
        ev = clinvar.by_ids(["16609"])
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"no network for live ClinVar test: {exc}")
    assert ev and ev[0].citation_id == "clinvar:VCV000016609"
    assert ev[0].payload["entrez_gene_id"] == "1956"

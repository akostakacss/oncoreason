"""CIViC connector tests.

Two layers:
- Offline (always run): mapping/filtering/ordering logic, with canned nodes injected so no
  network is touched. This is the behaviour worth locking down.
- Live (network-gated): one real query, skipped cleanly when offline/CI has no internet.
"""
from __future__ import annotations

import urllib.error

import pytest

from oncoreason.datasources import EvidenceKind, EvidenceQuery, get_source
from oncoreason.datasources.civic import CIViCSource

# two canned CIViC nodes: one accepted somatic level-A, one submitted (must be filtered out)
_NODES = [
    {
        "id": 2994, "evidenceType": "PREDICTIVE", "evidenceLevel": "A",
        "evidenceDirection": "SUPPORTS", "significance": "SENSITIVITYRESPONSE",
        "evidenceRating": 5, "status": "ACCEPTED", "variantOrigin": "SOMATIC",
        "therapies": [{"name": "Erlotinib"}],
        "disease": {"name": "Lung Non-small Cell Carcinoma", "doid": "3908"},
        "molecularProfile": {"name": "EGFR L858R", "variants": [
            {"id": 33, "name": "L858R", "alleleRegistryId": "CA126713",
             "clinvarIds": ["16609", "376282"],
             "hgvsDescriptions": ["NP_005219.2:p.Leu858Arg"]},
        ]},
        "source": {"citationId": "24868098", "sourceType": "PUBMED"},
    },
    {
        "id": 99999, "evidenceType": "PREDICTIVE", "evidenceLevel": "B",
        "evidenceDirection": "SUPPORTS", "significance": "SENSITIVITYRESPONSE",
        "evidenceRating": 2, "status": "SUBMITTED", "variantOrigin": "SOMATIC",
        "therapies": [{"name": "Experimental"}],
        "disease": {"name": "Lung Non-small Cell Carcinoma", "doid": "3908"},
        "molecularProfile": {"name": "EGFR L858R"},
        "source": {"citationId": "0", "sourceType": "PUBMED"},
    },
]


@pytest.fixture
def offline_civic(monkeypatch):
    src = CIViCSource(use_cache=False)
    monkeypatch.setattr(src, "_fetch_nodes", lambda mp, first: list(_NODES))
    return src


def test_maps_node_to_evidence(offline_civic):
    ev = offline_civic.retrieve(EvidenceQuery(gene="EGFR", variant="p.L858R", top_k=5))
    assert len(ev) == 1  # SUBMITTED node dropped by accepted_only
    e = ev[0]
    assert e.citation_id == "civic:EID2994"
    assert e.source == "civic" and e.kind == EvidenceKind.PREDICTIVE
    assert e.evidence_level == "A" and e.is_somatic is True
    assert "Erlotinib" in e.summary and e.payload["source_citation_id"] == "24868098"
    # canonical cross-database join keys captured from the variant record
    assert e.payload["caid"] == "CA126713"
    assert "16609" in e.payload["clinvar_ids"]


def test_accepted_only_can_be_disabled(offline_civic):
    offline_civic.accepted_only = False
    ev = offline_civic.retrieve(EvidenceQuery(gene="EGFR", variant="L858R", top_k=5))
    assert len(ev) == 2
    # ordering: level A before level B regardless of input order
    assert [e.evidence_level for e in ev] == ["A", "B"]


def test_requires_gene_and_variant(offline_civic):
    with pytest.raises(ValueError):
        offline_civic.retrieve(EvidenceQuery(gene="EGFR"))
    with pytest.raises(ValueError):
        offline_civic.retrieve(EvidenceQuery(variant="p.L858R"))


@pytest.mark.parametrize("gene,variant,drug", [("EGFR", "p.L858R", "Osimertinib"),
                                               ("KRAS", "p.G12C", "Sotorasib")])
def test_live_query(gene, variant, drug):
    """Real CIViC round-trip. Skips (not fails) when there is no network."""
    civic = get_source("civic")
    try:
        ev = civic.retrieve(EvidenceQuery(gene=gene, variant=variant,
                                          tumor_type="lung", top_k=10))
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"no network for live CIViC test: {exc}")
    assert ev, f"expected CIViC evidence for {gene} {variant}"
    drugs = {d.lower() for e in ev for d in e.payload.get("therapies", [])}
    assert drug.lower() in drugs
    assert all(e.citation_id.startswith("civic:EID") for e in ev)

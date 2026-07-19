"""Smoke test: the package imports, schemas construct, config loads, and the
controlled-data guard fires. No network, no GPU, no heavy deps beyond pyyaml.

Run:  pytest -q
"""
from __future__ import annotations

import pytest


def test_package_imports():
    import oncoreason
    from oncoreason import agents, cases, datasources, evaluation, retrieval
    from oncoreason import supervision, training  # noqa: F401

    assert oncoreason.__version__


def test_registry_lists_sources():
    from oncoreason.datasources import CONTROLLED_SOURCES, PUBLIC_SOURCES, available_sources

    names = available_sources()
    for n in PUBLIC_SOURCES + CONTROLLED_SOURCES:
        assert n in names


def test_case_schema_constructs():
    from oncoreason.cases import Alteration, Case, ClinicalContext

    case = Case(
        case_id="demo-1",
        alterations=[Alteration(gene="EGFR", variant="p.L858R", escat_tier="I-A")],
        context=ClinicalContext(tumor_type="lung", stage="IV"),
    )
    assert case.alterations[0].gene == "EGFR"
    assert case.gold is None  # not labelled yet


def test_trace_schema_constructs():
    from oncoreason.agents import Citation, ReasoningStep, Trace

    tr = Trace(case_id="demo-1")
    tr.steps.append(
        ReasoningStep(index=0, text="EGFR L858R is an activating mutation.",
                      citations=[Citation(citation_id="civic:123", source="civic",
                                          claim="L858R is activating")])
    )
    assert len(tr.all_citations()) == 1
    assert tr.abstained is False


def test_public_source_typed_and_validates():
    """CIViC is a real connector now (Phase 1). Check typing + input validation only —
    the functional behaviour is tested offline in test_civic.py (no network here)."""
    from oncoreason.datasources import EvidenceQuery, get_source

    civic = get_source("civic")
    assert civic.name == "civic" and civic.is_controlled is False
    with pytest.raises(ValueError):
        civic.retrieve(EvidenceQuery(gene="EGFR"))  # variant missing -> hard error, not empty


def test_controlled_source_guards_without_data():
    """The reserved slot must refuse to run without licensed data present."""
    from oncoreason.datasources import ControlledSourceNotConfigured, EvidenceQuery, get_source

    nccn = get_source("nccn")
    assert nccn.is_controlled is True
    with pytest.raises(ControlledSourceNotConfigured):
        nccn.retrieve(EvidenceQuery(free_text="lung adenocarcinoma first line"))


def test_config_loads():
    from oncoreason import load_config

    cfg = config = load_config()
    assert config["seed"] == 17
    assert "civic" in cfg["sources"]["public"]
    assert cfg["data"]["split_level"] == "case"

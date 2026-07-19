"""Orchestrator: planner -> specialists -> synthesizer -> abstain. Offline with fakes."""
from __future__ import annotations

from oncoreason.agents import DeterministicLLM, Orchestrator
from oncoreason.agents.guideline_index import index_docs
from oncoreason.cases.schema import Alteration, Case, ClinicalContext
from oncoreason.datasources.base import Evidence, EvidenceKind
from oncoreason.retrieval.base import BM25Retriever

LUAD = ClinicalContext(tumor_type="lung adenocarcinoma", stage="IV")


class _FakeCIViC:
    """Returns a canned CIViC-shaped Evidence for EGFR L858R; empty otherwise."""
    name = "civic"

    def retrieve(self, query):
        if query.gene == "EGFR":
            return [Evidence(
                source="civic", kind=EvidenceKind.PREDICTIVE, citation_id="civic:EID1",
                summary="EGFR L858R SUPPORTS SENSITIVITY to osimertinib in lung adenocarcinoma",
                evidence_level="A", is_somatic=True,
                payload={"evidence_direction": "SUPPORTS", "significance": "SENSITIVITYRESPONSE",
                         "therapies": ["osimertinib"], "disease": "Lung Adenocarcinoma",
                         "clinvar_ids": ["376280"], "caid": "CA126713"},
            )]
        return []


class _FakeClinVar:
    name = "clinvar"

    def by_ids(self, variation_ids):
        return [Evidence(source="clinvar", kind=EvidenceKind.PATHOGENICITY,
                         citation_id=f"clinvar:{variation_ids[0]}",
                         summary="germline: Likely pathogenic", is_somatic=False,
                         payload={"entrez_gene_id": 1956})]


def _guideline():
    r = BM25Retriever(source="esmo_index")
    r.index(index_docs())
    return r


def test_actionable_case_recommends_and_cites():
    orch = Orchestrator(llm=DeterministicLLM(), sources={"civic": _FakeCIViC(),
                        "clinvar": _FakeClinVar()}, guideline_retriever=_guideline())
    t = orch.run(Case("c1", [Alteration("EGFR", "L858R")], LUAD))

    assert not t.abstained
    assert t.recommendation and t.recommendation[0].lower() == "osimertinib"
    assert t.confidence >= 0.9                      # CIViC level A + guideline I-A agree
    assert t.model == "deterministic-v0"
    # citations resolve to real-shaped ids from both a variant and a guideline source
    cids = [c.citation_id for c in t.all_citations()]
    assert any(c.startswith("civic:") for c in cids)
    assert any(c.startswith("guideline:") for c in cids)
    # every tool call was logged and succeeded
    calls = [c for s in t.steps for c in s.tool_calls]
    assert calls and all(c.ok for c in calls)


def test_abstains_when_no_evidence():
    # no guideline retriever, empty CIViC -> nothing to stand on -> defer
    orch = Orchestrator(sources={"civic": _FakeCIViC()}, guideline_retriever=None)
    t = orch.run(Case("c2", [Alteration("AKT3", "P100L")], LUAD))
    assert t.abstained
    assert t.recommendation == []
    assert t.confidence < 0.5


def test_no_targeted_driver_falls_back_to_chemo_not_hallucination():
    orch = Orchestrator(sources={"civic": _FakeCIViC()}, guideline_retriever=_guideline())
    t = orch.run(Case("c3", [Alteration("TP53", "R175H")], LUAD))
    # TP53 has no targeted option -> must NOT return a targeted drug like osimertinib
    joined = " ".join(t.recommendation).lower()
    assert "osimertinib" not in joined
    assert any(x in joined for x in ("chemotherapy", "pembrolizumab", "platinum"))


def test_trace_structure_is_auditable():
    orch = Orchestrator(sources={"civic": _FakeCIViC(), "clinvar": _FakeClinVar()},
                        guideline_retriever=_guideline())
    t = orch.run(Case("c4", [Alteration("EGFR", "L858R")], LUAD))
    assert t.steps[0].index == 0                    # planner first
    assert t.steps[-1].text.startswith("Synthesis") # synthesizer last
    assert t.metadata["specialists_run"] == ["variant", "guideline"]
    assert all(s.prm_score is None for s in t.steps) # unscored until Phase 5

"""Retrieval: BM25 ranking + Hybrid RRF fusion. Offline, no model download."""
from __future__ import annotations

from oncoreason.retrieval.base import BM25Retriever, HybridRetriever, RetrievedChunk

DOCS = [
    ("d1", "EGFR L858R lung adenocarcinoma osimertinib first line"),
    ("d2", "ALK rearrangement lung adenocarcinoma alectinib"),
    ("d3", "KRAS G12C lung adenocarcinoma sotorasib adagrasib"),
    ("d4", "BRAF V600E melanoma dabrafenib trametinib"),
]


def test_bm25_ranks_relevant_doc_first():
    r = BM25Retriever()
    r.index(DOCS)
    hits = r.search("EGFR osimertinib", top_k=3)
    assert hits and hits[0].doc_id == "d1"
    assert all(isinstance(h, RetrievedChunk) for h in hits)
    assert hits[0].score > 0


def test_bm25_empty_on_no_overlap():
    r = BM25Retriever()
    r.index(DOCS)
    assert r.search("pancreatic gemcitabine", top_k=5) == []


class _FakeDense:
    """Deterministic stand-in for the dense arm: returns a fixed ranking."""
    name = "dense"

    def __init__(self, order):
        self._order = order

    def index(self, docs):
        self._docs = dict(docs)

    def search(self, query, top_k=10):
        return [RetrievedChunk(d, self._docs[d], 1.0 - i * 0.01, "dense")
                for i, d in enumerate(self._order)][:top_k]


def test_hybrid_rrf_fuses_both_arms():
    sparse = BM25Retriever()
    # dense disagrees with sparse; RRF should reward docs ranked well by BOTH
    dense = _FakeDense(order=["d3", "d1", "d2", "d4"])
    hybrid = HybridRetriever(sparse, dense)
    hybrid.index(DOCS)
    hits = hybrid.search("EGFR osimertinib lung", top_k=4)
    ids = [h.doc_id for h in hits]
    # d1 is top for sparse and 2nd for dense -> should fuse to the front
    assert ids[0] == "d1"
    assert set(ids) <= {"d1", "d2", "d3", "d4"}
    assert all(h.source == "hybrid" for h in hits)

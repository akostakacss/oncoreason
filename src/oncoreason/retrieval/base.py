"""Retrieval — sparse (BM25), dense (MedCPT/BGE), and hybrid (RRF).

Build all three and compare (covers the dense/sparse/hybrid requirement by doing
each). Everything here runs on CPU. Phase 3.2.

BM25 is implemented in pure Python (Okapi BM25) so the sparse path has no third-party
dependency and the repo stays runnable offline; it is a faithful, standard BM25 and can be
swapped for ``rank-bm25`` without changing the interface. The dense encoder needs a model
download (MedCPT/BGE via sentence-transformers) — it is wired but raises a clear "enable"
message until that stack is present (it runs on Kaggle/Colab). Hybrid RRF fuses any sparse +
any dense retriever and is fully implemented and testable with either.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str          # stable id for citation verification
    text: str
    score: float
    source: str          # e.g. "esmo_index", "pubmed"


@runtime_checkable
class Retriever(Protocol):
    name: str

    def index(self, docs: list[tuple[str, str]]) -> None:
        """Index (doc_id, text) pairs. For guidelines: derived chunks only, not raw text."""
        ...

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        ...


class BM25Retriever:
    """Okapi BM25, pure Python. Standard defaults (k1=1.5, b=0.75)."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75, source: str = "index") -> None:
        self.k1 = k1
        self.b = b
        self.source = source
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._df: Counter = Counter()
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def index(self, docs: list[tuple[str, str]]) -> None:
        self._ids, self._texts, self._tf, self._len = [], [], [], []
        self._df = Counter()
        for doc_id, text in docs:
            toks = _tokenize(text)
            tf = Counter(toks)
            self._ids.append(doc_id)
            self._texts.append(text)
            self._tf.append(tf)
            self._len.append(len(toks))
            self._df.update(tf.keys())
        n = len(self._ids)
        self._avgdl = (sum(self._len) / n) if n else 0.0
        # BM25 idf with the standard +0.5 smoothing, floored positive so a term present in
        # every doc still contributes a little rather than going negative.
        self._idf = {
            term: max(1e-9, math.log(1 + (n - df + 0.5) / (df + 0.5)))
            for term, df in self._df.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        q = _tokenize(query)
        scored: list[RetrievedChunk] = []
        for i, doc_id in enumerate(self._ids):
            tf, dl = self._tf[i], self._len[i]
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
                s += self._idf.get(term, 0.0) * (f * (self.k1 + 1)) / denom
            if s > 0:
                scored.append(RetrievedChunk(doc_id, self._texts[i], s, self.source))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]


class DenseRetriever:
    """Dense bi-encoder retrieval (MedCPT / BGE) via sentence-transformers.

    Needs the model weights; runs on CPU but downloads on first use. Kept behind a clear
    enable-path so the offline/CI path never blocks on a download.
    """

    name = "dense"

    def __init__(self, model_name: str = "ncbi/MedCPT-Query-Encoder", source: str = "index") -> None:
        self.model_name = model_name
        self.source = source
        self._model = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._emb = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - environment-dependent
                raise NotImplementedError(
                    "Dense retrieval needs `sentence-transformers` + the encoder weights "
                    f"({self.model_name}). Install it and run on Kaggle/Colab, or use "
                    "BM25Retriever for the offline path. Phase 3.2."
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index(self, docs: list[tuple[str, str]]) -> None:
        model = self._load()
        self._ids = [d for d, _ in docs]
        self._texts = [t for _, t in docs]
        self._emb = model.encode(self._texts, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        import numpy as np

        model = self._load()
        if self._emb is None:
            return []
        q = model.encode([query], normalize_embeddings=True)[0]
        sims = np.asarray(self._emb) @ np.asarray(q)  # cosine (both normalized)
        order = sims.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(self._ids[i], self._texts[i], float(sims[i]), self.source)
            for i in order
        ]


class HybridRetriever:
    """Reciprocal Rank Fusion over a sparse + a dense retriever.

    RRF is rank-based (scale-free), so it fuses BM25's unbounded scores with dense cosine
    without normalization. score(doc) = sum over retrievers of 1/(rrf_k + rank).
    """

    name = "hybrid"

    def __init__(self, sparse: Retriever, dense: Retriever, rrf_k: int = 60) -> None:
        self.sparse = sparse
        self.dense = dense
        self.rrf_k = rrf_k

    def index(self, docs: list[tuple[str, str]]) -> None:
        self.sparse.index(docs)
        self.dense.index(docs)

    def search(self, query: str, top_k: int = 10) -> list[RetrievedChunk]:
        # over-fetch each arm so the fusion has enough depth to reorder
        depth = max(top_k * 4, 20)
        runs = [self.sparse.search(query, depth), self.dense.search(query, depth)]
        fused: dict[str, float] = {}
        chunk_by_id: dict[str, RetrievedChunk] = {}
        for run in runs:
            for rank, chunk in enumerate(run):
                fused[chunk.doc_id] = fused.get(chunk.doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
                chunk_by_id.setdefault(chunk.doc_id, chunk)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(doc_id, chunk_by_id[doc_id].text, score, "hybrid")
            for doc_id, score in ranked
        ]

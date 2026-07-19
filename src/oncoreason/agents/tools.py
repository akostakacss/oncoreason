"""Tool layer — every external lookup goes through a logged, timed wrapper.

Each call yields a ``ToolCall`` record (ok / latency_ms / error / n_results) so the trace
carries tool-use-reliability data for Phase 6 metrics. Tools are thin: they call a
``DataSource`` or ``Retriever`` and hand back findings; interpretation is the agent's job.

Specialists wired in v1: **variant** (CIViC somatic + ClinVar germline) and **guideline**
(the derived index). **trial** (ClinicalTrials.gov) and **literature** (PubMed) are optional —
if no source is provided they are skipped with a logged, non-fatal note, matching the plan's
scoping (trial matching = biomarker retrieval only; literature optional). Phase 3.1.
"""
from __future__ import annotations

import time
from typing import Callable

from ..datasources.base import Evidence, EvidenceQuery
from ..retrieval.base import RetrievedChunk, Retriever
from .guideline_index import CHUNKS_BY_ID
from .trace import Citation, ToolCall


def call_tool(tool: str, fn: Callable, args: dict) -> tuple[object, ToolCall]:
    """Run ``fn(**args)`` timed and guarded; return (result, ToolCall).

    On error the result is an empty list and the ToolCall records the failure — a specialist
    can carry on and the reliability metric sees the failed call rather than a crash.
    """
    t0 = time.perf_counter()
    try:
        result = fn(**args)
        n = len(result) if hasattr(result, "__len__") else None
        call = ToolCall(tool=tool, args=args, ok=True,
                        latency_ms=(time.perf_counter() - t0) * 1000, n_results=n)
        return result, call
    except Exception as e:  # noqa: BLE001 - deliberately broad: log, don't crash the run
        call = ToolCall(tool=tool, args=args, ok=False,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        error=f"{type(e).__name__}: {e}")
        return [], call


# ---- variant specialist: CIViC (somatic) + ClinVar (germline), joined on canonical ids ----
def variant_lookup(gene, variant, tumor_type, civic, clinvar, top_k=5):
    """Return (evidence, tool_calls). ClinVar is reached via the ClinVar ids CIViC returns —
    a canonical-id join, never a name-string match (Phase 0 Gate 3)."""
    calls: list[ToolCall] = []
    q = EvidenceQuery(gene=gene, variant=variant, tumor_type=tumor_type, top_k=top_k)
    civ, c1 = call_tool("civic.retrieve", civic.retrieve, {"query": q})
    calls.append(c1)
    evidence: list[Evidence] = list(civ)

    clinvar_ids = sorted({i for e in evidence for i in e.payload.get("clinvar_ids", [])})
    if clinvar is not None and clinvar_ids:
        germ, c2 = call_tool("clinvar.by_ids", clinvar.by_ids, {"variation_ids": clinvar_ids})
        calls.append(c2)
        evidence += list(germ)
    return evidence, calls


# ---- guideline specialist: retrieval over the derived index -------------------------------
def guideline_lookup(gene, tumor_type, retriever: Retriever, top_k=3):
    """Return (chunks, tool_call). Retrieves author-summarized recommendations (derived index,
    never raw copyrighted text)."""
    query = f"{gene} {tumor_type}".strip()
    chunks, call = call_tool("guideline.search", retriever.search,
                             {"query": query, "top_k": top_k})
    return list(chunks), call


# ---- citation builders --------------------------------------------------------------------
def evidence_citations(evidence: list[Evidence]) -> list[Citation]:
    return [Citation(citation_id=e.citation_id, source=e.source, claim=e.summary)
            for e in evidence]


def guideline_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    out: list[Citation] = []
    for ch in chunks:
        gc = CHUNKS_BY_ID.get(ch.doc_id)
        claim = gc.recommendation if gc else ch.text
        out.append(Citation(citation_id=f"guideline:{ch.doc_id}", source="guideline_index",
                            claim=claim))
    return out

"""A small DERIVED guideline index — author-summarized, license-safe.

Copyright discipline (Phase 1.3 / risk register): ESMO and NCCN guideline *text*
is copyrighted and is NEVER committed. What lives here are **derived representations** —
short, author-written structured summaries of well-known lung recommendations, each with a
provenance pointer to the public source it summarizes. This is the same distinction the
controlled-data connector enforces: a derived index is shippable; raw text is loaded at
runtime from a git-ignored path via a connector.

This seed exists so the guideline specialist is functional in the PoC and so the retrieval
ablation (BM25/dense/hybrid) has something real to search. In production this is replaced by
a richer index built from the licensed text the lab holds — dropped into the same shape.

Each entry: a stable id, the gene/context it concerns, an author-written recommendation, an
ESCAT tier, and a provenance note. Nothing here is copied guideline prose.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuidelineChunk:
    chunk_id: str
    gene: str
    tumor_type: str
    recommendation: str          # author-written summary, NOT copied text
    therapies: tuple[str, ...]
    escat_tier: str
    provenance: str              # points at the public source it summarizes


# NSCLC first-line biomarker-directed recommendations (author-summarized).
GUIDELINE_CHUNKS: tuple[GuidelineChunk, ...] = (
    GuidelineChunk(
        "gl-egfr-sens", "EGFR", "lung adenocarcinoma",
        "Activating sensitizing EGFR mutation (exon 19 deletion or L858R): first-line "
        "osimertinib is preferred.",
        ("osimertinib",), "I-A",
        "Derived from ESMO NSCLC living guideline, EGFR section; author summary.",
    ),
    GuidelineChunk(
        "gl-egfr-t790m", "EGFR", "lung adenocarcinoma",
        "EGFR T790M resistance mutation after earlier-generation TKI: osimertinib.",
        ("osimertinib",), "I-A",
        "Derived from ESMO NSCLC living guideline, EGFR resistance; author summary.",
    ),
    GuidelineChunk(
        "gl-alk", "ALK", "lung adenocarcinoma",
        "ALK rearrangement: first-line next-generation ALK inhibitor (alectinib, "
        "brigatinib, or lorlatinib).",
        ("alectinib", "brigatinib", "lorlatinib"), "I-A",
        "Derived from ESMO NSCLC living guideline, ALK section; author summary.",
    ),
    GuidelineChunk(
        "gl-ros1", "ROS1", "lung adenocarcinoma",
        "ROS1 fusion: first-line entrectinib or crizotinib.",
        ("entrectinib", "crizotinib"), "I-A",
        "Derived from ESMO NSCLC living guideline, ROS1 section; author summary.",
    ),
    GuidelineChunk(
        "gl-braf", "BRAF", "lung adenocarcinoma",
        "BRAF V600E: dabrafenib plus trametinib.",
        ("dabrafenib", "trametinib"), "I-B",
        "Derived from ESMO NSCLC living guideline, BRAF section; author summary.",
    ),
    GuidelineChunk(
        "gl-kras-g12c", "KRAS", "lung adenocarcinoma",
        "KRAS G12C after prior systemic therapy: sotorasib or adagrasib.",
        ("sotorasib", "adagrasib"), "I-B",
        "Derived from ESMO NSCLC living guideline, KRAS G12C section; author summary.",
    ),
    GuidelineChunk(
        "gl-met", "MET", "lung adenocarcinoma",
        "MET exon 14 skipping: capmatinib or tepotinib.",
        ("capmatinib", "tepotinib"), "I-B",
        "Derived from ESMO NSCLC living guideline, MET section; author summary.",
    ),
    GuidelineChunk(
        "gl-ret", "RET", "lung adenocarcinoma",
        "RET fusion: selpercatinib or pralsetinib.",
        ("selpercatinib", "pralsetinib"), "I-B",
        "Derived from ESMO NSCLC living guideline, RET section; author summary.",
    ),
    GuidelineChunk(
        "gl-erbb2", "ERBB2", "lung adenocarcinoma",
        "HER2 (ERBB2) activating mutation after prior therapy: trastuzumab deruxtecan.",
        ("trastuzumab deruxtecan",), "I-B",
        "Derived from ESMO NSCLC living guideline, HER2 section; author summary.",
    ),
    GuidelineChunk(
        "gl-ntrk", "NTRK1", "lung adenocarcinoma",
        "NTRK fusion: larotrectinib or entrectinib.",
        ("larotrectinib", "entrectinib"), "I-C",
        "Derived from ESMO tumour-agnostic NTRK recommendation; author summary.",
    ),
    GuidelineChunk(
        "gl-nodriver-luad", "", "lung adenocarcinoma",
        "No targetable driver: platinum-doublet chemotherapy +/- immunotherapy by PD-L1.",
        ("pembrolizumab", "platinum-doublet chemotherapy"), "N/A",
        "Derived from ESMO NSCLC living guideline, non-oncogene-addicted; author summary.",
    ),
    GuidelineChunk(
        "gl-nodriver-lusc", "", "lung squamous cell carcinoma",
        "Squamous NSCLC without targetable driver: platinum-doublet chemotherapy +/- "
        "immunotherapy by PD-L1.",
        ("pembrolizumab", "platinum-doublet chemotherapy"), "N/A",
        "Derived from ESMO NSCLC living guideline, squamous; author summary.",
    ),
)


def index_docs() -> list[tuple[str, str]]:
    """(doc_id, text) pairs for a Retriever. Text = gene + context + recommendation."""
    return [
        (c.chunk_id, f"{c.gene} {c.tumor_type} {c.recommendation} {' '.join(c.therapies)}")
        for c in GUIDELINE_CHUNKS
    ]


CHUNKS_BY_ID: dict[str, GuidelineChunk] = {c.chunk_id: c for c in GUIDELINE_CHUNKS}

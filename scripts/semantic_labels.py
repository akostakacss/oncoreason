#!/usr/bin/env python3
"""Semantic step labels — does the cited record actually *support* the claim?

Stages B-D established that the structural label is the binding constraint. It asks only
"does this citation resolve, and does it belong to this case", which is bookkeeping: the
negative class collapsed to 95% "cited nothing", and inlining evidence text could not help
because there was no relational question in the label for the extra information to answer.

MTBBench (Vasilev, ..., Moor, Bunne; NeurIPS 2025 D&B) reports that models "frequently
hallucinate ... and fail to reconcile conflicting evidence". None of that is visible to a
label that only checks id membership. This module makes it visible, using structured fields
CIViC already ships:

    significance        SENSITIVITYRESPONSE | RESISTANCE
    evidence_direction  SUPPORTS | DOES_NOT_SUPPORT
    therapies           ['Erlotinib', ...]
    disease             'Lung Non-small Cell Carcinoma'

Three failure modes become mechanically detectable without any human annotation:

  - **contradiction** the step claims benefit from a therapy while citing a record that says
    RESISTANCE, or DOES_NOT_SUPPORT. This is the "fails to reconcile conflicting evidence"
    failure, and it is a reasoning error rather than a clerical one.
  - **therapy mismatch** the step names a therapy the cited record is not about, so the
    citation cannot support the specific claim being made.
  - **disease mismatch** the cited record is about a different tumour type — the machine
    analogue of the TP53 tumor-type mismatch the project was built around.

These are proxies, not clinician judgements. They are stated as such wherever reported, and
the honest upgrade path is MTBBench's expert-verified QA pairs (`docs/MTBBENCH_INTEGRATION.md`).
"""
from __future__ import annotations

import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

# Claim direction. Deliberately conservative: a step that signals neither way yields None and
# is left unjudged rather than guessed at, because a wrong label is worse than a missing one.
_BENEFIT = re.compile(
    r"\b(sensitiv\w*|respond\w*|benefit\w*|effective|efficacious|indicated|recommend\w*|"
    r"should be (?:considered|offered|used)|treat(?:ed|ment)? with|first[- ]line|"
    r"appropriate|suitable|eligible)\b", re.I)
_RESIST = re.compile(
    r"\b(resistan\w*|refractory|not effective|ineffective|does not respond|no benefit|"
    r"lack of (?:response|benefit)|contraindicat\w*|should not be|avoid|unlikely to benefit|"
    r"progress\w* on)\b", re.I)


def claim_direction(text: str) -> str | None:
    """SENSITIVITY, RESISTANCE, or None when the step does not commit either way."""
    resist = bool(_RESIST.search(text))
    benefit = bool(_BENEFIT.search(text))
    if resist and not benefit:
        return "RESISTANCE"
    if benefit and not resist:
        return "SENSITIVITY"
    return None                      # silent, or both — do not guess


def record_direction(payload: dict) -> str | None:
    """Collapse CIViC's (significance, evidence_direction) pair into one claim direction.

    DOES_NOT_SUPPORT inverts the significance: "does not support sensitivity" is evidence
    against benefit, which is what a reader would take from it.
    """
    sig = (payload.get("significance") or "").upper()
    direction = (payload.get("evidence_direction") or "").upper()
    if sig.startswith("SENSITIVITY"):
        base = "SENSITIVITY"
    elif sig.startswith("RESISTANCE"):
        base = "RESISTANCE"
    else:
        return None
    if direction.startswith("DOES_NOT") or direction.startswith("DOESNOT"):
        return "RESISTANCE" if base == "SENSITIVITY" else "SENSITIVITY"
    return base


def _therapies_in(text: str, vocabulary: set[str]) -> set[str]:
    low = text.lower()
    return {t for t in vocabulary if t and t.lower() in low}


def label_step(step_text: str, cited_ids: list[str], records: dict,
               tumor_type: str, therapy_vocab: set[str]) -> dict:
    """Judge one step against the records it cites. Returns the label plus its reasons.

    `records` maps citation_id -> the evidence payload dict (guideline chunks have no
    structured fields and are treated as non-judgeable rather than as failures).
    """
    reasons: list[str] = []
    claim_dir = claim_direction(step_text)
    named = _therapies_in(step_text, therapy_vocab)

    contradicted = therapy_mismatch = disease_mismatch = False
    supporting = 0
    judgeable = 0

    for cid in cited_ids:
        payload = records.get(cid)
        if not payload:
            continue                                  # guideline chunk or unresolved
        judgeable += 1
        rec_dir = record_direction(payload)
        rec_therapies = {t for t in (payload.get("therapies") or [])}
        rec_disease = (payload.get("disease") or "").lower()

        if claim_dir and rec_dir and claim_dir != rec_dir:
            contradicted = True
            reasons.append(f"{cid}: step claims {claim_dir.lower()}, record says {rec_dir.lower()}")
        elif claim_dir and rec_dir and claim_dir == rec_dir:
            supporting += 1

        if named and rec_therapies and not (named & {t.lower() for t in rec_therapies} or
                                            named & rec_therapies):
            # only a mismatch when the step named a therapy at all
            if not {n.lower() for n in named} & {t.lower() for t in rec_therapies}:
                therapy_mismatch = True
                reasons.append(f"{cid}: step names {sorted(named)}, record is about "
                               f"{sorted(rec_therapies)}")

        if rec_disease and tumor_type and tumor_type.lower() not in rec_disease:
            disease_mismatch = True
            reasons.append(f"{cid}: record disease '{payload.get('disease')}' "
                           f"vs case tumour type '{tumor_type}'")

    sound = bool(cited_ids) and not contradicted and not therapy_mismatch
    return {
        "label_semantic_sound": sound,
        "claim_direction": claim_dir,
        "n_judgeable_citations": judgeable,
        "n_supporting": supporting,
        "contradicted": contradicted,
        "therapy_mismatch": therapy_mismatch,
        "disease_mismatch": disease_mismatch,
        "reasons": reasons[:4],
    }


def build_record_index() -> tuple[dict, set[str], dict]:
    """Return (citation_id -> payload, therapy vocabulary, case_id -> tumour type).

    Guideline chunks are included alongside CIViC evidence. They carry `therapies`, `gene`
    and `tumor_type` of their own, so they can be judged on the same three axes — without
    them only ~23% of steps cite anything structured, because the policy cites guidelines far
    more often than CIViC records. A guideline chunk is by construction a SUPPORTS/SENSITIVITY
    statement: it recommends a therapy for a context.
    """
    import json

    from oncoreason.agents.guideline_index import GUIDELINE_CHUNKS
    from oncoreason.agents.tools import variant_lookup
    from oncoreason.datasources import get_source

    civic, clinvar = get_source("civic"), get_source("clinvar")
    records: dict = {}
    vocab: set[str] = set()
    tumors: dict = {}

    for ch in GUIDELINE_CHUNKS:
        records[f"guideline:{ch.chunk_id}"] = {
            "therapies": list(ch.therapies),
            "disease": ch.tumor_type,
            "gene": ch.gene,
            "significance": "SENSITIVITYRESPONSE",
            "evidence_direction": "SUPPORTS",
            "escat_tier": ch.escat_tier,
            "source_kind": "guideline",
        }
        vocab |= set(ch.therapies)

    with open(os.path.join(REPO, "data", "cases", "cases.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            tumor = d["context"].get("tumor_type", "lung")
            tumors[d["case_id"]] = tumor
            for a in d["alterations"]:
                ev, _ = variant_lookup(a["gene"], a["variant"], tumor, civic, clinvar)
                for e in ev:
                    records[e.citation_id] = e.payload
                    vocab |= {t for t in (e.payload.get("therapies") or [])}
    return records, vocab, tumors


if __name__ == "__main__":
    import json

    records, vocab, tumors = build_record_index()
    print(f"{len(records)} structured records | {len(vocab)} distinct therapies")
    print("sample therapies:", sorted(vocab)[:12])

    demo = [
        ("EGFR L858R is sensitive to erlotinib and it should be offered first-line.",
         "SENSITIVITY (benefit language)"),
        ("This variant confers resistance to erlotinib, so it should be avoided.",
         "RESISTANCE"),
        ("The patient has an EGFR mutation.", "None — no direction committed"),
    ]
    for text, expect in demo:
        print(f"\n  {text!r}\n    -> {claim_direction(text)}  (expected {expect})")

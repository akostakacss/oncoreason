"""Variant normalization — the Gate 3 layer, shared by every connector.

The silent-failure trap this prevents: the same protein change is written three ways
across sources — CIViC uses one-letter with no prefix (``L858R``), HGVS/cBioPortal use a
``p.`` prefix (``p.L858R``), ClinVar uses three-letter (``p.Leu858Arg``). A connector that
queries with the wrong spelling gets an *empty result*, not an error, so a normalization
bug looks exactly like "no evidence exists". Every connector routes variant strings through
here so a lookup that misses is a real miss.

This module is the **display / spelling** layer only — HGVS one-letter <-> three-letter and
CIViC's prefixless short form. It is deliberately NOT the cross-database *identity* layer:
matching the same allele across sources is done with canonical identifiers (ClinGen Allele
Registry CAID, GA4GH VRS computed ids), which CIViC already exposes per variant
(``alleleRegistryId``, ``clinvarIds``, ``hgvsDescriptions`` — see civic.py). String
re-spelling must never be the join mechanism; use it for the CIViC molecular-profile name
and for human-readable output, nothing more. Gene-symbol normalization (aliases / previous
symbols) is a separate axis handled by HGNC, not here.

Scope (deliberately small and honest): reliable for **protein substitutions** (the bulk of
actionable lung drivers: EGFR L858R/T790M, KRAS G12C, BRAF V600E). Non-substitution events
(``exon 19 deletion``, ``L747_P753delinsS``, fusions like ``EML4-ALK``) are passed through
unchanged rather than mangled — three-letter codes inside them are still converted, but the
structure is preserved. Full HGVS validation (transcript, genome build) is a documented
upgrade path (VEP / GA4GH VRS), not done here.
"""
from __future__ import annotations

import re

# Standard amino acids + stop, three-letter -> one-letter.
_AA_3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Sec": "U", "Xaa": "X",
}
_AA_1TO3 = {v: k for k, v in _AA_3TO1.items() if v not in ("U", "X")}

_THREE_RE = re.compile("|".join(sorted(_AA_3TO1, key=len, reverse=True)))
# one-letter substitution: <AA><pos><AA|*>, the only form I round-trip to three-letter.
_SUBST_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY*])$")


def _strip_prefix(v: str) -> str:
    """Remove a leading ``p.`` and any wrapping parentheses (``p.(Leu858Arg)``)."""
    s = v.strip()
    if s.lower().startswith("p."):
        s = s[2:]
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return s.strip()


def three_to_one(protein: str) -> str:
    """Convert three-letter amino-acid codes to one-letter, preserving everything else.

    ``p.Leu858Arg`` -> ``L858R``   ·   ``Leu747_Pro753delinsSer`` -> ``L747_P753delinsS``
    Input already in one-letter form is returned unchanged (idempotent).
    """
    return _THREE_RE.sub(lambda m: _AA_3TO1[m.group(0)], _strip_prefix(protein))


def one_to_three(protein: str) -> str | None:
    """Best-effort inverse for simple substitutions (for ClinVar-style ``p.Leu858Arg``).

    Returns ``None`` for anything that is not a plain substitution, rather than guessing —
    the caller then falls back to the one-letter form or a broader query.
    """
    s = three_to_one(protein)  # normalize first so mixed input still works
    m = _SUBST_RE.match(s)
    if not m:
        return None
    ref, pos, alt = m.groups()
    alt3 = "Ter" if alt == "*" else _AA_1TO3[alt]
    return f"p.{_AA_1TO3[ref]}{pos}{alt3}"


def civic_variant_name(variant: str) -> str:
    """The spelling CIViC uses in a molecular-profile name: one-letter, no ``p.`` prefix.

    ``p.L858R`` / ``p.Leu858Arg`` / ``L858R`` all -> ``L858R``.
    """
    return three_to_one(variant)


def civic_profile_name(gene: str, variant: str) -> str:
    """CIViC single-variant molecular-profile name, e.g. ``EGFR L858R``."""
    return f"{gene.strip().upper()} {civic_variant_name(variant)}"


def is_substitution(variant: str) -> bool:
    """True if the variant is a plain single-residue protein substitution."""
    return bool(_SUBST_RE.match(three_to_one(variant)))

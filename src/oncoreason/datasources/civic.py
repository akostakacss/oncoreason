"""CIViC connector (PUBLIC, CC0) — the mandatory evidence backbone.

CIViC = expert-curated *somatic* clinical interpretations. It is NOT a lookup table: each
evidence item carries an evidence **level** (A strongest … E weakest), a **direction**
(SUPPORTS / DOES_NOT_SUPPORT), a **significance** (e.g. SENSITIVITYRESPONSE, RESISTANCE),
and a **type** (predictive / prognostic / diagnostic / …). Turning that into "this variant
implies this drug" is *interpretation* and happens in the agent (Stage 1), not here. This
connector's job is to return the raw, cited evidence faithfully.

Design notes:
- Variant strings are normalized (Gate 3) before querying — CIViC writes ``L858R``, callers
  may pass ``p.L858R`` / ``p.Leu858Arg``. See ``oncoreason.variants``.
- Results are cached on disk (CC0 data, safe to keep) so a given query is reproducible and
  we don't hammer the public API. Delete the cache dir to refresh.
- ``variantOrigin`` sets ``is_somatic`` per item, keeping germline (PREDISPOSING) items
  honestly separated from somatic therapeutic evidence downstream.

API: https://civicdb.org/api/graphql   ·   Phase 1 (connector) + Phase 2 (cases).
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

from ..variants import civic_profile_name
from .base import DataSource, Evidence, EvidenceKind, EvidenceQuery

_API_URL = "https://civicdb.org/api/graphql"

# CIViC evidenceType -> our EvidenceKind. Predisposing/oncogenic/functional are about the
# variant's nature, mapped to PATHOGENICITY; therapeutic evidence is PREDICTIVE.
_TYPE_TO_KIND = {
    "PREDICTIVE": EvidenceKind.PREDICTIVE,
    "PROGNOSTIC": EvidenceKind.PROGNOSTIC,
    "DIAGNOSTIC": EvidenceKind.DIAGNOSTIC,
    "PREDISPOSING": EvidenceKind.PATHOGENICITY,
    "ONCOGENIC": EvidenceKind.PATHOGENICITY,
    "FUNCTIONAL": EvidenceKind.PATHOGENICITY,
}

_QUERY = """
query($mp:String, $first:Int){
  evidenceItems(molecularProfileName:$mp, first:$first){
    nodes{
      id evidenceType evidenceLevel evidenceDirection significance evidenceRating
      status variantOrigin
      therapies{ name }
      disease{ name doid }
      molecularProfile{
        name
        variants{
          id name
          ... on GeneVariant { alleleRegistryId clinvarIds hgvsDescriptions }
        }
      }
      source{ citationId sourceType }
    }
  }
}
"""


def _canonical_ids(molecular_profile: dict) -> dict:
    """Collect the standard cross-database identifiers CIViC stores on the variant(s).

    The CAID (ClinGen Allele Registry canonical id, e.g. ``CA126713``) is the join key we
    use to match the same allele across sources — string re-spelling is NOT the mechanism.
    ClinVar ids and full HGVS descriptions come along for free. Compound profiles (>1 variant)
    contribute all of theirs.
    """
    caids, clinvar, hgvs = [], [], []
    for v in molecular_profile.get("variants") or []:
        if v.get("alleleRegistryId"):
            caids.append(v["alleleRegistryId"])
        clinvar += v.get("clinvarIds") or []
        hgvs += v.get("hgvsDescriptions") or []
    return {
        "caid": caids[0] if caids else None,   # primary join key
        "caids": caids,                        # all, for compound profiles
        "clinvar_ids": sorted(set(clinvar)),
        "hgvs": hgvs,
    }


def _origin_is_somatic(origin: str | None) -> bool | None:
    if not origin:
        return None
    o = origin.upper()
    if "GERMLINE" in o:
        return False
    if "SOMATIC" in o:
        return True
    return None  # MIXED / UNKNOWN / NA -> unknown, don't guess


class CIViCSource:
    name = "civic"
    is_controlled = False

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        accepted_only: bool = True,
        min_rating: int | None = None,
        timeout: float = 25.0,
    ) -> None:
        # Default cache under data/public/ (git-ignored, regenerable). CC0 data — safe.
        if cache_dir is None:
            from .. import REPO_ROOT

            cache_dir = REPO_ROOT / "data" / "public" / "cache" / "civic"
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.accepted_only = accepted_only  # drop SUBMITTED/REJECTED by default
        self.min_rating = min_rating        # optional evidence-rating (1-5) floor
        self.timeout = timeout

    # -- transport -----------------------------------------------------------------
    def _post(self, mp_name: str, first: int) -> list[dict]:
        body = json.dumps({"query": _QUERY, "variables": {"mp": mp_name, "first": first}})
        req = urllib.request.Request(
            _API_URL, data=body.encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            doc = json.loads(resp.read())
        if "errors" in doc:
            raise RuntimeError(f"CIViC GraphQL error for {mp_name!r}: {doc['errors']}")
        return doc["data"]["evidenceItems"]["nodes"]

    def _cache_path(self, mp_name: str, first: int) -> Path:
        key = hashlib.sha1(f"{mp_name}|{first}".encode()).hexdigest()[:16]
        slug = re.sub(r"[^A-Za-z0-9]+", "_", mp_name).strip("_")
        return self.cache_dir / f"{slug}__{key}.json"

    def _fetch_nodes(self, mp_name: str, first: int) -> list[dict]:
        cache = self._cache_path(mp_name, first)
        if self.use_cache and cache.exists():
            return json.loads(cache.read_text())
        nodes = self._post(mp_name, first)
        if self.use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(nodes))
        return nodes

    # -- interface -----------------------------------------------------------------
    def retrieve(self, query: EvidenceQuery) -> list[Evidence]:
        """Return CIViC evidence for ``query.gene`` + ``query.variant`` (both required).

        ``query.tumor_type`` is an optional *soft* filter: matching-disease items float to
        the top but non-matching items are still returned (adjudicating tumor-type relevance
        is the agent's job, not the connector's). Ordering: level (A→E), then rating.
        """
        if not query.gene or not query.variant:
            raise ValueError("CIViC retrieve needs both gene and variant (normalized HGVS ok).")
        mp_name = civic_profile_name(query.gene, query.variant)
        # over-fetch so post-filters (status/rating/disease) still leave top_k
        nodes = self._fetch_nodes(mp_name, max(query.top_k * 4, 20))

        out: list[Evidence] = []
        for n in nodes:
            if self.accepted_only and n.get("status") != "ACCEPTED":
                continue
            rating = n.get("evidenceRating")
            if self.min_rating is not None and (rating or 0) < self.min_rating:
                continue
            therapies = [t["name"] for t in (n.get("therapies") or [])]
            disease = (n.get("disease") or {}).get("name")
            level = n.get("evidenceLevel")
            direction = n.get("evidenceDirection")
            significance = n.get("significance")
            kind = _TYPE_TO_KIND.get(n.get("evidenceType"), EvidenceKind.LITERATURE)
            src = n.get("source") or {}
            ids = _canonical_ids(n.get("molecularProfile") or {})

            claim = f"{mp_name}: {direction} {significance}".replace("_", " ")
            if therapies:
                claim += " to " + " + ".join(therapies)
            if disease:
                claim += f" in {disease}"
            claim += f" (CIViC level {level}, rating {rating})"

            out.append(
                Evidence(
                    source=self.name,
                    kind=kind,
                    citation_id=f"civic:EID{n['id']}",
                    summary=claim,
                    evidence_level=level,
                    is_somatic=_origin_is_somatic(n.get("variantOrigin")),
                    payload={
                        "evidence_type": n.get("evidenceType"),
                        "evidence_direction": direction,
                        "significance": significance,
                        "evidence_rating": rating,
                        "therapies": therapies,
                        "disease": disease,
                        "disease_doid": (n.get("disease") or {}).get("doid"),
                        "molecular_profile": (n.get("molecularProfile") or {}).get("name"),
                        "source_type": src.get("sourceType"),
                        "source_citation_id": src.get("citationId"),
                        "variant_origin": n.get("variantOrigin"),
                        # canonical cross-database identifiers (the real join keys)
                        "caid": ids["caid"],
                        "caids": ids["caids"],
                        "clinvar_ids": ids["clinvar_ids"],
                        "hgvs": ids["hgvs"],
                    },
                )
            )

        # order: evidence level A(strong)->E(weak), then rating high->low
        level_rank = {c: i for i, c in enumerate("ABCDE")}
        out.sort(key=lambda e: (
            level_rank.get(e.evidence_level or "E", 9),
            -(e.payload.get("evidence_rating") or 0),
        ))
        # soft tumor-type preference: matching-disease items first, none dropped
        if query.tumor_type:
            toks = [t for t in re.split(r"\W+", query.tumor_type.lower()) if len(t) > 2]

            def disease_match(e: Evidence) -> int:
                d = (e.payload.get("disease") or "").lower()
                return 0 if any(t in d for t in toks) else 1

            out.sort(key=disease_match)  # stable: preserves level/rating order within groups
        return out[: query.top_k]


# static type check: this class satisfies the DataSource protocol
_: DataSource = CIViCSource()  # type: ignore[abstract]

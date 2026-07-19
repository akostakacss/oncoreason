"""ClinVar connector (PUBLIC) — variant pathogenicity, kept on its own axis.

ClinVar's job here is the **germline pathogenicity** classification, deliberately separate
from CIViC's somatic *therapeutic* evidence — pooling them is a category error a clinician
spots instantly. Since ClinVar's 2024 restructure a record carries up to three independent
classifications; I surface the germline one and keep the others in the payload:
  - ``germline_classification``     -> the primary Evidence (kind=PATHOGENICITY)
  - ``oncogenicity_classification`` -> payload (Oncogenic/Benign/…)
  - ``somatic_clinical_impact``     -> payload

Review status matters as much as the label: "reviewed by expert panel" (3★) is not the same
as "no assertion criteria" (0★), so it is carried on every item.

Joining, not re-spelling (see the identity-standards note): the primary entry point is
``by_ids`` using the ClinVar Variation IDs CIViC already provides (``clinvarIds``). The
``retrieve`` fallback searches by gene + the three-letter HGVS protein form (``p.Leu858Arg``)
that ClinVar indexes — produced by ``variants.one_to_three``. The record's Entrez ``geneid``
is captured as the gene-level join key.

Access: NCBI E-utilities (esearch + esummary, JSON). Phase 1 + Phase 2.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

from ..variants import one_to_three
from .base import DataSource, Evidence, EvidenceKind, EvidenceQuery

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


class ClinVarSource:
    name = "clinvar"
    is_controlled = False

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        email: str | None = None,
        api_key: str | None = None,
        timeout: float = 25.0,
    ) -> None:
        if cache_dir is None:
            from .. import REPO_ROOT

            cache_dir = REPO_ROOT / "data" / "public" / "cache" / "clinvar"
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        # NCBI etiquette: identify the tool; email/api_key lift rate limits. Never hard-code
        # a personal email into the repo — read it from the environment if the user set it.
        self.email = email or os.environ.get("NCBI_EMAIL")
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self.timeout = timeout

    # -- transport -----------------------------------------------------------------
    def _params(self, **p) -> dict:
        p["tool"] = "oncoreason"
        if self.email:
            p["email"] = self.email
        if self.api_key:
            p["api_key"] = self.api_key
        return p

    def _get(self, fn: str, **p) -> bytes:
        url = _EUTILS + fn + "?" + urllib.parse.urlencode(self._params(**p))
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            return resp.read()

    def _esearch(self, term: str, retmax: int) -> list[str]:
        d = json.loads(self._get("esearch.fcgi", db="clinvar", term=term,
                                  retmode="json", retmax=str(retmax)))
        return d.get("esearchresult", {}).get("idlist", [])

    def _esummary(self, ids: list[str]) -> dict:
        """Batch esummary for ClinVar variation ids, with a per-id disk cache (stable data)."""
        result: dict = {}
        missing: list[str] = []
        for i in ids:
            cache = self.cache_dir / f"{i}.json"
            if self.use_cache and cache.exists():
                result[i] = json.loads(cache.read_text())
            else:
                missing.append(i)
        if missing:
            d = json.loads(self._get("esummary.fcgi", db="clinvar",
                                     id=",".join(missing), retmode="json"))
            res = d.get("result", {})
            for i in missing:
                rec = res.get(i)
                if rec is None:
                    continue
                result[i] = rec
                if self.use_cache:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    (self.cache_dir / f"{i}.json").write_text(json.dumps(rec))
        return result

    # -- mapping -------------------------------------------------------------------
    @staticmethod
    def _to_evidence(uid: str, rec: dict) -> Evidence | None:
        germ = rec.get("germline_classification") or {}
        desc = germ.get("description")
        if not desc:
            return None  # no germline classification on this record
        genes = rec.get("genes") or []
        entrez = genes[0].get("geneid") if genes else None
        accession = rec.get("accession") or f"VCV{int(uid):09d}"
        review = germ.get("review_status")

        summary = f"{rec.get('title', accession)}: germline classification '{desc}'"
        if review:
            summary += f" ({review})"

        return Evidence(
            source="clinvar",
            kind=EvidenceKind.PATHOGENICITY,
            citation_id=f"clinvar:{accession}",
            summary=summary,
            evidence_level=review,          # ClinVar's confidence lives in the review status
            is_somatic=False,               # germline axis — kept distinct from CIViC (somatic)
            payload={
                "variation_id": uid,
                "accession": accession,
                "title": rec.get("title"),
                "protein_change": rec.get("protein_change"),
                "entrez_gene_id": entrez,   # gene-level join key
                "gene_symbol": genes[0].get("symbol") if genes else None,
                "germline_description": desc,
                "germline_review_status": review,
                "germline_last_evaluated": germ.get("last_evaluated"),
                "oncogenicity_classification": rec.get("oncogenicity_classification"),
                "somatic_clinical_impact": rec.get("somatic_clinical_impact"),
            },
        )

    # -- interface -----------------------------------------------------------------
    def by_ids(self, variation_ids: list[str]) -> list[Evidence]:
        """Primary path: resolve known ClinVar Variation IDs (e.g. from CIViC ``clinvarIds``)."""
        ids = [str(i) for i in variation_ids if str(i).strip()]
        if not ids:
            return []
        recs = self._esummary(ids)
        out = [self._to_evidence(i, recs[i]) for i in ids if i in recs]
        return [e for e in out if e is not None]

    def retrieve(self, query: EvidenceQuery) -> list[Evidence]:
        """Search by gene + variant. Prefer ``by_ids`` when ClinVar IDs are already known.

        Uses the three-letter HGVS protein form ClinVar indexes (``p.Leu858Arg``); falls back
        to the raw variant string for non-substitutions (e.g. ``exon 19 deletion``).
        """
        if not query.gene or not query.variant:
            raise ValueError("ClinVar retrieve needs both gene and variant (or use by_ids).")
        hgvs_p = one_to_three(query.variant) or query.variant
        term = f"{query.gene}[gene] AND {hgvs_p}"
        ids = self._esearch(term, retmax=max(query.top_k * 2, 10))
        return self.by_ids(ids)[: query.top_k]


_: DataSource = ClinVarSource()  # type: ignore[abstract]

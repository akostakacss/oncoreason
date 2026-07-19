"""cBioPortal connector (PUBLIC, per-study) — real molecular profiles for cases.

Purpose: sample REAL co-occurring alteration patterns (biological plausibility) instead of
hand-composing cases. cBioPortal is NOT an evidence source and NOT an outcome source here —
it supplies the raw tumor profiles that CIViC/ClinVar then annotate.

Hard rules (screened, not assumed):
  1. OPEN studies only. GENIE and other restricted-redistribution studies are excluded.
  2. NEVER read survival/treatment-outcome fields — retrospective outcome validation is a
     confounded causal-inference study, out of scope (Roadmap only).
  3. Panel study silence on a gene != wild-type — gate any "no mutation" inference on
     per-sample panel membership (see `gene_panel_ids` / `panel_genes`).

QC posture: this source is notoriously messy, so every record is emitted in a *normalized*
shape (canonical build label, Entrez id, placeholder '.' -> None) and passed through
`oncoreason.cases.qc.run_qc` before use. See the Phase-2 QC report.

API: https://www.cbioportal.org/api   ·   Phase 2.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from ..cases.qc import canonical_build, normalize_missing
from .base import DataSource, Evidence, EvidenceQuery

_API = "https://www.cbioportal.org/api"

# substrings that mark a study as restricted-redistribution or otherwise not for this repo.
_EXCLUDE_SUBSTR = ("genie",)

# Vetted open lung studies (curated, WES, hg19, one sample/patient). Overridable at init.
# Deliberately NOT the TCGA reprocessings (luad_tcga / luad_tcga_gdc) — same patients, a
# cross-study contamination trap (QC finding 4).
_DEFAULT_ALLOWED = (
    "luad_tcga_pan_can_atlas_2018",
    "lusc_tcga_pan_can_atlas_2018",
)

_LUNG_KEYS = ("lung", "nsclc", "luad", "lusc")


class CBioPortalSource:
    name = "cbioportal"
    is_controlled = False

    def __init__(
        self,
        allowed_studies: tuple[str, ...] | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        timeout: float = 90.0,
    ) -> None:
        self.allowed_studies = allowed_studies or _DEFAULT_ALLOWED
        if cache_dir is None:
            from .. import REPO_ROOT

            cache_dir = REPO_ROOT / "data" / "public" / "cache" / "cbioportal"
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.timeout = timeout

    # -- transport (cached) --------------------------------------------------------
    def _cache(self, key: str) -> Path:
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{h}.json"

    def _request(self, method: str, path: str, body: dict | None = None):
        key = f"{method} {path} {json.dumps(body, sort_keys=True) if body else ''}"
        cache = self._cache(key)
        if self.use_cache and cache.exists():
            return json.loads(cache.read_text())
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            _API + path, data=data, method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())
        if self.use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(out))
        return out

    # -- study scoping (step 1) ----------------------------------------------------
    def discover_lung_studies(self, include_restricted: bool = False) -> list[dict]:
        """List lung studies with build + size, GENIE/restricted excluded. Scoping helper —
        the shipped allow-list is curated from this, not taken wholesale."""
        out = []
        for s in self._request("GET", "/studies"):
            blob = (s.get("name", "") + s.get("cancerTypeId", "")).lower()
            if not any(k in blob for k in _LUNG_KEYS):
                continue
            if not include_restricted and any(x in s["studyId"].lower() for x in _EXCLUDE_SUBSTR):
                continue
            out.append({
                "study_id": s["studyId"],
                "name": s.get("name"),
                "build": canonical_build(s.get("referenceGenome")),
                "n_samples": s.get("allSampleCount", 0),
            })
        return sorted(out, key=lambda x: -x["n_samples"])

    def _assert_allowed(self, study_id: str) -> None:
        if any(x in study_id.lower() for x in _EXCLUDE_SUBSTR):
            raise ValueError(f"study {study_id!r} is restricted-redistribution (excluded).")
        if study_id not in self.allowed_studies:
            raise ValueError(
                f"study {study_id!r} not in allow-list {self.allowed_studies}. "
                "Add it explicitly after scoping (build/assay/open-status).")

    # -- mutations (step 2), emitted normalized ------------------------------------
    def fetch_mutations(self, study_id: str, entrez_ids: list[int]) -> list[dict]:
        """Return normalized somatic-mutation records for the given genes in one study."""
        self._assert_allowed(study_id)
        raw = self._request(
            "POST",
            f"/molecular-profiles/{study_id}_mutations/mutations/fetch?projection=DETAILED",
            {"sampleListId": f"{study_id}_all", "entrezGeneIds": list(entrez_ids)},
        )
        return [self._normalize(m) for m in raw]

    @staticmethod
    def _normalize(m: dict) -> dict:
        """cBioPortal record -> stable internal shape. Canonicalize build, resolve '.'->None,
        keep the Entrez id as the gene join key. Outcome/survival fields are never read."""
        gene = m.get("gene") or {}
        return {
            "study_id": m.get("studyId"),
            "sample_id": m.get("sampleId"),
            "patient_id": m.get("patientId"),
            "gene_symbol": gene.get("hugoGeneSymbol") or m.get("hugoGeneSymbol"),
            "entrez_gene_id": m.get("entrezGeneId"),
            "protein_change": normalize_missing(m.get("proteinChange")),
            "mutation_type": normalize_missing(m.get("mutationType")),
            "chrom": m.get("chr"),
            "start": m.get("startPosition"),
            "end": m.get("endPosition"),
            "ref": m.get("referenceAllele"),
            "alt": m.get("variantAllele"),
            "build": m.get("ncbiBuild"),  # canonicalized by qc.canonical_build on use
            "tumor_alt_count": m.get("tumorAltCount"),
            "tumor_ref_count": m.get("tumorRefCount"),
            "mutation_status": normalize_missing(m.get("mutationStatus")),
            "validation_status": normalize_missing(m.get("validationStatus")),
        }

    # -- panel coverage (for the 'absence != wild-type' gate) ----------------------
    def gene_panel_ids(self, study_id: str) -> list[str]:
        self._assert_allowed(study_id)
        data = self._request(
            "POST", f"/molecular-profiles/{study_id}_mutations/gene-panel-data/fetch",
            {"sampleListId": f"{study_id}_all"})
        return sorted({d.get("genePanelId") for d in data if d.get("genePanelId")})

    def panel_genes(self, panel_id: str) -> set[str]:
        d = self._request("GET", f"/gene-panels/{panel_id}")
        return {g["hugoGeneSymbol"] for g in d.get("genes", [])}

    # -- DataSource protocol: cBioPortal is a case source, not an evidence source ---
    def retrieve(self, query: EvidenceQuery) -> list[Evidence]:
        raise NotImplementedError(
            "cBioPortal supplies molecular profiles, not evidence. Use fetch_mutations() / "
            "discover_lung_studies(); annotate the profiles with CIViC/ClinVar for evidence.")


_: DataSource = CBioPortalSource()  # type: ignore[abstract]

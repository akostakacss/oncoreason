# Licensing & data governance

This proof of concept is built to a strict rule: **no patient data and no controlled or
licensed data are used or committed.** The architecture is nonetheless *ready to receive*
controlled data (as a licensed lab would have) via empty connector slots — see below.

## Principle: freely readable ≠ freely redistributable

Some sources are free to *read* but their content is copyrighted and must not be committed
to a public repository. The repo therefore stores only **derived representations** of such
sources (embeddings, chunk indices, or structured recommendations authored by us), never the
raw text, and loads any raw text at runtime from a git-ignored local path.

## Source status

| Source | In the repo? | Notes |
|--------|--------------|-------|
| **cBioPortal** (open studies) | Derived profiles only | Screen per study; **exclude GENIE** and restricted-terms studies. Outcomes not used. |
| **CIViC** | Yes (CC0) | Public-domain; the mandatory evidence backbone. |
| **ClinVar** | Yes | Public. Germline — kept separate from somatic (CIViC). |
| **ClinicalTrials.gov** | Query results only | Public API v2. |
| **PubMed / PMC** | Derived / OA subset | Respect the PMC Open Access subset for full text. |
| **ESMO / ESCAT** | **Derived index only** | Guideline **text is copyrighted** — never commit it. Raw text loaded at runtime. |
| **OncoKB** | **No** | Academic license/token required. Behind the controlled connector. |
| **NCCN** | **No** | Licensed. Behind the controlled connector. |
| **Institutional / kaiko multimodal data** | **No** | Not available here; the connector slot is reserved for it. |

## The controlled-data connector — reserved, and shipped empty

All evidence sources implement one interface (`oncoreason.datasources.base.DataSource`).

- **Public connectors** (`civic`, `clinvar`, `cbioportal`) are implemented and shipped.
- **Controlled connectors** (`nccn`, `oncokb_full`, `institutional`) are **stubs**: they read
  from a git-ignored path under `data/controlled/` (or an env var) and raise
  `ControlledSourceNotConfigured` with a pointer here if the data is absent.
- **`data/controlled/` is always empty in the repo** and is git-ignored in full.

A licensed user (e.g. the host lab) plugs their data into `data/controlled/<source>/` and the
same pipeline runs unchanged. The empty, clearly-labelled slot is deliberate: it demonstrates
the design accommodates institutional data **without distributing anything restricted.**

## Code license

Code is released under the MIT license (see `pyproject.toml`). This licenses the *code only*,
not any data retrieved through it — each data source retains its own terms.

# Phase 1 — Connectors & canonical-ID join

**Status:** ✅ done · **Compute:** CPU-local (network only)

## Goal

Build the license-safe `DataSource` connector layer and — the load-bearing part — make the
evidence sources **join on canonical identifiers, never on name strings**. Phase 0 Gate 3
proved that a name-string join fails silently (a mismatch returns *no evidence*, not an error),
so the whole downstream pipeline is only as trustworthy as this join.

## What was built

### The `DataSource` interface + registry
A common `retrieve(query) -> Evidence[]` contract ([`datasources/base.py`](../src/oncoreason/datasources/base.py)),
a registry ([`datasources/registry.py`](../src/oncoreason/datasources/registry.py), `get_source(...)`),
and the empty controlled-source slots ([`datasources/controlled.py`](../src/oncoreason/datasources/controlled.py))
that raise a clear "controlled source not configured — see LICENSING.md" rather than ever
shipping restricted data. The empty slot is a deliberate application signal: *the architecture
is ready to receive the lab's data without breaking a single license.*

### CIViC connector — the mandatory backbone
[`datasources/civic.py`](../src/oncoreason/datasources/civic.py). Somatic, therapeutic evidence
(CC0) via the GraphQL API. Queries `evidenceItems(molecularProfileName:$mp)`, and — the key
extension — pulls each variant's `alleleRegistryId` (CAID), `clinvarIds`, and `hgvsDescriptions`
so the canonical identifiers ride along on `Evidence.payload`. Filters to accepted evidence,
orders by evidence level (A→E) then rating, soft-sorts by tumor type. Maps
PREDICTIVE/PROGNOSTIC/DIAGNOSTIC to evidence kinds and folds PREDISPOSING/ONCOGENIC/FUNCTIONAL
into pathogenicity. Cached under `data/public/cache/civic/`.

### ClinVar connector — germline, kept separate
[`datasources/clinvar.py`](../src/oncoreason/datasources/clinvar.py). Germline pathogenicity via
NCBI E-utilities. The **primary path is `by_ids(...)`** — fed the ClinVar IDs CIViC already
returned, so it is a canonical-ID lookup with **no re-spelling**. Surfaces the post-2024
three-axis classification (germline / oncogenicity / somatic-clinical-impact), the review-status
star rating, and the Entrez gene ID as the join key. Reads `NCBI_EMAIL`/`NCBI_API_KEY` from the
environment — the user's personal email is never hardcoded. Cached under `data/public/cache/clinvar/`.
Germline evidence is kept **separate** from CIViC's somatic evidence — pooling them is a category
error a clinician spots instantly.

### The variant display layer (Gate 3, explicitly *not* the identity layer)
[`variants.py`](../src/oncoreason/variants.py). Reliable three-letter↔one-letter protein
conversion (`p.Leu858Arg` ↔ `L858R`) for substitutions, pass-through for everything else. This
is a **display / re-spelling** concern only; it is documented in its own docstring as explicitly
*not* how identity is established (that is the CAID/ClinVar-ID join above).

## Key decisions

- **Identity = CAID / VRS / Entrez, display = strings.** The single most important architectural
  choice, carried from Gate 3. See the memory note `variant-gene-identity-standards`.
- **Somatic (CIViC) and germline (ClinVar) never pooled** — separated by purpose.
- **License gating is structural, not a convention** — controlled connectors raise rather than
  read absent data; nothing restricted can be committed by construction.
- **Everything is cached and offline-reproducible** under `data/public/cache/` (git-ignored,
  regenerable).

## Outputs

- **Live end-to-end join demonstrated:** a CIViC query returns CAID `CA126713` + ClinVar IDs,
  which drive the ClinVar lookup directly — no name string ever compared. The HER2/ERBB2 alias
  trap (same gene, two symbols) is handled correctly because the join is on the Entrez ID, not
  the symbol.
- **Tests:** `tests/test_variants.py`, `tests/test_civic.py`, `tests/test_clinvar.py` (offline,
  fixture-driven).

## Status

Done and tested. Both public evidence sources are live and join on canonical IDs. Controlled
slots ship empty by design.

## Next

→ [Phase 2 — Case set](PHASE2_case_set.md): use these connectors to annotate real molecular
profiles sampled from cBioPortal.

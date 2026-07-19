# Phase 2 — Case set

**Status:** ✅ done · **Compute:** CPU-local (network only)

## Goal

Turn real, messy public molecular data into a clean, patient-disjoint set of annotated cases —
the target the reasoning model is trained and evaluated against. cBioPortal is the source of
biologically plausible co-occurring alteration patterns; its characteristic failure mode is a
*plausible-looking wrong value*, not a crash, so the whole phase is run with a data-QC-analyst
posture: **treat the source as guilty until checked.**

The full, reviewer-grade account of exactly how cBioPortal data is handled lives in
[`docs/DATA_HANDLING.md`](../docs/DATA_HANDLING.md) — this phase document is the summary.

## What was built

### Connector — cBioPortal ([`datasources/cbioportal.py`](../src/oncoreason/datasources/cbioportal.py))
Allow-list gated (`_assert_allowed` refuses anything not explicitly listed). Ships with two
curated, whole-exome, single-build lung cohorts: `luad_tcga_pan_can_atlas_2018` and
`lusc_tcga_pan_can_atlas_2018`. Deliberately **excludes** GENIE (restricted redistribution), the
TCGA reprocessings (same patients → leakage), and MSK panel cohorts (the "absence ≠ wild-type"
problem). Normalizes to a stable internal shape: Entrez ID as gene join key, `proteinChange`
as-is, `.`→`None`, and it *never reads* survival/outcome fields. `retrieve()` intentionally
raises — this is a **case source, not an evidence source**.

### QC battery ([`cases/qc.py`](../src/oncoreason/cases/qc.py))
Every ingest runs `run_qc`, which **flags, never silently drops**, with severity grades. Each
check encodes a real anomaly seen in the live data:

| Check | Severity | Catches |
|---|---|---|
| `build_mixed` | high | >1 genome build in a study (coordinates not comparable) |
| `type_vs_protein` | medium | `mutationType` disagreeing with the proteinChange consequence (curation error) |
| `vaf_zero_alt` | medium | a called mutation with zero supporting alt reads |
| `duplicate_row` | low | identical (sample, gene, proteinChange) listed twice |
| `hypermutator` | low | ≥1000 mutations in a sample (passenger swamping) |

The high-consequence traps (build mixing, placeholder status, cross-study contamination, panel
inference) are handled by **selection and design**, so they never reach the ingested data.

### Case builder ([`cases/build.py`](../src/oncoreason/cases/build.py))
`build_profiles` → `annotate_alteration` → `assemble_case` → `split_cases`:
- **Drop vs flag:** `duplicate_row` and `vaf_zero_alt` records are dropped (they corrupt
  aggregation); `type_vs_protein` is kept but recorded on the profile (the variant is real, only
  its label is suspect).
- **One representative sample per patient = the richest profile** (most alterations, deterministic
  tie-break) — not shuffle-dependent, and clinically sensible.
- **Cross-study patient de-dup** — the structural defence against the TCGA-reprocessing leakage
  class, independent of the allow-list.
- **Annotation attaches evidence, not verdicts** — CIViC (somatic) + ClinVar (germline) joined on
  canonical IDs. `actionable` is a screening flag (CIViC level A/B predictive sensitivity), **not**
  an ESCAT tier. `gold = None` by design — turning evidence into a tiered recommendation is
  interpretation, deferred to Phase 4.
- **Patient-disjoint split** — because each case is already one patient, a seeded random split
  cannot leak a patient across train/val/test/generalization.

## Outputs (the generated artifacts)

### QC report on the shipped studies → [`qc/`](../qc/)
Run [`20260718-2203-cbio-qc.md`](../qc/20260718-2203-cbio-qc.md) over 1,871 records:

| Study | Records | Patients | Build | Findings (high/med/low) |
|---|---|---|---|---|
| `luad_tcga_pan_can_atlas_2018` | 1,087 | 499 | GRCh37 | 7 (0 / 4 / 3) |
| `lusc_tcga_pan_can_atlas_2018` | 784 | 435 | GRCh37 | 2 (0 / 2 / 0) |

**0 high-severity findings**, ~0.5% flag rate, all low-grade curation noise, none touching the
driver variants cases are built around. Indexed in [`qc/INDEX.tsv`](../qc/INDEX.tsv); machine-
readable copy in [`20260718-2203-cbio-qc.json`](../qc/20260718-2203-cbio-qc.json). Regenerate
with [`scripts/cbio_qc_report.py`](../scripts/cbio_qc_report.py).

### The case set → [`data/cases/`](../data/cases/)
Built by [`scripts/build_cases.py`](../scripts/build_cases.py) (seed 17, deterministic).
Summary from [`manifest.json`](../data/cases/manifest.json):

- **50 cases · 99 alterations · 13 actionable alterations · 0 cases with QC flags surviving**
- Splits: **train 35 · val 5 · test 5 · generalization 5** (patient-disjoint by construction)
- Even across cohorts: 25 LUAD · 25 LUSC
- 16 driver genes (EGFR, KRAS, ALK, ROS1, BRAF, MET, RET, ERBB2, NTRK1, PIK3CA, TP53, STK11,
  KEAP1, NF1, RB1, NRAS)
- Cases in [`cases.jsonl`](../data/cases/cases.jsonl) (git-ignored, regenerable). Count is one
  flag away from the plan's 200–300 target (`--per-study N`); kept at 50 to prove the pipeline
  without a long annotation run.

### Spot-check that the canonical-ID join actually works
ERBB2 (≡ HER2) Y772_A775dup resolves to CAID `CA135369`, CIViC **level-A** sensitivity to
trastuzumab deruxtecan, and a ClinVar "Likely pathogenic" germline record — the alias trap
handled correctly because the join is on IDs, not the `HER2`/`ERBB2` string.

## Key decisions

- **cBioPortal is a case source only** — outcomes deliberately unused (confounded by indication;
  a full causal-inference study, documented as future work).
- **Flag, don't silently drop**, with a narrow, defensible drop policy for the two checks that
  corrupt aggregation.
- **Cases carry evidence, not gold labels** — the honest boundary between Phase 2 (attachment)
  and Phase 4 (labelling).
- **Point mutations only in v1**; clinical context is honest histology-only (`None` for stage /
  prior lines / performance status rather than imputed).

## Status

Done and tested (`tests/test_qc.py`, `tests/test_cbioportal.py`, `tests/test_build.py`). Pipeline
runs end to end on real data. Full documentation in [`docs/DATA_HANDLING.md`](../docs/DATA_HANDLING.md).

## Next

→ **Phase 3 — Reasoning scaffolding** (agents, tools, retrieval), or scale the case count toward
200–300 first. Phase 4 (guideline-verified gold labelling) is where `gold = None` gets resolved
into ESCAT-tiered recommendations.

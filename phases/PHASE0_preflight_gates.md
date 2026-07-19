# Phase 0 — Pre-flight gates

**Status:** ✅ done · **Compute:** Kaggle T4 (Gate 1), CPU-local (rest)

## Goal

De-risk the load-bearing assumptions with cheap spikes *before* committing weeks to
architecture. Five gates, each ~half a day, each with an explicit stop/continue decision.
The point of a gate is to fail cheaply: if a foundational assumption is wrong, find out now,
not after building on top of it.

## The five gates and their verdicts

### Gate 1 — Can the base model even hold a tool chain together?
Ran the base policy on hand-built lung cases with a tool interface, on a real T4.

- **Escalation decision made:** Qwen2.5-**1.5B**-Instruct was not reliable enough at holding
  a multi-step tool chain; escalated to **Qwen2.5-3B-Instruct** (still T4-feasible, 4-bit
  available if memory bites). This is exactly the "decide the model before building around it"
  purpose of the gate.
- **Evidence it works:** on the 3B model the traces show clean multi-step tool use *and* —
  critically — **correct deferral**. Case 3 (KRAS G12C, post-platinum) returned
  `Recommendation: defer` with a stated reason, rather than confabulating a therapy. That the
  base model already knows when to abstain is the signal that the whole PRM→confidence→abstain
  loop has something real to sharpen.
- **Verdict: CONTINUE** on Qwen2.5-3B-Instruct.
- **Artifacts:** [`runs/20260718-1903-v6-3b-cachetest.md`](../runs/20260718-1903-v6-3b-cachetest.md)
  (full stdout of the three-case probe), [`runs/INDEX.tsv`](../runs/INDEX.tsv),
  [`notebooks/phase0_gate1_smoke_test.ipynb`](../notebooks/phase0_gate1_smoke_test.ipynb).
  Confirmed the GPU path end to end: 2×Tesla T4, torch 2.10+cu128, local model mount, ~13–21 s/case.

### Gate 2 — Build 20 cases by hand, define "correct"
Establish what "correct" means, the answer space, and a reproducible scoring rubric *before*
automating any labels. This is the pre-registered gold standard the later PRM is measured against.
- **Verdict: CONTINUE** — rubric drafted; the answer space is molecular-interpretation →
  therapy recommendation with an explicit abstain option, matching the case schema built in
  Phase 2.

### Gate 3 — Variant-normalization spike
The silent-failure trap: variant nomenclature mismatches return *empty results, not errors*,
so a broken join is invisible. Probed matching variants across cBioPortal / CIViC / ClinVar.
- **Key finding that reshaped the architecture:** identity must be carried on **canonical IDs**
  (ClinGen CAID, GA4GH VRS, Entrez gene ID), never on name strings. CIViC hands back the CAID
  and ClinVar IDs for free, so the join is an ID lookup, not string munging. Name re-spelling
  (`p.Leu858Arg` ↔ `L858R`) is a **display-layer** concern only.
- **Verdict: CONTINUE** — implemented as the Phase 1 `variants.py` display layer + the CAID/
  ClinVar-ID join. Recorded in the memory note `variant-gene-identity-standards`.

### Gate 4 — Frontier baseline
Run a frontier model zero-shot on the gold cases to establish the ceiling and pre-answer
"why not just use a frontier model?".
- **Verdict: CONTINUE** — the teacher/policy split is set: `claude-opus-4-8` / `claude-sonnet-5`
  as data-generation *teachers* only; the open-weights Qwen policy is the object actually
  post-trained (you cannot post-train a closed API model — the reason the project exists).

### Gate 5 — Data + licensing scoping
Confirm the data sources are usable and license-clean.
- **Verdicts:** cBioPortal lung studies screened per-study (WES, single build) — see Phase 2;
  ESMO/NCCN guideline **text is copyrighted → derived indices only, never committed**;
  controlled sources (NCCN, full OncoKB, institutional) reserved behind empty connector slots
  ([`LICENSING.md`](../LICENSING.md)).
- **Verdict: CONTINUE.**

## Net outcome

All five gates passed. The consequential decisions that flowed into the build:
1. **Policy = Qwen2.5-3B-Instruct** (escalated from 1.5B at Gate 1).
2. **Identity via canonical IDs, not name strings** (Gate 3) — the backbone of Phase 1.
3. **Teacher ≠ policy**; frontier models label/generate, open weights get trained (Gate 4).
4. **Copyright discipline**: derived indices only; controlled slots ship empty (Gate 5).

## Next

→ [Phase 1 — Connectors & canonical-ID join](PHASE1_connectors.md).

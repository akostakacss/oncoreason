# Phase log

A running, human-readable record of what was built in each phase of the project — one
document per phase, in order, so the whole trajectory can be read start to finish.

Each phase document follows the same shape: **Goal → What was built → Key decisions →
Outputs (with the concrete numbers and the files they live in) → Status → Next**. Where a
phase produced durable artifacts (QC reports, the case set, run logs), the document links
straight to them.

For the condensed, evaluative read — what each phase actually **found**, good or bad, one page —
see [`../summary/FINDINGS.md`](../summary/FINDINGS.md) and its companion
[`results_figure.png`](../summary/results_figure.png).

The *why* behind each design decision is recorded in the phase documents themselves; for the
data, see [`docs/DATA_HANDLING.md`](../docs/DATA_HANDLING.md).

## Status at a glance

| Phase | Title | Status | Headline output |
|---|---|---|---|
| 0 | [Pre-flight gates](PHASE0_preflight_gates.md) | ✅ done | Base model probe passed on 3B; tool-chain + deferral confirmed on a T4 |
| 1 | [Connectors & canonical-ID join](PHASE1_connectors.md) | ✅ done | CIViC + ClinVar live, joined on CAID/ClinVar IDs — no name-string matching |
| 2 | [Case set](PHASE2_case_set.md) | ✅ done | 50 real, QC-filtered, patient-disjoint annotated cases from cBioPortal |
| 3 | [Reasoning scaffolding (agents, tools, retrieval)](PHASE3_scaffolding.md) | 🟡 core built | Planner→specialists→synthesizer; auditable traces; abstain path; BM25/hybrid retrieval |
| 4 | [Supervision (guideline-verified step labels)](PHASE4_supervision.md) | 🟡 core built | RAG-as-a-Judge step labelling; PRM-dataset builder; audit kappa |
| 5 | [Post-training (PRM, verifier-guided)](PHASE5_posttraining.md) | 🟡 core built on CPU, **GPU stage started** | PRM trained+calibrated, held-out acc **0.914**; min-rule Best-of-N; DPO pairs + reward-hacking guard; ModernBERT + real policy sampling + LoRA RFT/DPO now running on Kaggle T4 |
| 6 | [Clinical evaluation](PHASE6_evaluation.md) | 🟡 core built | 8 metrics with bootstrap CIs, run on 50 real cases; found the abstention defect |
| 7 | [Packaging](PHASE7_packaging.md) | ✅ done | README rewritten around the headline numbers (incl. the two failures); reproducibility re-verified fresh (71/71 tests, pipeline output matched to the last digit) |

Legend: ✅ done · 🟡 in progress · ⬜ not started.

## How to read this alongside the code

- **What the pipeline does** → the phase documents here.
- **How to run it** → [`README.md`](../README.md#quickstart-local-cpu).
- **Why each decision was made** → the *Key decisions* section of each phase document.
- **The actual generated outputs** → [`qc/`](../qc/), [`data/cases/`](../data/cases/),
  [`runs/`](../runs/), each linked from the relevant phase document.

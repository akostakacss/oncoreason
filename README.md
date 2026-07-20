# oncoreason — oncology reasoning model (proof of concept)

> **What this is.** A proof-of-concept home project built independently by Ákos Takács, outside
> any employer, as part of an application for the Postdoctoral Researcher position in Michael
> Moor's lab (Multimodal Reasoning Models for Oncology, ETH D-BSSE Basel). It claims no original
> methodology. It is a small-scale synthesis and implementation of **the lab's own published
> research direction** — Med-PRM, Process Reward Agents (PRA), AgentClinic, Almanac, RadAgent,
> MTBBench, Med-Flamingo, GMAI — built to demonstrate genuine engagement with that work ahead of
> an interview, not to claim authorship of the underlying ideas. Full attribution, paper by paper,
> is in [`summary/FINDINGS.md`](summary/FINDINGS.md#what-this-is-and-whose-work-it-is-based-on).

A small, reproducible **post-training + evaluation pipeline** for oncology reasoning:
an agent that answers lung-cancer molecular-interpretation questions using retrieval and
tools, shows its reasoning, cites its evidence, and abstains when unsure — plus a trained
**process reward model (PRM)** used for verifier-guided post-training (RFT / DPO) and a
clinical-grade evaluation harness.

![The system in one page: population evidence and an individual molecular profile are
adjudicated into an auditable recommendation](concept_figure.png)

> **Status: all 7 phases built and tested on CPU; the GPU stage is run and measured.**
> Data connectors, case construction, agent scaffolding, retrieval, guideline-verified process
> supervision, a trained and calibrated process reward model, and an 8-metric clinical
> evaluation harness all run end to end on CPU in one command (`python scripts/run_pipeline.py`),
> with **71 passing tests**. The GPU stage (ModernBERT PRM backend, verifier-guided rejection
> sampling, LoRA RFT/DPO) has been started on a Kaggle T4 — those steps were wired behind
> explicit enable-gates that raise a clear message rather than pretending, and are now being
> exercised for real. See [`phases/`](phases/) for a per-phase write-up including
> results, trade-offs and honest limitations, and [`phases/PHASE5_posttraining.md`](phases/PHASE5_posttraining.md)
> for the GPU-stage log as it's updated.

## Headline results, including the two that are bad news

Every number below is produced by `scripts/run_pipeline.py` on 50 real, patient-disjoint cases
and lands in `results/`, so none of it is asserted without an artifact behind it.

| Result | Number | Reading it |
|---|---|---|
| PRM held-out accuracy | ~~0.914~~ → **0.881** [0.798, 0.951] | **Superseded, and the correction is the result.** 0.914 was measured on *constructed* negatives. Sampling a real policy and re-measuring gave **0.530** — barely above chance. Redesigning the label (citation *correctness*, not *presence*) recovered 0.881. Caveat: ~65% of that label is tumour-type consistency. [Full trail](results/gpu_stage/). |
| Verifier-guided selection (Best-of-N) | +0.089 [**−0.059**, +0.238] | **Not distinguishable from zero.** Step-level accuracy did not translate into demonstrable trace-level selection at n=15 held-out cases. Reported because it is the honest answer, not the hoped-for one. |
| Guideline concordance | 0.96 top-1 | The **least** meaningful number here: the gold label is derived from the same guideline index the agent retrieves over. Reported with that caveat attached, not as a headline. |
| **Abstention** | fixed: **0/50 → 24/50** | **The harness found a real bug and I fixed it.** The system never abstained — traced to a strict `<` where a fallback guideline's confidence (0.50) tied the abstain threshold (0.50) exactly. One-character fix; molecular interpretation agreement moved 0.24 → **0.72**. Fixing it surfaced a subtler, still-open finding: 14/38 non-actionable cases still recommend, because they're guideline-backed even without top-tier CIViC evidence — a gold-standard definition gap, not a hallucination. See [Phase 6 addendum](phases/PHASE6_evaluation.md#addendum--the-abstain-threshold-fix). |
| Calibration (ECE) | **0.304** | Poor. Confidence tracks *which guideline branch fired*, not *how likely the answer is to be right*. Stated plainly rather than smoothed over. |
| Information gathering | r = −0.256 (p = 0.073) | Opposite sign to what MTBBench found for LLM agents — here, more evidence means a harder, more contested case, not better gathering. Diagnosed, not hidden. |

I would rather report a system that finds and names its own failures — and then fixes what's
fixable and precisely characterizes what isn't — than one that only reports flattering numbers.
The full reasoning, trade-offs and alternatives considered for each of these are in the per-phase
documents linked above and in [`summary/FINDINGS.md`](summary/FINDINGS.md).

> **Scope & safety.** Proof of concept, research use only — **not** a medical device and not
> for clinical decisions. **No patient data** and **no controlled/licensed data** are used or
> committed; see [`LICENSING.md`](LICENSING.md).

## The pipeline

![Build roadmap: eight phases from pre-flight gates to packaging, with the status and
headline artifact of each](roadmap_figure.png)

Both figures are generated, not drawn: `python scripts/make_concept_figure.py` and
`python scripts/make_roadmap_figure.py` read the latest `results/*-pipeline.json`, so the
numbers on them cannot drift from the pipeline.

| Stage | Package | What it does |
|------|---------|--------------|
| 1 Data & cases | `oncoreason.datasources`, `oncoreason.cases` | Real public molecular profiles + evidence → cases |
| 2 Scaffolding | `oncoreason.retrieval`, `oncoreason.agents` | Multi-agent retrieval, tools, auditable trace |
| 3 Supervision | `oncoreason.supervision` | Guideline-verified step labels |
| 4 Post-training | `oncoreason.training` | PRM, RFT, DPO, best-of-N, abstention |
| 5 Evaluation | `oncoreason.evaluation` | Concordance, calibration, deferral, citations |

## Quickstart (local, CPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .               # installs the oncoreason package
pip install -r requirements.txt
pytest -q                      # 71 tests: retrieval, agents, labelling, PRM, evaluation
python scripts/run_pipeline.py # runs Phases 2-6 end to end, reproduces every number above
```

`run_pipeline.py` writes a timestamped JSON to `results/` (indexed in `results/INDEX.tsv`) —
the headline numbers in this README are read straight off that file, not hand-copied.

GPU steps (ModernBERT PRM backend, policy sampling, RFT/DPO LoRA) run on **Kaggle T4** — see
[`phases/PHASE5_posttraining.md`](phases/PHASE5_posttraining.md) for the hybrid local-dev /
Kaggle-GPU workflow, and `notebooks/phase0_gate1_smoke_test.ipynb` for a first Kaggle run.

## Where to start

1. Read [`summary/FINDINGS.md`](summary/FINDINGS.md) — one page, what each phase actually found
   (not just what was built), plus [`summary/results_figure.png`](summary/results_figure.png).
2. Read [`phases/README.md`](phases/README.md) — what was built, phase by phase, with results,
   trade-offs and honest limitations for each.
3. Skim [`publications/REFERENCES.md`](publications/REFERENCES.md) — the published work this
   builds on.
4. Read [`docs/MTBBENCH_INTEGRATION.md`](docs/MTBBENCH_INTEGRATION.md) — what is next and why:
   a non-circular gold standard, an abstention fix, and longitudinal evaluation.

## What's next (backlog, not a phase)

All 7 phases are built; nothing further is required for the pipeline to be complete and honest
about itself. What remains is scoped, dated backlog:

- **Non-circular gold standard** from MTBBench, to replace the guideline-derived label that Phase
  6 flagged as partially circular — and to reconcile with the narrower CIViC-only "actionable"
  definition that the abstention fix (below) showed disagrees with the guideline specialist's own
  evidence base on 14/50 cases.
- **The Phase 4.3 human audit kappa**, currently asserted rather than measured.
- **A baseline arm** (untrained policy or frontier model), so the headline numbers can be called
  an improvement over something.

~~The abstention fix for the 0/50 finding~~ — done: a strict-inequality tie at the abstain
threshold, fixed in one line, molecular interpretation agreement 0.24 → 0.72. See the
[Phase 6 addendum](phases/PHASE6_evaluation.md#addendum--the-abstain-threshold-fix).

Details and reasoning for each: [`docs/MTBBENCH_INTEGRATION.md`](docs/MTBBENCH_INTEGRATION.md),
and the Honest limitations section of every phase document.

## Repo layout

```
src/oncoreason/
  datasources/   DataSource interface; public connectors + empty controlled stubs
  cases/         Case schema
  retrieval/     BM25 / dense / hybrid retrievers
  agents/        planner → specialists → synthesizer; Trace schema
  supervision/   step segmentation + guideline-verified labelling
  training/      PRM (trained, CPU), RFT/DPO data construction (LoRA training gated to Kaggle)
  evaluation/    8 metrics — concordance, calibration, deferral, citations, information gathering
configs/         default.yaml
notebooks/       Kaggle Gate 1 smoke test
data/public/     public data (gitignored; .gitkeep only)
data/controlled/ controlled data slot — ALWAYS empty in the repo
```

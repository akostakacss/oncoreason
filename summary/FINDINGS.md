# Findings — what each phase actually found

This is the evaluative summary: not what was built (that's [`phases/`](../phases/)) but **what
we found out** at each stage, and how good or bad news it was. Read this first; go to the
per-phase document linked in each section for the method, the numbers, and the trade-offs.

The one-page visual companion to this document is [`results_figure.png`](results_figure.png)
(source: [`results_figure.pdf`](results_figure.pdf), regenerate with
`python scripts/make_results_figure.py`) — see **Standard figures**, below, for what each panel
is and why it's the right chart for that result.

## What this is, and whose work it is based on

This is a **proof-of-concept home project**, built independently by Ákos Takács, outside any
employer, as part of an application for the Postdoctoral Researcher position in Michael Moor's
lab (Multimodal Reasoning Models for Oncology, ETH D-BSSE Basel). It is not original research and
claims no new methodology. It is a **small-scale synthesis and implementation of Prof. Moor's own
published research direction**, built to demonstrate genuine engagement with that body of work
ahead of an interview, not to claim authorship of the underlying ideas.

Every architectural choice below traces back to a specific paper from the lab, cited here rather
than left implicit:

- **Med-PRM** (Yun et al., EMNLP 2025) — guideline-verified, step-level process reward supervision
  via RAG-as-a-Judge. Direct source of Phase 4's labelling method and Phase 5's PRM.
- **Process Reward Agents (PRA)** (ICML 2026) — online, frozen-policy step steering with selective
  retrieval. The documented roadmap beyond this project's Best-of-N / RFT / DPO.
- **AgentClinic** (npj Digital Medicine 2026) — tool-using clinical agent evaluation as a
  first-class axis, not static QA. Direct source of Phase 6's tool-use-reliability metric.
- **Almanac** (Zakka et al., NEJM AI 2024) — retrieval-grounded, cited generation with a
  confidence cutoff feeding abstention. Direct source of the citation-grounding metric and the
  retrieval-confidence framing behind abstention.
- **RadAgent** (2026) — RL-trained, tool-using, evidence-grounded stepwise reasoning with an
  inspectable trace. The vision-modality analogue of this project's text-only scaffold.
- **MTBBench** (Vasilev et al., NeurIPS 2025 Datasets & Benchmarks) — multimodal, longitudinal,
  agentic molecular tumor board benchmark. Source of the information-gathering metric and the
  case for a non-circular, MTB-derived gold standard (`docs/MTBBENCH_INTEGRATION.md`).
- **Med-Flamingo** (Moor et al., ML4H 2023) and **GMAI** (Moor et al., Nature 2023) — the
  multimodal-foundation-model framing this project's imaging extension (deliberately not built)
  is scoped against.

The source papers are listed in [`publications/REFERENCES.md`](../publications/REFERENCES.md). Where the
sections below say "the harness caught X" or "we built Y," read that as *applying methods from
the papers above within a small demonstration*, not as a claim of new science. What is genuinely
this author's own contribution is narrower and stated plainly: the specific engineering of a
small, honest, reproducible pipeline that applies these ideas to lung-cancer molecular
interpretation, and the clinical and biostatistics judgment — from the author's own doctoral and
clinical-research background — brought to the evaluation design.

## Phase-by-phase

### Phase 0 — Pre-flight gates → [full doc](../phases/PHASE0_preflight_gates.md)
**Found:** the 1.5B base model could *not* reliably hold a multi-step tool chain — escalated to
3B before any architecture was built around the smaller model. Also found the 3B model already
knows how to abstain (Case 3, KRAS G12C, returned `defer` unprompted), which is the signal that
verifier-guided abstention has a real behavior to sharpen rather than teach from nothing. And:
identity across evidence sources must ride on canonical IDs, never gene-name strings, or a
mismatch fails **silently** (empty results, not an error).
**Verdict:** ✅ clean. Every gate produced a genuine go/no-go decision that changed the build
(policy size, ID-based joins, teacher/policy split, copyright scope) rather than rubber-stamping
a plan already decided.

### Phase 1 — Connectors & canonical-ID join → [full doc](../phases/PHASE1_connectors.md)
**Found:** the canonical-ID join works on a real alias trap — ERBB2/HER2, two gene symbols for
one Entrez ID — resolved correctly because the join never touches the symbol string. Germline
(ClinVar) and somatic (CIViC) evidence are structurally kept apart, which matters because pooling
them is a category error a clinician would catch immediately.
**Verdict:** ✅ clean. The one architectural bet from Phase 0 (ID-based identity) held up against
a real case, not just a synthetic test.

### Phase 2 — Case set → [full doc](../phases/PHASE2_case_set.md)
**Found:** cBioPortal's real failure mode is a plausible-looking wrong value, not a crash — the
QC battery (5 checks, severity-graded) confirmed this by finding real curation noise (mixed
genome builds, type/protein mismatches) in the raw data, all at low severity and none touching
the driver variants the cases are built around. **0 high-severity findings** across 1,871 records.
**Verdict:** ✅ clean, with an honest caveat: 50 cases against a 200–300 target, kept small to
prove the pipeline rather than run a long annotation job. Scaling is one flag away, not a redesign.

### Phase 3 — Reasoning scaffolding → [full doc](../phases/PHASE3_scaffolding.md)
**Found a real defect on the first live run.** TP53 R175H recommended Pazopanib+Vorinostat at
0.53 — a *sarcoma*-context CIViC item — outranking the correct lung chemo/IO guideline, because
every CIViC predictive item for that variant happens to be from a different tumor type. This is
the tumor-type-specificity trap named in the risk register, caught by the system on the first
real case it ever ran, not anticipated in advance.
**Verdict:** 🟡 mixed, in the useful direction — the defect was found and fixed (mismatch penalty
raised 0.15 → 0.30), but its existence is evidence that the deterministic weighting is brittle in
ways a learned verifier (Phase 5) needs to replace, not just patch.

### Phase 4 — Supervision (guideline-verified step labels) → [full doc](../phases/PHASE4_supervision.md)
**Found:** the RAG-as-a-Judge labeller actually discriminates — a step with a fabricated,
non-resolving citation is correctly labelled unsound, it doesn't rubber-stamp every step. That
single test result is what makes Phase 5 possible at all.
**Verdict:** 🟡 mixed — the mechanism works, but the audit kappa (agreement with a human-scored
subset) that would make the label quality a *measured* number rather than an *asserted* one has
not been run. This is the single most consequential unfinished measurement in the project.

### Phase 5 — Post-training (PRM) → [full doc](../phases/PHASE5_posttraining.md)
**Found something that reshaped the phase before any model was trained:** labelling 50 real
traces produced 298 steps, **298 of them sound** — one class, un-trainable. This is not a data
bug, it's a structural property: a deterministic scaffold that only cites real retrievals cannot
produce an ungrounded step. Counterfactual negatives (strip-citations / swap-citations) fixed it:
550 examples, 54.3% positive, held-out **accuracy 0.914** (TP 86 / FP 12 / TN 62 / FN 2), and the
reward-hacking check (does it just prefer longer or more-cited traces?) came back clean.
**Verdict:** 🟡 mixed, honestly reported — 0.914 is real but measured on a semi-synthetic
distribution; the verifier has never scored an actual policy hallucination, because no sampling
policy has run yet. **Update: the GPU stage has been started** (ModernBERT fine-tune, real
policy sampling, LoRA RFT/DPO on Kaggle T4) — see the log at the top of
[`PHASE5_posttraining.md`](../phases/PHASE5_posttraining.md) for results as they land.

### Phase 6 — Clinical evaluation → [full doc](../phases/PHASE6_evaluation.md)
**Found the project's most important defect, then fixed it.** The system originally **never
abstained — 0 of 50 cases** — even though only 12 carry actionable (level A/B) evidence. This was
**invisible to guideline concordance**, which looked excellent at 0.96. Root cause, found by
reading the orchestrator: a fallback guideline's confidence (0.50) tied the abstain threshold
(0.50) exactly, and the comparison was a strict `<`, so the tie always resolved to "answer, don't
abstain." One-character fix (`<` → `<=`); **abstention went 0/50 → 24/50 and molecular
interpretation agreement went 0.24 → 0.72.** That fix then surfaced a second, subtler finding: 14
of the 38 non-actionable cases *still* recommend, not because of a bug but because they're backed
by a real gene-specific guideline tier even without top-tier CIViC evidence — the ground-truth
"actionable" flag and the guideline specialist's own evidence base define actionability slightly
differently. That reconciliation is now the sharper backlog item, folded into the MTBBench work.
Also found, unaffected by the fix: calibration is poor (ECE 0.304 — confidence encodes which
guideline branch fired, not correctness), and information-gathering correlates *negatively* with
correctness (r = −0.256, opposite sign to MTBBench's finding for LLM agents) — more evidence here
means a harder, more contested case, not better information use.
**Verdict:** 🟡 mixed, in the useful direction — the harness found a real bug, the bug was fixable
and got fixed, and fixing it did not paper over the deeper problem, it sharpened it into a
precise, named gap. A concordance-only report would have shipped a false picture of a working
system the whole way through.

### Phase 7 — Packaging → [full doc](../phases/PHASE7_packaging.md)
**Found:** every number quoted anywhere in the project's documents reproduced exactly on a fresh
run (`pytest` 71/71; `run_pipeline.py` matched to the last digit against Phases 5 and 6). Nothing
in packaging itself required a code or metric change — its job was to put the two failures from
Phase 6 on the front page instead of three clicks deep behind a clean-looking phase table. (The
abstention fix described in Phase 6's addendum was made afterward, once this front page was
already surfacing the defect prominently enough for someone to go and root-cause it.)
**Verdict:** ✅ clean. The most important thing packaging found is negative: no drift between what
is claimed and what is committed.

## The shape of the whole project, in one line

Phases 0–4 mostly confirmed the architecture was sound while surfacing real, fixable defects
(tumor-type mismatch in Phase 3, degenerate labels in Phase 5). Phases 5–6 are where the project
earns its claim to rigor: they found a verifier that only knows a semi-synthetic distribution, and
a system that never said "I don't know" — which turned out to be a one-line bug, fixed, which then
exposed a genuinely subtler definitional gap underneath. Nothing here was hidden, and fixing what
was fixable didn't mean the story became "everything works" — it became a more precise account of
what does and doesn't. That is the intended reading of this whole log: not "everything worked,"
but "everything was checked, fixed where fixable, and here is exactly what checking found."

## Standard figures used here, and why

Six results needed a chart; each was picked for the specific statistical claim it has to carry,
not for variety. None of these are novel — they are the standard forms this class of result is
reported in the calibration / selective-prediction / clinical-ML literature, which is deliberate:
a reviewer from that literature should recognize the shape of each panel immediately.

| Panel | Standard form | Why this one, not something flashier |
|---|---|---|
| **A — PRM confusion matrix** | 2×2 confusion heatmap | The direct, standard way to show a binary classifier's error *asymmetry* (FN 2 vs FP 12) — a single accuracy number hides which direction the errors fall, and that direction is a design choice (Phase 5) worth seeing at a glance. |
| **B — Calibration** | Reliability diagram (observed accuracy vs mean confidence per bin, against the diagonal) | The standard calibration plot (Guo et al. 2017 and the whole ECE literature). The diagonal is the only honest reference: it shows *where* confidence and correctness diverge, not just that ECE is 0.304. |
| **C — Deferral curve** | Risk-coverage / accuracy-coverage curve | Standard in selective prediction (El-Yaniv & Wiener). Shows the one genuinely useful property found in Phase 6 — the ranking is monotone — even though the absolute confidence values are miscalibrated (panel B). Coverage/accuracy is the right pair of axes; anything else would hide the trade-off a deferral threshold actually makes. |
| **D — Forest plot with bootstrap CIs** | Forest plot (point estimate + 95% CI per metric) | Standard in clinical-trial and meta-analysis reporting for exactly this situation: several rates, n = 50, where a bare point estimate invites over-reading. Color status (good/caveated/concerning) carries the *interpretation* Phase 6 argues for, not just the number. |
| **E — Information gathering** | Grouped strip/jitter plot with group means | The honest form for n = 48 vs 2 — a bar chart of the two means would have erased the sample-size imbalance that matters for reading the correlation; showing every point (including the very small "incorrect" group) is the non-deceptive choice. |
| **F — Abstention decision** | 2×2 confusion heatmap (ground truth × abstained/recommended) | Started as a plain two-bar chart (12 actionable vs 0 abstained) — replaced with a second confusion matrix once the bug was fixed, because a single rate could no longer carry the result: it needed to show 0 misses, 12+24 correct, *and* the 14-case residual gap in one honest picture, which is exactly what a 2×2 table is for. |

**What was deliberately not used:** an ROC curve (the PRM's operating point is fixed by the
labelling threshold, not swept — a confusion matrix is the more honest artifact here); a bare bar
chart of headline metrics without CIs (would misrepresent precision at n = 50); a radar/spider
chart (the standard way multi-metric summaries get harder to read, not easier — a forest plot
sorts and compares far better). Regenerating the figure after a fresh pipeline run
(`python scripts/run_pipeline.py && python scripts/make_results_figure.py`) is the same
reproducibility discipline as the rest of the project: the figure is derived from
`results/*.json`, never hand-drawn from remembered numbers.

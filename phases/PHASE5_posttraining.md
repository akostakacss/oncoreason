# Phase 5 — Post-training (the process reward model)

**Status:** 🟡 core built on CPU; **GPU stage started on Kaggle T4** ·
**Compute:** CPU-local (below) + Kaggle T4 (in progress)

> **GPU-stage log.** The CPU-provable path below (data construction, degenerate-label finding,
> counterfactual negatives, the TF-IDF PRM) is done and unchanged. Now started: the
> `backend="modernbert"` fine-tune, real (non-deterministic) policy sampling for Best-of-N, and
> LoRA RFT/DPO on Qwen2.5-3B-Instruct, all previously gated behind explicit enable-flags because
> `transformers`/GPU were unavailable locally. Estimated 2-4 T4-hours; this section will be
> updated with the held-out numbers once the run completes. See `scripts/run_pipeline.py --help`
> for the CPU path this extends.

## Aim

Turn the Phase-4 step labels into a **trained verifier** that can (a) score the soundness of a
reasoning step, (b) rank candidate traces so the best one can be selected at inference time,
and (c) construct the data for preference-based post-training. This is the deep-learning core
of the project: process-level supervision and verifier-guided training.

## Method

| Component | What it does |
|---|---|
| `PRM` (`training/prm.py`) | Step-soundness classifier. Input `(step text + evidence ids)` → calibrated `P(sound)`. |
| `split_by_case` | Held-out split at **case level**, never step level. |
| `_fit_temperature` | Temperature scaling on the held-out split, fitted by minimising NLL. |
| `trace_score` | Aggregates step scores to a trace score by **min**, not mean. |
| `best_of_n` | Scores N candidate traces, returns the best. Training-free use of the verifier. |
| `select_rft_traces` | RFT: keeps traces above a score threshold for LoRA-SFT. |
| `build_dpo_pairs` | DPO: builds (chosen, rejected) pairs per case, dropping near-ties. |
| `reward_hacking_report` | Checks the verifier is not simply preferring longer / more-cited traces. |

## Reasoning: the four decisions that mattered

**1. Case-level splitting, not step-level.** Steps from one case share evidence, phrasing and
topic. A random step-level split puts near-duplicates on both sides and inflates the score.
This is the same leakage discipline as the patient-level split in Phase 2, one level down.

**2. Min over steps, not mean.** A trace is only as sound as its weakest step. A mean lets one
strong step paper over a broken one, which is precisely the "right answer, wrong reasoning"
failure the whole project exists to catch. This follows Med-PRM and PRA.

**3. Temperature scaling.** An uncalibrated verifier score cannot drive abstention. Calibration
is what converts a ranking signal into a confidence signal, so it is a requirement, not a
polish step.

**4. Counterfactual negatives (the interesting one).** See Results.

## Results (real, `results/20260719-1650-pipeline.json`, seed 17)

**The finding that forced a design change.** Labelling the 50 real traces produced **298 step
examples, of which 298 were sound. 100%.** The PRM could not be trained at all: one class.

This is not a bug, it is a property of the system. A *deterministic* scaffold emits citations
only from real retrievals, so it **cannot produce an ungrounded step**. Genuine negatives
require a generative policy that can hallucinate, which is the GPU path I have not run.

Fix: `supervision.mine_negatives` constructs counterfactual unsound steps by
- **strip**: same claim, citations removed → an assertion with no supporting record;
- **swap**: citations replaced with ids from a *different* case → evidence that exists but does
  not support this claim, the machine analogue of the TP53 tumor-type mismatch.

That yields **252 synthetic negatives → 550 examples, 54.3% positive.**

**Trained verifier:**

| Metric | Value |
|---|---|
| Held-out accuracy | **0.914** (35 train cases / 15 unseen test cases) |
| Confusion | TP 86 · FP 12 · TN 62 · FN 2 |
| Fitted temperature | 0.28 |
| Trace scores (min-rule) | min 0.413 · mean 0.715 · max 0.963 |
| Reward-hacking check | chosen-longer 0.0, chosen-more-cited 0.0 → **not suspicious** |

## Interpretation

- **0.914 is real but it is accuracy on a semi-synthetic distribution.** The positives are
  genuine scaffold steps; the negatives are constructed. The honest reading is "the verifier
  reliably separates grounded from ungrounded claims", not "the verifier is 91% accurate on
  policy output". That second number cannot exist until a policy generates traces.
- **Temperature 0.28 (< 1) means the raw classifier was under-confident** and calibration
  sharpened it. Worth knowing before the score is used as a confidence.
- **FN 2 vs FP 12**: it errs toward calling things unsound. For a safety-oriented verifier that
  is the better direction to fail in, but it is a choice, not an accident of the data.
- **The reward-hacking check passing is a real negative result worth reporting.** The verifier
  is not just counting citations or length, which is the standard way this kind of preference
  signal goes wrong.

## Trade-offs, and what I did not do

| Decision | Alternative | Why I chose this |
|---|---|---|
| **TF-IDF + logistic regression** as the shipped backend | ModernBERT-base fine-tune (the plan's original choice) | `transformers` is unavailable locally and torch is CPU-only. A working baseline that actually trains and is testable in CI beats a transformer path that only runs elsewhere. ModernBERT stays wired behind `backend="modernbert"` for Kaggle. **Cost:** the reported number is a bag-of-ngrams verifier, not a contextual one; it will look different at 8k context. |
| **Counterfactual negatives** | (a) accept the degenerate labels and skip the PRM; (b) hand-write negatives; (c) wait for policy samples | (a) abandons the core novelty; (b) does not scale and injects my own priors; (c) blocks all of Phase 5 on GPU access. Counterfactual mining is standard contrastive practice and the two strategies map onto failure modes I have actually observed. **Cost:** synthetic distribution, stated everywhere it is reported. |
| **min** aggregation | mean, or a learned aggregator | mean hides broken steps; a learned aggregator needs trace-level labels I do not have. `mean` is retained as an ablation switch. |
| **Data construction implemented, LoRA training gated** | implement RFT/DPO training loops that cannot run | Selection logic and pair construction carry the intellectual content and are testable; the training call is mechanical `trl` plumbing. Shipping an untested GPU loop would be the overclaiming this project is built to avoid. |
| **No PPO/GRPO** | full RL | Infeasible on a free T4. GRPO on Alps with the PRM as reward model is the documented scale-up, and it is what RadAgent actually does. |

## Honest limitations

1. The verifier has never seen policy-generated text. Its accuracy on real hallucinations is
   **unmeasured**.
2. Best-of-N is implemented and tested but **not meaningfully demonstrated**: the deterministic
   scaffold produces one trace per case, so there is nothing to choose between. It needs a
   sampling policy, which is the GPU path.
3. The audit kappa from Phase 4.3 has not been run against a human-scored subset. Until it is,
   the label quality underpinning all of this is asserted rather than measured.

## Tests

`tests/test_prm.py` (7): case-disjoint splitting; the verifier learns the signal and ranks a
grounded step above an ungrounded one; temperature is fitted; min-vs-mean aggregation;
Best-of-N picks the grounded trace; RFT selection and DPO pairs with the hacking guard; save
and load round-trip. Part of **71 passing tests**.

## Next

→ [Phase 6 — Clinical evaluation](PHASE6_evaluation.md).

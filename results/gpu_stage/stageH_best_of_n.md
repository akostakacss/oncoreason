# GPU stage H — Best-of-N, finally demonstrated

**Run:** CPU, seed 17 · `scripts/stage_h_best_of_n.py`
**Artifacts:** `stageH_best_of_n.json`
**Input:** stage E hard-sampled candidates (8 per case), TF-IDF verifier trained on semantic labels

## Aim

Phase 5's second honest limitation: *"Best-of-N is implemented and tested but **not meaningfully
demonstrated**: the deterministic scaffold produces one trace per case, so there is nothing to
choose between. It needs a sampling policy, which is the GPU path."* Stage E supplied the
candidates and stage F supplied a verifier past the gate. This retires that limitation with a
number.

## Method

Trace **quality** is the fraction of steps the semantic labeller judges sound. The verifier's
own score decides only *which* trace is selected — it is deliberately kept out of the outcome
metric, because scoring selection with the same signal that drove selection would report the
verifier's confidence rather than its usefulness.

Three arms on the 15 held-out cases the verifier never saw:

| Arm | Meaning |
|---|---|
| random | mean quality over all 8 candidates — what one sample gets you in expectation |
| best-of-N | quality of the candidate the verifier ranked highest |
| oracle | quality of the best candidate present — the ceiling selection could reach |

`best_of_n` and `score_trace_with_prm` are imported from `oncoreason.training`, so this
exercises the shipped code path rather than a reimplementation. Deltas are bootstrapped over
**cases**, 1000 iterations, paired.

## Results

| Arm | Trace quality |
|---|---|
| worst candidate | 0.000 |
| random (mean of 8) | 0.167 |
| **best-of-N** | **0.256** |
| oracle (max of 8) | 0.458 |

| Contrast | Estimate | 95% CI |
|---|---|---|
| **Best-of-N over random** | **+0.089** | **[−0.059, +0.238]** |
| Oracle over Best-of-N | +0.202 | [+0.071, +0.371] |

## Interpretation — two findings, one of them negative

**1. The selection lift is not statistically distinguishable from zero.** +0.089 is a 53%
relative improvement over a random sample, and the point estimate is in the right direction,
but the interval **crosses zero**. At 15 held-out cases this is not evidence that the verifier
picks better traces. It is evidence that, if it does, the effect is smaller than this
experiment can resolve.

This is a genuine negative result, and it qualifies the stage F gate. A verifier at 0.849–0.881
*step-level* balanced accuracy does **not** translate into demonstrably reliable *trace-level*
selection. Step accuracy and ranking utility are different quantities, and only the first was
measured before now.

**2. The oracle ceiling is 0.458.** Even the best of eight candidates has fewer than half its
steps judged sound. The gap from Best-of-N to oracle *is* significant ([+0.071, +0.371]), so
there is real headroom selection is not capturing — but the headroom itself is modest, because
the policy is weak.

That second number matters more than it first appears. **RFT trains a policy to imitate its own
selected traces.** If the best available trace is 46% sound, rejection-sampling fine-tuning is
imitating a low ceiling, and the achievable gain is bounded by it before any training runs.

## What this implies for RFT/DPO

Stage F's gate passed on step-level accuracy and permitted RFT/DPO. Stage H tempers that:

- The selection signal driving RFT trace choice has **no demonstrated lift** at this n.
- The imitation target has a **0.458 quality ceiling**.
- DPO is somewhat better placed: it uses the *contrast* between best and worst (0.458 vs 0.000
  is a wide margin, comfortably clearing `build_dpo_pairs`' 0.05 threshold), so it extracts
  signal from the spread rather than from the absolute quality of the winner.

The honest ordering is therefore: **a baseline arm first**, then DPO ahead of RFT, and any
result reported against the measured selection lift rather than against the assumption that
Best-of-N works.

## Honest limitations

1. **n = 15 cases.** The interval is wide enough that a real effect of +0.09 would not be
   detectable. More held-out cases, or repeated splits, is the fix — not a different verifier.
2. **TF-IDF backend**, not ModernBERT. Stage F found the two statistically indistinguishable
   (0.849 [0.768, 0.921] vs 0.881 [0.798, 0.951]), so this is unlikely to be the limiting
   factor, but it is not the best available verifier.
3. **Quality inherits the semantic label's caveat**: ~65% of it is tumour-type consistency, and
   part of that is a retrieval artifact rather than policy error (stage G).
4. **Single N and temperature.** Whether N=16 or a different temperature widens the spread
   enough for selection to bite is untested.

## Next

→ A **baseline arm** (untrained policy through the Phase 6 harness) — Phase 6's third honest
limitation, and the missing reference point for every number in this directory.

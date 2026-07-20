# GPU stage C — does the verifier transfer from constructed to real negatives?

**Run:** CPU, seed 17 · `scripts/stage_c_prm_real.py` ·
**Artifacts:** `stageC_prm_transfer.json`
**Input:** `runs/20260720-0912-b-n8-t08-artifacts/candidates-relabelled.jsonl` (stage B)

![Stage C: balanced accuracy falls from 0.908 on constructed negatives to 0.530 on real ones,
barely above chance; recall on unsound steps falls from 0.838 to 0.114](stageC_figure.png)

## Aim

Phase 5's first honest limitation reads: *"The verifier has never seen policy-generated text.
Its accuracy on real hallucinations is unmeasured."* Stage B produced real hallucinations.
This measures it.

## Method

Two verifiers, one shared case-level split (15 of 50 cases held out, seed 17), evaluated on
the **identical** real test set — so the only variable is the training distribution.

| Arm | Trained on | Tested on |
|---|---|---|
| A | synthetic negatives from `mine_negatives`, train cases | real policy steps, test cases |
| B | real policy steps, train cases | real policy steps, test cases |

Arm A is the number Phase 5 could not report. Arm B is the ceiling for this feature set.
Balanced accuracy is reported alongside raw accuracy because the classes are skewed and raw
accuracy flatters a model that simply says "sound".

## Results

| Arm | Accuracy | Balanced | Recall (sound) | **Recall (unsound)** |
|---|---|---|---|---|
| A on synthetic holdout *(control)* | 0.914 | 0.908 | 0.977 | 0.838 |
| **A: synthetic → real** | **0.585** | **0.530** | 0.945 | **0.114** |
| B: real → real | 0.947 | 0.950 | 0.929 | 0.971 |

The control arm reproduces Phase 5's published **0.914** exactly, which confirms the setup is
faithful to the original pipeline before anything is concluded from the other two rows.

## Interpretation

**The transfer fails, and it fails almost completely.** Balanced accuracy 0.530 against a
chance floor of 0.500. The decisive number is recall on unsound steps: the verifier catches
**83.8% of constructed negatives and 11.4% of real ones**. On the real test set it returns
"sound" for 297 of 323 steps — it has learned the shape of the counterfactuals rather than the
concept of groundedness.

This retires Phase 5's limitation #1 by answering it: the honest figure for the shipped
verifier on real policy output is **0.53 balanced accuracy, not 0.914**. The README's existing
caution — *"the honest reading is 'the verifier reliably separates grounded from ungrounded
claims', not 'the verifier is 91% accurate on policy output'"* — turns out to have been
correct and, if anything, understated.

**Arm B's 0.947 should not be read as a good verifier either.** Stage B found that 95% of real
unsound steps are simply uncited, so arm B is largely detecting an empty citation slot. It
establishes that the real distribution is learnable, not that the resulting model is
discriminating. The 26 steps carrying genuinely invented ids are too few to measure separately
at this n.

**Why the transfer fails** is visible in stage B's second finding. `mine_negatives` builds two
kinds of negative: `strip` (citations removed) and `swap` (citations from another case). The
policy produces the first kind constantly and the second kind **never** — zero `off_case`
steps in 1076. Half the synthetic training signal describes a failure mode that does not occur.

## Trade-offs, and what I did not do

| Decision | Alternative | Why |
|---|---|---|
| TF-IDF backend for both arms | ModernBERT for both | The comparison is between *training distributions*; holding the model fixed isolates that. A ModernBERT run changes both variables at once. |
| Balanced accuracy as the headline | raw accuracy | At a 0.567 positive rate, a constant "sound" predictor scores 0.567 raw. Arm A's 0.585 is barely above that, which raw accuracy alone disguises. |
| Same case split for both arms | independent splits | Independent splits would confound distribution shift with split luck at n=15 test cases. |

## Honest limitations

1. **n = 15 test cases**, 323 steps. No confidence intervals computed here; the effect is
   large enough that the direction is not in doubt, but the magnitude is loosely estimated.
2. Arm B's high score rests on an easy feature. A verifier that must judge *semantic support*
   rather than *citation presence* is not yet measured and would score lower.
3. Both arms use bag-of-ngrams features over step text plus evidence **ids** — not evidence
   text. `PRM._featurize` never shows the model what the cited record says, which caps what
   any backend can learn here (see Next).
4. Single policy, single temperature. Whether a stronger or differently-prompted policy
   produces harder negatives is untested.

## Next

1. **Fix `_featurize` to inline evidence text** (`src/oncoreason/training/prm.py:78`). It
   currently passes `civic:EID2994` as an opaque token, so neither TF-IDF nor an 8k-context
   transformer can assess whether the evidence supports the claim. This gates any meaningful
   ModernBERT run — as wired, the transformer would have nothing extra to attend to.

   > **Done, and it made things worse — see `stageD_featurize_ablation.md`.** Inlining the
   > text lowered balanced accuracy in all three arms and pushed this transfer arm to 0.458,
   > below chance. The prediction above was wrong: bag-of-ngrams cannot use the information,
   > because judging support is a relational question. The gating claim survives, the
   > "it will help" implication does not.
2. **Harder negatives**: require a citation on every step so "uncited" stops dominating the
   negative class, forcing the distinction toward *correct* citation rather than *any*.
3. Retire or re-scope the `swap` strategy in `mine_negatives`, which stage B showed models a
   failure mode this policy never exhibits.

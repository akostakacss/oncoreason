# GPU stage D — inlining evidence text: a prediction that failed

**Run:** CPU, seed 17 · `scripts/stage_c_prm_real.py --with-evidence-text`
**Artifacts:** `stageC_ids_only.json`, `stageC_with_text.json`, `stageD_figure.png`

![Stage D: inlining the cited records' text lowers balanced accuracy in all three arms, and
pushes the synthetic-to-real transfer below chance](stageD_figure.png)

## Aim

Stage C's first "Next" item asserted that `PRM._featurize` was the binding constraint: it
passed `civic:EID2994` as an opaque token and never showed the model what the record said, so
no backend could assess whether the evidence *supported* the claim. The stated expectation was
that inlining evidence text would raise the ceiling and was a precondition for a useful
ModernBERT run.

`_featurize` now accepts an optional `evidence_text` field (`src/oncoreason/training/prm.py:78`),
backward-compatible so callers holding only ids are unaffected. This measures what that buys.

## Method

The stage C design unchanged — same case-level split, same three arms, same TF-IDF backend —
run twice, with the feature string as the only variable. Evidence text resolved for **570 of
1076** steps (the rest cite nothing, so there is nothing to inline).

## Results

| Arm | ids only | + evidence text | Δ |
|---|---|---|---|
| A on synthetic holdout *(control)* | 0.908 | 0.896 | −0.012 |
| **A: synthetic → real** | 0.530 | **0.458** | **−0.072** |
| B: real → real | 0.950 | 0.917 | −0.033 |

*(balanced accuracy; chance = 0.500)*

**The prediction was wrong.** Adding evidence text made every arm worse, and pushed the
synthetic→real transfer from marginally-above-chance to **below** it. In that arm false
negatives rose from 10 to 61: the model began calling genuinely sound steps unsound.

Verified this is not a plumbing bug — 570 steps carry populated text and it appears correctly
in the feature string (checked by hand against `_featurize` output).

## Interpretation

**Bag-of-ngrams cannot use this information, because judging support is a relational
question.** TF-IDF sees the union of the claim's words and the evidence's words; it has no
representation of whether they correspond. Three things follow:

1. **The text is nearly constant across correct and incorrect citations.** CIViC summaries are
   formulaic — `GENE VARIANT: SUPPORTS SENSITIVITYRESPONSE to DRUG in Lung...`. A citation
   swapped in from another case produces text that is lexically almost indistinguishable from
   the right one.
2. **It dilutes what was working.** The discriminative signal was largely the `NO_EVIDENCE`
   token and step-level phrasing. Adding 10–30 tokens of shared vocabulary to every cited step
   pushes sound and unsound steps closer together in feature space.
3. **So the intervention is necessary but not sufficient.** The information genuinely is
   required to judge support — it simply cannot be exploited by a model that cannot attend
   across the claim/evidence boundary.

**This sharpens the ModernBERT hypothesis rather than removing it.** The stage C claim that
`_featurize` gates a meaningful transformer run was correct; the implication that inlining
text would help on its own was not. The experiment is now well-posed: hold the feature string
fixed at "step + ids + evidence text" and vary only the model class. If ModernBERT beats
TF-IDF on identical inputs, cross-attention is doing the work — and that is a clean, single-
variable result. If it does not, the problem is the label distribution, not the architecture.

## What I got wrong, and why it is worth recording

I stated the `_featurize` fix as the gating item with more confidence than the evidence
supported, reasoning from the mechanism (ids are opaque) to a predicted outcome (inlining
helps) without noting that the mechanism only pays off under a model class the project was not
yet running. The measurement cost about two minutes of CPU and reversed the conclusion. Left
unmeasured, it would have been carried into the ModernBERT run as an assumption, where a null
result would have been much harder to attribute.

## Honest limitations

1. One backend, one feature construction. "Evidence text hurts TF-IDF" is established;
   "evidence text helps a transformer" is a hypothesis, not a result.
2. Evidence text is inlined raw. No truncation, ordering, or claim/evidence separator token —
   any of which could change the outcome for either model class.
3. Same n = 15 test cases as stage C, so the same caveat on magnitude applies.

## Next

1. **ModernBERT with the feature string held fixed** — the single-variable test the above sets
   up. This is the run that needs a GPU.
2. **Harder negatives** (stage B finding): require a citation on every step, so the negative
   class stops being 95% "cited nothing" and the task becomes citation *correctness*.
3. Retire or re-scope `mine_negatives`' `swap` strategy, which stage B showed models a failure
   mode this policy never exhibits.

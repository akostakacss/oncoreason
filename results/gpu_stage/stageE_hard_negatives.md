# GPU stage E — making the negatives hard instead of merely real

**Run:** `runs/20260720-1535-e-hard-n8-t08` · Kaggle T4 · 36.2 min · kernel
`ta1010/oncoreason-policy-sampling-hard` · Qwen2.5-3B-Instruct, N=8, temperature 0.8, seed 17
**Artifacts:** `candidates.jsonl` (400 samples, 1236 steps), `sampling_report.json`

## Aim

Stage B replaced constructed negatives with observed ones, but stage C showed the resulting
verifier still sat at 0.530 balanced accuracy on real output. The reason was in stage B's own
numbers: **481 of 507 unsound steps simply carried no citation**, so "unsound" was very nearly
"the citation slot is empty" — the same trivially separable signal the `strip` counterfactual
already provided. Sampling had made the negatives real without making them *hard*.

Stage B also found **zero** off-case citations in 1076 steps. The `swap` strategy in
`mine_negatives` models a policy citing another case's evidence, and the policy never did it —
because it was only ever shown one case's evidence and had no other ids available to misuse.

Two changes address both findings at once.

## Method

1. **Distractor evidence.** Four records drawn from *other* cases are mixed into each prompt
   pool and shuffled in, so position does not leak which are real. Mean pool size rises from
   ~5.7 to **9.7** records. Citing a distractor is a genuine error that stays mechanically
   detectable, because the true pool is retained separately for labelling — what the policy is
   *shown* and what a citation is *judged against* are deliberately different objects in the
   code.
2. **Mandatory citations.** The system prompt now requires a `[CITE: ...]` list on every step,
   and states explicitly that the evidence list contains records which may not bear on this
   patient. This closes the "cite nothing" escape, leaving citing the *wrong* record as the
   failure still available.

Everything else is held fixed from stage B: same model, N, temperature, seed, parser, and
notebook builder (`scripts/build_sampling_notebook.py hard`).

## Results

| Metric | Stage B | **Stage E** |
|---|---|---|
| Samples / steps | 400 / 1076 | 400 / **1236** |
| Parse rate | 1.000 | **1.000** |
| **Uncited steps** | 481 | **43** |
| **Cited another case's evidence** | **0** | **559** |
| Invented ids | 61 | 106 |
| Unsound step rate | 0.488 | 0.557 |
| Within-case spread | 0.967 | 0.640 |
| Steps per sample | 2.69 | 3.09 |
| Runtime | 25.2 min | 36.2 min |

**Both interventions did what they were designed to do.** The degenerate negative collapsed
(481 → 43 uncited), and the negative class is now dominated by a failure of *citation
correctness* rather than *citation presence*.

## Interpretation

**The headline finding is 0 → 559.** Given four irrelevant records per case, the policy cites
one in **45% of all steps**. Stage B's zero was not evidence that the policy is careful about
provenance; it was an artefact of never being given the opportunity to be careless. This is a
substantive result about a 3B policy under evidence pressure, and it echoes MTBBench's
observation that models "fail to reconcile conflicting evidence".

It also settles the open question from stage B about `mine_negatives`' `swap` strategy. `swap`
is not modelling a fictional failure mode after all — it models one that only appears when the
policy's context contains material it should reject. That is a fair description of any
realistic retrieval setting, and it means the counterfactual was better motivated than stage B
concluded from its own zero.

**Within-case spread fell 0.967 → 0.640.** Expected: mandating citations constrains the output
space. It remains comfortably high enough for Best-of-N ranking and for DPO pairs to clear the
0.05 margin, so the diversity that made stage B useful has not been spent.

**Runtime rose 25 → 36 min**, tracking the rise in steps per sample. Mandatory citations make
the policy write more.

## Honest limitations

1. **Four distractors per case is a chosen difficulty, not a measured one.** No sweep was run,
   so 45% off-case citation is the rate at *this* contamination level and does not extrapolate.
2. **The distractors are drawn uniformly**, not adversarially. Records that are lexically or
   clinically close to the true pool would be a harder and more realistic test than a random
   draw.
3. **Off-case is still a structural judgement.** It detects "this record belongs to a different
   case", not "this record fails to support the claim". The semantic layer
   (`stageG_efficacy_mtbbench.md`) is what addresses that, and stage F is where the two label
   definitions are compared head to head.
4. Invented ids rose 61 → 106. Some of that is a larger pool giving more surface to
   mis-transcribe; it has not been separated from genuine fabrication.

## Next

→ `stageF_modernbert.md` — the 2×2 over {TF-IDF, ModernBERT} × {structural, semantic labels},
feature string held fixed, which is the decision gate for whether RFT/DPO are worth running.

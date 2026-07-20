# GPU stage F — the decision gate: capacity, or labels?

**Run:** `runs/20260720-1604-f-2x2-semantic` · Kaggle T4 · kernel
`ta1010/oncoreason-prm-modernbert` · seed 17
**Artifacts:** `stageF_modernbert.json`, `prm_modernbert_semantic/` (weights, git-ignored)
**Input:** stage E hard-sampled candidates, via `kernel_sources`

## Aim

Stage C measured the shipped verifier at 0.530 balanced accuracy on real policy output. Stage D
tested the obvious fix — inline what each cited record says — and it made things *worse*, with
the diagnosis that judging support is **relational** and bag-of-ngrams sees only the union of
two word bags. Stage E then made the negatives hard, and stage G built a semantic label.

This is the 2×2 that separates the two candidate explanations. Feature string is frozen at
`step + ids + evidence text`; split, seed and data are identical in every cell. Only the
**backend** and the **label definition** vary.

Confidence intervals resample **cases**, not steps — 1000 iterations, 95% — matching
MTBBench's evaluation discipline and preserving the leakage guard the split enforces.

## Results

| Backend | Label | Balanced accuracy | 95% CI | Positive rate |
|---|---|---|---|---|
| TF-IDF | structural | 0.659 | [0.562, 0.763] | 0.404 |
| ModernBERT | structural | 0.767 | [0.683, 0.853] | 0.404 |
| TF-IDF | semantic | 0.849 | [0.768, 0.921] | 0.172 |
| **ModernBERT** | **semantic** | **0.881** | **[0.798, 0.951]** | 0.172 |

| Effect | Size |
|---|---|
| Model class, structural labels | +0.108 |
| Model class, semantic labels | +0.032 |
| **Label definition, TF-IDF** | **+0.190** |
| **Label definition, ModernBERT** | **+0.114** |

## Verdict: PASS — with the reason stated precisely

**The label mattered more than the model.** Changing the label bought +0.190 for TF-IDF;
changing the backend bought +0.032 once the label was right. Stage D's diagnosis is
vindicated: the binding constraint was that the label posed no relational question, not that
the model lacked capacity to answer one.

**The model-class effect is not statistically distinguishable here.** ModernBERT's
[0.798, 0.951] overlaps TF-IDF's [0.768, 0.921] almost entirely. At 15 held-out cases, +0.032
is well inside noise. The honest claim is *"cross-attention did not clearly help once the label
was fixed"*, not *"ModernBERT is better"*. Reporting the CI is what makes that visible; the
point estimate alone would have oversold it.

**The gate passes** at 0.881 [0.798, 0.951], lower bound comfortably above the 0.55 floor set
in advance. A verifier at this level is usable for trace selection, so RFT/DPO can proceed —
with the caveat below attached to the reward signal.

## The caveat that must travel with the 0.881

**`disease_mismatch` accounts for 806 of 1236 steps (65%) of the semantic label's negatives.**
The semantic label is therefore substantially *"did the cited record concern the right tumour
type"*, which is closer to a matching task than to judging clinical support. Two consequences:

1. **0.881 should not be read as "the verifier understands whether evidence supports a
   claim."** It largely measures tumour-type consistency. The genuinely relational signals —
   contradiction (93) and therapy mismatch (142) — are real but a minority of the label.
2. **Part of that signal is a retrieval artifact.** CIViC returns cross-disease evidence for a
   variant, so some mismatches originate in what the retriever surfaced rather than in what the
   policy reasoned. Flagged in stage G and unresolved here.

That TF-IDF alone reaches 0.849 is consistent with this reading: a largely lexical
disease-matching task is exactly what bag-of-ngrams can do well, which is also why the
transformer adds so little on top.

## A comparison that is not apples-to-apples

The two label columns describe **different classification tasks**, with different positive
rates (0.404 vs 0.172). Balanced accuracy corrects for prevalence within a task but does not
make cross-task comparison rigorous. "+0.190 from the label" therefore means *"the semantic
task is more learnable from these features"*, not *"the same task got easier"*. The
model-class comparison, within a fixed label, is the only clean single-variable contrast in
this table.

## Honest limitations

1. **n = 15 held-out cases**, 361 test steps. Every CI here is wide; nothing separated by less
   than ~0.1 should be treated as a real difference.
2. **One seed.** No repeated splits, so split luck is uncontrolled.
3. **Semantic labels are proxies** — regex claim-direction over step text against structured
   record fields. Not clinician judgements. The upgrade path remains MTBBench's expert-verified
   QA pairs.
4. **Three epochs, no early stopping, no hyperparameter search** for ModernBERT. Its number is
   a first fit, not a tuned one, so the model-class effect could be understated.
5. `max_length=512` rather than the 8k ModernBERT supports — appropriate for these short steps,
   but it means the long-context property that motivated choosing ModernBERT is untested.

## Next

The gate permits RFT/DPO. The reward signal must be described wherever those results are
reported as: *a ModernBERT step-verifier at 0.881 [0.798, 0.951] balanced accuracy on
hard-sampled policy traces, whose label is 65% tumour-type consistency.*

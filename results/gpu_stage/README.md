# The GPU stage — an audit trail

Phase 5 shipped a process reward model with **0.914 held-out accuracy** and an honest caveat
attached: the negatives were *constructed*, not observed, so the number described a
semi-synthetic distribution and the verifier's accuracy on real policy output was
**unmeasured**. This directory is the record of measuring it.

Every stage here is reproducible from a committed script, every Kaggle run is archived under
`runs/` with its stdout and artifacts, and every claim below links to the file that produced
it. Where a prediction turned out wrong, the wrong prediction is left in place with the
correction attached rather than quietly edited out — the sequence of reasoning is the point.

## How to read this

| Stage | Question | Verdict | Files |
|---|---|---|---|
| **B** | What does a real policy actually get wrong? | Real hallucinations exist, but 95% of them are "cited nothing" | [`stageB_sampling.md`](stageB_sampling.md) · [figure](stageB_figure.png) |
| **C** | Does a verifier trained on constructed negatives transfer? | **No — 0.530 balanced accuracy, barely above chance** | [`stageC_prm_transfer.md`](stageC_prm_transfer.md) · [figure](stageC_figure.png) |
| **D** | Does showing the model the evidence *text* fix it? | **No — it made every arm worse.** A prediction of mine that failed | [`stageD_featurize_ablation.md`](stageD_featurize_ablation.md) · [figure](stageD_figure.png) |
| **E** | Can we make the negatives *hard* instead of merely real? | **Yes — uncited 481→43, off-case 0→559** | [`stageE_hard_negatives.md`](stageE_hard_negatives.md) |
| **G** | Can the label ask about *support* rather than bookkeeping? | Yes — semantic labels from structured fields, coverage 53% | [`stageG_efficacy_mtbbench.md`](stageG_efficacy_mtbbench.md) |
| **F** | Is the bottleneck model capacity, or the labels? | **The labels.** +0.190 from the label, +0.032 from the model | [`stageF_modernbert.md`](stageF_modernbert.md) · [figure](stageF_figure.png) |
| — | **Decision gate**: is the verifier usable as a reward signal? | **PASS** — 0.881 [0.798, 0.951], with a caveat | see below |
| **H** | Does verifier-guided selection pick better traces? | **Not measurably** — +0.089 [−0.059, +0.238] | [`stageH_best_of_n.md`](stageH_best_of_n.md) |

## The thread, in order

**1. The original problem.** A deterministic scaffold emits citations only from real
retrievals, so it *cannot* produce an ungrounded step. Labelling its 50 traces gave 298 step
examples of which 298 were sound — one class, nothing to train on. Phase 5's workaround was
`supervision.mine_negatives`, which constructs unsound steps two ways: `strip` (remove the
citations) and `swap` (replace them with ids from a different case).

**2. Stage B — observe instead of construct.** Sampled 8 traces per case from
Qwen2.5-3B-Instruct at temperature 0.8. Parse rate 1.000, within-case score spread 0.967 (so
Best-of-N finally has candidates to rank — Phase 5 limitation #2). But two findings undercut
the headline:

- **481 of 507 unsound steps simply carry no citation.** Sampling made the negatives real
  without making them *hard*; "cited nothing" is very nearly the signal `strip` already gave.
- **Zero off-case citations in 1076 steps.** The `swap` strategy models a failure mode this
  policy never exhibits, because it only ever sees one case's evidence and has no other
  case's ids available to misuse. Half the synthetic training signal describes nothing real.

**3. Stage C — measure the transfer.** Two verifiers, one shared case-level split, identical
real test set; only the training distribution differs.

| Arm | Balanced accuracy | Recall on unsound steps |
|---|---|---|
| trained on synthetic, tested on synthetic *(control)* | 0.908 | 0.838 |
| **trained on synthetic, tested on real** | **0.530** | **0.114** |
| trained on real, tested on real | 0.950 | 0.971 |

The control reproduces Phase 5's published 0.914 exactly, confirming the setup is faithful
before anything is concluded. Then: the verifier catches 84% of constructed negatives and
**11% of real ones**. It learned the shape of the counterfactuals, not groundedness.

Arm B's 0.950 is *not* evidence of a good verifier — stage B showed 95% of real negatives are
uncited, so it is largely detecting an empty citation slot.

**4. Stage D — the obvious fix, which failed.** Stage C's top recommendation was to inline
what each cited record says, since `_featurize` passed `civic:EID2994` as an opaque token.
Implemented it; it lowered balanced accuracy in all three arms and pushed the transfer arm to
0.458, *below* chance.

The diagnosis: judging support is **relational**, and bag-of-ngrams sees only the union of
two word bags. CIViC summaries are formulaic, so a citation swapped from another case reads
almost identically to the right one. The information is necessary but unusable without a
model that can attend across the claim/evidence boundary.

I had stated that recommendation with more confidence than the evidence supported. Two
minutes of CPU reversed it. Left unmeasured it would have entered stage F as an assumption,
where a null result would have been much harder to attribute.

**5. Stage E — make the negatives hard.** Two changes to sampling: mix **distractor evidence
from other cases** into each prompt pool (4 per case, shuffled so position does not leak), and
make a citation **mandatory** on every step. Citing a distractor is a genuine error that
remains mechanically detectable — and it resurrects the `swap` failure mode by finally giving
the policy other cases' ids to misuse.

**6. Stage F — capacity or labels?** With the feature string frozen at
`step + ids + evidence text`, the only variable is the model class. ModernBERT-base against
TF-IDF on identical inputs. If the transformer wins, cross-attention was the missing piece. If
it does not, the bottleneck is the label distribution and no capacity fixes it.

**7. The decision gate.** RFT and DPO use the verifier to select traces and build preference
pairs. A verifier near chance yields a near-random reward signal, and training on it would
produce an adapter whose numbers mean nothing. So stage F decides whether stages G/H run at
all — and a "no" is a legitimate result to report, not a failure to hide.

## Reproducing

```bash
# CPU — relabelling and the transfer measurement
python scripts/relabel_candidates.py runs/<run>-artifacts/candidates.jsonl
python scripts/stage_c_prm_real.py runs/<run>-artifacts/candidates-relabelled.jsonl
python scripts/stage_c_prm_real.py <same> --with-evidence-text -o stageC_with_text.json

# GPU — every Kaggle run goes through krun.sh so nothing is left unarchived
scripts/krun.sh policy_sampling       b-n8-t08
scripts/krun.sh policy_sampling_hard  e-hard-n8-t08
scripts/krun.sh prm_modernbert        f-modernbert

# figures (read the saved JSON, so they cannot drift from the results)
python scripts/make_gpu_stage_figures.py
```

Notebooks are **generated** from `scripts/build_*_notebook.py` rather than hand-edited, so the
sampling configuration lives in version control as readable Python. The builders refuse to
write a notebook whose cells do not compile — added after the first stage B run died on the
GPU from a cell-serialization bug that a naive local check had missed.

## Run archive

Every attempt, including the failures, is in `runs/INDEX.tsv`. The three failed stage B runs
(nbformat cell serialization, Kaggle archive expansion, mount-path discovery) cost ~8 minutes
of T4 time between them and are kept deliberately: a tooling record that only contains
successes is not an audit trail.


## The gate result, and what it does and does not license

Stage F's 2x2 lands at **0.881 [0.798, 0.951]** for ModernBERT on semantic labels, clearing
the 0.55 CI-lower-bound floor set in advance. RFT/DPO may proceed.

Three things must travel with that number.

**1. The label, not the model, did the work.** Changing the label definition bought +0.190 for
TF-IDF; changing the backend bought +0.032 once the label was right. Stage D's diagnosis —
that the constraint was a label posing no relational question, rather than model capacity —
is what the 2x2 confirms.

**2. The model-class effect is inside noise.** ModernBERT's [0.798, 0.951] overlaps TF-IDF's
[0.768, 0.921] almost entirely. At 15 held-out cases the honest statement is "cross-attention
did not clearly help once the label was fixed", not "the transformer is better".

**3. The semantic label is ~65% tumour-type consistency.** `disease_mismatch` accounts for 806
of 1236 steps, so 0.881 substantially measures whether the cited record concerned the right
tumour type — closer to matching than to judging clinical support, and partly a *retrieval*
artifact because CIViC returns cross-disease evidence for a variant. That TF-IDF alone reaches
0.849 is consistent with a largely lexical task.

The reward signal for any downstream RFT/DPO must therefore be described as: *a ModernBERT
step-verifier at 0.881 [0.798, 0.951] on hard-sampled policy traces, whose label is 65%
tumour-type consistency.*

## Stage H tempers the gate

Stage F passed on **step-level** accuracy. Stage H then measured **trace-level** selection and
found the Best-of-N lift over a random sample to be +0.089 **[−0.059, +0.238]** — the interval
crosses zero. Step accuracy and ranking utility are different quantities, and only the first
had been measured when the gate was set.

The oracle ceiling is also low: even the best of 8 candidates has only **0.458** of its steps
judged sound. Since RFT trains a policy to imitate its own selected traces, that ceiling bounds
what rejection-sampling fine-tuning could achieve before any training runs.

## Still open

- **RFT / DPO** (`train_rft`, `train_dpo` in `training/posttrain.py`) remain stubs. The gate
  permits them, but stage H argues a baseline arm should come first, and that DPO is better
  placed than RFT because it extracts signal from the best/worst *contrast* (0.458 vs 0.000)
  rather than from the winner's absolute quality.
- **No baseline arm.** Phase 6's third honest limitation stands: nothing here can yet be called
  an improvement over an untrained policy.
- **`disease_mismatch` conflates policy error with retrieval artifact.** Separating them is the
  single highest-value fix to the label.
- **One seed, 15 test cases.** No repeated splits; every interval here is wide.

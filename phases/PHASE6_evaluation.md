# Phase 6 — Clinical evaluation

**Status:** 🟡 core built (8 metrics implemented and run on real data) · **Compute:** CPU-local

> **Addendum, same day:** the abstention defect described below (§2) has been root-caused and
> fixed. See **"Addendum — the abstain-threshold fix"** at the end of this document for the root
> cause, the one-line fix, the corrected numbers, and a second, subtler finding the fix exposed.
> The Results and Interpretation sections immediately below are kept as originally written —
> the record of what the harness found before the fix — rather than edited after the fact.

## Aim

Measure the system on **every dimension this project targets**, with statistics that survive
scrutiny at n = 50. This is the pillar that must be done properly rather than gestured at:
evaluation is what separates a demonstrable result from an assertion.

## Method

Eight metrics, each returning a rate with a **percentile bootstrap 95% CI** (2000 resamples,
seeded) rather than a bare point estimate.

| Metric | Question it answers |
|---|---|
| `guideline_concordance` | Does the recommendation match the guideline? (top-1 and any-match) |
| `molecular_interpretation_accuracy` | Does it recommend exactly when the evidence supports one? |
| `reasoning_step_accuracy` | Are the intermediate steps sound? (catches right-answer-wrong-reasoning) |
| `tool_use_reliability` | Do the tools work? Success, latency, redundant calls, empty returns |
| `citation_grounding` | Does every citation resolve to a real record? |
| `calibration` | Brier, ECE, reliability curve. Does confidence track correctness? |
| `deferral_curve` | Accuracy vs coverage as the abstain threshold moves |
| `information_gathering` | Does gathering more evidence actually help? (added after MTBBench) |

Supporting: `bootstrap_ci` and `bonferroni` for the many-metrics multiplicity problem.

## Results (50 cases, all with guideline-derived gold, seed 17)

| Metric | Result |
|---|---|
| Guideline concordance (top-1) | **0.96** [0.90, 1.00] |
| Guideline concordance (any-match) | **1.00** [1.00, 1.00] |
| Molecular interpretation agreement | **0.24** [0.12, 0.36] · 12/50 cases actionable |
| Step soundness | **1.00** (n = 298) |
| Tool reliability | 223 calls, **100% success**, median 0.09 ms, **90 redundant**, **72 empty** |
| Citation grounding | 226 citations, **100% resolved** |
| Calibration | Brier **0.141** [0.104, 0.178], **ECE 0.304** |
| Deferral | monotone ✅ · coverage 1.00 → acc 0.96 · coverage 0.48 → acc **1.00** |
| Information gathering | **r = −0.256** (p = 0.073); 6.7 evidence items when correct vs **13.5 when incorrect** |

## Interpretation: four of these are bad news, and that is the point

**1. Guideline concordance 0.96 is the least meaningful number here.** The gold label is derived
from the same guideline index the guideline specialist retrieves over, so this is **partially
circular** and a system that merely echoed the guideline would score ~1.0 by construction. The
only informative part is that it is *not* 1.0: in 2/50 cases the adjudicator diverged from the
guideline because CIViC evidence outranked it. Reported with the caveat attached, in the metric
docstring and in `supervision.label_case_outcome`.

**2. Molecular interpretation agreement 0.24 is a genuine failure, and the harness caught it.**
The system **never abstained** (0/50). Because the "no targetable driver" guideline chunk always
fires with weight 0.50, confidence never drops below the 0.50 abstain threshold, so the agent
recommends something for all 50 cases even though only 12 have level A/B actionable evidence.
The metric is essentially reporting the actionable base rate, which is the signature of a system
with **no abstention discrimination**. This is the single most important defect found, and it
was invisible to concordance, which looked excellent.

**3. Step soundness 1.00 and citation grounding 1.00 are trivially true.** The deterministic
scaffold cannot emit an ungrounded step (Phase 5), and the resolver checks citations against the
trace's own ids, which is structural rather than independent verification. Both are reported as
**degenerate-by-construction**, not as achievements.

**4. ECE 0.304 is poor calibration.** The reliability curve shows why: 24 cases sit at confidence
0.50 with accuracy 1.00 (the chemo/IO cases, systematically *under*-confident) and 24 at 0.92 with
accuracy 1.00. Confidence encodes *which guideline branch fired*, not *how likely the answer is
to be right*. The deferral curve is nevertheless monotone, so the ordering is useful even though
the absolute values are not.

**5. Information gathering r = −0.256 runs opposite to MTBBench.** They found files-accessed
correlates *positively* with accuracy for LLM agents. Here, cases with more evidence are more
often wrong. The explanation is that in a deterministic system "more evidence" does not mean
better information gathering, it means a harder multi-alteration case where competing and
tumor-mismatched evidence accumulates and degrades the adjudication. Their metric, imported
unchanged, exposed a weakness of my synthesizer.

**6. 90/223 redundant and 72/223 empty tool calls** is a real inefficiency. Note the metric
counts redundancy globally across traces, so the same gene queried for different patients counts
as redundant; that definition is generous and should be tightened before it is quoted.

## Trade-offs, and what I did not do

| Decision | Alternative | Why I chose this |
|---|---|---|
| **Bootstrap CIs on everything** | point estimates | At n = 50 a bare rate is close to meaningless. Costs nothing on CPU and is the honest presentation. |
| **Guideline-derived gold** | MTB-derived gold; expert re-annotation | No non-circular gold existed when this was built. Now **MTBBench** supplies one, and adopting it is the top backlog item (see `docs/MTBBENCH_INTEGRATION.md`). Chose to ship with the circular label *plus a loud caveat* rather than have no outcome metric. |
| **Report degenerate metrics** (step soundness 1.00) | suppress them | Suppressing them would make the harness look better and be less true. They are reported with "by construction" attached. |
| **Structural citation resolver** | semantic support-checking (does the record support the claim?) | Semantic checking needs the Claude teacher, which is left unwired. The interface takes a resolver returning `(resolves, supports)` so the upgrade is a drop-in. |
| **Point-biserial correlation** for information gathering | regression controlling for case complexity | With 50 cases and no complexity covariate, a partial correlation would be over-fitted. The confound (more alterations → more evidence *and* harder) is stated instead of modelled. |
| **No LLM-as-judge anywhere** | judge model scoring free text | Circularity: an LLM judging LLM output, flagged in the risk register. Every metric here is computed from structured trace fields. |

## Honest limitations

1. **The headline metric is circular** until MTBBench is wired in.
2. **No human audit** has been run, so label quality is asserted, not measured.
3. **No baseline comparison.** There is no untrained-policy or frontier-model arm yet, so none
   of these numbers can be called an improvement over anything.
4. **n = 50, single tumour type, single timepoint.** No longitudinal evaluation yet, though the
   schema now supports it.

## Tests

`tests/test_evaluation.py` (9): bootstrap CI brackets the mean; Bonferroni scales and clips;
concordance top-1 vs any-match with unlabelled cases skipped; molecular agreement rewards
matching actionability; step accuracy counts unlabelled separately; tool reliability flags
failures, redundancy and empty returns; citation grounding with and without a resolver;
calibration and deferral consistency; graceful degradation with no labels. Part of **71 passing
tests**.

## Next

→ **Phase 7 (packaging)**, and the MTBBench backlog: a non-circular gold standard, and
longitudinal evaluation. The abstention fix, originally listed here as backlog, is now done —
see the addendum below.

## Addendum — the abstain-threshold fix

**Root cause.** `Orchestrator._pick_guideline`'s "no targetable driver" fallback (the generic
chemo/IO chunk, ESCAT tier `"N/A"`) carries confidence weight `_GL_TIER_W["N/A"] = 0.50`
([`agents/orchestrator.py:28`](../src/oncoreason/agents/orchestrator.py#L28)) — and the abstain
decision was a **strict** inequality, `confidence < self.abstain_threshold`, with
`abstain_threshold = 0.5`. `0.50 < 0.50` is `False`, so the fallback guideline — which fires on
*every* case with no gene-specific targeted-therapy guideline — always cleared its own threshold
by exactly zero margin. That is why the system recommended something on all 50 cases regardless
of whether the evidence actually supported one: not a modeling failure, a tie broken the wrong
way by a strict `<` against two numbers deliberately set equal.

**Fix.** One-character change, `<` → `<=` ([`agents/orchestrator.py:145`](../src/oncoreason/agents/orchestrator.py#L145)),
with a comment recording why. Checked that no other weight in `_GL_TIER_W` or `_CIVIC_LVL_W`
sits at exactly 0.50 (the nearest are 0.55 and 0.45), so the change affects only the intended
case and does not newly abstain on any weak-but-real signal.

**Corrected numbers** (`results/20260719-2005-pipeline.json`, same 50 cases, seed 17):

| Metric | Before | After |
|---|---|---|
| Abstained | 0 / 50 | **24 / 50** |
| Molecular interpretation agreement | 0.24 [0.12, 0.36] | **0.72 [0.60, 0.84]** |
| Actionable cases correctly recommended | 12 / 12 | **12 / 12** (unchanged — no actionable case was ever missed, before or after) |
| Non-actionable cases correctly abstained | 0 / 38 | **24 / 38** |

Every other metric (guideline concordance, calibration, deferral curve, tool reliability,
citation grounding, PRM accuracy, information gathering) is **numerically identical** before and
after, because the fix touches only the abstain decision, not the confidence scores or
recommendations themselves — a useful cross-check that the fix is narrowly scoped.

**A second, subtler finding the fix exposed.** 0.72, not 1.00 — because **14 of the 38
non-actionable cases still get a confident, non-abstained recommendation.** Tracing those cases:
each has a gene-specific ESMO/NCCN guideline entry at tier I-A/I-B/I-C (e.g. EGFR, ALK, RET, MET
exon 14, KRAS G12C — all established standard-of-care targeted therapies), which clears the
abstain threshold on its own (weights 0.70–0.95), independent of whether CIViC separately carries
a level A/B item for that exact variant/tumor pairing. `n_actionable_cases` in this metric is
defined narrowly — the Phase 2 screening flag, CIViC level A/B only — so a system correctly
following an established guideline can still register as "wrong" against that narrower
definition. This is **not a new system defect**: 0/38 false negatives (an actionable case that
gets abstained) both before and after the fix, and the 14 "residual gap" cases are all recommending
genuinely-guideline-backed therapies, not hallucinations. It is a **gold-standard definition gap**
— molecular_interpretation_accuracy's ground truth and the guideline specialist's own evidence
base encode "actionable" slightly differently, and the metric should arguably be measured against
the guideline-derived gold (`case.gold`, from Phase 4) rather than the narrower CIViC-only
screening flag. That reconciliation is now the more precise successor item on the backlog, folded
into the MTBBench gold-standard work in `docs/MTBBENCH_INTEGRATION.md`.

**Verified reproducible:** `python scripts/run_pipeline.py` regenerates these exact numbers;
`python scripts/make_results_figure.py` regenerates panel F of
[`summary/results_figure.png`](../summary/results_figure.png) as a 2×2 abstention-decision matrix
(miss / correct / correct / residual-gap) rather than the old two-bar chart, so the residual
finding is visible, not just the headline rate. All 71 tests still pass; the fix added no new
test-breaking behavior. See [`summary/FINDINGS.md`](../summary/FINDINGS.md) for the updated,
project-wide narrative.

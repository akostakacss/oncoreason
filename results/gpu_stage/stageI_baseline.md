# GPU stage I — the baseline arm, and what it reveals about the metric

**Run:** CPU, seed 17 · `scripts/stage_i_baseline.py`
**Artifacts:** `stageI_baseline.json`
**Input:** stage E hard-sampled candidates · Phase 6 harness imported unchanged

## Aim

Phase 6's third honest limitation: *"No baseline comparison. There is no untrained-policy or
frontier-model arm yet, so none of these numbers can be called an improvement over anything."*

Stage E sampled 400 traces from an untrained Qwen2.5-3B-Instruct. That is the missing arm.
Running it through the same eight metrics, the same 50 cases and the same guideline-derived
gold gives every other number in this project a reference point.

## Results

| Metric | Scaffold | Policy ×1 | Policy + BoN |
|---|---|---|---|
| Guideline concordance, top-1 | 0.960 | 0.060 | 0.100 |
| Guideline concordance, any-match | 1.000 | 0.080 | 0.120 |
| Molecular interpretation agreement | 0.720 | 0.680 | 0.700 |
| Step soundness | 0.000 | 0.184 | 0.268 |
| Citation grounding | 1.000 | 1.000 | 1.000 |
| Calibration (ECE) | 0.304 | 0.424 | 0.448 |
| Abstained | 24 / 50 | 31 / 50 | 23 / 50 |

## The headline is not the gap — it is that the metric cannot referee this comparison

Guideline concordance shows 0.960 against 0.060, which reads as the scaffold annihilating the
policy. That number should not be reported without the following, because most of the gap is
measurement rather than capability.

**1. The metric is circular in the scaffold's favour, and Phase 6 already said so.** The gold
label is derived from the same guideline index the scaffold retrieves over, so the scaffold
emits canonical therapy strings *drawn from the thing it is scored against*. Phase 6 called
0.96 "the least meaningful number here". Scoring a free-text policy against that same gold does
not make it a fair contest; it makes the scaffold's structural advantage explicit.

**2. Exact string matching punishes phrasing, not reasoning.** Inspecting the misses among
non-abstained traces:

```
policy 'platinum doublet chemotherapy +'     gold 'platinum doublet chemotherapy'   <- trailing '+'
policy 'platinum based doublet chemotherapy with potential consideration of
        immune checkpoint inhibitors based on the absence of other actionable drivers'
                                             gold 'platinum doublet chemotherapy'   <- verbose, correct
policy 'rank therapy options based on the strength of evidence'                      <- genuine failure
```

Only the third is a real error. The first two are clinically right and lexically wrong.

**3. Abstention is asymmetric between the arms.** The policy writes `defer` as its
recommendation on 31 of 50 cases, which can never match a therapy gold. The scaffold abstains
24 times but still emits a ranked list. Restricting to the 19 non-abstained policy traces gives
**3/19 = 0.158 top-1**, not 0.060 — still well below the scaffold, but nearly three times the
headline figure.

**4. My own parser cost the policy a factor of three.** The first run of this script scored the
policy at **0.020** because the recommendation string still carried its trailing `[CITE: ...]`
block and qualifiers like "first-line". `clean_recommendation` recovers 0.020 → 0.060. That
correction is recorded here rather than silently applied, because a baseline that understates
the comparator by 3× is worse than no baseline.

## What the other metrics say, and they are more informative

**Step soundness inverts the story: 0.000 scaffold vs 0.268 policy + BoN.** The scaffold scores
zero because the semantic labeller judges *its* steps too — and the scaffold cites guideline
chunks whose tumour type does not match the case, which `disease_mismatch` flags. That is the
same TP53-tumor-type failure mode the project was built around, found in the scaffold rather
than the policy. It is also partly an artifact of the label (stage G), so it should not be
over-read in either direction — but it is the first evidence in this project that the
deterministic scaffold has a reasoning-quality problem its own metrics were blind to.

**Molecular interpretation agreement is close: 0.720 vs 0.680 vs 0.700.** On the metric that is
*not* derived from the guideline index, the untrained policy is within four points of the
scaffold. That is the single most important row in the table and it directly contradicts what
concordance implies.

**Best-of-N helps consistently but modestly** — concordance 0.060 → 0.100, step soundness
0.184 → 0.268, abstention 31 → 23. Directionally positive on every row, which is more
encouraging than stage H's null on trace quality alone, though stage H's confidence interval
still applies.

**Calibration is worse for the policy** (0.424 vs 0.304), and both are poor.

## Conclusion

The baseline arm exists, so Phase 6's limitation #3 is retired. But its main finding is
methodological: **guideline concordance cannot fairly compare a deterministic scaffold against
a generative policy**, because the gold is derived from the scaffold's own retrieval corpus and
scored by exact string match. On the less circular metric the two arms are nearly level.

This is the strongest argument yet for the MTBBench integration already scoped in
`docs/MTBBENCH_INTEGRATION.md`: an externally authored, clinician-validated gold with
structured answers would make this comparison meaningful, where the current one cannot be.

## Honest limitations

1. **One sample per case** for the policy arm; a different sample would give different numbers.
2. **The Best-of-N verifier saw 35 of the 50 cases in training.** Its arm is optimistic on
   those and honest only on the 15 held out; stage H is the clean measurement.
3. **Step soundness for the scaffold inherits `disease_mismatch`'s conflation** of policy error
   with retrieval artifact (stage G), so 0.000 overstates the scaffold's problem.
4. **No frontier-model arm.** "Untrained policy" here means untrained *Qwen2.5-3B*, not a
   strong general model, so this bounds the low end rather than the state of the art.

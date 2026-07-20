# GPU stage B — policy sampling

**Run:** `runs/20260720-0912-b-n8-t08` · Kaggle T4 · 25.2 min · kernel
`ta1010/oncoreason-policy-sampling` · Qwen2.5-3B-Instruct, N=8, temperature 0.8, seed 17
**Artifacts:** `candidates.jsonl` (400 samples), `candidates-relabelled.jsonl`,
`stageB_sampling.json`

![Stage B: 400 traces produced at 100% parse rate; the negative class is 95% uncited steps,
and the in-notebook labeller over-counted hallucinations 2.3x](stageB_figure.png)

## Aim

Phase 5 could not train a verifier on real data: a deterministic scaffold emits citations
only from real retrievals, so it **cannot** produce an ungrounded step, and the 298 labelled
steps came back 298/298 sound. The fix at the time was `supervision.mine_negatives`, which
constructs negatives by stripping citations or swapping in ids from another case. That made
the PRM trainable but left its headline 0.914 sitting on a semi-synthetic distribution.

This stage replaces construction with observation: sample from a generative policy and record
what it actually gets wrong.

## Method

Per case, the same CIViC / ClinVar / guideline evidence the CPU pipeline retrieves is placed
in the prompt with explicit ids, and the policy is asked for numbered steps each carrying a
`[CITE: ...]` list. Eight samples per case are drawn in one batched `generate` call. Every
cited id is then checked against that case's evidence pool.

## Results

| Metric | Value |
|---|---|
| Samples | 400 (50 cases × 8) |
| Parse rate | **1.000** |
| Steps | 1076 (mean 2.69 per sample) |
| Within-case score spread | **0.967** |
| Abstain rate | 0.67 |
| Unsound step rate | 0.471 (after relabelling) |

**Temperature 0.8 is the right setting.** A within-case spread of 0.967 means the eight
samples genuinely differ, which is what Best-of-N needs in order to be more than machinery —
Phase 5 limitation #2 recorded that it was implemented but not meaningfully demonstrated,
because the deterministic scaffold yields exactly one trace per case.

### The relabelling correction

The in-notebook labeller matched cited strings exactly and split them on commas only, which
over-counted hallucinations roughly threefold. `scripts/relabel_candidates.py` separates
three things it had conflated:

| Category | Before | After | What it actually is |
|---|---|---|---|
| Steps with invented ids | 61 | **26** | a well-formed id resolving to nothing — a real hallucination |
| Distinct invented strings | 42 | **16** | |
| Prefix dropped (`gl-ret` for `guideline:gl-ret`) | counted as invented | resolved | real evidence, parser too strict |
| Delimiter (`a and b`, `a & b`, `a; b`) | counted as one invented string | split and resolved | real evidence, parser too strict |
| Prose in the CITE slot (`no specific evidence found`) | counted as invented | counted as uncited | a meta-statement, not a fabrication |

## Two findings that matter more than the headline

**1. The real negative class is 95% "cited nothing".** Of 507 unsound steps, **481 are
uncited** and only 26 carry an invented id. A classifier can score well here by detecting an
empty citation slot — which is very nearly the signal the `strip` counterfactual already
provided. Sampling made the negatives real, but it did not make them *hard*.

**2. `off_case` count is zero.** Across 1076 steps the policy never once cited a real id
belonging to a different case. The `swap` strategy in `mine_negatives` — the one modelled on
the TP53 tumor-type mismatch — therefore tests a failure mode this policy does not exhibit,
because it only ever sees one case's evidence in its context. That is a genuine negative
result about the counterfactual design, and it holds regardless of what stage C finds.

## Honest limitations

1. Mean 2.69 steps per sample against ~6 for the deterministic scaffold: the policy reasons
   much more briefly than the scaffold it is meant to replace.
2. "Sound" here means *the citation resolves and belongs to this case*. Whether the cited
   record actually **supports** the claim is not checked — that is semantic verification and
   it is still unwired.
3. Single temperature, single N, single model. No sweep, so 0.8 is defensible rather than
   optimal.
4. Three failed runs preceded this one (nbformat cell serialization, Kaggle archive
   expansion, mount-path discovery). All are archived in `runs/INDEX.tsv`; none affected the
   result, and total wasted T4 time was ~8 minutes.

## Next

→ `stageC_prm_transfer.md` — whether a verifier trained on the constructed negatives
transfers to these real ones.

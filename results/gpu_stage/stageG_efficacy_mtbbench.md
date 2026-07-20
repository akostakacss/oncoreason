# GPU stage G — raising efficacy, guided by MTBBench

**Status:** semantic labelling implemented and measured on CPU · protocol changes designed
**Artifacts:** `scripts/semantic_labels.py`
**Reading:** [`docs/MTBBENCH_INTEGRATION.md`](../../docs/MTBBENCH_INTEGRATION.md) ·
Vasilev K\*, Misrahi A\*, Jain E\*, Cheng P, Liakopoulos P, Michielin O, **Moor M**‡,
**Bunne C**‡ — *MTBBench*, NeurIPS 2025 Datasets & Benchmarks

## The diagnosis stages B–D arrived at

The verifier is not failing because it is too small. It is failing because **the label asks
the wrong question**.

`label_sound` currently means *"this citation resolves and belongs to this case."* That is
bookkeeping. It produced a negative class that was 95% "cited nothing" (stage B), it did not
transfer to real policy output at 0.530 balanced accuracy (stage C), and it left no relational
question for evidence text to answer, which is why inlining that text made TF-IDF *worse*
rather than better (stage D).

MTBBench names the failure modes that actually matter, and none of them are visible to an id
membership check. Their models *"frequently hallucinate, struggle with reasoning from
time-resolved data, and **fail to reconcile conflicting evidence or different modalities**."*

## Solution 1 — semantic labels from structured fields *(implemented, measured)*

CIViC records already ship the fields needed to ask the right question, and so do this
project's own guideline chunks:

| Field | Values |
|---|---|
| `significance` | `SENSITIVITYRESPONSE` \| `RESISTANCE` |
| `evidence_direction` | `SUPPORTS` \| `DOES_NOT_SUPPORT` |
| `therapies` | `['Erlotinib', ...]` |
| `disease` | `'Lung Non-small Cell Carcinoma'` |

`scripts/semantic_labels.py` compares the **claim the step makes** against the **record it
cites**, on three axes:

- **contradiction** — the step claims benefit while citing a record that says `RESISTANCE`, or
  `DOES_NOT_SUPPORT`. This is MTBBench's "fails to reconcile conflicting evidence", detected
  without annotation.
- **therapy mismatch** — the step names a therapy the cited record is not about, so the
  citation cannot support that specific claim.
- **disease mismatch** — the record concerns a different tumour type. This is the machine
  analogue of the TP53 tumor-type mismatch the whole project is framed around.

Claim direction is read from the step text with deliberately conservative patterns: a step
that commits to neither direction returns `None` and is left **unjudged rather than guessed
at**, because a wrong label is worse than a missing one.

### Coverage, and the fix that mattered

Judging only CIViC records covered **23.3%** of steps — the policy cites guideline chunks far
more often than CIViC evidence. Guideline chunks carry `therapies`, `gene` and `tumor_type` of
their own, and a guideline recommendation is by construction a SUPPORTS/SENSITIVITY statement.
Including them lifted coverage to **53.0%**.

### Yield on the 1076 stage B steps

| Signal | Count |
|---|---|
| Steps with a judgeable citation | 570 (53.0%) |
| **Disease mismatch** | **257** |
| **Therapy mismatch** | **23** |
| **Contradiction** | **15** |

Real examples, all mechanically detected:

```
gl-ret (lung adenocarcinoma) cited for a lung SQUAMOUS cell carcinoma case
gl-nodriver-lusc cited while the step names selpercatinib / pralsetinib
    → the record is about pembrolizumab / platinum-doublet chemotherapy
civic:EID1893 cited to claim sensitivity → the record says RESISTANCE to Dacomitinib
```

**Why this should raise efficacy.** The negative class stops being "an empty citation slot"
and becomes "a citation that does not support the claim". That is a question evidence text can
actually answer, which restores the premise stage D falsified — and it gives a transformer
something to attend across that a bag of ngrams provably cannot represent.

### Honest limitations

1. **Proxies, not clinician judgements.** Direction is inferred from regex patterns over step
   text. Stated as such wherever reported; the upgrade path is MTBBench's expert-verified QA
   pairs (§3.3 of the integration note).
2. **53% coverage, not 100%.** Steps citing nothing remain judgeable only structurally.
3. **Disease mismatch is partly a retrieval artifact.** CIViC returns cross-disease evidence
   for a variant, so some mismatches are the retriever's doing rather than the policy's. It is
   reported separately from soundness for that reason, not folded silently into the label.
4. Not yet used to train anything — the yield measurement above is the evidence that it is
   worth doing, not a result about a verifier.

## Solution 2 — MTBBench's agentic protocol *(designed)*

Their setup makes evidence **non-persistent**: at each turn the agent sees a query and a set
of modality files, must *actively request* a subset, and **files do not carry over**, forcing
deliberate information gathering rather than dumping everything into context.

Adopting that would change the negatives qualitatively. Right now the policy is handed its
evidence, so the only errors available are misusing what it was given. Under their protocol it
can also **fail to request what it needed** — an under-gathering failure that is invisible in
the current setup and is exactly what their information-gathering analysis measures.

This matters because of a result Phase 6 already has: my `information_gathering` metric
returns **r = −0.256**, the *opposite* sign to MTBBench's finding that files-accessed
correlates positively with accuracy. The diagnosis was that in a deterministic scaffold "more
evidence" means "harder case", not "better gathering". A request-based protocol is what would
make that metric measure gathering rather than case difficulty.

## Solution 3 — bootstrap CIs on verifier numbers *(designed)*

MTBBench evaluates with **1000-iteration bootstrap resampling and 95% CIs**. Phase 6 already
applies that discipline to its eight metrics, but the stage C and F verifier numbers are bare
point estimates on **15 held-out cases**. Given how much weight the 0.530 figure carries, it
should be reported as an interval. `evaluation.bootstrap_ci` already exists; this is wiring,
not new method.

## Solution 4 — non-circular labels *(designed)*

The deeper fix is the one the integration note already identifies: MTBBench supplies
clinician-validated, externally authored ground truth, replacing labels derived from the same
guideline index the agent retrieves over. For the verifier specifically, that would replace
*my* soundness rule — regex direction matching and id membership — with expert judgement, and
turn every proxy above into a validated target.

## Priority

1. **Semantic labels** — implemented; next step is retraining the verifier on them and
   re-running the stage C transfer measurement.
2. **Bootstrap CIs** — cheap, and the reported numbers need them.
3. **Agentic protocol** — the largest change; would make both the negatives and the
   information-gathering metric meaningful.
4. **MTBBench gold** — removes the circularity that Phase 6 flags as its top limitation.

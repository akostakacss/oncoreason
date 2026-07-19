# MTBBench: why it is the most important paper for this project, and how it changes it

**Paper.** Vasilev K\*, Misrahi A\*, Jain E\*, Cheng P, Liakopoulos P, Michielin O, **Moor M**‡,
**Bunne C**‡. *MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology.*
NeurIPS 2025, Datasets & Benchmarks. ETH Zürich / EPFL / HUG (Geneva University Hospitals).
Code: `github.com/bunnelab/MTBBench` · Data: `huggingface.co/datasets/EeshaanJain/MTBBench`

Read in full. This document is the answer to "how do I fold it into the project".

---

## 1. Why this one outranks the rest

Of the thirteen papers read, this is the closest to what this project targets, for five reasons.

1. **It is the target task, in benchmark form.** The goal here is multimodal reasoning for
   oncology supporting diagnosis, molecular interpretation, treatment selection and
   **longitudinal care**. MTBBench is precisely a *multimodal, longitudinal, agentic* oncology
   decision benchmark. No other paper in the set covers all three axes at once (their Table 1
   makes this claim explicitly against nine competing benchmarks).
2. **It formalises molecular tumor board practice.** An MTB report interprets an NGS profile
   into a therapy recommendation; MTBBench simulates that deliberation directly. The alignment
   is with clinical practice rather than with any modelling choice.
3. **It resolves my biggest open methodological problem.** My gold labels are derived from
   my own guideline index, which is partially circular (documented in
   `supervision.label_case_outcome`). MTBBench provides **clinician-validated, externally
   authored ground truth**, released publicly. It is the non-circular evaluation target the
   roadmap said I needed and could not find.
4. **It runs on Alps under the same programme.** The acknowledgements credit the Swiss AI
   Initiative / CSCS **Alps** allocation, SWISS AI Large Grant No. 46, *"Virtual Patient
   Platform"*. That is the compute environment and the umbrella project this work targets.
5. **It defines the kaiko interface.** MTBBench exposes **pathology foundation models as
   callable tools** (CONCH for H&E, UNI2 + ABMIL for IHC quantification). That is exactly the
   socket kaiko's encoders plug into, and exactly the Pattern-A hook my plan reserved.

## 2. What the paper actually does

- **Two tracks.** *Multimodal*: 26 head-and-neck cases from HANCOCK (CC BY 4.0), 390 expert-
  verified QA pairs over H&E, IHC, hematology. *Longitudinal*: 40 cases from **MSK-CHORD**
  (CC BY-NC-ND 4.0), 183 QA pairs over genomics plus structured clinical timelines.
- **Agentic protocol.** Multi-turn. At turn *t* the agent sees query *q_t* and a set of
  modality files *F_t*, and must **actively request** a subset. Crucially, **files do not
  persist across turns** and must be re-requested, which forces deliberate information
  gathering rather than dumping everything into context.
- **Foundation models as tools.** CONCH (1.17M H&E image-caption pairs) for image-text
  similarity; UNI2 embeddings + attention-based multiple-instance learning (ABMIL) to regress
  percent-positive staining for IHC; plus PubMed (top-30 retrieval reranked by
  `bge-reranker-v2-m3`, top-3 abstracts returned) and DrugBank.
- **Findings.**
  - Frontier models are weak: best multimodal overall 69.1% (internvl3-78b), and
    **outcome/recurrence prediction sits near chance (50%)**. Longitudinal recurrence and
    progression likewise near chance.
  - **Information gathering beats scale**: the number of files accessed correlates with
    accuracy more strongly than model size does.
  - Tools help: up to **+9.0%** multimodal, **+11.2%** longitudinal.
  - Models "frequently hallucinate, struggle with reasoning from time-resolved data, and fail
    to reconcile conflicting evidence or different modalities."
  - Evaluation uses **bootstrap resampling, 1000 iterations, 95% CIs** — the same small-n
    discipline my Phase 6 harness already applies.
- **Stated next direction** (their conclusion): "integration of medical foundation models with
  capabilities for analyzing complex **longitudinal** data, enabling deeper temporal reasoning
  and personalized decision support."

## 3. What it changes in this project

### 3.1 It exposes a real gap: I had no time axis

My `Case` was a single timepoint: a molecular profile plus histology. MTBBench (and
"longitudinal care") makes sequential reasoning central. **Implemented**: `TimelineEvent` and
`ClinicalTimeline` in `cases/schema.py`, with `Case.timeline`. The important method is
`ClinicalTimeline.until(t)`, which returns only events at or before *t* — the guard that stops
future information leaking into a decision that was made before it existed. That is the
temporal analogue of the patient-level split I already enforce.

### 3.2 It supplies a metric I did not have, and that metric immediately found a fault

**Implemented**: `evaluation.information_gathering`, my analogue of their files-accessed
analysis (distinct citations plus non-empty tool calls, correlated with correctness).

Result on my 50 cases: **point-biserial r = −0.256 (p = 0.073)**; mean evidence gathered was
**6.7 when correct** and **13.5 when incorrect**.

The correlation runs *opposite* to MTBBench's. The interpretation is not that they are wrong,
it is that my system fails differently: in a deterministic scaffold, "more evidence" does not
mean "better information gathering", it means "a harder, multi-alteration case where competing
and tumor-mismatched evidence accumulates". So more evidence degrades the adjudication instead
of improving it. That is a genuine weakness of my synthesizer, surfaced by importing their
metric, and it is a better thing to report than another number that flatters the system.

### 3.3 It gives me a non-circular evaluation target

My guideline-derived gold is partly circular. MTBBench is expert-validated and externally
authored. **Planned** (`datasources/mtbbench.py`): a read-only adapter that loads the public
HuggingFace dataset and maps its QA items onto my `Case`/`Trace` contract so the existing
Phase 6 harness can score against it unchanged. Licence discipline applies as everywhere else:
HANCOCK is CC BY 4.0, the MSK-CHORD-derived track is **CC BY-NC-ND 4.0**, so it is
research-use, non-commercial, no-derivatives; it goes behind the connector, is never vendored
into the repo, and the NC/ND terms are recorded in `LICENSING.md`.

### 3.4 It specifies the kaiko socket precisely

**Planned** (`agents/tools.py`): a `FoundationModelTool` protocol —
`describe(image, candidates) -> ranked descriptors` and `quantify(image, marker) -> value` —
matching how MTBBench wraps CONCH and UNI2+ABMIL. Ships unimplemented, like every other
controlled slot, but it means "kaiko's encoders plug in here" points at a signature rather than
at a sentence.

### 3.5 It reframes the roadmap

The plan's MTB-concordance item was gated on "do published MTB studies release case-level
tables?". That gate is now **resolved**: MTBBench releases exactly this. The roadmap item
changes from *"check whether the data exists"* to *"run my scaffold on MTBBench-Longitudinal
and report kappa against the clinician-validated answers"*, which is a concrete first-year task.

The design notes from the plan still hold and should be carried over: report agreement against
the board as a second rater, cite the inter-MTB concordance ceiling, keep guideline concordance
and MTB concordance separate, and version-match guidelines to the study era.

## 4. Where each paper now sits

MTBBench does not displace Med-PRM and PRA as the *method* backbone; it becomes the *target*.

| Role | Paper |
|---|---|
| **Target task and evaluation** | **MTBBench** (multimodal, longitudinal, agentic MTB decisions) |
| Supervision method | Med-PRM (guideline-verified process rewards) |
| Post-training method | PRA (online, frozen-policy step rewards) |
| Vision-side agent template | RadAgent (RL, tool orchestration, checklist, faithfulness) |
| Retrieval and citation grounding | Almanac, MIRIAD |
| Multimodal fusion architecture | Med-Flamingo, GenMI |
| Framing | GMAI |

Read as a programme: **MTBBench says what to be good at; Med-PRM and PRA say how to get good
at it; RadAgent shows it done in another modality.**

## 5. Concrete backlog this creates

Implemented now:
- [x] `ClinicalTimeline` / `TimelineEvent` and `Case.timeline` (§3.1)
- [x] `evaluation.information_gathering` (§3.2), wired into `scripts/run_pipeline.py`

Next, in priority order:
- [ ] `datasources/mtbbench.py` adapter over the public HF dataset (§3.3)
- [ ] `FoundationModelTool` protocol for the kaiko slot (§3.4)
- [ ] Multi-turn orchestration with non-persistent evidence access, mirroring their protocol
- [ ] Longitudinal case construction from MSK-CHORD (available via cBioPortal, allow-list gated)
- [ ] Report kappa against MTBBench ground truth as the non-circular headline number

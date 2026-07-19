# Phase 3 — Reasoning scaffolding

**Status:** 🟡 core built (variant + guideline specialists live; trial/literature + dense
encoder deferred) · **Compute:** CPU-local

## Goal

Build the target agent architecture: a **planner** that decomposes a case, **specialist
sub-agents** each with their own context stream, and a **synthesizer** that produces an
auditable, citation-grounded recommendation with an explicit abstain path — plus the retrieval
layer (sparse/dense/hybrid) and a logged tool layer for reliability metrics.

## What was built

### Retrieval — BM25, hybrid RRF (dense wired) — [`retrieval/base.py`](../src/oncoreason/retrieval/base.py)
- **BM25Retriever** — a faithful Okapi BM25 in pure Python (no third-party dep), so the sparse
  path runs offline. Standard k1=1.5, b=0.75.
- **HybridRetriever** — Reciprocal Rank Fusion (rank-based, scale-free) over a sparse + a dense
  arm. Fully implemented and tested.
- **DenseRetriever** — MedCPT/BGE via sentence-transformers, wired behind a clear enable-path
  (needs the model weights; runs on Kaggle/Colab). The offline/CI path never blocks on a
  download. This covers the "dense / sparse / hybrid" requirement by building each.

### Tool layer — [`agents/tools.py`](../src/oncoreason/agents/tools.py)
Every external lookup goes through `call_tool`, which **times and guards** the call and emits a
`ToolCall` record (ok / latency_ms / error / n_results). A failed call is logged and returns
empty — the run continues — so the trace carries the tool-use-reliability data Phase 6 needs,
and a network hiccup never crashes a case.

### Derived guideline index — [`agents/guideline_index.py`](../src/oncoreason/agents/guideline_index.py)
A small, **author-summarized** set of NSCLC biomarker→therapy recommendations with ESCAT tiers
and provenance notes. This is a **derived representation, never copyrighted guideline text** —
the exact copyright discipline from Phase 0 Gate 5 / the risk register. It exists so the
guideline specialist is functional and the retrieval ablation has something real to search; in
production it is replaced by a richer index built from the licensed text the lab holds, dropped
into the same shape.

### Multi-agent orchestrator — [`agents/orchestrator.py`](../src/oncoreason/agents/orchestrator.py)
Planner → specialists → synthesizer, emitting a `Trace`:
- **planner** decomposes the case into per-alteration sub-questions and records which
  specialists ran;
- **variant specialist** joins CIViC (somatic) + ClinVar (germline) on canonical IDs (Phase 1);
- **guideline specialist** retrieves over the derived index, picking the gene-specific
  recommendation or, if none exists, the tumor-type "no targetable driver" (chemo/IO) chunk —
  **never an unrelated gene's chunk**;
- **synthesizer** merges the guideline prior with CIViC predictive-sensitivity evidence into a
  ranked recommendation, confidence, and abstain flag.

**The decision is deterministic and evidence-derived, not LLM-authored.** An `LLMClient`
([`agents/clients.py`](../src/oncoreason/agents/clients.py)) only *narrates* each step, so a run
is reproducible offline with `DeterministicLLM`; `ClaudeLLM` is the teacher slot, deliberately
left unimplemented so no unreviewed API integration ships. This
is the auditable "guideline = prior, evidence = signal, model = adjudicator" framing from the
plan, in miniature.

## The demo run — [`traces/`](../traces/)

`python scripts/run_scaffold.py` on three canonical cases (real cached CIViC/ClinVar data):

| Case | Recommendation (top) | Confidence | Note |
|---|---|---|---|
| EGFR L858R | **osimertinib** | 0.97 | CIViC level A + guideline I-A agree |
| KRAS G12C | **sotorasib**, adagrasib | 0.95 | CIViC level A + guideline I-B |
| TP53 R175H | **pembrolizumab / platinum-doublet** | 0.50 | no targeted driver → chemo/IO |

Each trace has a planner step, per-alteration variant + guideline steps with logged tool calls
and resolvable citations, and a synthesis step. Saved as `.jsonl` (machine) + `.md` (readable)
+ `INDEX.tsv`, mirroring the `runs/` and `qc/` archive policy.

## A real finding the scaffold surfaced (worth reading)

The first TP53 run recommended **Pazopanib + Vorinostat at 0.53**, edging out the chemo
guideline. Inspecting the evidence showed why: every CIViC predictive-sensitivity item for TP53
R175H is for a **different tumor type** — Pazopanib+Vorinostat in *sarcoma*, EAP in *stomach*,
doxorubicin in *breast*. This is precisely the **tumor-type-specificity trap** in the risk
register ("BRAF V600E is actionable differently across tumors"), and it showed the mismatch
penalty was too weak. Fix: a tumor-mismatched CIViC item is now discounted enough (0.30) that it
cannot outrank the tumor-matched guideline prior. Post-fix, TP53 correctly defers to lung
chemo/IO. This is the kind of silent-wrongness the whole project is built to catch, caught by
the scaffold on its first real run — and the deeper structural answer to it is Phase 4 + the PRM.

## Tests

`tests/test_retrieval.py` (BM25 ranking, no-overlap empties, RRF fusion) and
`tests/test_agents.py` (actionable recommend + citations + tool logging; **abstain fires when
there is no evidence**; no-targeted-driver falls back to chemo not a hallucinated targeted drug;
trace structure is auditable). **51 tests pass** (was 44).

## Scope — built vs. deferred (stated, not hidden)

- **Built & live:** variant + guideline specialists, BM25 + hybrid retrieval, logged tools,
  auditable trace, deterministic synthesizer with abstain.
- **Wired, enable-on-Kaggle:** dense (MedCPT) retrieval — needs model weights.
- **Deferred (plan-scoped):** trial (ClinicalTrials.gov) and literature (PubMed) specialists —
  the interface accepts them; v1 ships variant + guideline. Trial matching is scoped to
  biomarker retrieval, not eligibility parsing (an unsolved NLP problem — risk register).
- **Teacher API:** `ClaudeLLM` intentionally unimplemented; the interface is defined.

## Known limitations

- The guideline index is gene-level; for EGFR it may cite the T790M chunk for an L858R case
  (both recommend osimertinib, so the recommendation is right, the rationale text is imprecise).
  Variant-aware guideline matching is a documented upgrade.
- Synthesis weights are hand-set (the auditable prior); Phase 5's PRM is what learns to score
  step soundness rather than relying on these fixed weights.

## Next

→ **Phase 4 — Supervision**: segment these traces into steps and attach guideline-verified step
labels (this is where `gold = None` and unscored `prm_score` start getting resolved), then
**Phase 5** trains the PRM on them. The scaffold already emits traces in the exact shape Phase 4
consumes.

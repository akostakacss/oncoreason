# Phase 4 — Process supervision (guideline-verified step labelling)

**Status:** 🟡 core built (deterministic RAG-as-a-Judge; semantic teacher-judge reserved) ·
**Compute:** CPU-local

## Goal

Turn the auditable traces from Phase 3 into **step-level training signal** — the load-bearing
novelty of the whole project. This is the Med-PRM idea (Yun et al., EMNLP 2025): **the guideline
supplies the label.** Each reasoning step is auto-verified against the evidence it rests on
(sound / unsound), so no human annotation team is needed; a human then audits a held-out subset
and the agreement (**Cohen's kappa**) is *reported* as the honest foundation of the claim.

## What was built ([`supervision/labelling.py`](../src/oncoreason/supervision/labelling.py))

- **Fixed segmentation rule** (`segment_steps`) — one `ReasoningStep` = one step unit, pinned
  once (segmentation changes the labels, so it must be mechanical — Phase 4.1).
- **RAG-as-a-Judge label** (`label_step_against_guideline`) — a step is **sound iff its
  evidential claims are backed by resolvable retrieved evidence**. A step that cites a record
  which does **not** resolve (a hallucinated / unverifiable citation) is labelled **unsound** —
  exactly the citation-grounding failure the PRM must learn to catch. A step with no evidential
  citation (the planner) is structurally sound (nothing to verify).
- **Trace labelling + write-back** (`label_trace`, `annotate_trace_with_labels`) — labels every
  step and writes `sound` onto `ReasoningStep.label_sound` (consumed by the Phase-5 PRM).
- **PRM dataset builder** (`build_prm_examples`) — emits `(case_id, step_index, step_text,
  evidence_ids, label_sound)` rows: the training data Phase 5 consumes.
- **Audit** (`audit_agreement`) — Cohen's kappa between auto-labels and a manual audit subset,
  aligned by `(case_id, step_index)`. This is the number to report.

This is the **deterministic, offline core** of Med-PRM's RAG-as-a-Judge. The *semantic* version
— does the record actually **support** the claim, not merely resolve — is the reserved teacher
(Claude) slot, left unwired so no unreviewed API integration ships. The
verifier here already catches the load-bearing failure mode (unverifiable citations).

## Demonstration (on the real Phase-3 trace)

Labelling the saved EGFR L858R trace:

| Step | Kind | Sound | Verified citations |
|---|---|---|---|
| 0 | planner | ✅ (structural) | 0 — no evidential claim |
| 1 | variant specialist | ✅ | 6 resolvable CIViC/ClinVar records |
| 2 | guideline specialist | ✅ | 1 resolvable guideline record |
| 3 | synthesizer | ✅ (structural) | 0 — no evidential claim |

→ 4 PRM training rows, 4 sound. A step citing a non-resolvable record (tested with a fabricated
trial id) is correctly labelled **unsound** — the verifier discriminates, it does not rubber-stamp.

## Why this matters (the connection to the lab)

This is the exact mechanism behind the two Moor-lab backbone papers:
- **Med-PRM** — labels each step against *retrieved* evidence (RAG-as-a-Judge); reports expert
  agreement (kappa 0.74/0.71). My labeller is the same idea; the audit kappa is the honest
  number to report once a human subset is scored.
- **PRA (2026)** — the online successor that steers a *frozen* policy with these step rewards.
  My labels → Phase-5 PRM → Best-of-N is the Med-PRM rung; a PRA-style online reward agent is
  the documented roadmap.

See [`../publications/REFERENCES.md`](../publications/REFERENCES.md) for the source papers.

## Tests

`tests/test_labelling.py` — planner is structurally sound; resolvable citation → sound while a
hallucinated citation → unsound; PRM-row shape; audit kappa (perfect = 1.0, partial < 1.0).
**55 tests pass** (was 51).

## Scope — built vs. deferred

- **Built:** deterministic guideline-verified labelling, PRM-dataset builder, audit-kappa.
- **Deferred (reserved teacher slot):** the semantic "does the evidence *support* the claim"
  judge (Claude) — left unwired; the interface is ready.
- **Deferred to Phase 5:** training the PRM (ModernBERT) on `build_prm_examples` output;
  trace score = **min over step scores**; verifier-guided Best-of-N; RFT/DPO.

## Next

→ **Phase 5 — Post-training:** train the ModernBERT PRM on the labelled steps, then verifier-
guided inference (Best-of-N) and RFT/DPO. The label pipeline already emits data in the exact
shape Phase 5 consumes.

"""Process supervision: guideline-verified step labelling + audit kappa. Offline."""
from __future__ import annotations

from oncoreason.agents.trace import Citation, ReasoningStep, Trace
from oncoreason.supervision import (
    StepLabel,
    audit_agreement,
    build_prm_examples,
    label_trace,
    resolvable_evidence_ids,
)


def _trace() -> Trace:
    return Trace(
        case_id="c1",
        steps=[
            ReasoningStep(index=0, text="Plan: consult specialists."),  # no citations
            ReasoningStep(index=1, text="EGFR L858R: CIViC level A.",
                          citations=[Citation("civic:EID1", "civic", "sensitivity to osimertinib")]),
            ReasoningStep(index=2, text="Guideline: osimertinib (ESCAT I-A).",
                          citations=[Citation("guideline:gl-egfr-sens", "guideline_index", "osimertinib")]),
            # a step that cites a record which does NOT resolve -> should be unsound
            ReasoningStep(index=3, text="Trial match found.",
                          citations=[Citation("trial:NCT_HALLUCINATED", "trial", "made-up trial")]),
        ],
    )


def test_planner_step_is_structurally_sound():
    t = _trace()
    labels = {l.step_index: l for l in label_trace(t)}
    assert labels[0].sound is True
    assert labels[0].note == "no evidential claim"
    assert labels[0].verifying_citation_ids == []


def test_resolvable_citation_is_sound_hallucinated_is_unsound():
    t = _trace()
    # resolvable set is built from the trace's real citations; the hallucinated id is in it too
    # here, so simulate the real check by passing only the genuinely-resolvable ids.
    resolvable = {"civic:EID1", "guideline:gl-egfr-sens"}
    labels = {l.step_index: l for l in label_trace(t, resolvable)}
    assert labels[1].sound is True and labels[1].verifying_citation_ids == ["civic:EID1"]
    assert labels[2].sound is True
    assert labels[3].sound is False                       # cites an unresolvable record
    assert labels[3].note == "cites no resolvable evidence"


def test_build_prm_examples_shape():
    t = _trace()
    rows = build_prm_examples(t, {"civic:EID1", "guideline:gl-egfr-sens"})
    assert len(rows) == 4
    r3 = next(r for r in rows if r["step_index"] == 3)
    assert r3["label_sound"] is False
    assert r3["evidence_ids"] == ["trial:NCT_HALLUCINATED"]
    assert set(rows[0]) == {"case_id", "step_index", "step_text", "evidence_ids", "label_sound"}


def test_audit_kappa_perfect_and_partial():
    auto = [StepLabel("c1", i, s, []) for i, s in [(0, True), (1, True), (2, False), (3, False)]]
    # manual agrees on all 4 -> kappa 1.0
    manual_same = [StepLabel("c1", i, s, []) for i, s in [(0, True), (1, True), (2, False), (3, False)]]
    assert audit_agreement(auto, manual_same) == 1.0
    # manual flips two -> kappa < 1
    manual_diff = [StepLabel("c1", i, s, []) for i, s in [(0, True), (1, False), (2, True), (3, False)]]
    assert audit_agreement(auto, manual_diff) < 1.0

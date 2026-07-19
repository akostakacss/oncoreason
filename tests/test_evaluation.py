"""Evaluation harness: every metric the harness defines, with small-n discipline. Offline."""
from __future__ import annotations

from oncoreason.agents.trace import Citation, ReasoningStep, ToolCall, Trace
from oncoreason.cases.schema import Alteration, Case, ClinicalContext, GoldRecommendation
from oncoreason.evaluation import (
    bonferroni,
    bootstrap_ci,
    calibration,
    citation_grounding,
    deferral_curve,
    guideline_concordance,
    molecular_interpretation_accuracy,
    reasoning_step_accuracy,
    tool_use_reliability,
)

LUAD = ClinicalContext(tumor_type="lung adenocarcinoma")


def _case(cid, gold=None, actionable=False):
    return Case(
        case_id=cid,
        alterations=[Alteration("EGFR", "L858R")],
        context=LUAD,
        gold=GoldRecommendation(recommended=[gold]) if gold else None,
        provenance={"evidence": {"EGFR L858R": {"actionable": actionable}}},
    )


def _trace(cid, rec, conf, abstained=False, sound=None, calls=(), cites=()):
    step = ReasoningStep(
        index=0, text="step",
        tool_calls=list(calls),
        citations=[Citation(c, "civic", "claim") for c in cites],
    )
    step.label_sound = sound
    return Trace(case_id=cid, steps=[step], recommendation=list(rec),
                 confidence=conf, abstained=abstained)


def test_bootstrap_ci_brackets_the_mean():
    mean, lo, hi = bootstrap_ci([1.0] * 8 + [0.0] * 2, seed=1)
    assert lo <= mean <= hi
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


def test_bonferroni_scales_and_clips():
    assert bonferroni([0.01, 0.02]) == [0.02, 0.04]
    assert bonferroni([0.6, 0.9]) == [1.0, 1.0]


def test_guideline_concordance_top1_and_skips_unlabelled():
    cases = [_case("a", gold="osimertinib"), _case("b", gold="osimertinib"), _case("c")]
    traces = [
        _trace("a", ["osimertinib", "erlotinib"], 0.9),   # top1 hit
        _trace("b", ["erlotinib", "osimertinib"], 0.8),   # any-match only
        _trace("c", ["whatever"], 0.5),                   # unlabelled -> skipped
    ]
    out = guideline_concordance(traces, cases)
    assert out["top1"]["n"] == 2 and out["top1"]["rate"] == 0.5
    assert out["any_match"]["rate"] == 1.0
    assert out["skipped_unlabelled"] == 1


def test_molecular_interpretation_rewards_matching_actionability():
    cases = [_case("a", actionable=True), _case("b", actionable=False)]
    traces = [
        _trace("a", ["osimertinib"], 0.9),          # recommends, is actionable -> correct
        _trace("b", [], 0.2, abstained=True),        # abstains, not actionable -> correct
    ]
    out = molecular_interpretation_accuracy(traces, cases)
    assert out["agreement"]["rate"] == 1.0
    assert out["n_actionable_cases"] == 1


def test_reasoning_step_accuracy_counts_unlabelled_separately():
    traces = [_trace("a", ["x"], 0.9, sound=True), _trace("b", ["x"], 0.9, sound=None)]
    out = reasoning_step_accuracy(traces)
    assert out["step_soundness"]["n"] == 1
    assert out["unlabelled_steps"] == 1


def test_tool_use_reliability_flags_failures_and_redundancy():
    calls = [
        ToolCall("civic.retrieve", {"q": 1}, True, 10.0, None, 5),
        ToolCall("civic.retrieve", {"q": 1}, True, 12.0, None, 5),   # redundant
        ToolCall("clinvar.by_ids", {"q": 2}, False, 5.0, "HTTPError", None),
        ToolCall("civic.retrieve", {"q": 3}, True, 8.0, None, 0),    # succeeded but empty
    ]
    out = tool_use_reliability([_trace("a", ["x"], 0.9, calls=calls)])
    assert out["n_calls"] == 4
    assert out["success"]["rate"] == 0.75
    assert out["redundant_calls"] == 1
    assert out["successful_but_empty"] == 1
    assert "HTTPError" in out["errors"][0]


def test_citation_grounding_with_and_without_resolver():
    traces = [_trace("a", ["x"], 0.9, cites=["civic:EID1", "civic:MISSING"])]
    structural = citation_grounding(traces)
    assert structural["n_citations"] == 2 and structural["resolved"] is None

    resolved = citation_grounding(traces, resolver=lambda cid: cid != "civic:MISSING")
    assert resolved["resolved"]["rate"] == 0.5


def test_calibration_and_deferral_are_consistent():
    # confident-and-right, unconfident-and-wrong: well ordered
    cases = [_case(str(i), gold="osimertinib") for i in range(4)]
    traces = [
        _trace("0", ["osimertinib"], 0.9), _trace("1", ["osimertinib"], 0.9),
        _trace("2", ["wrong"], 0.2), _trace("3", ["wrong"], 0.2),
    ]
    cal = calibration(traces, cases)
    assert cal["n"] == 4 and 0.0 <= cal["ece"] <= 1.0
    assert cal["brier"]["value"] < 0.1          # confidence tracks correctness well

    dc = deferral_curve(traces, cases)
    assert dc["n"] == 4
    high = next(p for p in dc["points"] if p["threshold"] == 0.7)
    assert high["accuracy"] == 1.0 and high["coverage"] == 0.5
    assert dc["accuracy_monotone_in_threshold"] is True


def test_metrics_degrade_gracefully_without_labels():
    traces = [_trace("a", ["x"], 0.9)]
    assert calibration(traces, [_case("a")])["n"] == 0
    assert deferral_curve(traces, [_case("a")])["n"] == 0

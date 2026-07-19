"""PRM: case-level splitting, training, calibration, and verifier-guided selection. Offline."""
from __future__ import annotations

from oncoreason.agents.trace import Citation, ReasoningStep, Trace
from oncoreason.training import (
    PRM,
    PRMConfig,
    best_of_n,
    build_dpo_pairs,
    reward_hacking_report,
    score_trace_with_prm,
    select_rft_traces,
    split_by_case,
    trace_score,
    train_prm,
)


def _examples():
    """Sound steps cite evidence; unsound steps do not. A learnable signal."""
    rows = []
    for i in range(12):
        rows.append({"case_id": f"c{i}", "step_index": 0,
                     "step_text": "Variant EGFR L858R shows CIViC level A sensitivity evidence.",
                     "evidence_ids": [f"civic:EID{i}"], "label_sound": True})
        rows.append({"case_id": f"c{i}", "step_index": 1,
                     "step_text": "Trial match asserted with no supporting record at all.",
                     "evidence_ids": [], "label_sound": False})
    return rows


def _trace(case_id, texts_and_cites):
    steps = [
        ReasoningStep(index=i, text=txt,
                      citations=[Citation(c, "civic", "claim") for c in cites])
        for i, (txt, cites) in enumerate(texts_and_cites)
    ]
    return Trace(case_id=case_id, steps=steps)


def test_split_is_case_disjoint():
    train, test, test_cases = split_by_case(_examples(), test_frac=0.3, seed=1)
    train_cases = {e["case_id"] for e in train}
    assert test_cases and not (train_cases & test_cases), "case leaked across the split"
    assert len(train) + len(test) == len(_examples())


def test_prm_trains_and_separates_sound_from_unsound():
    model, report = train_prm(_examples(), PRMConfig(seed=1))
    assert report.n_test_cases > 0
    assert report.accuracy >= 0.8, f"PRM failed to learn the signal: {report.summary()}"
    good = model.score_step("CIViC level A sensitivity evidence.", ["civic:EID1"])
    bad = model.score_step("Trial match asserted with no supporting record at all.", [])
    assert good > bad, "verifier does not rank a grounded step above an ungrounded one"


def test_temperature_scaling_is_fitted():
    _, report = train_prm(_examples(), PRMConfig(seed=1, temperature_scale=True))
    assert report.temperature > 0


def test_trace_score_uses_min_not_mean():
    # one broken step must drag the whole trace down
    assert trace_score([0.9, 0.9, 0.1], "min") == 0.1
    assert trace_score([0.9, 0.9, 0.1], "mean") > 0.6
    assert trace_score([], "min") == 0.0


def test_best_of_n_prefers_the_grounded_trace():
    model, _ = train_prm(_examples(), PRMConfig(seed=1))
    grounded = _trace("x", [("CIViC level A sensitivity evidence.", ["civic:EID1"])])
    ungrounded = _trace("x", [("Trial match asserted with no supporting record at all.", [])])
    best, scores = best_of_n(model, [ungrounded, grounded])
    assert best is grounded
    assert len(scores) == 2


def test_rft_selection_and_dpo_pairs_with_hacking_guard():
    model, _ = train_prm(_examples(), PRMConfig(seed=1))
    grounded = _trace("x", [("CIViC level A sensitivity evidence.", ["civic:EID1"])])
    ungrounded = _trace("x", [("Trial match asserted with no supporting record at all.", [])])

    kept = select_rft_traces(model, [grounded, ungrounded], threshold=0.5)
    assert grounded in kept and ungrounded not in kept

    pairs = build_dpo_pairs(model, {"x": [grounded, ungrounded]})
    assert len(pairs) == 1 and pairs[0]["chosen"] is grounded
    rep = reward_hacking_report(pairs)
    assert rep["n_pairs"] == 1 and "suspicious" in rep


def test_prm_roundtrip_save_load(tmp_path):
    model, _ = train_prm(_examples(), PRMConfig(seed=1), out_dir=str(tmp_path))
    reloaded = PRM.load(str(tmp_path))
    a = model.score_step("CIViC level A sensitivity evidence.", ["civic:EID1"])
    b = reloaded.score_step("CIViC level A sensitivity evidence.", ["civic:EID1"])
    assert abs(a - b) < 1e-9

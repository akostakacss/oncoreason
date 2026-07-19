"""Clinical evaluation. CPU; reproducible; pre-registered.

Every dimension this project targets. Metrics and thresholds are fixed here *before* running,
which is what makes this a pre-registered analysis plan rather than a fishing expedition.

Small-n discipline (this is the whole point of the section): every rate is reported with a
**bootstrap confidence interval**, never as a bare point estimate, because at 50 cases a
point estimate is close to meaningless. Where many metrics are compared at once, apply
`bonferroni` to the p-values before drawing conclusions.

The four Almanac axes (factuality, completeness, adversarial safety, preference) and the
AgentClinic emphasis on tool use are reflected in the metric set. Phase 6.
"""
from __future__ import annotations

import random
import statistics

from ..agents.trace import Trace
from ..cases.schema import Case

# --- small-n statistics ------------------------------------------------------


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 17) -> tuple[float, float, float]:
    """Return (mean, lo, hi) percentile bootstrap CI. Reported with EVERY rate."""
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(statistics.fmean(rng.choices(values, k=n)))
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (statistics.fmean(values), lo, hi)


def _rate(flags: list[bool], seed: int = 17) -> dict:
    vals = [1.0 if f else 0.0 for f in flags]
    mean, lo, hi = bootstrap_ci(vals, seed=seed)
    return {"n": len(vals), "rate": round(mean, 4),
            "ci95": [round(lo, 4), round(hi, 4)]}


def bonferroni(pvalues: list[float]) -> list[float]:
    """Bonferroni-correct a family of p-values (many metrics x conditions)."""
    m = len(pvalues)
    return [min(1.0, p * m) for p in pvalues]


def _norm(s: str) -> str:
    return " ".join((s or "").lower().replace("-", " ").split())


def _recommended_set(trace: Trace) -> set[str]:
    return {_norm(r) for r in (trace.recommendation or []) if r}


def _gold_set(case: Case) -> set[str]:
    if not case.gold:
        return set()
    return {_norm(r) for r in (case.gold.recommended or []) if r}


# --- accuracy / concordance --------------------------------------------------

def guideline_concordance(traces: list[Trace], cases: list[Case]) -> dict:
    """Agreement with the guideline recommendation. Keep DISTINCT from MTB concordance.

    Scored top-1 (is the guideline's therapy the model's first choice) and any-match
    (does it appear anywhere in the ranked list). Cases without a gold label are skipped
    and counted, never silently treated as failures.
    """
    by_id = {c.case_id: c for c in cases}
    top1, anyhit, skipped = [], [], 0
    for t in traces:
        c = by_id.get(t.case_id)
        gold = _gold_set(c) if c else set()
        if not gold:
            skipped += 1
            continue
        rec = [_norm(r) for r in (t.recommendation or [])]
        top1.append(bool(rec) and rec[0] in gold)
        anyhit.append(any(r in gold for r in rec))
    return {"top1": _rate(top1), "any_match": _rate(anyhit),
            "skipped_unlabelled": skipped}


def molecular_interpretation_accuracy(traces: list[Trace], cases: list[Case]) -> dict:
    """Variant -> actionability against the attached CIViC/ClinVar evidence.

    A case is 'actionable' if its provenance carries level A/B predictive sensitivity
    evidence; the model is scored on whether it produced a (non-abstained) recommendation
    exactly when the evidence supports one.
    """
    by_id = {c.case_id: c for c in cases}
    correct, n_actionable = [], 0
    recommended_when_actionable = abstained_when_actionable = 0
    recommended_when_not_actionable = abstained_when_not_actionable = 0
    for t in traces:
        c = by_id.get(t.case_id)
        if c is None:
            continue
        ev = (c.provenance or {}).get("evidence", {}) or {}
        actionable = any(v.get("actionable") for v in ev.values())
        n_actionable += int(actionable)
        predicted = bool(t.recommendation) and not t.abstained
        correct.append(predicted == actionable)
        if actionable and predicted:
            recommended_when_actionable += 1
        elif actionable and not predicted:
            abstained_when_actionable += 1
        elif not actionable and predicted:
            recommended_when_not_actionable += 1
        else:
            abstained_when_not_actionable += 1
    return {
        "agreement": _rate(correct),
        "n_actionable_cases": n_actionable,
        # a 2x2 breakdown of the abstention decision itself, not just the pass/fail rate:
        # recommended_when_not_actionable is the residual gap even after the abstain-threshold
        # tie fix — cases where a gene-specific guideline tier alone (independent of the
        # narrower CIViC-level-A/B "actionable" screening flag) clears the abstain threshold.
        "recommended_when_actionable": recommended_when_actionable,
        "abstained_when_actionable": abstained_when_actionable,
        "recommended_when_not_actionable": recommended_when_not_actionable,
        "abstained_when_not_actionable": abstained_when_not_actionable,
    }


def reasoning_step_accuracy(traces: list[Trace]) -> dict:
    """Step-level soundness. Catches 'right answer, wrong reasoning', which outcome-only
    evaluation is blind to. Requires traces labelled in Phase 4 (`step.label_sound`)."""
    flags = [bool(s.label_sound) for t in traces for s in t.steps if s.label_sound is not None]
    unlabelled = sum(1 for t in traces for s in t.steps if s.label_sound is None)
    return {"step_soundness": _rate(flags), "unlabelled_steps": unlabelled}


# --- reliability -------------------------------------------------------------

def tool_use_reliability(traces: list[Trace]) -> dict:
    """Call success rate, latency, and redundant calls (from the logged ToolCalls).

    AgentClinic's point: effective tool use is a first-class axis, not an implementation
    detail. A silent tool failure yields a confident but empty answer.
    """
    calls = [c for t in traces for s in t.steps for c in s.tool_calls]
    if not calls:
        return {"n_calls": 0, "success": _rate([]), "note": "no tool calls logged"}
    ok = [bool(c.ok) for c in calls]
    lat = [c.latency_ms for c in calls if c.latency_ms is not None]
    seen, redundant = set(), 0
    for c in calls:
        key = (c.tool, repr(sorted((c.args or {}).items(), key=lambda kv: kv[0])))
        if key in seen:
            redundant += 1
        seen.add(key)
    empty = [c for c in calls if c.ok and (c.n_results == 0)]
    return {
        "n_calls": len(calls),
        "success": _rate(ok),
        "median_latency_ms": round(statistics.median(lat), 2) if lat else None,
        "redundant_calls": redundant,
        "successful_but_empty": len(empty),
        "errors": [c.error for c in calls if c.error][:5],
    }


def citation_grounding(traces: list[Trace], resolver=None) -> dict:
    """Each citation must resolve to a real record, and (when a resolver is supplied) the
    record must support the claim.

    `resolver` is a callable id -> bool (resolves) or id -> (resolves, supports). Without
    one, only structural grounding is reported and that limitation is stated in the output.
    """
    cites = [c for t in traces for c in t.all_citations()]
    if not cites:
        return {"n_citations": 0, "note": "no citations emitted"}
    if resolver is None:
        per_trace = [len(t.all_citations()) > 0 for t in traces]
        return {
            "n_citations": len(cites),
            "traces_with_citations": _rate(per_trace),
            "resolved": None,
            "note": "no resolver supplied; structural grounding only, support not verified",
        }
    resolved, supported = [], []
    for c in cites:
        r = resolver(c.citation_id)
        if isinstance(r, tuple):
            resolved.append(bool(r[0]))
            supported.append(bool(r[1]))
        else:
            resolved.append(bool(r))
    out = {"n_citations": len(cites), "resolved": _rate(resolved)}
    if supported:
        out["supports_claim"] = _rate(supported)
    return out


# --- calibration & deferral (biostatistics strength) -------------------------

def _correct_flags(traces: list[Trace], cases: list[Case]) -> list[tuple[float, bool]]:
    """(confidence, correct) pairs for labelled cases only."""
    by_id = {c.case_id: c for c in cases}
    out = []
    for t in traces:
        c = by_id.get(t.case_id)
        gold = _gold_set(c) if c else set()
        if not gold or t.confidence is None:
            continue
        rec = [_norm(r) for r in (t.recommendation or [])]
        out.append((float(t.confidence), bool(rec) and rec[0] in gold))
    return out


def calibration(traces: list[Trace], cases: list[Case], n_bins: int = 5) -> dict:
    """Reliability curve, Brier score, expected calibration error, with CIs.

    A confidence that does not track correctness cannot justify deferral, so this metric
    gates the abstention story rather than decorating it.
    """
    pairs = _correct_flags(traces, cases)
    if not pairs:
        return {"n": 0, "note": "no labelled cases with confidence; calibration undefined"}
    briers = [(p - (1.0 if y else 0.0)) ** 2 for p, y in pairs]
    mean_b, lo_b, hi_b = bootstrap_ci(briers)
    bins = []
    ece = 0.0
    for i in range(n_bins):
        lo_e, hi_e = i / n_bins, (i + 1) / n_bins
        sel = [(p, y) for p, y in pairs if (p >= lo_e and (p < hi_e or i == n_bins - 1))]
        if not sel:
            bins.append({"bin": [lo_e, hi_e], "n": 0})
            continue
        conf = statistics.fmean([p for p, _ in sel])
        acc = statistics.fmean([1.0 if y else 0.0 for _, y in sel])
        ece += (len(sel) / len(pairs)) * abs(acc - conf)
        bins.append({"bin": [lo_e, hi_e], "n": len(sel),
                     "mean_confidence": round(conf, 4), "accuracy": round(acc, 4)})
    return {
        "n": len(pairs),
        "brier": {"value": round(mean_b, 4), "ci95": [round(lo_b, 4), round(hi_b, 4)]},
        "ece": round(ece, 4),
        "reliability_curve": bins,
    }


def deferral_curve(traces: list[Trace], cases: list[Case],
                   thresholds: list[float] | None = None) -> dict:
    """Accuracy vs coverage as the abstain threshold moves.

    The loop the whole project closes: verifier score -> confidence -> abstain -> coverage.
    A useful system trades coverage for accuracy monotonically; if it does not, say so.
    """
    pairs = _correct_flags(traces, cases)
    if not pairs:
        return {"n": 0, "note": "no labelled cases with confidence; deferral undefined"}
    ths = thresholds or [0.0, 0.3, 0.5, 0.7, 0.9]
    points = []
    for th in ths:
        kept = [(p, y) for p, y in pairs if p >= th]
        if not kept:
            points.append({"threshold": th, "coverage": 0.0, "accuracy": None, "n": 0})
            continue
        acc_vals = [1.0 if y else 0.0 for _, y in kept]
        mean_a, lo_a, hi_a = bootstrap_ci(acc_vals)
        points.append({
            "threshold": th,
            "coverage": round(len(kept) / len(pairs), 4),
            "accuracy": round(mean_a, 4),
            "accuracy_ci95": [round(lo_a, 4), round(hi_a, 4)],
            "n": len(kept),
        })
    accs = [p["accuracy"] for p in points if p["accuracy"] is not None]
    monotone = all(a <= b + 1e-9 for a, b in zip(accs, accs[1:]))
    return {"n": len(pairs), "points": points, "accuracy_monotone_in_threshold": monotone}


def information_gathering(traces: list[Trace], cases: list[Case]) -> dict:
    """Does gathering more evidence actually make the answer better?

    MTBBench's headline empirical finding (Vasilev, ..., Moor, Bunne, NeurIPS 2025 D&B) is
    that the number of modality files an agent chooses to access correlates with accuracy
    *more strongly than model scale does*: "effective information gathering, rather than raw
    scale, is a key determinant of accuracy."

    My analogue of "files accessed" is evidence actually retrieved: distinct citations plus
    non-empty tool calls. I report the point-biserial correlation with correctness, and the
    mean evidence gathered for correct vs incorrect cases. A near-zero or negative
    correlation is itself informative: it means the agent is retrieving without benefiting,
    which is the failure mode this metric exists to expose.
    """
    by_id = {c.case_id: c for c in cases}
    ev_counts, correct = [], []
    for t in traces:
        c = by_id.get(t.case_id)
        gold = _gold_set(c) if c else set()
        if not gold:
            continue
        n_cites = len({x.citation_id for x in t.all_citations()})
        n_useful = sum(1 for s in t.steps for tc in s.tool_calls
                       if tc.ok and (tc.n_results or 0) > 0)
        rec = [_norm(r) for r in (t.recommendation or [])]
        ev_counts.append(float(n_cites + n_useful))
        correct.append(bool(rec) and rec[0] in gold)

    if len(ev_counts) < 3 or len(set(correct)) < 2:
        return {"n": len(ev_counts),
                "note": "need >=3 labelled cases and both outcomes for a correlation"}

    hit = [e for e, y in zip(ev_counts, correct) if y]
    miss = [e for e, y in zip(ev_counts, correct) if not y]
    try:
        from scipy.stats import pointbiserialr
        r, p = pointbiserialr([1 if y else 0 for y in correct], ev_counts)
        r, p = float(r), float(p)
    except Exception:  # pragma: no cover - scipy optional
        r, p = float("nan"), float("nan")

    return {
        "n": len(ev_counts),
        "n_correct": len(hit),
        "n_incorrect": len(miss),
        "evidence_when_correct": hit,
        "evidence_when_incorrect": miss,
        "mean_evidence_when_correct": round(statistics.fmean(hit), 3) if hit else None,
        "mean_evidence_when_incorrect": round(statistics.fmean(miss), 3) if miss else None,
        "pointbiserial_r": None if r != r else round(r, 4),
        "p_value": None if p != p else round(p, 4),
        "interpretation": (
            "more evidence gathered tracks with being correct" if (r == r and r > 0.1)
            else "gathering more evidence does not track with being correct"
        ),
    }


#: the harness runs this ordered set; see configs/default.yaml
METRIC_REGISTRY = {
    "guideline_concordance": guideline_concordance,
    "molecular_interpretation_accuracy": molecular_interpretation_accuracy,
    "reasoning_step_accuracy": reasoning_step_accuracy,
    "tool_use_reliability": tool_use_reliability,
    "citation_grounding": citation_grounding,
    "calibration": calibration,
    "deferral_curve": deferral_curve,
    "information_gathering": information_gathering,
}

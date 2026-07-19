"""Post-training — verifier-guided inference and the two compared methods.

This is the "comparison of RL training approaches", at honest scale:
  - **Best-of-N** (Phase 5.2): sample N traces, score each with the PRM, keep the best.
    Training-free, and the first thing to report (does the verifier actually help?).
  - **RFT** (rejection-sampling fine-tuning): PRM selects high-scoring traces -> LoRA-SFT.
  - **DPO**: PRM builds preference pairs -> LoRA-DPO (trl).

What is CPU-runnable and therefore implemented here: the **trace scoring rule**, Best-of-N
selection, the RFT trace filter, the DPO pair construction, and the reward-hacking guard.
These are the parts that carry the intellectual content and can be tested offline.

What needs a GPU and therefore stays gated: the actual LoRA-SFT / LoRA-DPO training calls
(`trl` + `peft` on Kaggle). They raise a clear enable-message rather than pretending.

Trace scoring uses **min over step scores** (Med-PRM / PRA): a reasoning trace is only as
sound as its weakest step. A mean would let one strong step paper over a broken one.

No PPO/GRPO here (infeasible on a free T4). GRPO on Alps is the roadmap scale-up, with the
PRM as reward model. Phase 5.2 / 5.3.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class PolicyConfig:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"  # escalated at Gate 1
    load_in_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lr: float = 1e-4
    epochs: int = 1
    seed: int = 17


# ---- 5.2 verifier-guided inference ---------------------------------------------------
def trace_score(step_probs: list[float], rule: str = "min") -> float:
    """Aggregate step scores into one trace score.

    `min` is the default and the one to report: a trace is only as sound as its weakest
    step (Med-PRM, PRA). `mean` is offered for the ablation, not as the headline.
    """
    if not step_probs:
        return 0.0
    if rule == "mean":
        return float(statistics.fmean(step_probs))
    if rule == "min":
        return float(min(step_probs))
    raise ValueError(f"unknown rule {rule!r} (use 'min' or 'mean')")


def score_trace_with_prm(prm, trace, rule: str = "min") -> float:
    """Score one Trace by scoring each of its steps with the PRM and aggregating."""
    examples = [
        {"step_text": s.text, "evidence_ids": [c.citation_id for c in s.citations]}
        for s in trace.steps
    ]
    if not examples:
        return 0.0
    return trace_score(prm.predict_proba(examples), rule)


def best_of_n(prm, candidates: list, rule: str = "min"):
    """Pick the highest-scoring trace among N candidates. Returns (best_trace, scores).

    Report honestly whether this beats greedy/first-sample. A negative result cleanly
    reported is stronger than a fabricated positive (risk register).
    """
    if not candidates:
        raise ValueError("best_of_n needs at least one candidate trace")
    scores = [score_trace_with_prm(prm, t, rule) for t in candidates]
    best_i = max(range(len(candidates)), key=lambda i: scores[i])
    return candidates[best_i], scores


# ---- 5.3 post-training data construction --------------------------------------------
def select_rft_traces(prm, traces: list, threshold: float = 0.7, rule: str = "min") -> list:
    """RFT selection: keep only traces the verifier scores above `threshold` for LoRA-SFT."""
    return [t for t in traces if score_trace_with_prm(prm, t, rule) >= threshold]


def build_dpo_pairs(prm, traces_by_case: dict, rule: str = "min",
                    margin: float = 0.05) -> list[dict]:
    """Build (chosen, rejected) preference pairs per case from PRM scores.

    Only pairs separated by more than `margin` are kept; near-ties carry no preference
    signal and mostly inject noise into DPO.
    """
    pairs = []
    for case_id, traces in traces_by_case.items():
        if len(traces) < 2:
            continue
        scored = sorted(((score_trace_with_prm(prm, t, rule), t) for t in traces),
                        key=lambda x: x[0], reverse=True)
        hi, best = scored[0]
        lo, worst = scored[-1]
        if hi - lo > margin:
            pairs.append({"case_id": case_id, "chosen": best, "rejected": worst,
                          "chosen_score": hi, "rejected_score": lo})
    return pairs


def reward_hacking_report(pairs: list[dict]) -> dict:
    """Check the PRM is not simply preferring *longer* or *more-cited* traces.

    The risk register's DPO failure mode: preference pairs that merely encode verbosity or
    citation-stuffing produce a policy that games the verifier. If the chosen traces are
    systematically longer or carry more citations, treat the DPO result as suspect.
    """
    def _len(t):
        return sum(len(s.text) for s in t.steps)

    def _cites(t):
        return sum(len(s.citations) for s in t.steps)

    if not pairs:
        return {"n_pairs": 0, "chosen_longer_frac": None, "chosen_more_cited_frac": None,
                "suspicious": False, "note": "no pairs"}

    longer = sum(1 for p in pairs if _len(p["chosen"]) > _len(p["rejected"])) / len(pairs)
    cited = sum(1 for p in pairs if _cites(p["chosen"]) > _cites(p["rejected"])) / len(pairs)
    suspicious = longer > 0.8 or cited > 0.8
    return {
        "n_pairs": len(pairs),
        "chosen_longer_frac": round(longer, 3),
        "chosen_more_cited_frac": round(cited, 3),
        "suspicious": suspicious,
        "note": ("chosen traces are mostly just longer/more-cited; verifier may be gameable"
                 if suspicious else "no strong length/citation bias detected"),
    }


# ---- GPU-gated training calls --------------------------------------------------------
def train_rft(policy_cfg: PolicyConfig, selected_traces: list, out_dir: str) -> str:  # pragma: no cover
    """LoRA-SFT on PRM-selected traces. Needs trl + peft + GPU (Kaggle)."""
    raise NotImplementedError(
        "RFT LoRA-SFT runs on Kaggle (trl SFTTrainer + peft) - Phase 5.3. "
        "Trace selection itself is implemented CPU-side in `select_rft_traces`."
    )


def train_dpo(policy_cfg: PolicyConfig, pairs: list[dict], out_dir: str) -> str:  # pragma: no cover
    """LoRA-DPO on PRM-derived preference pairs. Needs trl + peft + GPU (Kaggle)."""
    raise NotImplementedError(
        "DPO LoRA training runs on Kaggle (trl DPOTrainer + peft) - Phase 5.3. "
        "Pair construction and the reward-hacking guard are implemented CPU-side."
    )

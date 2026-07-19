"""Post-training: PRM + verifier-guided inference + RFT/DPO. GPU steps run on Kaggle."""
from .posttrain import (
    PolicyConfig,
    best_of_n,
    build_dpo_pairs,
    reward_hacking_report,
    score_trace_with_prm,
    select_rft_traces,
    trace_score,
    train_dpo,
    train_rft,
)
from .prm import PRM, PRMConfig, PRMReport, split_by_case, train_prm

__all__ = [
    "PRM",
    "PRMConfig",
    "PRMReport",
    "train_prm",
    "split_by_case",
    "PolicyConfig",
    "trace_score",
    "score_trace_with_prm",
    "best_of_n",
    "select_rft_traces",
    "build_dpo_pairs",
    "reward_hacking_report",
    "train_rft",
    "train_dpo",
]

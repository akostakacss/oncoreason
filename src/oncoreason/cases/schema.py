"""Case schema — the unit the whole pipeline consumes.

A Case is one lung-cancer molecular-interpretation scenario: a real (public) molecular
profile + clinical context + the gold recommendation used for evaluation.

Phase 2. Split at the CASE level (never across train/test — leakage).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    GENERALIZATION = "generalization"  # disjoint real cBioPortal profiles


@dataclass(frozen=True)
class Alteration:
    gene: str
    variant: str                     # normalized HGVS preferred (Gate 3)
    kind: str = "mutation"           # mutation | cna | fusion
    is_somatic: bool = True
    escat_tier: str | None = None    # I-A ... X, if actionable


@dataclass(frozen=True)
class ClinicalContext:
    tumor_type: str = "lung"         # actionability is tumor-type-specific
    stage: str | None = None
    prior_lines: int | None = None
    performance_status: int | None = None  # ECOG 0-4


@dataclass(frozen=True)
class TimelineEvent:
    """One dated clinical event. The unit of longitudinal reasoning.

    Added after MTBBench (Vasilev, ..., Moor, Bunne; NeurIPS 2025 D&B), which shows that real
    MTB decision-making is *sequential*: evidence accumulates over time and provisional
    decisions are revised as it arrives. "Longitudinal care" is an explicit design target.
    """

    timepoint: str                   # ISO date, or a relative index like "t0", "t1"
    kind: str                        # diagnosis | sequencing | treatment | progression | lab | pathology
    detail: str = ""
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ClinicalTimeline:
    """An ordered patient history. `until()` is what makes sequential evaluation honest:
    at decision point t the agent may only see events up to t, never the future."""

    events: tuple[TimelineEvent, ...] = ()

    def ordered(self) -> list[TimelineEvent]:
        return sorted(self.events, key=lambda e: e.timepoint)

    def until(self, timepoint: str) -> list[TimelineEvent]:
        """Events at or before `timepoint`. Prevents leaking future information into a
        decision that was made before it existed."""
        return [e for e in self.ordered() if e.timepoint <= timepoint]

    def decision_points(self) -> list[str]:
        return sorted({e.timepoint for e in self.events})


@dataclass(frozen=True)
class GoldRecommendation:
    """Guideline-derived ground truth for evaluation. Keep the evidence tier."""

    recommended: list[str] = field(default_factory=list)  # therapy/class options
    escat_tier: str | None = None
    guideline_source: str = "ESMO"
    guideline_version: str | None = None  # version-match to avoid temporal drift
    rationale: str | None = None


@dataclass(frozen=True)
class Case:
    case_id: str
    alterations: list[Alteration]
    context: ClinicalContext
    gold: GoldRecommendation | None = None   # None until labelled
    split: Split | None = None
    timeline: ClinicalTimeline | None = None  # longitudinal history (MTBBench-style); None = single timepoint
    provenance: dict = field(default_factory=dict)  # e.g. {"cbioportal_study": ...}

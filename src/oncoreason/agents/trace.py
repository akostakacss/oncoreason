"""Trace schema — the auditable reasoning record (a first-class output).

Every run emits a Trace: ordered reasoning steps, each with the tool calls it made and the
citations it rests on, plus a final recommendation, a confidence, and an explicit abstain
flag. This is the object supervised in Stage 3, scored in Stage 5, and audited by a clinician.

Phases 3 (produced), 4 (labelled/segmented), 5 (scored).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolCall:
    tool: str                 # e.g. "civic.retrieve"
    args: dict
    ok: bool
    latency_ms: float | None = None
    error: str | None = None
    n_results: int | None = None


@dataclass(frozen=True)
class Citation:
    citation_id: str          # must resolve to a real record (verified in Stage 5)
    source: str
    claim: str                # the claim this citation supports


@dataclass
class ReasoningStep:
    """One reasoning step. Segmentation rule is fixed once (Phase 4.1)."""

    index: int
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    # filled by the PRM in Stage 4 (None until labelled/scored):
    prm_score: float | None = None
    label_sound: bool | None = None


@dataclass
class Trace:
    case_id: str
    steps: list[ReasoningStep] = field(default_factory=list)
    recommendation: list[str] = field(default_factory=list)  # ranked options
    confidence: float | None = None
    abstained: bool = False
    model: str | None = None          # policy id + version (reproducibility)
    metadata: dict = field(default_factory=dict)

    def all_citations(self) -> list[Citation]:
        return [c for s in self.steps for c in s.citations]

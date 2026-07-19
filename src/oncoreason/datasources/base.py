"""DataSource interface — the license-safe connector contract.

Every evidence source (public or controlled) implements one interface so the pipeline
is agnostic to whether data is public or licensed. Public connectors ship functional;
controlled connectors ship as stubs that raise `ControlledSourceNotConfigured` unless a
licensed user has placed data under `data/controlled/`. See LICENSING.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class EvidenceKind(str, Enum):
    """What a piece of evidence speaks to. Keep somatic/germline distinct downstream."""

    PREDICTIVE = "predictive"      # therapy response (e.g. CIViC)
    PROGNOSTIC = "prognostic"
    DIAGNOSTIC = "diagnostic"
    PATHOGENICITY = "pathogenicity"  # variant pathogenicity (e.g. ClinVar, germline)
    GUIDELINE = "guideline"        # recommendation text / ESCAT tier
    TRIAL = "trial"                # clinical trial match
    LITERATURE = "literature"      # publication


@dataclass(frozen=True)
class EvidenceQuery:
    """A structured request to a source. Extend as needed per connector."""

    gene: str | None = None
    variant: str | None = None          # normalized HGVS preferred (Gate 3)
    tumor_type: str | None = None        # actionability is tumor-type-specific
    free_text: str | None = None
    top_k: int = 10


@dataclass(frozen=True)
class Evidence:
    """One retrieved evidence item. `citation_id` must resolve to a real record."""

    source: str                          # connector name, e.g. "civic"
    kind: EvidenceKind
    citation_id: str                     # stable id used for citation verification
    summary: str                         # short human-readable claim
    payload: dict = field(default_factory=dict)  # raw structured fields
    evidence_level: str | None = None    # e.g. CIViC A-E, ESCAT tier, OncoKB level
    is_somatic: bool | None = None       # True somatic / False germline / None n/a


class ControlledSourceNotConfigured(RuntimeError):
    """Raised when a controlled connector is used without its licensed data present."""


@runtime_checkable
class DataSource(Protocol):
    """The one interface every connector implements."""

    name: str
    is_controlled: bool

    def retrieve(self, query: EvidenceQuery) -> list[Evidence]:
        """Return evidence matching `query`. Public connectors hit an API/local dump;
        controlled connectors read `data/controlled/<name>/` or raise."""
        ...

"""Controlled-data connectors — RESERVED SLOTS, shipped empty.

These represent licensed sources a host lab would have (NCCN, full OncoKB, institutional /
kaiko multimodal data). They implement the same DataSource interface as the public
connectors, but read from a git-ignored path under data/controlled/ and raise
`ControlledSourceNotConfigured` if the licensed data is absent — which it always is in this
public repo. NOTHING controlled is ever committed. See LICENSING.md.

The empty slot is deliberate: it shows the architecture is ready to receive institutional
data without distributing anything restricted.
"""
from __future__ import annotations

import os
from pathlib import Path

from .base import (
    ControlledSourceNotConfigured,
    DataSource,
    Evidence,
    EvidenceQuery,
)

CONTROLLED_ROOT = Path(
    os.environ.get("ONCOREASON_CONTROLLED_DIR", "data/controlled")
)


class _ControlledSource:
    """Base for licensed connectors: guards on the presence of local licensed data."""

    name = "controlled"
    is_controlled = True

    def _require_data(self) -> Path:
        path = CONTROLLED_ROOT / self.name
        if not path.exists() or not any(path.iterdir()):
            raise ControlledSourceNotConfigured(
                f"Controlled source '{self.name}' is not configured. "
                f"A licensed user places its data in {path}/ — it is never committed. "
                f"See LICENSING.md."
            )
        return path

    def retrieve(self, query: EvidenceQuery) -> list[Evidence]:
        self._require_data()
        raise NotImplementedError(
            f"Controlled connector '{self.name}' retrieval is implemented only by a "
            f"licensed user against their local data."
        )


class NCCNSource(_ControlledSource):
    """NCCN guidelines — licensed. Use ESMO/ESCAT (public, derived index) in the PoC."""

    name = "nccn"
    is_controlled = True


class OncoKBFullSource(_ControlledSource):
    """Full OncoKB annotations — academic token required. CIViC is the public backbone."""

    name = "oncokb_full"
    is_controlled = True


class InstitutionalSource(_ControlledSource):
    """Institutional / kaiko multimodal patient-level data — the ETH/kaiko slot."""

    name = "institutional"
    is_controlled = True


# protocol conformance
_a: DataSource = NCCNSource()          # type: ignore[abstract]
_b: DataSource = OncoKBFullSource()    # type: ignore[abstract]
_c: DataSource = InstitutionalSource() # type: ignore[abstract]

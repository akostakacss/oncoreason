"""Connector registry — wires source names to implementations.

Public sources are enabled by default. Controlled sources are registered but remain inert
until a licensed user supplies data (they raise ControlledSourceNotConfigured otherwise).
"""
from __future__ import annotations

from .base import DataSource
from .cbioportal import CBioPortalSource
from .civic import CIViCSource
from .clinvar import ClinVarSource
from .controlled import InstitutionalSource, NCCNSource, OncoKBFullSource

#: name -> factory. Adding a source is one line here.
_REGISTRY: dict[str, type] = {
    # public
    "civic": CIViCSource,
    "clinvar": ClinVarSource,
    "cbioportal": CBioPortalSource,
    # controlled (reserved slots — inert without licensed data)
    "nccn": NCCNSource,
    "oncokb_full": OncoKBFullSource,
    "institutional": InstitutionalSource,
}

PUBLIC_SOURCES = ("civic", "clinvar", "cbioportal")
CONTROLLED_SOURCES = ("nccn", "oncokb_full", "institutional")


def get_source(name: str, **kwargs) -> DataSource:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown source '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_sources(include_controlled: bool = True) -> list[str]:
    names = list(PUBLIC_SOURCES)
    if include_controlled:
        names += list(CONTROLLED_SOURCES)
    return names

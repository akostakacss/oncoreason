"""oncoreason — oncology reasoning model (proof of concept).

A post-training and evaluation pipeline for oncology reasoning: data connectors, case
construction, agent scaffolding, retrieval, process supervision, a process reward model,
and a clinical evaluation harness. Import submodules as needed:

    from oncoreason.datasources import get_source, available_sources
    from oncoreason.cases import Case
    from oncoreason.agents import Trace, Orchestrator
"""
from __future__ import annotations

from pathlib import Path

__version__ = "0.0.1"

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config (defaults to configs/default.yaml)."""
    import yaml

    if path is None:
        path = REPO_ROOT / "configs" / "default.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)


__all__ = ["__version__", "load_config", "PACKAGE_ROOT", "REPO_ROOT"]

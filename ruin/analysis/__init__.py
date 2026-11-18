"""Analysis of the structure that forms inside D."""

from __future__ import annotations

from typing import Any

from .organization import StepFields, StructureRecorder, label_components

__all__ = [
    "StepFields",
    "StructureRecorder",
    "aggregate",
    "label_components",
    "plot_organization",
    "run_organization",
]

_REPORT = {"aggregate", "plot_organization", "run_organization"}


def __getattr__(name: str) -> Any:
    # `report` runs shifts, so it imports the engine, and the engine imports this package
    # for the recorder. Loading `report` only when it is asked for keeps the import graph
    # acyclic whichever module a user imports first.
    if name in _REPORT:
        from . import report

        return getattr(report, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

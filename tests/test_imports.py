"""Every module must import on its own, in a fresh interpreter, whatever else is or isn't loaded.

Circular imports hide behind import order: a test suite that always imports the CLI first
never sees them, while a user who imports the engine first hits them at once.
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

import ruin


def _modules() -> list[str]:
    names = ["ruin"]
    names += [m.name for m in pkgutil.walk_packages(ruin.__path__, prefix="ruin.")]
    return sorted(names)


def _import_alone(name: str) -> tuple[str, int, str]:
    done = subprocess.run(
        [sys.executable, "-I", "-c", f"import {name}"], capture_output=True, text=True, timeout=120
    )
    return name, done.returncode, done.stderr.strip().splitlines()[-1] if done.stderr else ""


def test_every_module_imports_on_its_own():
    names = _modules()
    assert len(names) > 20  # the walk found the package
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_import_alone, names))
    failures = [(n, err) for n, code, err in results if code != 0]
    assert not failures, failures


@pytest.mark.parametrize(
    "name",
    ["aggregate", "plot_organization", "run_organization", "StructureRecorder", "label_components"],
)
def test_analysis_package_exposes_its_public_names(name):
    import ruin.analysis as analysis

    assert callable(getattr(analysis, name))


def test_analysis_package_rejects_unknown_names():
    import ruin.analysis as analysis

    with pytest.raises(AttributeError, match="no attribute"):
        analysis.does_not_exist  # noqa: B018

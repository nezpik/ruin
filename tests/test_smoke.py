"""Smoke tests for RUIN v0.2 — ensure the package still imports and basic flow works."""

from __future__ import annotations

import pytest

from ruin import cli
from ruin.config import load_config
from ruin.core.qdot import QDot, QDotType
from ruin.simulation.agent_based import run_trajectory


def test_package_imports():
    """Core modules must be importable."""
    assert cli is not None
    from ruin.state_space.probability_square import ProbabilitySquare  # noqa: F401
    from ruin.risk.ruin_probability import run_monte_carlo  # noqa: F401


def test_load_golden_config(golden_config_path):
    """The existing example YAML must still parse."""
    cfg = load_config(golden_config_path)
    assert "simulation" in cfg
    assert cfg["simulation"]["grid_width"] == 30
    assert cfg["qdots"]["n_standard"] == 120


def test_small_config_runs(small_config):
    """A minimal config should complete a short trajectory without crashing."""
    result = run_trajectory(small_config, seed=42, max_steps=20)
    assert isinstance(result, dict)
    assert "ruined" in result
    assert "final_d_state" in result
    assert "cumulative_loss" in result
    assert result["ruin_time"] is None or isinstance(result["ruin_time"], int)


def test_qdot_step_basic():
    """QDot.step must run and correctly set ruined state when time expires."""
    q = QDot(
        id=0,
        type=QDotType.STANDARD,
        x=3,
        y=4,
        time_window=5.0,
        delay_penalty=1.0,
        drift_multiplier=1.0,
        volatility_multiplier=1.0,
    )
    rng = __import__("random").Random(0)

    # Force many steps so it goes late
    for _ in range(20):
        penalty = q.step(width=10, height=10, base_drift=0.8, volatility=0.2, rng=rng)
        if q.is_ruined:
            break

    assert q.is_ruined is True
    assert q.late is True
    assert penalty > 0

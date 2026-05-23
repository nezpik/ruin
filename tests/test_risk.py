"""Tests for Monte Carlo risk analysis (v0.2)."""

from __future__ import annotations

import pytest

from ruin.config import load_config
from ruin.risk.ruin_probability import run_monte_carlo, bootstrap_ci


def test_bootstrap_ci_basic():
    values = [0.0, 1.0, 0.0, 1.0, 0.0]
    lower, upper = bootstrap_ci(values, confidence=0.95, n_boot=100)
    assert 0.0 <= lower <= upper <= 1.0


def test_run_monte_carlo_small(golden_config):
    # Use a small number of paths for speed in tests
    result = run_monte_carlo(golden_config, paths=8, max_steps=30)
    assert result["paths"] == 8
    assert 0.0 <= result["ruin_probability"] <= 1.0
    assert "ruin_probability_ci" in result
    assert len(result["ruin_probability_ci"]) == 2
    assert result["mean_loss"] is not None
    assert result["value_at_risk"] is not None


def test_run_monte_carlo_golden_small(golden_config):
    result = run_monte_carlo(golden_config, paths=20, max_steps=50)
    assert "ruin_probability_ci" in result
    assert result["paths"] == 20
    # With golden config we expect high ruin probability in short horizon
    assert result["ruin_probability"] > 0.3

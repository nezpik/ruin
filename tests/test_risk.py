"""Tests for Monte Carlo risk analysis (v0.2) with parallel execution."""

from __future__ import annotations

import pytest

from ruin.config import load_config
from ruin.risk.ruin_probability import run_monte_carlo, bootstrap_ci


def test_bootstrap_ci_basic():
    values = [0.0, 1.0, 0.0, 1.0, 0.0]
    lower, upper = bootstrap_ci(values, confidence=0.95, n_boot=100)
    assert 0.0 <= lower <= upper <= 1.0


def test_run_monte_carlo_small_parallel(golden_config):
    # Test parallel execution with small paths (should still work correctly)
    result = run_monte_carlo(golden_config, paths=12, max_steps=30, n_jobs=2)
    assert result["paths"] == 12
    assert 0.0 <= result["ruin_probability"] <= 1.0
    assert "ruin_probability_ci" in result
    assert "n_jobs" in result
    assert result["n_jobs"] >= 1
    assert result["mean_loss"] is not None


def test_run_monte_carlo_golden_parallel(golden_config):
    result = run_monte_carlo(golden_config, paths=30, max_steps=50, n_jobs=4)
    assert "ruin_probability_ci" in result
    assert result["paths"] == 30
    assert result["ruin_probability"] > 0.3
    assert result.get("n_jobs") is not None

"""Tests for Monte Carlo risk analysis (v0.2) with parallel execution."""

from __future__ import annotations

import pytest

from ruin.risk.ruin_probability import bootstrap_ci, run_monte_carlo


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


def test_monte_carlo_is_reproducible_and_independent_of_workers(golden_ruin_config):
    serial = run_monte_carlo(golden_ruin_config, paths=8, max_steps=80, n_jobs=1)
    parallel = run_monte_carlo(golden_ruin_config, paths=8, max_steps=80, n_jobs=3)
    serial.pop("n_jobs")
    parallel.pop("n_jobs")
    assert serial == parallel


@pytest.mark.slow
def test_golden_scenario_ruin_is_uncertain(golden_ruin_config):
    """The reference scenario must neither always ruin nor never ruin."""
    result = run_monte_carlo(golden_ruin_config, paths=120)
    assert 0.02 < result["ruin_probability"] < 0.40
    low, high = result["ruin_probability_ci"]
    assert low < result["ruin_probability"] < high
    assert result["expected_shortfall"] >= result["value_at_risk"] >= result["mean_loss"]
    assert result["mean_on_time_rate"] > 0.9


# v0.2 typed config tests — exercise direct RuinConfig path
def test_run_monte_carlo_small_parallel_typed(golden_ruin_config):
    """Same small parallel test but passing RuinConfig model directly."""
    result = run_monte_carlo(golden_ruin_config, paths=12, max_steps=30, n_jobs=2)
    assert result["paths"] == 12
    assert 0.0 <= result["ruin_probability"] <= 1.0
    assert "ruin_probability_ci" in result
    assert "n_jobs" in result

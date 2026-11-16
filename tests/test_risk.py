"""Tests for Monte Carlo risk analysis (v0.2) with parallel execution."""

from __future__ import annotations

import pytest

from ruin.risk.ruin_probability import (
    bootstrap_ci,
    bootstrap_tail_ci,
    expected_shortfall,
    run_monte_carlo,
    value_at_risk,
    wilson_interval,
)


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


def test_wilson_interval_known_values():
    low, high = wilson_interval(50, 100, 0.95)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)
    assert low + high == pytest.approx(1.0)


def test_wilson_interval_is_informative_at_the_extremes():
    """Zero ruins in 200 paths does not mean ruin is impossible."""
    low, high = wilson_interval(0, 200, 0.95)
    assert low == 0.0
    assert 0.015 < high < 0.025
    low, high = wilson_interval(200, 200, 0.95)
    assert high == 1.0
    assert 0.975 < low < 0.985
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_narrows_with_more_paths():
    widths = [high - low for low, high in (wilson_interval(n // 10, n) for n in (50, 500, 5000))]
    assert widths == sorted(widths, reverse=True)


def test_value_at_risk_is_the_empirical_quantile():
    losses = [float(i) for i in range(1, 101)]  # 1..100
    assert value_at_risk(losses, 0.95) == 95.0
    assert value_at_risk(losses, 0.99) == 99.0
    assert value_at_risk([float(i) for i in range(1, 31)], 0.95) == 29.0  # ceil(28.5)
    assert expected_shortfall(losses, 0.95) == pytest.approx(97.5)
    assert value_at_risk([], 0.95) == 0.0


def test_bootstrap_tail_ci_brackets_the_estimates():
    import numpy as np

    losses = np.random.default_rng(0).exponential(100.0, size=800).tolist()
    var_ci, es_ci = bootstrap_tail_ci(losses, 0.95, 0.95, n_boot=500)
    assert var_ci[0] <= value_at_risk(losses, 0.95) <= var_ci[1]
    assert es_ci[0] <= expected_shortfall(losses, 0.95) <= es_ci[1]
    assert es_ci[0] >= var_ci[0]
    # True values for Exp(100): VaR = 100*ln(20) = 299.6, ES = VaR + 100.
    assert var_ci[0] < 299.6 < var_ci[1]
    assert es_ci[0] < 399.6 < es_ci[1]


def test_monte_carlo_reports_intervals_and_uncensored_losses(golden_ruin_config):
    cfg = golden_ruin_config.model_copy(deep=True)
    cfg.ruin.initial_buffer = 5.0
    cfg.disruption_field.shock_arrival_rate *= 3
    result = run_monte_carlo(cfg, paths=10, max_steps=150, n_jobs=1, ci_boot=200)
    assert result["ruin_probability"] > 0
    for key in ("ruin_probability", "mean_loss", "value_at_risk", "expected_shortfall"):
        low, high = result[f"{key}_ci"]
        assert low <= result[key] <= high
    # Paths run on after ruin, so losses are not capped near the buffer.
    assert result["mean_loss"] > cfg.ruin.initial_buffer
    assert result["mean_min_surplus"] < 0
    assert sum(result["loss_share_by_type"].values()) == pytest.approx(1.0, abs=1e-3)

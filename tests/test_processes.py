"""Tests for stochastic processes (v0.2 vectorized + legacy)."""

from __future__ import annotations

import numpy as np
import pytest

from ruin.core.processes import (
    make_rng,
    sample_gbm_displacements,
    sample_compound_poisson_jumps,
    sample_shock_arrivals,
    sample_travel_step,
    PoissonArrivalProcess,
    JumpDiffusionProcess,
)


def test_make_rng_reproducible():
    rng1 = make_rng(123)
    rng2 = make_rng(123)
    a = rng1.standard_normal(5)
    b = rng2.standard_normal(5)
    np.testing.assert_array_almost_equal(a, b)


def test_gbm_displacements_shape_and_stats():
    rng = make_rng(42)
    disp = sample_gbm_displacements(5000, drift=1.2, volatility=0.4, dt=1.0, rng=rng)
    assert disp.shape == (5000,)
    assert np.all(np.isfinite(disp))
    # Mean should be close to drift * dt
    assert abs(disp.mean() - 1.2) < 0.05


def test_compound_poisson_jumps():
    rng = make_rng(99)
    jumps = sample_compound_poisson_jumps(
        2000, intensity=0.08, mean_jump_size=3.0, dt=1.0, rng=rng
    )
    assert jumps.shape == (2000,)
    non_zero = jumps[jumps != 0]
    if len(non_zero) > 0:
        assert np.all(non_zero > 0)
        assert non_zero.mean() > 1.0  # rough sanity


def test_shock_arrivals_mask():
    rng = make_rng(7)
    mask = sample_shock_arrivals(300, rate=0.05, dt=1.0, rng=rng)
    assert mask.dtype == bool
    assert mask.shape == (300,)
    rate_est = mask.mean()
    assert 0.02 < rate_est < 0.08  # probabilistic band


def test_scalar_travel_step():
    py_rng = __import__("random").Random(42)
    val = sample_travel_step(1.2, 0.4, 0.08, 3.0, dt=1.0, rng=py_rng)
    assert isinstance(val, float)
    assert np.isfinite(val)


def test_legacy_classes_still_work():
    py_rng = __import__("random").Random(123)
    p = PoissonArrivalProcess(0.1, py_rng)
    assert isinstance(p.event_occurs(dt=1.0), bool)

    j = JumpDiffusionProcess(1.0, 0.5, 0.1, 2.0, __import__("random").Random(456))
    jump = j.sample_jump()
    assert isinstance(jump, float)


def test_processes_with_small_config_fixture(small_ruin_config):
    """Smoke that the new primitives can be called with realistic parameters from a config."""
    cfg = small_ruin_config
    p = cfg.processes

    rng = make_rng(42)
    disp = sample_gbm_displacements(10, p.travel_drift, p.travel_volatility, dt=1.0, rng=rng)
    jumps = sample_compound_poisson_jumps(
        10, p.jump_intensity, p.jump_mean_size, dt=1.0, rng=rng
    )

    assert len(disp) == 10
    assert len(jumps) == 10
    assert np.all(np.isfinite(disp + jumps))

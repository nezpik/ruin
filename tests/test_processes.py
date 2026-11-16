"""Tests for the vectorized stochastic processes."""

from __future__ import annotations

import numpy as np
import pytest

from ruin.core.processes import (
    EXPOSURE_SATURATION,
    MIN_DRAG,
    batch_qdot_travel_step,
    field_drag,
    make_rng,
    sample_compound_poisson_jumps,
    sample_gbm_displacements,
    sample_shock_events,
)


def test_make_rng_reproducible():
    a = make_rng(123).standard_normal(5)
    b = make_rng(123).standard_normal(5)
    np.testing.assert_array_equal(a, b)


def test_gbm_displacements_shape_and_stats():
    disp = sample_gbm_displacements(5000, drift=1.2, volatility=0.4, dt=1.0, rng=make_rng(42))
    assert disp.shape == (5000,)
    assert abs(disp.mean() - 1.2) < 0.05
    assert abs(disp.std() - 0.4) < 0.05


def test_compound_poisson_jumps_mean():
    jumps = sample_compound_poisson_jumps(
        50_000, intensity=0.08, mean_jump_size=3.0, dt=1.0, rng=make_rng(99)
    )
    assert jumps.shape == (50_000,)
    assert np.all(jumps >= 0)
    # E[compound Poisson] = intensity * mean size
    assert abs(jumps.mean() - 0.24) < 0.02


def test_compound_poisson_rejects_unknown_distribution():
    with pytest.raises(ValueError):
        sample_compound_poisson_jumps(5, 0.1, 1.0, distribution="cauchy", rng=make_rng(0))


def test_shock_events_rate_and_bounds():
    rng = make_rng(7)
    counts = []
    for _ in range(2000):
        xs, ys, intensities = sample_shock_events(0.5, 4.0, width=6, height=9, rng=rng)
        counts.append(len(xs))
        assert np.all((xs >= 0) & (xs < 6))
        assert np.all((ys >= 0) & (ys < 9))
        assert np.all(intensities >= 0)
    assert abs(np.mean(counts) - 0.5) < 0.06


def test_field_drag_bounds():
    drag = field_drag(np.array([0.0, EXPOSURE_SATURATION / 2, EXPOSURE_SATURATION, 1e6]))
    assert drag[0] == pytest.approx(1.0)
    assert drag[2] == pytest.approx(MIN_DRAG)
    assert drag[3] == pytest.approx(MIN_DRAG)
    assert np.all(np.diff(drag) <= 0)


def _travel(n=400, exposure=0.0, seed=1, steps=1):
    rng = make_rng(seed)
    x = rng.integers(0, 20, n)
    y = rng.integers(0, 20, n)
    dest_x = rng.integers(0, 20, n)
    dest_y = rng.integers(0, 20, n)
    carry = np.zeros(n)
    start = (x.copy(), y.copy())
    total_moved = np.zeros(n, dtype=bool)
    arrived = np.zeros(n, dtype=bool)
    for _ in range(steps):
        x, y, carry, moved, arrived = batch_qdot_travel_step(
            x, y, dest_x, dest_y, carry,
            np.full(n, exposure), np.ones(n), np.ones(n),
            1.2, 0.4, 0.05, 2.0, rng,
        )
        total_moved |= moved
    return start, (x, y), (dest_x, dest_y), arrived


def test_travel_never_overshoots_and_closes_distance():
    (x0, y0), (x1, y1), (dx, dy), _ = _travel()
    before = np.abs(dx - x0) + np.abs(dy - y0)
    after = np.abs(dx - x1) + np.abs(dy - y1)
    assert np.all(after <= before)
    assert after.sum() < before.sum()
    # Each coordinate stays between where it started and where it is going.
    assert np.all((x1 >= np.minimum(x0, dx)) & (x1 <= np.maximum(x0, dx)))
    assert np.all((y1 >= np.minimum(y0, dy)) & (y1 <= np.maximum(y0, dy)))


def test_travel_uses_both_axes():
    (x0, y0), (x1, y1), _, _ = _travel(steps=3)
    assert np.any(x1 != x0)
    assert np.any(y1 != y0)


def test_travel_eventually_arrives():
    _, _, _, arrived = _travel(steps=200)
    assert arrived.all()


def test_blocked_cells_slow_travel():
    (x0, y0), (x1, y1), _, _ = _travel(steps=5, exposure=0.0)
    free = (np.abs(x1 - x0) + np.abs(y1 - y0)).sum()
    (x0, y0), (x1, y1), _, _ = _travel(steps=5, exposure=EXPOSURE_SATURATION)
    blocked = (np.abs(x1 - x0) + np.abs(y1 - y0)).sum()
    assert blocked < 0.5 * free


def test_travel_handles_empty_population():
    empty = np.zeros(0, dtype=int)
    out = batch_qdot_travel_step(
        empty, empty, empty, empty, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),
        1.0, 0.3, 0.05, 2.0, make_rng(0),
    )
    assert all(len(a) == 0 for a in out)

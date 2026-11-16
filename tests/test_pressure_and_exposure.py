"""Tests for pressures, D-state selection, the disruption field, and Q-dot exposure."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from ruin.config_models import RuinConfig
from ruin.core.processes import EXPOSURE_SATURATION, make_rng, sample_shock_events
from ruin.metrics.pressure import (
    DState,
    choose_d_state,
    global_order_pressure,
    local_order_pressure,
)
from ruin.state_space.disruption_field import DisruptionField
from ruin.state_space.probability_square import ProbabilitySquare


def _field(**kw):
    params = {
        "width": 21,
        "height": 21,
        "memory_weight": 0.95,
        "propagation_radius": 2,
        "propagation_decay": 0.55,
        "temporal_decay": 0.03,
        "recovery_rate": 0.02,
        "propagation_share": 0.4,
        **kw,
    }
    return DisruptionField(**params)


def test_field_shock_fades_and_spreads():
    field = _field()
    field.inject(10, 10, 100.0)
    totals = []
    for _ in range(6):
        field.step()
        totals.append(field.total())
    # Away from the boundary, total intensity shrinks by exactly `gain` each step.
    assert totals[0] == pytest.approx(100.0 * field.gain)
    assert all(b == pytest.approx(a * field.gain) for a, b in pairwise(totals))
    assert field.grid[10, 11] > 0 and field.grid[12, 10] > 0  # spread to neighbours
    assert field.grid[10, 10] == field.grid.max()


@pytest.mark.parametrize("mode", ["reflect", "wrap"])
def test_field_keeps_disruption_inside_d(mode):
    """With reflect or wrap, nothing leaks out: total shrinks by exactly `gain`."""
    field = _field(width=5, height=4, boundary_mode=mode)
    field.inject(0, 0, 50.0)
    field.inject(4, 3, 20.0)
    for _ in range(20):
        before = field.total()
        field.step()
        assert field.total() == pytest.approx(before * field.gain)
        assert field.grid.min() >= 0


def test_field_absorbing_boundary_loses_intensity():
    field = _field(width=5, height=4, boundary_mode="absorb")
    field.inject(0, 0, 50.0)
    before = field.total()
    field.step()
    assert field.total() < before * field.gain


def test_field_wrap_reaches_the_opposite_edge():
    wrapped = _field(width=9, height=9, boundary_mode="wrap")
    reflected = _field(width=9, height=9, boundary_mode="reflect")
    for field in (wrapped, reflected):
        field.inject(0, 4, 10.0)
        field.step()
    assert wrapped.grid[4, 8] > 0
    assert reflected.grid[4, 8] == 0
    assert reflected.grid[4, 1] > wrapped.grid[4, 1]  # folded back instead


def test_field_rejects_unknown_boundary_and_oversized_radius():
    with pytest.raises(ValueError):
        _field(boundary_mode="teleport")
    with pytest.raises(ValueError):
        _field(width=3, height=3, propagation_radius=4)


def test_field_long_run_level_matches_theory():
    """Shock noise in, geometric decay out: mean total = rate * mean * gain / (1 - gain)."""
    field = _field(width=12, height=12)
    rng = make_rng(11)
    rate, mean_intensity = 0.5, 40.0
    totals = []
    for step in range(30_000):
        xs, ys, intensities = sample_shock_events(rate, mean_intensity, 12, 12, rng=rng)
        field.inject_many(xs, ys, intensities)
        field.step()
        if step >= 500:
            totals.append(field.total())
    expected = rate * mean_intensity * field.gain / (1 - field.gain)
    assert np.mean(totals) == pytest.approx(expected, rel=0.05)


def test_field_rejects_non_decaying_parameters():
    with pytest.raises(ValueError):
        _field(memory_weight=1.0, temporal_decay=0.0, recovery_rate=0.0)


def test_config_rejects_non_decaying_field(small_config):
    small_config["disruption_field"].update(
        {"memory_weight": 1.0, "temporal_decay": 0.0, "recovery_rate": 0.0}
    )
    with pytest.raises(Exception, match="never decay"):
        RuinConfig.model_validate(small_config)


def test_config_rejects_unordered_thresholds(small_config):
    small_config["probability_square"]["d_state_thresholds"]["stressed_chaos_pressure"] = 0.9
    with pytest.raises(Exception, match="stressed <= disrupted <= chaotic"):
        RuinConfig.model_validate(small_config)


def test_field_stays_bounded_over_a_long_run(golden_ruin_config):
    square = ProbabilitySquare(golden_ruin_config, seed=3, store_qdot_snapshots=False)
    cfg = golden_ruin_config.disruption_field
    # Stationary mean of total intensity is injection / (1 - gain); allow a wide margin.
    ceiling = 20 * cfg.shock_arrival_rate * cfg.mean_initial_intensity / (1 - cfg.gain)
    for _ in range(400):
        square.step()
        assert square.field.total() < ceiling


def test_qdot_exposure_multiplier_is_applied(small_ruin_config):
    def exposures(multiplier):
        cfg = small_ruin_config.model_copy(deep=True)
        cfg.disruption_field.qdot_exposure_multiplier = multiplier
        cfg.disruption_field.shock_arrival_rate = 0.0
        square = ProbabilitySquare(cfg, seed=5)
        square.field.grid[:] = 2.0
        square.step()
        return np.array([q.field_exposure for q in square.qdots])

    base = exposures(1.0)
    tripled = exposures(3.0)
    assert base.max() > 0
    np.testing.assert_allclose(tripled, 3.0 * base)


def test_qdot_order_pressure_reflects_local_population(small_ruin_config):
    square = ProbabilitySquare(small_ruin_config, seed=42)
    for _ in range(5):
        square.step()
    assert square.qdots
    assert all(q.order_pressure >= 1 for q in square.qdots)


def test_pressure_helpers():
    assert local_order_pressure(5, 2, 1) == 6
    assert local_order_pressure(1, 0, 10) == 0
    assert global_order_pressure(10, 10, 0) == pytest.approx(1.0)
    assert global_order_pressure(10, 0, 10) == pytest.approx(-1.0)
    assert global_order_pressure(0, 0, 0) == 0.0


def test_choose_d_state():
    args = {
        "stressed_threshold": 0.3,
        "disrupted_threshold": 0.5,
        "chaotic_threshold": 0.7,
        "recovering_decay_ratio": 0.2,
    }
    assert choose_d_state(True, 0.0, 0.0, **args) is DState.RUINED
    assert choose_d_state(False, 0.1, 0.1, **args) is DState.STABLE
    assert choose_d_state(False, 0.35, 0.34, **args) is DState.STRESSED
    assert choose_d_state(False, 0.55, 0.5, **args) is DState.DISRUPTED
    assert choose_d_state(False, 0.8, 0.5, **args) is DState.CHAOTIC
    # A sharp fall from a stressed-or-worse level is recovery...
    assert choose_d_state(False, 0.35, 0.6, **args) is DState.RECOVERING
    assert choose_d_state(False, 0.1, 0.4, **args) is DState.RECOVERING
    # ...but a still-chaotic grid is chaotic however fast it is falling,
    assert choose_d_state(False, 0.75, 2.0, **args) is DState.CHAOTIC
    # and noise at a calm level is not recovery.
    assert choose_d_state(False, 0.01, 0.05, **args) is DState.STABLE


def test_base_risk_intensity_is_felt_everywhere(small_ruin_config):
    cfg = small_ruin_config.model_copy(deep=True)
    cfg.disruption_field.shock_arrival_rate = 0.0
    cfg.probability_square.base_risk_intensity = 1.5
    square = ProbabilitySquare(cfg, seed=5)
    snapshot = square.step()
    assert all(q.field_exposure == pytest.approx(1.5) for q in square.qdots)
    # Every cell is 30% blocked, so chaos pressure can not be lower than that.
    assert snapshot["chaos_pressure"] >= 1.5 / EXPOSURE_SATURATION


def test_congestion_memory_remembers_stalls(small_ruin_config):
    def run(memory, blocked_steps=6, free_steps=4):
        cfg = small_ruin_config.model_copy(deep=True)
        cfg.disruption_field.shock_arrival_rate = 0.0
        cfg.processes.demand_arrival_rate = 0.0
        cfg.probability_square.congestion_memory = memory
        square = ProbabilitySquare(cfg, seed=9)
        peak = 0.0
        for step in range(blocked_steps + free_steps):
            # Hold the whole grid blocked, then release it.
            square.field.grid[:] = 1000.0 if step < blocked_steps else 0.0
            square.step()
            assert square.congestion.min() >= 0 and square.congestion.max() <= 1
            if step == blocked_steps - 1:
                peak = float(square.congestion.sum())
        return peak, float(square.congestion.sum())

    peak, after = run(memory=0.9)
    assert peak > 0
    assert 0 < after < peak  # fading, but still remembered
    _, after_no_memory = run(memory=0.0)
    assert after_no_memory < after


def test_snapshot_interval_thins_heavy_snapshots(small_ruin_config):
    cfg = small_ruin_config.model_copy(deep=True)
    cfg.probability_square.snapshot_interval = 4
    square = ProbabilitySquare(cfg, seed=1, store_field_snapshots=True)
    for _ in range(12):
        square.step()
    assert len(square.snapshots) == 12  # light paths are kept every step
    with_qdots = [s["time"] for s in square.snapshots if "qdots" in s]
    assert with_qdots == [4, 8, 12]
    assert [s["time"] for s in square.snapshots if "field" in s] == [4, 8, 12]
    assert "chaos_field" in square.snapshots[3] and "order_field" in square.snapshots[3]


def test_config_rejects_unsupported_time_step(small_config):
    small_config["simulation"]["dt"] = 0.5
    with pytest.raises(Exception, match="dt"):
        RuinConfig.model_validate(small_config)

"""Tests for pressures, D-state selection, the disruption field, and Q-dot exposure."""

from __future__ import annotations

import numpy as np
import pytest

from ruin.config_models import RuinConfig
from ruin.metrics.pressure import (
    DState,
    choose_d_state,
    global_order_pressure,
    local_order_pressure,
)
from ruin.state_space.disruption_field import DisruptionField
from ruin.state_space.probability_square import ProbabilitySquare


def _field(**kw):
    params = dict(
        width=21,
        height=21,
        memory_weight=0.95,
        propagation_radius=2,
        propagation_decay=0.55,
        temporal_decay=0.03,
        recovery_rate=0.02,
        propagation_share=0.4,
    )
    params.update(kw)
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
    assert all(b == pytest.approx(a * field.gain) for a, b in zip(totals, totals[1:]))
    assert field.grid[10, 11] > 0 and field.grid[12, 10] > 0  # spread to neighbours
    assert field.grid[10, 10] == field.grid.max()


def test_field_never_grows_at_boundary():
    field = _field(width=4, height=4)
    field.inject(0, 0, 50.0)
    previous = field.total()
    for _ in range(20):
        field.step()
        assert field.total() <= previous * field.gain + 1e-9
        previous = field.total()


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
    args = dict(
        stressed_threshold=0.3,
        disrupted_threshold=0.5,
        chaotic_threshold=0.7,
        recovering_decay_ratio=0.2,
    )
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

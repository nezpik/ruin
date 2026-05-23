"""Smoke tests for RUIN v0.2 — ensure the package still imports and basic flow works."""

from __future__ import annotations

import pytest

from ruin import cli
from ruin.config import load_config, load_ruin_config
from ruin.config_models import RuinConfig
from ruin.core.qdot import QDot, QDotType
from ruin.simulation.agent_based import run_trajectory


def test_package_imports():
    """Core modules must be importable."""
    assert cli is not None
    from ruin.state_space.probability_square import ProbabilitySquare  # noqa: F401
    from ruin.risk.ruin_probability import run_monte_carlo  # noqa: F401
    from ruin.config_models import RuinConfig  # noqa: F401


def test_load_golden_config(golden_config_path):
    """The existing example YAML must still parse (legacy path)."""
    cfg = load_config(golden_config_path)
    assert "simulation" in cfg
    assert cfg["simulation"]["grid_width"] == 30
    assert cfg["qdots"]["n_standard"] == 120


def test_load_golden_ruin_config(golden_ruin_config):
    """Pydantic v0.2 typed loader must validate the golden config."""
    cfg: RuinConfig = golden_ruin_config
    assert isinstance(cfg, RuinConfig)
    assert cfg.simulation.grid_width == 30
    assert cfg.qdots.n_standard == 120
    assert cfg.disruption_field.shock_arrival_rate == 0.05
    assert cfg.ruin.sla_threshold == 0.12


def test_small_config_runs(small_config):
    """A minimal config should complete a short trajectory without crashing."""
    result = run_trajectory(small_config, seed=42, max_steps=20)
    assert isinstance(result, dict)
    assert "ruined" in result
    assert "final_d_state" in result
    assert "cumulative_loss" in result
    assert result["ruin_time"] is None or isinstance(result["ruin_time"], int)


def test_small_ruin_config_runs(small_ruin_config):
    """Typed small config must also drive a trajectory (pass RuinConfig directly)."""
    # v0.2: now pass the model object directly — run_trajectory accepts ConfigLike
    result = run_trajectory(small_ruin_config, seed=123, max_steps=15)
    assert "ruined" in result
    assert "final_d_state" in result


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


def test_config_strict_validation():
    """Pydantic must reject unknown top-level keys (extra='forbid')."""
    bad_cfg = {
        "simulation": {"name": "bad", "seed": 1, "grid_width": 5, "grid_height": 5, "shift_duration": 10},
        "qdots": {"n_standard": 1, "n_express": 0, "n_bulky": 0, "standard": {"time_window": 10, "delay_penalty": 1, "drift_multiplier": 1, "volatility_multiplier": 1}, "express": {"time_window": 10, "delay_penalty": 1, "drift_multiplier": 1, "volatility_multiplier": 1}, "bulky": {"time_window": 10, "delay_penalty": 1, "drift_multiplier": 1, "volatility_multiplier": 1}},
        "processes": {"demand_arrival_rate": 1, "travel_drift": 1, "travel_volatility": 0.1, "jump_intensity": 0.01, "jump_mean_size": 1},
        "probability_square": {"d_state_thresholds": {"stressed_chaos_pressure": 0.3, "disrupted_chaos_pressure": 0.5, "chaotic_chaos_pressure": 0.7, "recovering_decay_ratio": 0.3}},
        "disruption_field": {"shock_arrival_rate": 0.1},
        "ruin": {"initial_buffer": 10},
        "risk": {},
        "visualization": {},
        "evil_extra_key": "should_fail",
    }
    with pytest.raises(Exception) as exc:
        RuinConfig.model_validate(bad_cfg)
    assert "extra" in str(exc.value).lower() or "validation error" in str(exc.value).lower()

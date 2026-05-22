"""Pytest fixtures for RUIN v0.2 test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ruin.config import load_config


@pytest.fixture(scope="session")
def golden_config_path() -> Path:
    """Path to the canonical urban_ruin_shift example config."""
    return Path(__file__).parent.parent / "examples" / "urban_ruin_shift.yaml"


@pytest.fixture(scope="session")
def golden_config(golden_config_path: Path) -> dict[str, Any]:
    """Parsed golden config dict (v0.1 compatible)."""
    return load_config(golden_config_path)


@pytest.fixture
def small_config() -> dict[str, Any]:
    """Minimal config for fast unit tests."""
    return {
        "simulation": {
            "name": "tiny_test",
            "seed": 123,
            "grid_width": 8,
            "grid_height": 8,
            "shift_duration": 30,
            "dt": 1,
        },
        "qdots": {
            "n_standard": 5,
            "n_express": 2,
            "n_bulky": 1,
            "standard": {
                "time_window": 40,
                "delay_penalty": 1.0,
                "drift_multiplier": 1.0,
                "volatility_multiplier": 1.0,
            },
            "express": {
                "time_window": 25,
                "delay_penalty": 2.0,
                "drift_multiplier": 1.2,
                "volatility_multiplier": 1.1,
            },
            "bulky": {
                "time_window": 60,
                "delay_penalty": 1.5,
                "drift_multiplier": 0.8,
                "volatility_multiplier": 0.9,
            },
        },
        "processes": {
            "demand_arrival_rate": 3.0,
            "travel_drift": 1.0,
            "travel_volatility": 0.3,
            "jump_intensity": 0.05,
            "jump_mean_size": 2.0,
            "jump_size_distribution": "exponential",
        },
        "probability_square": {
            "base_risk_intensity": 0.0,
            "qdot_feedback_weight": 0.5,
            "d_state_thresholds": {
                "stressed_chaos_pressure": 0.35,
                "disrupted_chaos_pressure": 0.55,
                "chaotic_chaos_pressure": 0.75,
            },
        },
        "disruption_field": {
            "enabled": True,
            "shock_arrival_rate": 0.04,
            "mean_initial_intensity": 1.5,
            "memory_weight": 0.8,
            "propagation_radius": 1,
            "propagation_decay": 0.5,
            "temporal_decay": 0.15,
            "recovery_rate": 0.1,
        },
        "ruin": {
            "initial_buffer": 30,
            "barrier": 0,
            "sla_threshold": 0.25,
            "recovery_rate": 0.0,
        },
    }

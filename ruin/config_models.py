"""Pydantic v2 models for RUIN configuration (v0.2).

Provides strict validation (extra='forbid') while preserving full backward
compatibility with v0.1 YAML files via dict → model_validate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --- Leaf / sub-models -------------------------------------------------------

class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)

    name: str
    seed: int = 42
    grid_width: int = Field(default=30, ge=1)
    grid_height: int = Field(default=30, ge=1)
    shift_duration: int = Field(ge=1)
    dt: float = Field(default=1.0, gt=0)


class QDotTypeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time_window: float = Field(gt=0)
    # Loss charged per step while a Q dot is late (scaled by local exposure).
    delay_penalty: float = Field(ge=0)
    drift_multiplier: float = Field(gt=0)
    volatility_multiplier: float = Field(ge=0)
    # Surplus earned when a Q dot reaches its destination on time.
    completion_reward: float = Field(default=0.0, ge=0)


class QDotsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_standard: int = Field(ge=0)
    n_express: int = Field(ge=0)
    n_bulky: int = Field(ge=0)
    initial_distribution: Literal["uniform"] = "uniform"
    standard: QDotTypeParams
    express: QDotTypeParams
    bulky: QDotTypeParams


class ProcessesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Expected number of new Q dots entering D per step (Poisson).
    demand_arrival_rate: float = Field(ge=0)
    # Mean cells travelled toward the destination per step, and its volatility.
    travel_drift: float = Field(gt=0)
    travel_volatility: float = Field(ge=0)
    # Per-Q-dot compound Poisson setbacks: probability-rate and mean size (cells).
    jump_intensity: float = Field(ge=0)
    jump_mean_size: float = Field(ge=0)
    jump_size_distribution: Literal["exponential"] = "exponential"


class DStateThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stressed_chaos_pressure: float = Field(ge=0, le=1)
    disrupted_chaos_pressure: float = Field(ge=0, le=1)
    chaotic_chaos_pressure: float = Field(ge=0, le=1)
    recovering_decay_ratio: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> DStateThresholds:
        if not (
            self.stressed_chaos_pressure
            <= self.disrupted_chaos_pressure
            <= self.chaotic_chaos_pressure
        ):
            raise ValueError(
                "d_state_thresholds must satisfy stressed <= disrupted <= chaotic"
            )
        return self


class ProbabilitySquareConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_risk_intensity: float = 0.0
    congestion_memory: float = Field(default=0.85, ge=0, le=1)
    boundary_mode: Literal["reflect", "wrap", "absorb"] = "reflect"
    snapshot_interval: int = Field(default=1, ge=1)
    initial_d_state: Literal["stable", "stressed", "disrupted", "chaotic", "recovering"] = "stable"
    d_state_thresholds: DStateThresholds
    order_pressure_weight: float = Field(default=1.0, ge=0)
    chaos_pressure_weight: float = Field(default=1.0, ge=0)
    qdot_feedback_weight: float = Field(default=0.5, ge=0)


class DisruptionFieldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Expected number of shocks hitting the whole grid per step (Poisson).
    shock_arrival_rate: float = Field(ge=0)
    shock_types: list[str] = Field(default_factory=list)
    mean_initial_intensity: float = Field(default=2.0, ge=0)
    propagation_radius: int = Field(default=2, ge=0)
    propagation_decay: float = Field(default=0.55, ge=0, le=1)
    # Fraction of a cell's surviving intensity that moves to its neighbours each step.
    propagation_share: float = Field(default=0.3, ge=0, le=1)
    temporal_decay: float = Field(default=0.12, ge=0, le=1)
    recovery_rate: float = Field(default=0.08, ge=0, le=1)
    memory_weight: float = Field(default=0.85, ge=0, le=1)
    qdot_exposure_multiplier: float = Field(default=1.0, ge=0)

    @property
    def gain(self) -> float:
        """Fraction of total field intensity that survives one step."""
        return self.memory_weight * max(0.0, 1.0 - self.temporal_decay - self.recovery_rate)

    @model_validator(mode="after")
    def _subcritical(self) -> DisruptionFieldConfig:
        if self.temporal_decay + self.recovery_rate > 1.0:
            raise ValueError("temporal_decay + recovery_rate must not exceed 1")
        if self.gain >= 1.0:
            raise ValueError(
                "disruption field would never decay: "
                "memory_weight * (1 - temporal_decay - recovery_rate) must be < 1"
            )
        return self


class RuinParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_buffer: float
    # Planned capacity recovered per step, independent of completions.
    recovery_rate: float = Field(default=0.0, ge=0)
    barrier: float = 0.0
    sla_threshold: float = Field(default=0.12, ge=0)
    rule: Literal["surplus_or_sla", "surplus", "sla"] = "surplus_or_sla"


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    monte_carlo_paths: int = Field(default=1000, ge=1)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    report_time_to_ruin: bool = True
    report_d_state_path: bool = True
    report_chaotic_organization: bool = True
    report_var: bool = True
    report_expected_shortfall: bool = True


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    frames: int = Field(default=300, ge=1)
    show_qdot_types: bool = True
    show_ruin_event: bool = True
    show_disruption_field: bool = True
    show_d_state: bool = True
    show_order_chaos_pressure: bool = True
    heatmap_field: Literal["disruption_intensity", "order_pressure", "chaos_pressure"] = "disruption_intensity"


# --- Root model --------------------------------------------------------------

class RuinConfig(BaseModel):
    """Complete validated RUIN configuration root."""

    model_config = ConfigDict(extra="forbid")

    simulation: SimulationConfig
    qdots: QDotsConfig
    processes: ProcessesConfig
    probability_square: ProbabilitySquareConfig
    disruption_field: DisruptionFieldConfig
    ruin: RuinParams
    risk: RiskConfig
    visualization: VisualizationConfig


# Convenience type alias
RuinConfigRoot = RuinConfig
"""Pydantic v2 models for RUIN configuration (v0.2).

Provides strict validation (extra='forbid') while preserving full backward
compatibility with v0.1 YAML files via dict → model_validate.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, List


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
    delay_penalty: float
    drift_multiplier: float
    volatility_multiplier: float


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

    demand_arrival_rate: float = Field(ge=0)
    travel_drift: float
    travel_volatility: float = Field(ge=0)
    jump_intensity: float = Field(ge=0)
    jump_mean_size: float
    jump_size_distribution: Literal["exponential", "normal"] = "exponential"


class DStateThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stressed_chaos_pressure: float = Field(ge=0, le=1)
    disrupted_chaos_pressure: float = Field(ge=0, le=1)
    chaotic_chaos_pressure: float = Field(ge=0, le=1)
    recovering_decay_ratio: float = Field(ge=0, le=1)


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
    shock_arrival_rate: float = Field(ge=0)
    shock_types: List[str] = Field(default_factory=list)
    mean_initial_intensity: float = 2.0
    propagation_radius: int = Field(default=2, ge=0)
    propagation_decay: float = Field(default=0.55, ge=0, le=1)
    temporal_decay: float = Field(default=0.12, ge=0, le=1)
    recovery_rate: float = Field(default=0.08, ge=0, le=1)
    memory_weight: float = Field(default=0.85, ge=0, le=1)
    qdot_exposure_multiplier: float = 1.0


class RuinParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_buffer: float
    recovery_rate: float = 0.0
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
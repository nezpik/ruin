"""Pydantic v2 models for RUIN configuration (v0.2).

Provides strict validation (extra='forbid') while preserving full backward
compatibility with v0.1 YAML files via dict → model_validate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

QDOT_TYPE_NAMES = ("standard", "express", "bulky")


def expand_parameter(parameter: str) -> list[str]:
    """Expand `*` in a dotted parameter path to every Q-dot type."""
    if "*" not in parameter:
        return [parameter]
    return [parameter.replace("*", name) for name in QDOT_TYPE_NAMES]


# --- Leaf / sub-models -------------------------------------------------------


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)

    name: str
    seed: int = 42
    grid_width: int = Field(default=30, ge=1)
    grid_height: int = Field(default=30, ge=1)
    shift_duration: int = Field(ge=1)
    # One step is one unit of time. Finer time steps are not supported yet, so
    # every rate, window, and decay in the config is "per step".
    dt: float = Field(default=1.0, ge=1.0, le=1.0)


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
    # Local rerouting: a Q dot choosing between its two forward moves (along x or
    # along y) weights each by exp(-avoidance * how blocked the next cell on that
    # axis is). 0 is no avoidance (axes are chosen at random in proportion to the
    # distance left on each); larger values steer around blocked cells.
    avoidance: float = Field(default=0.0, ge=0)


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
            raise ValueError("d_state_thresholds must satisfy stressed <= disrupted <= chaotic")
        return self


REGIME_NAMES = ("stable", "stressed", "disrupted", "chaotic")


class RegimeConfig(BaseModel):
    """The hidden regime of D: what the destiny frame is *doing*, as opposed to
    what its pressures say it is doing.

    D moves between four levels (stable, stressed, disrupted, chaotic) one
    step at a time. Each level changes how often disruption arrives, how big
    it is, and how noisy travel is, so disruption comes in persistent
    clusters instead of a constant drizzle. The regime is pulled by D's own
    pressures, which is how Q and D co-evolve: late deliveries raise chaos
    pressure, chaos pressure escalates the regime, and the regime brings more
    disruption.

    All hazards are per step. Lists are indexed by level, lowest first.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # Where each shift starts. `stationary` draws a level from the long-run mix
    # of the exogenous chain (a shift can begin on a bad day).
    start: Literal["stable", "stressed", "disrupted", "chaotic", "stationary"] = "stationary"
    # Spontaneous probability per step of moving up from level k to k+1 ...
    escalate: list[float] = Field(default_factory=lambda: [0.015, 0.02, 0.02])
    # ... and of moving down from level k+1 to k.
    relax: list[float] = Field(default_factory=lambda: [0.05, 0.04, 0.03])
    # What each level does, as multipliers (level 0 first).
    shock_rate: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    shock_size: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    # Multiplies travel volatility and the per-Q-dot setback rate.
    travel_noise: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    # Feedback from pressure: the regime is pulled toward the level D's chaos
    # pressure points to. A move toward that level is `pressure_pull` times
    # likelier and a move away from it `pressure_pull` times less likely.
    # Pressure inside the regime's own band leaves the chain untouched.
    pressure_pull: float = Field(default=1.0, ge=1.0)
    # Healthy flow (order pressure) resists escalation and speeds recovery:
    # escalation is scaled by (1 - order_support * o), relaxation by
    # (1 + order_support * o), where o is order pressure clipped to [0, 1].
    order_support: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _shapes(self) -> RegimeConfig:
        for name, length in (
            ("escalate", 3),
            ("relax", 3),
            ("shock_rate", 4),
            ("shock_size", 4),
            ("travel_noise", 4),
        ):
            values = getattr(self, name)
            if len(values) != length:
                raise ValueError(f"regime.{name} needs {length} values, got {len(values)}")
            if any(v < 0 for v in values):
                raise ValueError(f"regime.{name} must not be negative")
        if any(not 0 <= v < 1 for v in self.escalate):
            raise ValueError("regime.escalate entries must be probabilities in [0, 1)")
        if any(not 0 < v <= 1 for v in self.relax):
            raise ValueError("regime.relax entries must be probabilities in (0, 1]")
        return self

    def stationary(self) -> list[float]:
        """Long-run share of time at each level for the exogenous chain
        (pressure feedback off). The chain is a birth-death chain, so detailed
        balance gives it exactly: pi[k+1] / pi[k] = escalate[k] / relax[k]."""
        weights = [1.0]
        for up, down in zip(self.escalate, self.relax, strict=True):
            weights.append(weights[-1] * up / down)
        total = sum(weights)
        return [w / total for w in weights]

    def exogenous_mean_mass(self) -> float:
        """Long-run mean of shock_rate * shock_size under `stationary()`: the
        average disruption mass per unit of base rate and size."""
        pi = self.stationary()
        return sum(p * r * z for p, r, z in zip(pi, self.shock_rate, self.shock_size, strict=True))


class ProbabilitySquareConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Background disruption present in every cell at all times (5 = fully blocked).
    base_risk_intensity: float = Field(default=0.0, ge=0)
    # How long a cell remembers that Q dots stalled there (0 = no memory).
    congestion_memory: float = Field(default=0.85, ge=0, lt=1)
    # What happens to disruption spreading past the edge of D.
    boundary_mode: Literal["reflect", "wrap", "absorb"] = "reflect"
    # Keep Q-dot and field snapshots every this many steps (paths are kept every step).
    snapshot_interval: int = Field(default=1, ge=1)
    initial_d_state: Literal["stable", "stressed", "disrupted", "chaotic", "recovering"] = "stable"
    d_state_thresholds: DStateThresholds
    qdot_feedback_weight: float = Field(default=0.5, ge=0)
    # The hidden regime of D (off by default: disruption is then memoryless).
    regime: RegimeConfig = Field(default_factory=RegimeConfig)


class DisruptionFieldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Expected number of shocks hitting the whole grid per step (Poisson).
    shock_arrival_rate: float = Field(ge=0)
    shock_types: list[str] = Field(default_factory=list)
    mean_initial_intensity: float = Field(default=2.0, ge=0)
    # Spread of shock sizes. 0 gives exponential sizes (coefficient of variation 1);
    # above 0, sizes are lognormal with this log-scale sigma and the same mean.
    # sigma 1.5 gives a coefficient of variation of about 3.
    shock_size_sigma: float = Field(default=0.0, ge=0)
    propagation_radius: int = Field(default=2, ge=0)
    propagation_decay: float = Field(default=0.55, ge=0, le=1)
    # Fraction of a cell's surviving intensity that moves to its neighbours each step.
    propagation_share: float = Field(default=0.3, ge=0, le=1)
    temporal_decay: float = Field(default=0.12, ge=0, le=1)
    recovery_rate: float = Field(default=0.08, ge=0, le=1)
    memory_weight: float = Field(default=0.85, ge=0, le=1)
    qdot_exposure_multiplier: float = Field(default=1.0, ge=0)
    # Load-dependent persistence: where Q dots are stalled, disruption lingers.
    # Retention in a cell moves from `gain` toward `max_gain` in proportion to
    # `congestion_persistence` times the cell's congestion (0 to 1). Because
    # retention can never exceed `max_gain` < 1, the field still cannot amplify
    # itself, but under load it gets close to critical: jams persist.
    congestion_persistence: float = Field(default=0.0, ge=0, le=1)
    max_gain: float = Field(default=0.985, gt=0, lt=1)

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
        if self.congestion_persistence > 0 and self.max_gain < self.gain:
            raise ValueError("max_gain must be at least the field's base gain")
        return self


class RuinParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    initial_buffer: float
    # Planned capacity recovered per step, independent of completions.
    recovery_rate: float = Field(default=0.0, ge=0)
    barrier: float = 0.0
    # SLA breach: share of deadlines missed over the last `sla_window` steps,
    # missed / (missed + delivered on time), reaches `sla_threshold`.
    sla_threshold: float = Field(default=0.12, ge=0, le=1)
    sla_window: int = Field(default=60, ge=1)
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
    # Parameters that are only known approximately, by dotted path, with their
    # relative standard deviation: {"disruption_field.shock_arrival_rate": 0.1}.
    # `*` stands for every Q-dot type. Used by `ruin uncertainty`.
    parameter_uncertainty: dict[str, float] = Field(default_factory=dict)


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    frames: int = Field(default=300, ge=1)
    show_qdot_types: bool = True
    show_ruin_event: bool = True
    show_disruption_field: bool = True
    # Outline the jams (regions blocked at least half) on the heatmap.
    show_jams: bool = True
    show_d_state: bool = True
    show_order_chaos_pressure: bool = True
    heatmap_field: Literal[
        "disruption_intensity", "order_pressure", "chaos_pressure", "accumulated_loss"
    ] = "disruption_intensity"


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

    @model_validator(mode="after")
    def _uncertain_parameters_exist(self) -> RuinConfig:
        for parameter, spread in self.risk.parameter_uncertainty.items():
            if not 0 < spread <= 1:
                raise ValueError(
                    f"parameter_uncertainty[{parameter!r}] must be a relative spread in (0, 1]"
                )
            for path in expand_parameter(parameter):
                node: object = self
                for key in path.split("."):
                    if not hasattr(node, key):
                        raise ValueError(f"parameter_uncertainty: unknown parameter {path!r}")
                    node = getattr(node, key)
                if isinstance(node, bool) or not isinstance(node, float):
                    raise ValueError(
                        f"parameter_uncertainty: {path!r} is not a continuous parameter"
                    )
        return self

    @model_validator(mode="after")
    def _radius_fits_grid(self) -> RuinConfig:
        smallest = min(self.simulation.grid_width, self.simulation.grid_height)
        if self.disruption_field.propagation_radius > smallest:
            raise ValueError("propagation_radius must not exceed the smaller grid dimension")
        return self


# Convenience type alias
RuinConfigRoot = RuinConfig

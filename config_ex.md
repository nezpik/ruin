# RUIN Example Configuration

The authoritative schema is `ruin/config_models.py` (Pydantic, strict: unknown
keys are rejected). The reference scenario is `examples/urban_ruin_shift.yaml`,
reproduced here.

```yaml
# Urban Ruin Shift: the reference RUIN scenario.
#
# Calibrated so that ruin is a real question: about one shift in seven ruins
# over the 480-step horizon (see `ruin risk`). All quantities are per step.

simulation:
  name: urban_ruin_shift
  seed: 42
  grid_width: 30
  grid_height: 30
  shift_duration: 480
  dt: 1

qdots:
  # Initial population by type. The same mix is used for new arrivals.
  n_standard: 120
  n_express: 40
  n_bulky: 40
  initial_distribution: uniform
  standard:
    time_window: 70          # steps allowed to reach the destination
    delay_penalty: 1.0       # loss per step while late
    drift_multiplier: 1.0
    volatility_multiplier: 1.0
    completion_reward: 0.05  # surplus earned by an on-time delivery
  express:
    time_window: 46
    delay_penalty: 2.5
    drift_multiplier: 1.25
    volatility_multiplier: 1.2
    completion_reward: 0.12
  bulky:
    time_window: 104
    delay_penalty: 2.0
    drift_multiplier: 0.7
    volatility_multiplier: 0.9
    completion_reward: 0.08

processes:
  demand_arrival_rate: 8.0   # expected new Q dots per step
  travel_drift: 1.2          # mean cells travelled per step
  travel_volatility: 0.4
  jump_intensity: 0.05       # per-Q-dot rate of sudden setbacks
  jump_mean_size: 2.0        # mean setback, in cells of lost progress
  jump_size_distribution: exponential

probability_square:
  base_risk_intensity: 0.0
  congestion_memory: 0.85
  boundary_mode: reflect
  snapshot_interval: 1
  initial_d_state: stable
  d_state_thresholds:
    stressed_chaos_pressure: 0.20
    disrupted_chaos_pressure: 0.27
    chaotic_chaos_pressure: 0.34
    recovering_decay_ratio: 0.05
  order_pressure_weight: 1.0
  chaos_pressure_weight: 1.0
  qdot_feedback_weight: 0.5

disruption_field:
  enabled: true
  shock_arrival_rate: 0.4       # expected shocks per step, whole grid
  shock_types:
    - congestion_wave
    - accident
    - weather_cell
    - failed_handoff
    - demand_burst
  mean_initial_intensity: 285.0 # a cell is fully blocked at intensity 5
  propagation_radius: 2
  propagation_decay: 0.55
  propagation_share: 0.4        # share of surviving intensity that spreads
  temporal_decay: 0.03
  recovery_rate: 0.02
  memory_weight: 0.95
  qdot_exposure_multiplier: 1.0

ruin:
  initial_buffer: 120
  recovery_rate: 0.0
  barrier: 0.0
  sla_threshold: 0.05
  rule: surplus_or_sla

risk:
  monte_carlo_paths: 1000
  confidence_level: 0.95
  report_time_to_ruin: true
  report_d_state_path: true
  report_chaotic_organization: true
  report_var: true
  report_expected_shortfall: true

visualization:
  enabled: true
  frames: 300
  show_qdot_types: true
  show_ruin_event: true
  show_disruption_field: true
  show_d_state: true
  show_order_chaos_pressure: true
  heatmap_field: disruption_intensity
```

## Interpretation

- `simulation` defines the shift horizon and the resolution of the destiny frame.
- `qdots` defines the initial population and the per-type time window, delay
  penalty, speed, and on-time completion reward. New arrivals use the same mix.
- `processes` defines demand arrivals and the jump-diffusion that moves each
  Q dot toward its destination.
- `probability_square` defines the D-state thresholds on chaos pressure, how
  strongly late or failed Q dots feed back into the field, the background risk
  in every cell, how long cells remember stalled Q dots (`congestion_memory`),
  what happens to disruption at the edge of `D` (`boundary_mode`), and how
  often heavy snapshots are kept.
- `disruption_field` defines shock arrivals, spread, and decay. The field is
  subcritical by construction: `memory_weight * (1 - temporal_decay -
  recovery_rate)` must be below 1, so every shock eventually fades.
- `ruin` defines the surplus buffer, the barrier, the SLA threshold, and which
  rule applies (`surplus_or_sla`, `surplus`, or `sla`).
- `risk` defines the Monte Carlo path count and confidence level.
- `visualization` defines what the GIF shows: the heatmap quantity
  (`disruption_intensity`, `chaos_pressure`, or `order_pressure`), the number
  of frames spread over the shift, and which overlays are drawn.

## Keys that are accepted but reserved

These are validated but do not change anything yet:

- `qdots.initial_distribution`: only `uniform` exists.
- `probability_square.order_pressure_weight`, `chaos_pressure_weight`: no
  current formula uses a weight; the D-state thresholds already set the scale.
- `disruption_field.shock_types`: reserved for shock attribution.
- `risk.report_*`: every statistic is always reported.
- `visualization.enabled`: the `--visualize` flag decides.

`simulation.dt` must be 1: one step is one unit of time.

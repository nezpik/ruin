# RUIN v0.1 Example Configuration

This file describes the intended `examples/urban_ruin_shift.yaml` configuration shape for the first implementation.

```yaml
simulation:
  name: urban_ruin_shift_v0_1
  seed: 42
  grid_width: 30
  grid_height: 30
  shift_duration: 480
  dt: 1

qdots:
  n_standard: 120
  n_express: 40
  n_bulky: 40
  initial_distribution: uniform
  standard:
    time_window: 120
    delay_penalty: 1.0
    drift_multiplier: 1.0
    volatility_multiplier: 1.0
  express:
    time_window: 60
    delay_penalty: 2.5
    drift_multiplier: 1.25
    volatility_multiplier: 1.2
  bulky:
    time_window: 180
    delay_penalty: 2.0
    drift_multiplier: 0.7
    volatility_multiplier: 0.9

processes:
  demand_arrival_rate: 8.0
  travel_drift: 1.2
  travel_volatility: 0.4
  jump_intensity: 0.08
  jump_mean_size: 3.0
  jump_size_distribution: exponential

probability_square:
  base_risk_intensity: 0.0
  congestion_memory: 0.85
  boundary_mode: reflect
  snapshot_interval: 1
  initial_d_state: stable
  d_state_thresholds:
    stressed_chaos_pressure: 0.35
    disrupted_chaos_pressure: 0.55
    chaotic_chaos_pressure: 0.75
    recovering_decay_ratio: 0.30
  order_pressure_weight: 1.0
  chaos_pressure_weight: 1.0
  qdot_feedback_weight: 0.5

disruption_field:
  enabled: true
  shock_arrival_rate: 0.05
  shock_types:
    - congestion_wave
    - accident
    - weather_cell
    - failed_handoff
    - demand_burst
  mean_initial_intensity: 2.0
  propagation_radius: 2
  propagation_decay: 0.55
  temporal_decay: 0.12
  recovery_rate: 0.08
  memory_weight: 0.85
  qdot_exposure_multiplier: 1.0

ruin:
  initial_buffer: 120
  recovery_rate: 0.0
  barrier: 0.0
  sla_threshold: 0.12
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

- `simulation` defines the abstract shift horizon and destiny frame resolution.
- `qdots` defines the population of quantized logistics destinies.
- `processes` defines the stochastic drift, volatility, and jump structure.
- `probability_square` defines how the destiny grid is stored, stepped, and moved between D-states.
- `disruption_field` defines shock injection, local propagation, memory, decay, and recovery.
- D-state thresholds define when the grid becomes stressed, disrupted, chaotic, recovering, or ruined.
- `ruin` defines the system failure barrier.
- `risk` defines Monte Carlo and tail-risk outputs.
- `visualization` defines how the Probability Square should be rendered.

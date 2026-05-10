# RUIN v0.1 Architecture Specification

RUIN v0.1 should be built as a small Python package whose architecture mirrors the conceptual layers in `RUIN.md`: destiny state space, Q-dot entities, stochastic processes, network surplus, risk measurement, and visualization.

---

## Package Shape

```text
ruin/
├── __init__.py
├── core/
│   ├── qdot.py                 # QDot class + types + order pressure
│   ├── processes.py            # GBM, JumpDiffusion, Poisson arrivals
│   └── network.py              # Surplus process + ruin barrier
├── state_space/
│   ├── probability_square.py   # The 2D destiny frame + D-state
│   └── disruption_field.py     # Cell memory, propagation, decay, recovery
├── simulation/
│   ├── monte_carlo.py          # Fast vectorized paths (numpy)
│   └── agent_based.py          # Step-by-step Q-dot movement
├── risk/
│   └── ruin_probability.py     # Finite-time ruin (MC + analytical bounds)
├── metrics/
│   └── pressure.py             # Order pressure, chaos pressure, D-state rules
├── viz/
│   └── square.py               # Matplotlib/Plotly renderer
├── config/
│   └── urban_ruin_shift.yaml   # Example config
├── cli.py                      # `ruin simulate --config ... --visualize`
└── pyproject.toml
```

---

## Layer Responsibilities

### `core/qdot.py`

Defines the Q dot as the smallest unit of logistics destiny in v0.1.

Required concepts:

- `QDotType`: `STANDARD`, `EXPRESS`, `BULKY`.
- `QDot`: id, type, position, stochastic process, time-window slack, penalty, status, order pressure, D-state exposure.
- Q-dot state transition: destiny-bound movement, slack decay, delay detection, order pressure, local feedback, ruin contribution.

The Q dot should remain generic. It should not be hard-coded as only a parcel, only a vehicle, or only a Monte Carlo path. It should also never be allowed to leave `D`; it can only reveal possible order inside the destiny frame.

### `core/processes.py`

Defines stochastic processes used by Q dots, disruption shocks, and system-level events.

Required concepts:

- `GBMProcess`: smooth drift and volatility.
- `JumpDiffusionProcess`: diffusion plus compound Poisson jumps.
- `PoissonArrivalProcess`: stochastic order or disruption arrivals.
- `RecoveryProcess`: decay or clearing of local disruption intensity.
- seeded random generator support for reproducible experiments.

### `core/network.py`

Defines the logistics portfolio state.

Required concepts:

- `NetworkSurplus`: `U(t) = U(0) + C(t) - L(t)`.
- cumulative penalty accounting.
- SLA late-ratio accounting.
- combined ruin rule: `surplus_ruined OR sla_ruined`.

### `metrics/pressure.py`

Defines the first implementable pressure formulas.

Required concepts:

- `local_chaos_pressure`: disruption intensity plus congestion memory plus local failed-Q-dot ratio.
- `global_chaos_pressure`: mean local chaos pressure plus late ratio.
- `local_order_pressure`: active Q dots plus moving Q dots minus failed Q dots.
- `global_order_pressure`: active Q-dot utilization minus failed-Q-dot ratio.
- `DStateTransition`: threshold-based transition into stable, stressed, disrupted, chaotic, recovering, or ruined.

### `state_space/probability_square.py`

Defines the v0.1 representation of destiny frame `D`.

Required concepts:

- rectangular lattice dimensions.
- cell-level risk intensity.
- integration with the Disruption Propagation Field.
- D-state regime tracking: stable, stressed, disrupted, chaotic, recovering, ruined.
- order pressure and chaos pressure aggregation.
- feedback from Q-dot density, delay, clustering, recovery, and failure.
- Q-dot containment and boundary handling.
- step update over Q dots and grid state.
- snapshot export for visualization and Monte Carlo analysis.

### `state_space/disruption_field.py`

Defines the spatial memory of physical-flow uncertainty.

Required concepts:

- local shock injection from demand bursts, accidents, weather, failed handoffs, or congestion.
- propagation from a shocked cell to neighboring cells.
- decay and recovery of field intensity over time.
- congestion memory so repeated shocks accumulate instead of disappearing instantly.
- local exposure lookup for Q dots moving through `D`.
- event attribution so ruin analysis can distinguish absorbed shocks from cascading shocks.
- contribution to D-state transitions when disruption becomes the state of the grid.

### `simulation/agent_based.py`

Runs one explicit destiny trajectory.

Required concepts:

- step-by-step Q-dot movement.
- stochastic arrivals and disruptions.
- disruption-field propagation and recovery.
- Q-dot exposure to local field intensity.
- D-state transitions.
- order-vs-chaos pressure updates.
- co-evolution where Q-dot behavior reshapes the grid.
- network surplus updates.
- event log and state snapshots.

### `simulation/monte_carlo.py`

Runs many destiny trajectories.

Required concepts:

- repeated simulation with controlled seeds.
- finite-time ruin probability.
- distribution of time to ruin.
- vectorized or batched execution where possible.

### `risk/ruin_probability.py`

Converts simulation outputs into risk metrics.

Required concepts:

- empirical ruin probability.
- VaR and Expected Shortfall of cumulative loss.
- survival probability.
- late-ratio and surplus-path summaries.
- D-state path summaries.
- chaotic organization pattern summaries.
- pressure summaries for order pressure and chaos pressure.

### `viz/square.py`

Renders the Probability Square as analysis, not decoration.

Required concepts:

- Q-dot positions by type/status.
- grid heatmap for risk intensity or congestion pressure.
- D-state regime.
- order pressure and chaos pressure.
- chaotic organization patterns such as clusters, forced corridors, recovery pockets, and ruin zones.
- visual indication of ruin event.
- optional animation frames.

### `cli.py`

Exposes the first user-facing interface.

Required command shape:

```text
ruin simulate --config examples/urban_ruin_shift.yaml --visualize --frames 300
```

---

## Data Flow

```text
config
  -> process parameters
  -> ProbabilitySquare(D)
  -> DisruptionPropagationField
  -> Q dots
  -> agent-based trajectory
  -> NetworkSurplus
  -> ruin checks
  -> risk metrics
  -> visualization
```

Monte Carlo wraps this flow by repeating it across many possible destinies.

---

## v0.1 Invariants

- `D` is the state space; the Probability Square is the v0.1 implementation of it.
- `D` frames all Q dots; Q dots cannot leave the destiny frame.
- Q dots carry local destiny and order pressure; the network surplus carries global system health.
- The Disruption Propagation Field carries spatial memory of physical uncertainty.
- Q-dot movement must be affected by local field exposure, not only independent noise.
- Q-dot behavior must feed back into `D` through density, delay, clustering, recovery, and failure pressure.
- Disruption can be both an event and a D-state.
- Ruin must be measurable at every step.
- Every stochastic process must be seedable.
- Visualization must reflect model state, not invent separate visual-only behavior.
- The first implementation should prefer clarity and traceability over optimization.

---

## Future Expansion Boundaries

v0.1 should leave clean extension seams for:

- real map graphs,
- multiple depots,
- routing policies,
- hedging policies,
- live calibration,
- multimodal logistics,
- and reinforcement learning.

These are not part of the first concept implementation.
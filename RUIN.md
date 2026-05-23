# RUIN: Ruin-probability Unified In Networks

**Finance-grade ruin risk for the physical world.**

[![Python](https://img.shields.io/badge/Python-3.11--3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![v0.2 Foundation Hardening](https://img.shields.io/badge/Version-v0.2%20Foundation-green.svg)](https://github.com/nezpik/ruin)

RUIN is a framework for measuring and visualizing logistics systems as stochastic risk-bearing portfolios. It translates quantitative finance concepts such as Cramér-Lundberg ruin theory, jump diffusion, geometric Brownian motion, Monte Carlo paths, VaR, and Expected Shortfall into a logistics language built around physical uncertainty, service failure, and systemic collapse.

> **v0.2 Status**: The core simulation engine, vectorized primitives, typed configuration, parallel risk analysis, and real-field visualization have been hardened. The conceptual foundation in this document remains the north star.

The central object is the **Probability Square**: an abstract two-dimensional destiny frame `D` where many **Q dots** move through possible physical futures. RUIN is not trying to make logistics look deterministic, and it is not trying to predict destiny perfectly. RUIN frames chaos so that physical uncertainty can move, take shape, and become measurable.

---

## v0.1 Thesis

RUIN v0.1 focuses on a single research-grade use case:

**Urban Ruin Shift: disruption propagation across a last-mile destiny grid.**

The v0.1 world has no real maps, no GIS, and no optimization engine yet. It is a controlled abstract grid where the framework can define its language, risk mechanics, and visual model before expanding into real physical networks.

The shift begins with finite operational surplus. Q dots enter and move through `D` as delivery commitments, route obligations, or flow units. The Probability Square evolves as a disruption field: congestion, demand bursts, accidents, failed handoffs, and weather shocks leave memory in cells, spread locally, decay over time, and change the destiny of nearby Q dots. RUIN v0.1 succeeds when it makes this physical-flow unpredictability visible and measurable.

`D` frames all Q dots. A Q dot can drift, jump, stall, cluster, fail, recover, or contribute to ruin, but it cannot leave destiny. The quant layer estimates possible paths; the RUIN layer asks what kind of order can exist when those paths are trapped inside a chaotic physical world.

---

## Core Philosophy

- **Logistics as a portfolio**: A fleet, route set, warehouse, or delivery shift behaves like a portfolio exposed to correlated risk factors.
- **Shipments as risk instruments**: A parcel, vehicle, route, or service promise has drift, volatility, jump exposure, and loss contribution.
- **Uncertainty as first-class state**: RUIN does not hide uncertainty behind averages; it gives uncertainty a place to move.
- **Destiny before prediction**: Quant methods estimate possible futures, but RUIN's deeper object is the destiny frame that bounds physical chaos.
- **Q as order pressure**: Q dots are attempts to impose service, flow, promise, or completion inside a world that resists clean prediction.
- **Chaotic organization**: Disruption is not only disorder. It can produce temporary structures such as clusters, jams, funnels, forced corridors, recovery zones, and ruin zones.
- **Co-evolution of Q and D**: `D` shapes Q-dot motion, but Q-dot density, delay, failure, and recovery also reshape `D`.
- **Ruin as system truth**: A logistics system has failed when its operational surplus is exhausted or when its service-level barrier is breached.
- **Visualization as analysis**: The Probability Square is not decoration. It is a way to watch possible futures collide with physical constraints.

---

## Core Vocabulary

### Destiny frame `D`

`D` is the bounded destiny frame in which RUIN observes physical chaos and possible logistics futures. In v0.1, `D` is represented by a rectangular lattice with width, height, time, and regime state. Later versions can map `D` to graphs, real roads, warehouse networks, ports, or multimodal corridors.

`D` does not mean the future is fixed. It means uncertainty has a frame. Q dots cannot move outside this frame; they can only reveal how order forms, deforms, and sometimes fails inside it.

### Probability Square

The Probability Square is the v0.1 visual and computational representation of `D`. Each cell can carry risk intensity, congestion pressure, arrival density, volatility, disruption memory, or accumulated loss.

The Probability Square also stores the current **D-state**: the global regime of the grid. A shift can move through stable, stressed, disrupted, chaotic, recovering, and ruined regimes as chaos pressure and order pressure interact.

### Disruption Propagation Field

The Disruption Propagation Field is the v0.1 mechanism that makes uncertainty spatial and systemic. A disruption is not only a random jump applied to one Q dot. It can become a local field event that affects nearby cells, raises movement volatility, consumes slack, increases penalty exposure, and then either decays or cascades.

In v0.1, this field is abstract. It does not need real traffic feeds. It should capture the physical truth that logistics disruptions cluster, propagate, block, recover, and create correlated failures.

### Q dot

A Q dot is a quantized logistics destiny and a local attempt at order. In v0.1, a Q dot is a generic delivery commitment moving through the Probability Square. It can represent a parcel, route obligation, vehicle-task unit, or Monte Carlo realization depending on the simulation mode.

Each Q dot carries **order pressure**: the pressure to complete, deliver, satisfy a service promise, or preserve flow. That order pressure is always destiny-bound. It can resist disruption, contribute to congestion, create temporary organization, or fail into ruin, but it never exits `D`.

### Chaotic organization

Chaotic organization is the temporary structure that appears when Q dots attempt to create order inside a chaotic physical world. RUIN should make these structures visible instead of smoothing them away.

Examples include:

- Q-dot clusters around blocked regions,
- forced corridors through low-disruption cells,
- congestion pressure zones,
- recovery pockets after disruption decay,
- and ruin zones where order pressure can no longer overcome chaos pressure.

### D-state transition

A D-state transition is a regime change in the Probability Square. In v0.1, the intended regimes are:

| D-state | Meaning |
| --- | --- |
| `STABLE` | Q dots move with low field pressure and surplus is healthy |
| `STRESSED` | field pressure or late ratio rises but order remains coherent |
| `DISRUPTED` | disruption becomes a dominant local state of the grid |
| `CHAOTIC` | many Q dots are forced into emergent clusters, stalls, or unstable paths |
| `RECOVERING` | field intensity decays and order pressure begins to restore flow |
| `RUINED` | surplus or SLA barrier has been breached |

### Surplus process

The system surplus is the remaining operational buffer after disruptions and penalties are absorbed.

```text
U(t) = U(0) + C(t) - L(t)
```

Where:

- `U(t)` is logistics surplus at time `t`.
- `U(0)` is initial slack, buffer, reserve capacity, or service margin.
- `C(t)` is accumulated recovery capacity or planned service capacity.
- `L(t)` is cumulative disruption loss, delay penalty, missed SLA cost, or overload cost.

### Ruin

Ruin occurs when the system crosses a failure barrier:

```text
ruin = U(t) <= barrier
```

v0.1 also allows an SLA-based ruin trigger:

```text
ruin = late_qdots / active_qdots >= sla_threshold
```

The canonical v0.1 ruin rule is the combined rule:

```text
system_ruined = surplus_ruined OR sla_ruined
```

---

## Quant Finance Translation

RUIN does not borrow quant finance vocabulary as metaphor only. Each finance object must become an operational logistics object.

| Quant finance concept | RUIN logistics meaning | v0.1 object |
| --- | --- | --- |
| Cramér-Lundberg surplus | Buffer minus cumulative disruption claims | `NetworkSurplus` |
| Compound Poisson claims | Random disruption arrivals with random severity | `DisruptionProcess` |
| Geometric Brownian motion | Smooth stochastic drift in demand, speed, or lead time | `GBMProcess` |
| Jump diffusion | Normal movement plus sudden congestion or accident spikes | `JumpDiffusionProcess` |
| Bounded state space | Chaos has a destiny frame that Q dots cannot exit | `DestinyFrame` |
| Regime switching | The grid changes between stable, stressed, disrupted, chaotic, recovering, and ruined states | `DStateTransition` |
| Feedback loop | Q dots are shaped by D and also reshape D | `QDCoEvolution` |
| Emergent order | Temporary structure created by Q dots moving through disruption | `ChaoticOrganization` |
| Correlated risk factors | Neighboring cells and Q dots affected by the same physical shock | `DisruptionPropagationField` |
| Cascading failure | Local disruption becomes systemic service collapse | `PropagationRuinPath` |
| Buffer stock / reserves | Slack that absorbs shocks before they become ruin | `NetworkSurplus` |
| Recovery intensity | Decay, clearing, rerouting, or capacity restoration after a shock | `RecoveryProcess` |
| Monte Carlo paths | Many possible logistics destinies | `MonteCarloSimulation` |
| VaR | Quantile of shift-level logistics loss | `value_at_risk` |
| Expected Shortfall | Mean loss beyond the VaR tail | `expected_shortfall` |
| Barrier event | Operational bankruptcy or SLA collapse | `RuinEvent` |
| Hedging | Extra buffer, backup capacity, rerouting, safety stock | future policy layer |

---

## v0.1 Model Scope

### Included

- Abstract rectangular grid as `D`.
- Q dots with standard, express, and bulky types.
- Type-specific drift, volatility, time-window pressure, and penalty exposure.
- Poisson order/disruption arrivals.
- Jump-diffusion travel-time or movement shocks.
- Disruption Propagation Field with cell memory, local spread, decay, and recovery.
- Correlated Q-dot exposure to nearby physical-flow conditions.
- D-state regimes and transitions.
- Order pressure from Q dots and chaos pressure from the grid.
- Co-evolution where Q-dot movement changes the state of `D`.
- Surplus process and combined ruin barrier.
- Monte Carlo estimation of finite-time ruin probability.
- Probability Square animation semantics.
- YAML-like configuration structure.
- CLI shape for future implementation.

### Excluded

- Real OSM maps.
- Live traffic APIs.
- Vehicle routing optimization.
- Reinforcement learning.
- Multi-echelon supply chain graphs.
- Real-time dispatch decisions.
- Production-grade calibration from customer data.

---

## v0.1 Q Dot Types

| Q dot type | Physical meaning | Risk character |
| --- | --- | --- |
| `STANDARD` | Normal delivery commitment | Balanced drift, volatility, and penalty |
| `EXPRESS` | Faster, tighter time-window commitment | Higher SLA pressure and ruin contribution |
| `BULKY` | Slower or capacity-heavy commitment | Lower speed, higher drag, higher penalty if delayed |

In v0.1, the Q dot is intentionally generic. The first implementation should avoid overfitting it to only parcel, vehicle, or route. The framework should allow each project to bind Q dots to a physical interpretation.

---

## Probability Square Dynamics

At each simulation step:

1. New uncertainty enters through demand arrivals or disruption events.
2. Disruption shocks are written into the Probability Square as local field intensity.
3. The field propagates to nearby cells, decays, or accumulates as congestion memory.
4. The Probability Square updates its D-state from stable, stressed, disrupted, chaotic, recovering, or ruined.
5. Each active Q dot samples destiny-bound movement from its stochastic process and local field exposure.
6. Q dots apply order pressure by attempting to preserve service, flow, or completion.
7. Q-dot density, delay, and failure feed back into the grid and reshape `D`.
8. Each Q dot loses time-window slack according to movement, delay, and local disruption pressure.
9. Delayed or failed Q dots add penalty to cumulative system loss.
10. The surplus process is updated.
11. The system checks for ruin.

This produces two simultaneous views:

- **Microscopic view**: Q dots moving, delaying, bunching, jumping, or failing.
- **Macroscopic view**: system surplus approaching or avoiding ruin.

---

## MVP Algorithm

The first implementation should use a simple, traceable loop:

1. initialize `D` as a Probability Square with width, height, time, and initial D-state,
2. spawn Q dots with type, position, slack, penalty, process, and order pressure,
3. initialize `NetworkSurplus`,
4. sample new order arrivals and disruption events,
5. inject disruption events into the Disruption Propagation Field,
6. propagate, decay, and recover field intensity,
7. compute chaos pressure and order pressure,
8. update the D-state regime,
9. move each Q dot inside `D` using stochastic movement and local field exposure,
10. feed Q-dot density, delay, clustering, recovery, and failure pressure back into `D`,
11. update slack, penalties, late ratio, and surplus,
12. check surplus ruin and SLA ruin,
13. record snapshots, D-state path, surplus path, and output metrics.

This algorithm is intentionally not an optimizer. It is the minimum loop needed to make destiny-framed physical uncertainty visible.

---

## MVP Pressure and D-State Formulas

The first implementation should use transparent formulas before introducing calibration.

```text
local_chaos_pressure(cell, t)
  = disruption_intensity(cell, t)
  + congestion_memory(cell, t)
  + local_failed_qdot_ratio(cell, t)
```

```text
global_chaos_pressure(t)
  = mean(local_chaos_pressure(:, t))
  + late_ratio(t)
```

```text
local_order_pressure(cell, t)
  = active_qdots(cell, t)
  + moving_qdots(cell, t)
  - failed_qdots(cell, t)
```

```text
global_order_pressure(t)
  = active_qdots(t) / max_capacity
  - failed_qdots(t) / initial_qdots
```

Suggested D-state rule:

| Condition | D-state |
| --- | --- |
| `system_ruined == true` | `RUINED` |
| `global_chaos_pressure >= chaotic_threshold` | `CHAOTIC` |
| `global_chaos_pressure >= disrupted_threshold` | `DISRUPTED` |
| `global_chaos_pressure >= stressed_threshold` | `STRESSED` |
| `global_chaos_pressure is falling after disruption` | `RECOVERING` |
| otherwise | `STABLE` |

These formulas are not final theory. They are v0.1 scaffolding: simple enough to implement, inspect, and replace later.

---

## Expected v0.1 Outputs

- **Ruin probability**: Estimated probability that the shift ruins before time horizon.
- **Time-to-ruin distribution**: When ruin occurs across Monte Carlo destinies.
- **Surplus path**: `U(t)` over the shift.
- **Late ratio path**: delayed Q dots divided by active Q dots over time.
- **Tail loss metrics**: VaR and Expected Shortfall for cumulative penalty.
- **Q dot state snapshots**: positions, type, slack, penalty, and ruined state.
- **Disruption field snapshots**: cell intensity, propagation, decay, and memory over time.
- **D-state path**: the sequence of stable, stressed, disrupted, chaotic, recovering, and ruined regimes.
- **Order-vs-chaos pressure**: how Q-dot order pressure interacts with grid chaos pressure.
- **Chaotic organization patterns**: clusters, forced corridors, pressure zones, recovery pockets, and ruin zones.
- **Absorption vs cascade summary**: which shocks were absorbed by surplus and which contributed to ruin.
- **Probability Square frames**: visual state of the destiny grid through time.

---

## Research Identity

RUIN is closest to a hybrid of:

- stochastic process modeling,
- ruin theory,
- disruption propagation,
- destiny-framed state-space modeling,
- emergent chaotic organization,
- agent-based logistics simulation,
- resilience and buffer analysis,
- tail-risk measurement,
- and visual state-space analysis.

It should not become only a generic discrete-event simulator. It should keep its identity: **a framework for seeing when physical systems run out of destiny.**

---

## v0.1 Success Criteria

- The terminology around `D`, Q dots, Probability Square, surplus, and ruin is consistent.
- The quant-finance analogy is operational enough to implement.
- The first use case is narrow enough to build: abstract last-mile delivery on a grid.
- The Disruption Propagation Field makes physical uncertainty spatial, temporal, and systemic.
- `D` is understood as an inescapable destiny frame, not just a grid.
- Q dots are understood as order pressure moving inside bounded chaos.
- Disruption can be modeled as both an event and a D-state.
- The outputs are measurable: ruin probability, time-to-ruin, surplus, SLA failure, and tail loss.
- The visual model shows uncertainty as a living field, not a static chart.

---

## First Implementation Acceptance Criteria

The first code MVP is working when it can:

- run one visible Urban Ruin Shift from a YAML config,
- keep all Q dots bounded inside `D`,
- update the Disruption Propagation Field over time,
- compute order pressure, chaos pressure, and D-state path,
- produce Q-dot snapshots with position, type, slack, field exposure, D-state exposure, and local failure state,
- update surplus and late ratio at every step,
- detect surplus ruin and SLA ruin,
- run Monte Carlo paths with reproducible seeds,
- report ruin probability, time-to-ruin distribution, VaR, and Expected Shortfall,
- render or save Probability Square frames that reflect model state.

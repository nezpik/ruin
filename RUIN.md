# RUIN: Ruin-probability Unified In Networks

**Finance-grade ruin risk for the physical world.**

[![Python](https://img.shields.io/badge/Python-3.11--3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![v0.9.0](https://img.shields.io/badge/Version-v0.9.0-green.svg)](https://github.com/nezpik/ruin)

RUIN is a framework for measuring and visualizing logistics systems as stochastic risk-bearing portfolios. It translates quantitative finance concepts such as Cramér-Lundberg ruin theory, jump diffusion, geometric Brownian motion, Monte Carlo paths, VaR, and Expected Shortfall into a logistics language built around physical uncertainty, service failure, and systemic collapse.

> **Status (v0.9)**: This document is the conceptual foundation and north star. The engine now implements it with completable Q dots, a decaying disruption field, and a risk layer validated against classical ruin theory. Where the implemented formulas differ from the first sketches below, the section "Formulas as implemented" says how.

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
- **Unpredictability has structure**: Logistics disruption is not a constant drizzle of independent small shocks. It arrives in persistent bursts, its sizes are heavy-tailed, and the system's own state changes how likely the next burst is. A model that averages this away is predicting something the world does not do.
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

As measured (v0.9, `ruin organize`): jams, the first of these structures, are everywhere. About one forms every three steps, lives nine, occupies a tenth of cell-steps, and carries over half the loss. The hidden regime does not change typical jams but fattens the tail: ruin comes from one system-wide gridlock averaging over half the grid, not from many local jams adding up. Q dots pile up in jams and queue beside them; they do not funnel past them. Corridors and persistent pressure zones do not form in this model, because shocks land uniformly and trips are monotone, so they are left for a model with real topology. Where Q dots can steer (`processes.avoidance`), they get stuck less, but still open no routes.

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

### Regime of D

The D-state label says what the pressures *look like*. The **regime** says what D is *doing*, and it is what generates disruption. D is in one of four ordered regimes (`STABLE`, `STRESSED`, `DISRUPTED`, `CHAOTIC`), it moves one level at a time, and it stays where it is for a while, so disruption comes in persistent clusters. Each regime changes how often shocks arrive, how large they are, and how noisy travel is.

The regime is **hidden**. What an observer sees is its footprint in the pressures, which is why the D-state label and the regime can disagree: a stressed regime is often read as stable, and a crisis shows up in the pressures only after the field has had time to build. This makes remaining destiny an inference problem: how much destiny a shift has left depends on which regime it is in, and that is never known exactly. `ruin destiny --blind` measures what knowing the regime is worth.

The regime and the pressures **co-evolve**. Late deliveries raise chaos pressure; chaos pressure pulls the regime toward worse levels; a worse regime brings more and larger shocks; healthy flow (order pressure) resists escalation and speeds recovery. Where Q dots are stalled, disruption also lingers longer, so jams persist and the disruption field gets close to critical under load, though it can never amplify itself.

This is a hypothesis about how logistics systems behave, not an established fact, and it is tested rather than assumed: `ruin facts` checks that a scenario reproduces eight qualitative regularities (clustering, persistence, heavy-tailed shifts and trips, good days and bad days, the flaw of averages, partial observability, and co-evolution), and `ruin structure` shows how much a ruin probability depends on the shape of disruption at equal mean.

### Instinct (`Inst`)

`Inst` is the felt sense that today is not the usual day. It is computed as how far what is happening now sits from the routine, the typical value at this point of the shift over the days already lived, in units of how much those days normally differ. It is a cheap, model-free stand-in for what the observer computes exactly, and it is the human-facing reading of `D`. See [Destiny and the Instinct](#destiny-and-the-instinct).

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

### Formulas as implemented (v0.7)

The scaffolding above has been kept in spirit and made bounded and measurable:

```text
blocked(cell, t)           = min(disruption_intensity + base_risk_intensity, 5) / 5      in [0, 1]
congestion_memory(cell, t) = m * congestion_memory(cell, t-1)
                           + (1 - m) * stalled_qdots(cell, t) / qdots(cell, t)           in [0, 1]
local_chaos_pressure       = blocked + congestion_memory + late_qdots(cell) / qdots(cell)
global_chaos_pressure      = mean(local_chaos_pressure) + late_ratio
local_order_pressure       = max(0, qdots(cell) + moving_qdots(cell) - late_qdots(cell))
global_order_pressure      = moving_qdots / active_qdots - late_qdots / active_qdots
late_ratio                 = late_qdots / active_qdots        (Q dots still travelling)
sla_miss_rate              = deadlines_missed / (deadlines_missed + on_time_deliveries)
                             over the last sla_window steps
destiny(t)                 = P(no ruin before the horizon | everything known at t)
```

The hidden regime, with `L` the level chaos pressure points to (how many D-state floors it has reached), `o` order pressure clipped to [0, 1], and `s` the shock-size multiplier of the current level:

```text
P(up   from k) = escalate[k]  * pull(k, L) * (1 - order_support * o)
P(down from k) = relax[k-1]   * pull(k, L) * (1 + order_support * o)
pull(k, L)     = pressure_pull    if the move is toward L
               = 1/pressure_pull  if the move is away from L
               = 1                if L = k   (the regime's own band)
shocks / step  ~ Poisson(base_rate * shock_rate[k])
shock size     ~ lognormal(mean = base_size * shock_size[k], sigma)        (exponential if sigma = 0)
retention(cell) = gain + congestion_persistence * congestion(cell) * (max_gain - gain)
```

- A cell is fully blocked at intensity 5, where travel speed falls to 5% of normal.
- `CHAOTIC` and `DISRUPTED` are decided by thresholds first. `RECOVERING` is a sharp fall in chaos pressure from an at-least-stressed level.
- Retention can never exceed `max_gain` < 1, so total disruption between shocks still shrinks by at least that factor each step. Persistence makes jams linger; it does not let them grow.
- A Q dot earns its completion reward when delivered inside its time window. Once late it costs its delay penalty every step (up to double in a fully blocked cell) until it is delivered late or abandoned.
- SLA ruin is `sla_miss_rate >= sla_threshold`, judged once a full window has been observed. A deadline is missed at the moment a Q dot's time window runs out, not when it is finally delivered.
- Remaining destiny is estimated by branching many independent futures from the exact present state. A shift has run out of destiny when it reaches zero.
- `C(t)` in the surplus process is planned recovery plus completion rewards; `L(t)` is the accumulated delay penalty.
- Disruption obeys the same frame as Q dots: with the default `reflect` boundary it cannot leave `D`.

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

## What May Be Different

RUIN is mostly a recombination, and it should be read that way. Ruin theory is a century old. Disruption simulation, Markov models of supplier disruption and ripple effects (for example Hosseini, Ivanov and Dolgui, 2019), stress tests with a time-to-survive metric (the MIT and Accenture supply chain stress test), and Markov-modulated risk processes in actuarial ruin theory all exist. A critical review of disruption models ([Bugert and Lasch, 2018](https://www.bvl.de/lore/all-volumes--issues/volume-11/issue-1/supply-chain-disruption-models-a-critical-review)) is the best single map of what is already covered. Only a small part of that literature has been checked, so everything below is a claim about what looks different, not a finding that nobody has done it.

Six things may be different. For each, what would show it is wrong.

1. **Ruin with two doors, in a space where the disruption and the work feed each other.** A shift is ruined when its buffer is spent or when its promise to customers breaks (the rolling miss rate), whichever comes first, in a model where late work lengthens the disruption that makes more work late. The answer is a probability with a tail and an interval, not a scenario. *Wrong if* an existing tool already gives the same answers from the same inputs.
2. **Remaining destiny, with three ways of looking.** `Δ(t)` is the chance of reaching the end of the shift without ruin, from the exact present state. Computing it for someone who knows the hidden regime, someone who watches the pressures, and someone who knows nothing prices information about the regime in ruin units: the observer recovers about a third of what a clairvoyant knows. *Wrong if* the same quantity exists under another name (risk-to-go, runtime digital-twin forecasting) and has been measured this way.
3. **Shape before mean.** Four disruption structures with the same average load give 0%, 0%, 11.8% and 15.5% ruin (`ruin structure`). A plan built on averages cannot see this. *Wrong if* real disruption turns out not to be bursty: then the structures converge and the headline result goes away. This is the most falsifiable claim and the one real data should test first.
4. **The anatomy of a ruin.** In this model ruin is one system-wide gridlock, not many local jams adding up; the worst regimes are 9% of the time, 22% of the delay and 49% of the loss, because lateness compounds. *Wrong if* it is an artefact of the uniform shocks and monotone trips (it may be; the model says so).
5. **Validation as part of the object.** Agreement with the exact Cramér-Lundberg formula, a negative control for the stylized facts, intervals on every figure, and a regression test for each flaw found. This is good practice rather than a domain contribution, but it is what makes the other claims checkable.
6. **The Instinct.** Treating the ensemble of days a logistician has already lived as a dimension of `D`, and reading trouble as distance from the routine. This is the newest idea here and the least settled; the next section develops it.

What would settle all of this, in order: (a) check on real disruption data whether the clustering that point 3 depends on is there; (b) run RUIN head to head with time-to-survive and with simple warning rules on a network with real topology; (c) show a decision that changes because of what RUIN says, which needs the policy layer.

---

## Destiny and the Instinct

This section expands the concept. It separates what has been measured from what is a hypothesis.

### The linear day

A logistics operation is repeated. The same depots, the same cut-off times, the same routes, day after day, and the work of the people running it is to make that day better: tighter routes, thinner buffers, fewer surprises. The plan is linear in two senses. Inside a day it is an ordered sequence of steps. Across days it is a sequence converging on a fixed point, the optimised routine.

The plan is the expected day. It cannot say anything about the days that are not expected, and logistics is the business of the days that are not.

### Destiny as the remainder

**Destiny is what the linear plan cannot see: the bounded set of ways today could still go.** It is mysterious in a precise sense. It is there at every step (`Δ(t)` is defined from any state), it shapes what happens, and it is never observed directly. What can be seen is its footprint in the pressures and, afterwards, one realised day out of the many that were possible. The hidden regime makes this literal: a shift can be in a bad regime for a while before anything on the board says so.

This is why `D` is a frame and not a forecast. It does not say which day will happen. It says what kind of days can.

### The fourth dimension

`D` is drawn as a square, two dimensions of where. Add the step of the shift, and `D` is three-dimensional: a movie of one day. The fourth dimension is **which day**: the index `k` of the days that came before.

A logistician standing at step `t` of today (day `k`) is looking at a *fibre*: the many earlier days at the same step. The first three dimensions give the day as it is. The fourth gives what a day at this point usually is, and how much days differ. Today is a point in `(x, y, t)` and a position within the fibre over it. That position is what can feel wrong.

```text
routine    R_c(t)  = typical value of channel c at step t, over the days lived
spread     S_c(t)  = how much that value normally differs between days
deviation  z_c(t)  = (x_c(t) - R_c(t)) / S_c(t)
Inst(t)    = mean over the watched channels c of z_c(t)
```

`Inst` accumulates if wanted (an exponentially weighted average of `Inst(t)`), which is closer to a feeling that builds than to a sudden alarm.

In the engine, this fourth dimension is not new machinery. Monte Carlo and `fork` already produce the fibre of days over a state; "4D" is the perspective that makes them the thing a person stands inside. In the current model days are independent draws, so `k` carries no memory yet.

### Where `Inst` sits

| | Needs | Answers |
| --- | --- | --- |
| `Δ(t)`, remaining destiny | the model and the exact state | how much destiny is left |
| The observer | the model of the regimes, fitted on labelled history | which regime D is probably in |
| `Inst` | only the days already lived | how unusual today is, here |

`Inst` is what the observer would be if it had no model and only habit. It is weaker in principle and much cheaper, and it is how a person with experience of the operation can sense that something is off before they can say what. The observer is that instinct made exact.

### A first exploratory test

`experiments/instinct.py` simulates 1200 days of the reference scenario. The first 600 are the days lived and define the routine; the 600 after are new days (100 of them ruin). Every signal is allowed a false alarm on 10% of the shifts that do not ruin. The board is the surplus drawdown and the SLA miss rate, the things the ruin rule is built from.

| Signal | AUC (ruin within 40 steps) | Ruined shifts flagged | Median warning (steps before ruin) | False alarms |
| --- | --- | --- | --- | --- |
| The board | 0.960 | 100 of 100 | 22 | 0.10 |
| Late ratio, raw | 0.971 | 96 of 100 | 22 | 0.12 |
| Chaos pressure, raw | 0.979 | 89 of 100 | 49 | 0.09 |
| `Inst` on late ratio alone | 0.959 | 79 of 100 | 23 | 0.12 |
| `Inst` on late, field, chaos | 0.990 | 95 of 100 | 49 | 0.09 |
| The same, building up | 0.993 | 97 of 100 | 50 | 0.09 |
| The same with no routine, one overall scale | 0.990 | 94 of 100 | 43 | 0.08 |
| That, building up | 0.992 | 96 of 100 | 42 | 0.08 |

What this shows, and what it does not:

- **Watching what the board does not show gives about twice the warning.** At the same false-alarm rate a person who watches the pressure and the field flags a ruining shift about 50 steps ahead, against 22 for the board. This is partly by construction: the board is built from the breach condition, so it can only rise once lateness has piled up. It shows that waiting for the board is late, not that `Inst` forecasts well on a real operation.
- **Most of that comes from watching several channels, not from the routine.** Raw chaos pressure alone already warns 49 steps ahead. The learned routine adds a small amount on top (warning 49 to 50 steps against 42 to 43 without it, and 95 to 97 shifts flagged against 94 to 96). The direction is favourable; the difference was not tested for significance and rests on 100 ruined shifts, so it should be treated as within noise.
- **The routine has little to say here because this model has no daily rhythm.** After the first steps, the typical late ratio, field and chaos pressure are nearly flat over a shift (chaos pressure goes from 0.22 at step 30 to 0.24 at step 400). "What is normal for this time of day" is almost the same at every time of day. Real operations have a strong rhythm, with the morning wave, the cut-off and the evening return, and that is where the routine should earn its place. That is a hypothesis; it is not tested.

### The optimisation paradox

This one is a prediction from theory, not a measurement. Optimising a repeated operation does two things at once.

It makes days more alike, so the routine tightens (`S` falls) and `Inst` gets more sensitive to a departure. But it also pushes utilisation toward its limit and thins the buffer. In the classical model the ruin probability is `ψ(u) = ρ · exp(−(1−ρ) u / μ)`, where `ρ = λμ / c` is the load and `u` the buffer. Optimisation drives `ρ` toward 1, where the exponent goes to zero and `ψ` goes to 1, and it shrinks `u`. The optimised day looks calmer and sits closer to the edge.

So the instinct is trained on the calmest days there are, and it senses *departure from the routine*. It cannot sense *how thin the margin under the routine has become*. That is the job of `ψ(u)` and its sensitivity (`ruin sensitivity`). The two are complementary: instinct reads the day, ruin probability reads the margin. Whether this is what happens in real operations is untested.

### What is and is not claimed

- `Inst` is a computable stand-in. It is not a model of human cognition, and no claim is made that logisticians have such an instinct or that it can be measured. The name is a handle for the idea.
- The numbers above come from one illustrative scenario with no real data and no daily rhythm.
- The fourth dimension is a way of seeing the Monte Carlo ensemble, not a new object in the engine.

### Next steps this opens

- **Rhythm**: give demand and disruption a time-of-shift profile, then repeat the test, and test the routine's contribution for significance.
- **Memory across days**: let backlog, fatigue and learning carry over, so that `k` has structure and the fibre is not just independent draws.
- **A surprise map**: `Inst` per cell of the Probability Square, drawing where on the grid the day feels wrong.
- **Instinct and observer together**: whether `Inst` adds anything to the calibrated posterior, or only matches it.
- **Real logs**: the only place the claim "people sense trouble from deviation from the routine" can be checked.

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

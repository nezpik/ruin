# RUIN: Ruin-probability Unified In Networks

**Destiny-framed ruin risk for physical logistics flow.**

[![Python](https://img.shields.io/badge/Python-3.11--3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![v0.1 Concept MVP](https://img.shields.io/badge/Version-v0.1%20MVP-orange.svg)](https://github.com/nezpik/ruin)

RUIN is a research-grade framework for seeing when physical logistics systems run out of destiny.

It uses quantitative finance ideas like ruin theory, jump diffusion, Monte Carlo paths, Value at Risk, and Expected Shortfall, but it is not just a prediction engine. RUIN frames logistics uncertainty as motion inside a bounded destiny field `D`, where **Q dots** try to create order inside chaotic physical flow.

---

## Core Idea

Most logistics models ask:

> Can we predict the route, delay, or cost?

RUIN asks something different:

> When does physical flow lose the ability to preserve order?

In RUIN:

- **`D`** is the destiny frame: the bounded field where logistics futures unfold.
- **Q dots** are delivery commitments, route obligations, flow units, or Monte Carlo futures moving inside `D`.
- **Q dots cannot leave `D`**. They can drift, jump, cluster, stall, recover, fail, or ruin, but they remain destiny-bound.
- **Disruption can be an event or the state of the grid itself**.
- **Ruin** occurs when system surplus is exhausted or service-level failure crosses a barrier.

RUIN does not try to smooth chaos away. It gives chaos a frame so it can move, organize, propagate, and collapse visibly.

---

## v0.1 Use Case

### Urban Ruin Shift

RUIN v0.1 models one abstract last-mile delivery shift on a rectangular Probability Square.

The MVP does **not** use real maps, live traffic APIs, routing optimization, reinforcement learning, or customer data. It focuses on the smallest meaningful RUIN world:

1. Q dots spawn inside `D`.
2. They move through the Probability Square.
3. Disruptions appear as cell-level field intensity.
4. Disruptions propagate, decay, and recover.
5. Q dots experience field exposure and lose slack.
6. Q-dot density, delay, and failure reshape `D`.
7. The grid moves through D-states.
8. The surplus process tracks whether the system ruins.

---

## How It Works

RUIN v0.1 is built around a simple simulation loop.

### 1. Initialize `D`

`D` is represented by the `ProbabilitySquare`:

- width
- height
- time
- Q dots
- disruption field
- current D-state
- chaos pressure
- order pressure

The grid is not a map yet. It is a controlled physical-risk field.

### 2. Spawn Q dots

Q dots are typed commitments:

- `STANDARD`: balanced movement and penalty
- `EXPRESS`: tighter time window, higher SLA pressure
- `BULKY`: slower movement, higher penalty

Each Q dot carries:

- position inside `D`
- time-window slack
- delay penalty
- field exposure
- D-state exposure
- order pressure
- local ruined/late state

### 3. Inject disruption

The `DisruptionField` stores physical-flow pressure across cells.

Disruptions can represent:

- congestion waves
- accidents
- weather cells
- failed handoffs
- demand bursts
- local bottlenecks

Each shock has intensity, memory, propagation, decay, and recovery.

### 4. Compute pressure

RUIN tracks two opposing forces.

**Chaos pressure** measures disruption, congestion memory, and failed Q-dot pressure.

```text
local_chaos_pressure(cell, t)
  = disruption_intensity(cell, t)
  + congestion_memory(cell, t)
  + local_failed_qdot_ratio(cell, t)
```

**Order pressure** measures the attempt of Q dots to preserve service, flow, and completion.

```text
local_order_pressure(cell, t)
  = active_qdots(cell, t)
  + moving_qdots(cell, t)
  - failed_qdots(cell, t)
```

### 5. Update D-state

The Probability Square moves through regimes:

| D-state | Meaning |
| --- | --- |
| `STABLE` | Q dots move with low field pressure and surplus is healthy |
| `STRESSED` | pressure rises but flow remains coherent |
| `DISRUPTED` | disruption becomes dominant in part of the grid |
| `CHAOTIC` | clusters, stalls, forced paths, and unstable flow appear |
| `RECOVERING` | field intensity decays and order begins to return |
| `RUINED` | surplus or SLA barrier has been breached |

### 6. Move Q dots inside destiny

Each Q dot samples movement from the stochastic process, but movement remains bounded inside `D`.

Field exposure slows Q dots, consumes slack, and increases failure pressure.

### 7. Update surplus and ruin

RUIN uses a surplus process:

```text
U(t) = U(0) + C(t) - L(t)
```

Ruin occurs if:

```text
U(t) <= barrier
```

or if:

```text
late_qdots / active_qdots >= sla_threshold
```

---

## Install

Clone the repo:

```bash
git clone https://github.com/nezpik/ruin.git
cd ruin
```

Install editable:

```bash
python -m pip install -e .
```

RUIN v0.1 is currently dependency-free and runs with the Python standard library.

Recommended implementation target is Python 3.11 or 3.12. The current package metadata allows Python 3.11 through 3.14 while the MVP remains dependency-free.

---

## Run One Urban Ruin Shift

```bash
python -m ruin.cli simulate --config examples/urban_ruin_shift.yaml --max-steps 80 --frames 8 --output ruin_frames.txt
```

Example output:

```json
{
  "ruined": true,
  "ruin_time": 60,
  "ruin_reason": "surplus",
  "final_surplus": -19.936585,
  "final_d_state": "RUINED",
  "frames_written": "ruin_frames.txt"
}
```

The frame file shows sampled Probability Square states:

```text
t=1 D=STABLE chaos=0.0 order=0.222222 penalty=0.0
t=43 D=STABLE chaos=0.064419 order=0.222222 penalty=0.0
t=60 D=CHAOTIC chaos=0.810872 order=-0.022222 penalty=71.09539
```

---

## Run Monte Carlo Risk

```bash
python -m ruin.cli risk --config examples/urban_ruin_shift.yaml --paths 10 --max-steps 80
```

The risk command reports:

- ruin probability
- mean loss
- Value at Risk
- Expected Shortfall
- time-to-ruin samples
- mean time to ruin

---

## Current Package Layout

```text
ruin/
├── core/
│   ├── qdot.py
│   ├── network.py
│   └── processes.py
├── metrics/
│   └── pressure.py
├── state_space/
│   ├── probability_square.py
│   └── disruption_field.py
├── simulation/
│   └── agent_based.py
├── risk/
│   └── ruin_probability.py
├── viz/
│   └── square.py
├── config.py
└── cli.py
```

---

## What v0.1 Is

RUIN v0.1 is:

- a concept-first logistics ruin-risk MVP
- an abstract last-mile physical-flow simulator
- a destiny-frame model for Q dots
- a disruption propagation field
- a surplus and SLA ruin model
- a Monte Carlo tail-risk tool

## What v0.1 Is Not Yet

RUIN v0.1 is not yet:

- a real map or GIS system
- a vehicle routing optimizer
- a live traffic system
- a reinforcement learning environment
- a calibrated production logistics model
- a full visualization engine

---

## Roadmap

Near-term:

- add tests
- add real YAML parsing with `pyyaml`
- add NumPy-backed simulation paths
- add Matplotlib or Plotly animation
- add richer chaotic organization detection
- add package publishing metadata

Later:

- real road graphs
- routing policies
- depot and fleet constraints
- calibration from real logistics data
- hedging and backup-capacity policies
- live disruption feeds

---

## Academic Credit and Citation

RUIN was created by **Naji Zouiti**.

If you use RUIN, its concepts, terminology, diagrams, code, or experimental outputs in academic work, research prototypes, papers, theses, presentations, or derivative projects, please give clear credit to the original project and author.

Suggested citation:

```text
Zouiti, Naji. RUIN: Ruin-probability Unified In Networks. v0.1, 2026.
GitHub: https://github.com/nezpik/ruin
```

BibTeX:

```bibtex
@software{zouiti_ruin_2026,
  author = {Zouiti, Naji},
  title = {RUIN: Ruin-probability Unified In Networks},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/nezpik/ruin}
}
```

When discussing the framework academically, please preserve the core attribution that RUIN introduces a destiny-framed approach to logistics uncertainty using `D`, Q dots, chaotic organization, disruption propagation, and ruin probability.

---

## Philosophy

RUIN is built around one sentence:

> A physical system ruins when it runs out of destiny.

The Probability Square is not decoration. It is a way to watch order and chaos negotiate inside `D` until the system survives, recovers, or collapses.

# RUIN: Ruin-probability Unified In Networks

**Destiny-framed ruin risk for stochastic logistics flow.**

[![Python](https://img.shields.io/badge/Python-3.11--3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![v0.2 Foundation](https://img.shields.io/badge/Version-v0.2%20Foundation-green.svg)](https://github.com/nezpik/ruin)

RUIN is a research-grade framework for measuring when physical logistics systems run out of destiny. It translates quantitative finance concepts (ruin theory, jump-diffusion, Monte Carlo paths, VaR, Expected Shortfall, bootstrap confidence intervals) into a logistics language built around **physical uncertainty**, **service failure**, and **systemic collapse**.

The central object is the **Probability Square** — an abstract two-dimensional **destiny frame `D`** inside which many **Q dots** (delivery commitments, route obligations, flow units) move through possible futures.

RUIN does not try to predict the future perfectly. It gives chaos a bounded frame so that uncertainty can move, organize, propagate, and become measurable.

---

## Core Philosophy

> A physical system ruins when it runs out of destiny.

In RUIN:
- **`D`** (the destiny frame) bounds all possible futures. Q dots cannot leave `D`.
- **Q dots** are quantized attempts at order — they drift, jump, cluster, stall, recover, fail, or ruin inside the frame.
- **Disruption** is spatial and propagating: a field that intensifies, spreads, decays, and reshapes movement.
- **Ruin** occurs when operational surplus is exhausted or service-level barriers are breached.
- **Visualization** is analysis: the Probability Square makes the negotiation between order and chaos visible.

---

## v0.2 Highlights (Foundation Hardening)

RUIN v0.2 has moved from pure-stdlib concept to a credible, usable research instrument:

- **Vectorized stochastic processes** — NumPy-backed GBM, compound Poisson jumps, and shock arrivals for speed and reproducibility
- **Vectorized QDot batch travel** — hot loop replaced with `batch_qdot_travel_step()` for significant acceleration
- **Real disruption field storage** — full 30×30 field grids captured in snapshots (no more synthetic noise)
- **High-quality Matplotlib GIF visualization** — animated Probability Square + real propagating disruption field with Q-dot movement and ruin dynamics (`--visualize`)
- **Parallel Monte Carlo** — `ProcessPoolExecutor` with automatic core detection (`--jobs`)
- **Statistically credible risk analysis** — bootstrap confidence intervals for ruin probability + loss standard deviation
- **Typed configuration foundation** — Pydantic + NumPy/Matplotlib runtime dependencies
- **Comprehensive test suite** — 19+ passing tests guarding golden behavior

All core mechanics, D-state transitions, surplus ruin rules, and the original "Urban Ruin Shift" example remain faithful to the v0.1 vision.

See [CHANGELOG.md](CHANGELOG.md) for the full detailed list of v0.2 Foundation Hardening changes.

---

## Quickstart

### Install (with uv — recommended)

```bash
git clone https://github.com/nezpik/ruin.git
cd ruin
uv sync --dev
```

Alternatively (classic pip):

```bash
python -m pip install -e ".[test]"
```

Requires Python ≥ 3.11. Core dependencies: `numpy`, `pydantic`, `matplotlib`.

### Run One Visible Trajectory + GIF

```bash
uv run ruin simulate --config examples/urban_ruin_shift.yaml --max-steps 40 --visualize --output /tmp/ruin_shift.txt
```

This produces:
- Text frames (`/tmp/ruin_shift.txt`)
- Real animated GIF (`/tmp/ruin_shift.gif`) showing the actual disruption field propagating, Q-dots moving and ruining, D-state evolution, and pressure metrics

### Run Monte Carlo Risk Analysis (Parallel)

```bash
uv run ruin risk --config examples/urban_ruin_shift.yaml --paths 200 --max-steps 50 --jobs 4
```

Example output (truncated):

```json
{
  "paths": 200,
  "ruin_probability": 1.0,
  "ruin_probability_ci": [1.0, 1.0],
  "mean_loss": 265.0153,
  "loss_std": 82.4,
  "value_at_risk": 340.2,
  "expected_shortfall": 352.1,
  "mean_time_to_ruin": 21.8,
  "n_jobs": 4
}
```

The `--jobs` flag controls parallelism (defaults to available cores).

---

## AI Research Reports (optional)

RUIN can narrate a finished `simulate`/`risk` run into a markdown research
report — written in RUIN's own vocabulary (`D`, Q dots, D-state, surplus,
ruin) — using [OpenAI Codex](https://github.com/openai/codex) as a pure
text-generation engine. RUIN reads and summarizes the results itself,
builds the full prompt, and writes the report; Codex never touches the
filesystem (it runs in a read-only sandbox).

Install the optional `ai` extra:

```bash
python -m pip install "ruin[ai]"
```

This requires an authenticated local Codex session (`codex login`, or an
API key via `codex login --api-key`) — the SDK reuses your existing Codex
CLI credentials.

Then pass `--explain PATH` to either subcommand:

```bash
uv run ruin simulate --config examples/urban_ruin_shift.yaml --max-steps 40 --explain /tmp/trajectory_report.md
uv run ruin risk --config examples/urban_ruin_shift.yaml --paths 200 --max-steps 50 --explain /tmp/risk_report.md
```

**On reproducibility**: RUIN's simulations are seeded and deterministic, but
Codex's prose is not — the SDK exposes no seed/temperature control, so the
same `--explain` run can read differently each time. Rather than hide that,
each report brackets the narrative with two things RUIN keeps fully
deterministic and traceable instead:

- a **header** recording exactly which Codex turn produced it — requested
  model/effort, duration, token usage, and the thread/turn IDs — so the call
  itself is auditable even though its prose can't be regenerated; and
- an **appendix** echoing, verbatim, the exact compressed `result`/`config`
  summary Codex was given — generated by RUIN itself, not Codex, and
  reproducible byte-for-byte by re-running the same scenario and seed — so
  every claim in the narrative can be checked against the real numbers
  without leaving the document.

Add `--explain-effort LEVEL` (one of `none`, `minimal`, `low`, `medium`,
`high`, `xhigh`) to trade narrative depth for speed/cost — it's forwarded
straight through to Codex's reasoning-effort setting for that turn:

```bash
uv run ruin simulate --config examples/urban_ruin_shift.yaml --max-steps 40 \
    --explain /tmp/trajectory_report.md --explain-effort low
```

Omit it to use Codex's own default.

---

## Core Vocabulary (Preserved from v0.1)

- **Destiny frame `D`**: The bounded space in which all logistics futures unfold.
- **Probability Square**: The computational and visual representation of `D` (grid + field + state).
- **Q dot**: A quantized logistics commitment carrying time-window, penalty, drift/volatility multipliers, and field exposure.
- **Disruption Propagation Field**: Spatial memory of shocks that intensifies, propagates, decays, and recovers.
- **D-state**: Regime of the grid (`STABLE` → `STRESSED` → `DISRUPTED` → `CHAOTIC` → `RECOVERING` → `RUINED`).
- **Surplus process**: `U(t) = U(0) + C(t) - L(t)`. Ruin when `U(t) <= barrier` or SLA failure crosses threshold.

Full definitions and deeper philosophy are in `RUIN.md`.

---

## Current Architecture (v0.2)

```
ruin/
├── core/
│   ├── qdot.py              # Scalar QDot model (still used for state)
│   ├── network.py           # Surplus & SLA tracking
│   └── processes.py         # Vectorized GBM, jumps, shocks, batch travel
├── state_space/
│   ├── probability_square.py   # Main engine (now heavily vectorized)
│   └── disruption_field.py
├── simulation/
│   └── agent_based.py
├── risk/
│   └── ruin_probability.py     # Parallel MC + bootstrap CI + VaR/ES
├── viz/
│   └── square.py               # Matplotlib FuncAnimation + real-field GIFs
├── ai/
│   └── narrator.py             # Optional Codex-powered AI research reports (--explain)
├── metrics/
│   └── pressure.py
├── config.py
└── cli.py
```

---

## What v0.2 Is

- A **research instrument** for studying logistics ruin under bounded destiny
- A **visual + quantitative** tool for disruption propagation and order pressure
- A **fast, parallel, statistically grounded** Monte Carlo engine
- An **abstract but extensible** foundation (no real maps yet — by design)

---

## What v0.2 Is Not (Yet)

- A GIS / real-road simulator
- A vehicle routing optimizer
- A calibrated production forecasting tool
- A reinforcement-learning environment

These are explicitly deferred until the abstract mechanics are solid and reproducible.

---

## Roadmap (Post-v0.2)

**Foundation complete.** The following are now production-grade in v0.2:
- Typed Pydantic configuration with direct attribute access
- Vectorized NumPy primitives + batch Q-dot movement
- Real-field visualization with animated GIFs
- Parallel Monte Carlo with bootstrap confidence intervals

**Post-review fixes (Devin PR #1):** Critical regressions in pressure calculations (`active_by_cell`) and `qdot_exposure_multiplier` handling were identified during external review and have been corrected with dedicated regression tests.

**Next Foundation items (suggested order):**
- Performance benchmarking harness + profiling
- Structured logging + better error messages / validation UX
- Snapshot export (Parquet / JSONL) for external analysis
- Property-based + mutation testing expansion

**Later (when the abstract core is trusted):**
- Real road graphs and spatial topologies
- Fleet/depot constraints
- Calibration against real logistics data
- Hedging and backup-capacity policies

---

## Run the Test Suite

```bash
pytest -q
```

All tests must pass before any research use of a commit.

---

## Academic Credit and Citation

RUIN was created by **Naji Zouiti**.

If you use RUIN, its concepts, terminology, code, or experimental outputs in academic work, please cite the project.

Suggested citation:

> Zouiti, Naji. RUIN: Ruin-probability Unified In Networks. v0.2 Foundation, 2026.  
> GitHub: https://github.com/nezpik/ruin

BibTeX:

```bibtex
@software{zouiti_ruin_2026,
  author = {Zouiti, Naji},
  title = {RUIN: Ruin-probability Unified In Networks},
  year = {2026},
  version = {0.2.0-dev},
  url = {https://github.com/nezpik/ruin}
}
```

---

## Philosophy (Closing)

RUIN does not smooth chaos away. It gives chaos a frame (`D`) so that the negotiation between order pressure and disruption can be watched, measured, and understood — until the system either preserves its destiny or runs out of it.

The Probability Square is not decoration. It is the laboratory in which we study when physical flow loses the ability to keep its promises.

---

*Maintained by Naji Zouiti • MIT License • 2026*
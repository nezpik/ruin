# RUIN: Ruin-probability Unified In Networks

**Destiny-framed ruin risk for stochastic logistics flow.**

[![Python](https://img.shields.io/badge/Python-3.11--3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![v0.7.0](https://img.shields.io/badge/Version-v0.7.0-green.svg)](https://github.com/nezpik/ruin)

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

## How the Model Works

- **Q dots travel and can succeed.** Each Q dot enters `D` with an origin, a destination, and a time window. Delivered on time, it earns surplus. Once late, it drains surplus every step until it is delivered late or abandoned.
- **Demand keeps arriving.** New Q dots enter every step, so the shift is a flow, not a fixed population running down a clock.
- **Travel is a jump-diffusion.** Progress per step is drift slowed by local field drag, plus Gaussian noise, minus occasional compound Poisson setbacks.
- **D has a hidden regime.** D is stable, stressed, disrupted, or chaotic, and the regime is what generates disruption: in persistent bursts, with heavy-tailed sizes. You see its footprint in the pressures, never the regime itself. See [A Regime-Driven D](#a-regime-driven-d).
- **Disruption is a field of events.** Shocks land on the grid, spread to neighbouring cells, and fade. The field is subcritical by construction: a shock can never amplify itself, even where jams make it linger. By default it reflects at the edges: disruption, like Q dots, cannot leave `D`.
- **Q and D co-evolve.** The field slows Q dots; late and failed Q dots add disruption where they stand and push the regime toward worse levels; healthy flow resists escalation.
- **Ruin is uncertain.** With no disruption the reference shift never ruins. In the reference scenario about one shift in six does, at very different times, most by exhausting the surplus and some by breaching the SLA.

Everything is seeded and reproducible, Monte Carlo runs in parallel, and a 480-step path takes about 0.2 s.

See [CHANGELOG.md](CHANGELOG.md) for what changed in each version and how to migrate a config.

---

## A Regime-Driven D

Most simulators draw disruption as a constant drizzle of independent small shocks. Averaged over a shift that is almost deterministic, and it is not how logistics behaves: disruption arrives in bursts, a bad stretch is followed by another, and the system's own state changes how likely the next burst is. RUIN models `D` accordingly.

`D` is in a **regime**: stable, stressed, disrupted, or chaotic. The regime moves one level at a time and stays where it is for a while. Each level changes how often shocks arrive, how big they are, and how noisy travel is. The regime is **hidden**: what you observe is its footprint in the pressures (the D-state label), which is why the two disagree about a quarter of the time. In the reference scenario the D-state label matches the hidden regime exactly 73% of the time and is within one level 97% of the time.

The regime and the pressures **co-evolve**. Late deliveries raise chaos pressure, chaos pressure pulls the regime toward worse levels, a worse regime brings more and larger shocks, and healthy flow resists escalation and speeds recovery. Where Q dots are stalled, disruption also lingers, so jams persist, but retention is capped below 1 and the field still cannot amplify itself.

All of this is opt-in (`probability_square.regime.enabled`). With it off, disruption is memoryless and the model is exactly what it was in v0.6; `examples/urban_ruin_shift_thin.yaml` is that version of the scenario.

### Does it behave like logistics?

There is no data to calibrate against yet, so RUIN checks the model against qualitative regularities that logistics operations are widely reported to show. These are properties a model should have, not proof that it is right; I have not checked sources for them here.

```bash
uv run ruin facts --config examples/urban_ruin_shift.yaml
```

| Fact | Scenario | Memoryless version | Needed |
| --- | --- | --- | --- |
| Disruption clusters (Fano factor of events per 24 steps) | 3.41 | 1.0 | ≥ 1.5 (Poisson is 1) |
| Bad stretches persist (correlation of adjacent 12-step windows) | 0.48 | 0.01 | ≥ 0.15 |
| Shift loss is heavy-tailed (95th over 50th percentile) | 8.54 | 1.91 | ≥ 3 and ≥ 1.5× memoryless |
| Trip times are heavy-tailed (99th over 50th percentile) | 2.6 | 2.44 | ≥ 2 |
| Good days and bad days (variation of delayed-trip share across shifts) | 0.57 | 0.19 | ≥ 2× memoryless |
| Flaw of averages (mean loss over memoryless at equal mean disruption) | 1.82× | 1× | ≥ 1.1× |
| D-state follows the hidden regime without revealing it (correlation / exact match) | 0.65 / 0.73 | n/a | ≥ 0.5, exact < 0.9 |
| Pressure pushes the regime (escalation ratio when pressure points higher) | 1.55× | n/a | ≥ 1.3× |

The memoryless version of the same scenario passes one of the six facts that apply to it (trip tails, which it has by nature) and fails the other five, so the suite can fail.

### How much does the answer depend on the shape of disruption?

```bash
uv run ruin structure --config examples/urban_ruin_shift.yaml
```

runs the scenario under four structures that deliver the **same realised mean disruption** (148 units of shock mass per step, measured over whole shifts):

| Structure (same realised mean disruption) | Ruin probability | 95% interval | Mean loss | Expected shortfall |
| --- | --- | --- | --- | --- |
| Memoryless, exponential sizes | 0.0% | 0.0% to 0.9% | 84 | 194 |
| Memoryless, heavy-tailed sizes | 0.0% | 0.0% to 0.9% | 85 | 221 |
| Hidden regime, exponential sizes | 11.8% | 8.9% to 15.3% | 147 | 943 |
| Hidden regime, heavy tails, lingering jams (the scenario) | 15.5% | 12.3% to 19.4% | 183 | 1267 |

Two things stand out:

- **At the same average disruption, the memoryless world never ruins.** Ruin in this scenario is a property of crises, not of the average level of disruption. If real disruption were not clustered, this operation would be safe.
- **Heavy tails alone changed almost nothing for ruin** (a mild rise in the tail loss). A shock is already large enough to block a cell for a long time, so making the biggest ones bigger adds little; burstiness is what matters here.

A ruin probability should therefore be read as a statement about a structure, not as a number. The structural spread across these four, 0% to 16%, is larger than the Monte Carlo interval around any one of them.

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

Requires Python ≥ 3.11. Core dependencies: `numpy`, `pydantic`, `matplotlib`, `pyyaml`.

### Run One Visible Trajectory + GIF

```bash
uv run ruin simulate --config examples/urban_ruin_shift.yaml --max-steps 150 --visualize --output /tmp/ruin_shift.txt
```

This produces:
- Text frames (`/tmp/ruin_shift.txt`)
- Real animated GIF (`/tmp/ruin_shift.gif`) showing the disruption field propagating, Q dots travelling to their destinations (magenta when late, green on delivery, red on failure), the D-state, and the pressures

### Run Monte Carlo Risk Analysis (Parallel)

```bash
uv run ruin risk --config examples/urban_ruin_shift.yaml --jobs 4
```

Output for the 1,000 paths the config asks for (`time_to_ruin` list omitted):

```json
{
  "paths": 1000,
  "ruin_probability": 0.16,
  "ruin_probability_ci": [0.1386, 0.184],
  "ruin_by_surplus": 131,
  "ruin_by_sla": 29,
  "mean_loss": 188.469,
  "mean_loss_ci": [169.8396, 207.7931],
  "loss_std": 307.7357,
  "value_at_risk": 837.0866,
  "value_at_risk_ci": [651.9906, 943.6986],
  "expected_shortfall": 1260.2749,
  "expected_shortfall_ci": [1089.8598, 1430.3121],
  "loss_share_by_type": {"STANDARD": 0.2204, "EXPRESS": 0.7469, "BULKY": 0.0327},
  "expected_shortfall_by_type": {"STANDARD": 359.2645, "EXPRESS": 834.5985, "BULKY": 66.4119},
  "share_of_qdots_with_loss": 0.008035,
  "loss_share_of_top_1pct_qdots": 0.9639,
  "mean_regime_occupancy": {"STABLE": 0.7454, "STRESSED": 0.1645, "DISRUPTED": 0.0672, "CHAOTIC": 0.023},
  "mean_final_surplus": 198.6578,
  "mean_min_surplus": 30.5528,
  "mean_on_time_rate": 0.9917,
  "mean_time_to_ruin": 243.71,
  "confidence_level": 0.95,
  "n_jobs": 4
}
```

How to read it:

- Every `*_ci` is a confidence interval at `confidence_level`. The ruin probability uses a Wilson interval, which stays informative even when no path ruins. The loss figures use a bootstrap.
- Each path runs to the end of the shift even after it ruins, so the loss figures describe whole shifts and are not cut off at the moment of ruin.
- `ruin_by_surplus` / `ruin_by_sla` say which barrier was hit first. The SLA is breached when the share of deadlines missed over the last `sla_window` steps reaches `sla_threshold`.
- The loss is heavy-tailed: the expected shortfall is about seven times the mean loss, which is what bad days look like.
- `mean_regime_occupancy` is the share of time D spent in each hidden regime.
- `expected_shortfall_by_type` splits the Expected Shortfall among Q-dot types; the parts add up to the total.
- `share_of_qdots_with_loss` and `loss_share_of_top_1pct_qdots` say how concentrated the loss is: under 1% of Q dots carry essentially all of it.

The `--jobs` flag controls parallelism (defaults to available cores).

---

## Remaining Destiny

`ruin risk` answers "how often does a shift like this ruin?". `ruin destiny` answers the question the project is named for: *how much destiny does this shift have left, right now?*

```bash
uv run ruin destiny --config examples/urban_ruin_shift.yaml --seed 81 --blind --plot destiny.png
```

It follows one shift. Every 24 steps it freezes the exact present (every Q dot, every cell of the disruption field, the surplus, the SLA window, the hidden regime) and plays 64 independent futures forward from it. The share that reach the end of the shift without ruin is the shift's **remaining destiny** at that moment:

```text
destiny(t) = P(no ruin before the horizon | everything known at t)
```

![Remaining destiny of one shift, with the hidden regime](docs/destiny_regime_seed81.png)

This shift ran calm for over 200 steps at about 90% destiny. Then the regime escalated, from stressed to disrupted to chaotic, and the destiny fell to 56% at step 264 and 30% at step 288, then to nothing, and the shift ruined at step 339 by breaching the SLA. Look at the bottom panel: **surplus was at its highest (245) when the crisis began**. A surplus chart sees nothing wrong until it is too late; the buffer the shift would have needed for a 95% chance of safety jumped from about 250 to over 1,100.

The `--blind` curve is destiny for someone who knows everything about D except which regime it is in (the regime is redrawn from its long-run mix). At step 264 it still says 89%, against 56% for someone who knows the regime, and at step 288 it says 83% against 30%. The gap is what knowing the regime is worth. An observer who sees the pressures sits between the two curves: in this shift the D-state label read CHAOTIC at step 264, a regime ahead of the truth.

At `t = 0` the remaining destiny is one minus the ruin probability, so the two commands cross-check each other.

---

## Who Carried the Loss

Every Q dot has a ledger entry: where it entered, where it was going, how it ended, and how much loss it carried. `ruin simulate` reports the largest contributors and how concentrated the loss was:

```bash
uv run ruin simulate --config examples/urban_ruin_shift.yaml --seed 81
```

The full ledger is `ProbabilitySquare.ledger()`; a trajectory result carries `top_contributors`, `loss_concentration`, and `loss_field` (loss accrued per cell). Set `visualization.heatmap_field: accumulated_loss` to watch the loss map build up in the GIF.

---

## Parameter Sensitivity and Uncertainty

A confidence interval says how well the simulation has converged for one exact set of parameters. It says nothing about whether those parameters are right.

```bash
uv run ruin sensitivity --config examples/urban_ruin_shift.yaml
```

moves each parameter down and up by 10% and ranks them by effect on the odds of ruin (200 paths each):

| Parameter (moved ±10%) | Ruin at −10% | Ruin at +10% | Elasticity |
| --- | --- | --- | --- |
| `qdots.*.time_window` | 28.5% | 7.5% | -7.9 |
| `disruption_field.shock_arrival_rate` | 9.5% | 26.0% | +6.0 |
| `disruption_field.mean_initial_intensity` | 9.0% | 23.5% | +5.7 |
| `processes.travel_drift` | 26.0% | 13.0% | -4.3 |
| `qdots.*.delay_penalty` | 11.0% | 13.5% | +1.2 (noise) |
| `processes.jump_intensity` | 16.5% | 14.5% | -0.8 (noise) |
| `processes.demand_arrival_rate` | 14.5% | 16.5% | +0.8 (noise) |
| `qdots.*.completion_reward` | 13.5% | 12.5% | -0.4 (noise) |
| `ruin.initial_buffer` | 13.0% | 12.5% | -0.2 (noise) |

An elasticity of 6 means a 1% increase multiplies the odds of ruin by about 1.06. Effects inside Monte Carlo noise are flagged (`within_noise`); do not read a sign into them. Four levers matter: time windows, shock rate, shock size, and travel speed. The economics (penalties, rewards, demand) and the buffer do not move the answer by a detectable amount: a 10% bigger buffer is tiny next to the losses of a crisis.

The same ranking on the memoryless version of this scenario (v0.6) gave elasticities three to four times larger (time windows −32, travel speed −18, shock size +15, shock rate +13). That cliff was a property of memoryless disruption, not of logistics: when disruption arrives in bursts, a small change in the average moves ruin less.

```bash
uv run ruin uncertainty --config examples/urban_ruin_shift.yaml
```

goes one step further. The config lists how well each parameter is known (`risk.parameter_uncertainty`: here shock rate, shock size, and demand to 10%, travel speed to 5%, jump rate to 20%). The command draws 40 plausible parameter sets and estimates the ruin probability for each. For the reference scenario the ruin probability across draws has a median of 16.5%, a 5% to 95% range of 1% to 41%, and averages 19%, against 16% from the 1,000-path run with the parameters taken as exact. Not knowing the parameters widens the answer far more than Monte Carlo error does, but unlike in the memoryless model it does not push it toward certainty: no draw ruined more than half the time.

Two kinds of uncertainty are now visible side by side. **Parameter uncertainty** spreads the ruin probability from about 1% to 41%. **Structural uncertainty** (`ruin structure`) moves it from 0% to 16% depending on whether disruption is memoryless or clustered. Neither dominates, and both are much larger than the few points of sampling error around any single estimate.

---

## Validation Against Ruin Theory

The spatial model has no closed-form answer, but the layer underneath it does. Strip away the grid, feed the same surplus process a constant premium and compound Poisson claims with exponential sizes, and the ruin probability is known exactly (Cramér-Lundberg):

```text
psi(u) = (lam * mu / c) * exp(-(1/mu - lam/c) * u)
```

```bash
uv run ruin validate
```

compares Monte Carlo estimates with that formula and exits non-zero if any exact value falls outside its interval. With 20,000 paths, claim rate 1, mean claim 1, premium 1.5:

| Initial surplus | Exact | Monte Carlo | 99% interval |
| --- | --- | --- | --- |
| 0 | 0.6667 | 0.6699 | 0.6613 to 0.6784 |
| 2 | 0.3423 | 0.3400 | 0.3314 to 0.3487 |
| 5 | 0.1259 | 0.1236 | 0.1178 to 0.1298 |
| 10 | 0.0238 | 0.0231 | 0.0205 to 0.0259 |

This validates the surplus accounting, ruin detection, the Monte Carlo estimator, and the Wilson interval. It does not validate the spatial dynamics, which are checked by behavioural tests (no disruption means no ruin, the disruption field's long-run level matches its theoretical value, a larger buffer never ruins sooner, and so on).

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
uv run ruin simulate --config examples/urban_ruin_shift.yaml --explain /tmp/trajectory_report.md
uv run ruin risk --config examples/urban_ruin_shift.yaml --paths 200 --explain /tmp/risk_report.md
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
uv run ruin simulate --config examples/urban_ruin_shift.yaml \
    --explain /tmp/trajectory_report.md --explain-effort low
```

Omit it to use Codex's own default.

---

## Core Vocabulary (Preserved from v0.1)

- **Destiny frame `D`**: The bounded space in which all logistics futures unfold.
- **Probability Square**: The computational and visual representation of `D` (grid + field + state).
- **Q dot**: A quantized logistics commitment travelling from an origin to a destination, carrying a time window, a delay penalty, a completion reward, and field exposure.
- **Disruption Propagation Field**: Spatial memory of shocks that intensifies, propagates, decays, and recovers.
- **D-state**: Regime of the grid (`STABLE` → `STRESSED` → `DISRUPTED` → `CHAOTIC` → `RECOVERING` → `RUINED`).
- **Surplus process**: `U(t) = U(0) + C(t) - L(t)`. Ruin when `U(t) <= barrier` or SLA failure crosses threshold.

Full definitions and deeper philosophy are in `RUIN.md`.

---

## Current Architecture (v0.7.0)

```
ruin/
├── core/
│   ├── qdot.py              # Q-dot record, types, and statuses
│   ├── network.py           # Surplus & SLA tracking
│   └── processes.py         # Vectorized diffusion, jumps, shocks, batch travel
├── state_space/
│   ├── probability_square.py   # Main engine: Q dots as arrays, one step of D
│   ├── disruption_field.py     # Subcritical spread-and-decay field
│   └── regime.py               # The hidden regime of D
├── simulation/
│   └── agent_based.py
├── risk/
│   ├── ruin_probability.py     # Parallel MC, Wilson and bootstrap intervals, VaR/ES
│   ├── classical.py            # Cramér-Lundberg closed form and validation
│   ├── structure.py            # Ruin risk under mean-matched disruption structures
│   ├── stylized.py             # Stylized facts of logistics disruption
│   ├── destiny.py              # Remaining destiny: branching futures from the present
│   └── uncertainty.py          # Parameter sensitivity and uncertainty
├── viz/
│   └── square.py               # Matplotlib FuncAnimation + real-field GIFs
├── ai/
│   └── narrator.py             # Optional Codex-powered AI research reports (--explain)
├── metrics/
│   └── pressure.py
├── config.py                   # YAML loading and validation
├── config_models.py            # Pydantic schema
└── cli.py
```

---

## What RUIN Is

- A **research instrument** for studying logistics ruin under bounded destiny
- A **visual + quantitative** tool for disruption propagation and order pressure
- A **fast, parallel, reproducible** Monte Carlo engine
- An **abstract but extensible** foundation (no real maps yet — by design)

---

## What RUIN Is Not (Yet)

- A GIS / real-road simulator
- A vehicle routing optimizer
- A calibrated production forecasting tool
- A reinforcement-learning environment

These are explicitly deferred until the abstract mechanics are solid and reproducible.

---

## Roadmap

**Done in v0.4 (model validity):** Q dots with destinations, arrivals, and completion; earned surplus; a subcritical disruption field; 2D travel with jumps; honoured ruin rule; a single validated configuration path; a reference scenario where ruin is uncertain.

**Done in v0.5 (validation):** agreement with the closed-form Cramér-Lundberg ruin probability; Wilson interval for ruin probability and bootstrap intervals for mean loss, VaR, and Expected Shortfall; losses measured over whole shifts; continuous integration; boundary modes, background risk, congestion memory, snapshot interval, and the visualization options wired to the config.

**Done in v0.6 (making destiny measurable):** remaining destiny as a curve over the shift; a per-Q-dot loss ledger, loss map, and Expected Shortfall split by type; parameter sensitivity and parameter uncertainty; an SLA defined on missed deadlines; a rebalanced reference scenario where both ruin barriers are live.

**Done in v0.7 (a regime-driven D):** a hidden regime of D that generates disruption in persistent, heavy-tailed bursts and is pulled by D's own pressures; load-dependent persistence of disruption; destiny with and without knowledge of the regime; a stylized-facts suite (`ruin facts`) that the memoryless model fails; a structure ensemble (`ruin structure`) at equal mean disruption.

**Next (chaotic organization and hedging):**
- Detection of clusters, corridors, and ruin zones; shock attribution by `shock_types`
- A posterior over the hidden regime: an observer between "knows the regime" and "does not", from the D-state history
- Congestion that slows travel (density-dependent speed), which today only affects pressure and persistence
- Deadlines that depend on trip length: today a fixed promise makes long trips carry almost all the risk
- A policy layer: buffers, backup capacity, and rerouting compared by ruin probability reduced per unit cost

**Later (when the abstract core is trusted):**
- Real road graphs and spatial topologies
- Fleet/depot constraints
- Calibration against real logistics data
- Hedging and backup-capacity policies
- Regime and disruption parameters estimated from data (the regime chain is a natural fit for hidden-Markov methods)

---

## Run the Test Suite

```bash
uv run pytest -q          # full suite
uv run ruff check . && uv run ruff format --check .
uv run mypy ruin
```

All tests must pass before any research use of a commit.

---

## Academic Credit and Citation

RUIN was created by **Naji Zouiti**.

If you use RUIN, its concepts, terminology, code, or experimental outputs in academic work, please cite the project.

Suggested citation:

> Zouiti, Naji. RUIN: Ruin-probability Unified In Networks. v0.7.0, 2026.  
> GitHub: https://github.com/nezpik/ruin

BibTeX:

```bibtex
@software{zouiti_ruin_2026,
  author = {Zouiti, Naji},
  title = {RUIN: Ruin-probability Unified In Networks},
  year = {2026},
  version = {0.7.0},
  url = {https://github.com/nezpik/ruin}
}
```

---

## Philosophy (Closing)

RUIN does not smooth chaos away. It gives chaos a frame (`D`) so that the negotiation between order pressure and disruption can be watched, measured, and understood — until the system either preserves its destiny or runs out of it.

The Probability Square is not decoration. It is the laboratory in which we study when physical flow loses the ability to keep its promises.

---

*Maintained by Naji Zouiti • MIT License • 2026*
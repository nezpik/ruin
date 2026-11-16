# RUIN Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.0] - 2026-11

**Validation**: the risk layer now agrees with classical ruin theory, every
headline number carries a confidence interval, and the checks run in CI.

### Added

- **Cramér-Lundberg validation** (`ruin/risk/classical.py`, `ruin validate`).
  With the grid stripped away, the surplus process fed by a constant premium
  and compound Poisson exponential claims has a closed-form ruin probability.
  An exact event-driven simulation reproduces it at four surplus levels
  (for example exact 0.3423, estimated 0.3400 from 20,000 paths), and RUIN's
  own `NetworkSurplus`, stepped in time, converges to the same value.
- **Wilson interval** for ruin probability, replacing the bootstrap interval
  that collapsed to zero width whenever no path (or every path) ruined.
- **Bootstrap intervals** for mean loss, VaR, and Expected Shortfall
  (`mean_loss_ci`, `value_at_risk_ci`, `expected_shortfall_ci`).
- `loss_by_type` / `loss_share_by_type`, `min_surplus` / `mean_min_surplus`.
- `run_trajectory(..., stop_at_ruin=False)` to run a shift to its horizon.
- **Continuous integration** (`.github/workflows/ci.yml`): ruff lint and
  format, mypy, tests, and `ruin validate` on Python 3.11 and 3.13.
- Config keys that now do something:
  - `probability_square.boundary_mode`: what happens to disruption spreading
    past the edge of `D`. `reflect` (default) folds it back so disruption,
    like Q dots, cannot leave the frame; `wrap` makes the grid a torus;
    `absorb` loses it (the 0.4.0 behaviour).
  - `probability_square.base_risk_intensity`: background disruption in every cell.
  - `probability_square.congestion_memory`: each cell remembers the share of
    Q dots that stalled there; this is the congestion term of local chaos
    pressure defined in `RUIN.md`.
  - `probability_square.snapshot_interval`: keep Q-dot and field snapshots
    every N steps.
  - `visualization.frames`, `heatmap_field`, `show_qdot_types`,
    `show_ruin_event`, `show_disruption_field`, `show_d_state`,
    `show_order_chaos_pressure`. Frames are now spread over the whole
    trajectory instead of taking the first 120.
- Explicit ruff rule selection in `pyproject.toml`; the code base is clean
  under ruff (lint and format) and mypy.

### Changed

- **Monte Carlo losses cover the whole shift.** Paths used to stop at ruin, so
  the loss of every ruined path was capped near the buffer and VaR / Expected
  Shortfall understated the tail. On the reference scenario Expected
  Shortfall at 95% is about 540 measured over whole shifts.
- **VaR** is the empirical quantile `ceil(confidence * n)`; the previous
  index was one observation low when `confidence * n` was not an integer.
- `simulation.dt` must be 1. It was accepted and partly ignored.
- Reference scenario retuned for the reflecting boundary and the congestion
  term: mean shock intensity 285, D-state thresholds 0.20 / 0.27 / 0.34.
  1,000 paths give a ruin probability of 0.14 (95% interval 0.12 to 0.16),
  and the grid is `STABLE` about 70% of the time on the default seed.
- `QDotType`, `QDotStatus`, and `DState` are `StrEnum`s.

### What the validation showed

From 300-path sweeps around the reference scenario, changing one setting at a
time:

- **Buffer**: ruin probability falls roughly exponentially with the initial
  buffer (0.24, 0.13, 0.06, 0.03 at 60, 120, 200, 300), the shape classical
  ruin theory predicts.
- **Disruption level is a cliff**: half the shock rate gives 0.3% ruin, one
  and a half times gives 90%. Time windows 10% tighter give 57%, 10% looser
  1.3%. Parameter uncertainty dominates Monte Carlo error.
- **Concentration matters**: the same total disruption in half as many shocks
  of double size gives 19% ruin; in twice as many of half size, 6%.
- **Express Q dots carry about 90% of the loss.**
- **Q-to-D feedback is weak**: changing `qdot_feedback_weight` from 0 to 5
  moves ruin probability from 14% to 16%, inside the noise.
- **The SLA trigger is live only between 1% and 2%** late in-flight Q dots
  (67% and 3% ruin as the sole rule); at the configured 5% it never fires.

### Still reserved

`qdots.initial_distribution`, `order_pressure_weight`, `chaos_pressure_weight`,
`disruption_field.shock_types`, `risk.report_*`, and `visualization.enabled`
are validated but unused.

---

## [0.4.0] - 2026-11

**Model validity**: ruin is now a question instead of a certainty. This release
changes the simulation model itself, so results are not comparable with 0.3.0
and earlier, and configs need the changes listed under "Migration".

Before this release every run of the reference scenario ruined between steps
19 and 21 of 480 (ruin probability 1.0, CI [1.0, 1.0]): Q dots could never
complete, and the disruption field amplified itself by about 14% per step.
With the recalibrated reference scenario, 1,000 paths now give a ruin
probability of 0.12 (95% CI 0.10 to 0.14), with ruin times spread from step 63
to the end of the shift.

### Changed

- **Q dots have destinations and can succeed.** Each Q dot enters `D` at an
  origin with a destination, travels toward it along the grid on both axes,
  and leaves `D` only by terminating: delivered on time, delivered late, or
  failed (abandoned after a full extra time window of lateness).
- **Demand arrives.** `processes.demand_arrival_rate` is now used: new Q dots
  enter each step (Poisson), with the same type mix as the initial population.
- **Surplus can be earned.** On-time deliveries add `completion_reward` to the
  surplus, giving the `C(t)` term of `U(t) = U(0) + C(t) - L(t)` a meaning.
  A late Q dot drains `delay_penalty` every step it stays late (up to double
  in a fully blocked cell) instead of paying once and stopping.
- **Jumps act on Q dots.** `jump_intensity` / `jump_mean_size` are a per-Q-dot
  compound Poisson setback in travel progress. They were previously a second
  source of cell shocks and never touched Q dots.
- **The disruption field is subcritical by construction.** Each step a cell
  keeps `memory_weight * (1 - temporal_decay - recovery_rate)` of its
  intensity; `propagation_share` of that moves to neighbours through a
  normalised kernel. Total intensity can only shrink between shocks. Configs
  whose field would never decay are rejected.
- **Shocks are events, not noise.** `shock_arrival_rate` is now the expected
  number of shocks per step for the whole grid (it was a per-cell probability,
  which produced about 45 small shocks per step on the reference grid).
- **Field drag saturates.** A cell is fully blocked at intensity 5, where
  travel speed falls to 5% of normal.
- **Pressures.** Chaos pressure uses the blocked fraction of each cell (0 to 1)
  plus the late ratio. Global order pressure is the share of active Q dots
  that moved this step minus the share that is late.
- **D-state.** `CHAOTIC` and `DISRUPTED` now take precedence over
  `RECOVERING`, which requires a sharp fall from an at-least-stressed level.
  A ruined trajectory ends its `d_state_path` with `RUINED`.
- **Late ratio** is late travelling Q dots over travelling Q dots, as defined
  in `RUIN.md` (it was cumulative failures over the whole population).
- **`ruin.rule` is honoured** (`surplus_or_sla`, `surplus`, `sla`).
- **One configuration path.** YAML is read with PyYAML and always validated
  into `RuinConfig`; plain dicts passed to the API are validated the same way.
  The hand-written YAML parser and the separate dict code path are gone (they
  disagreed with the typed path on the same file and seed). The CLI now exits
  with an error on an invalid config instead of silently falling back.
- Monte Carlo results no longer depend on worker completion order, workers
  return only per-path summaries, and `n_jobs=1` runs in-process.
- Q dots are stored as arrays; one 480-step path of the reference scenario
  takes about 0.2 s.

### Added

- `completion_reward` per Q-dot type and `disruption_field.propagation_share`.
- `ruin simulate --seed`.
- Trajectory results: `cumulative_income`, `spawned`, `delivered`,
  `delivered_late`, `failed`, `on_time_rate`. Risk results: `ruin_by_surplus`,
  `ruin_by_sla`, `mean_final_surplus`, `mean_on_time_rate`.
- `QDotStatus` and a `destination` / `status` on every Q-dot snapshot.
- Behavioural tests (`tests/test_model_behaviour.py`): no disruption means no
  ruin, the field stays bounded, more disruption means more loss, a larger
  buffer never ruins sooner, the ruin rule is honoured, Q dots stay inside `D`
  and move on both axes, same seed gives the same destiny.
- Config validation: ordered D-state thresholds, non-negative rates and
  penalties, decaying field.

### Fixed

- Q dots never moved on the y axis and piled up on the right wall.
- Q-dot colours in the animation (type and status were never shown).

### Removed

- `QDot.step`, `to_dict`, the legacy scalar process classes
  (`PoissonArrivalProcess`, `JumpDiffusionProcess`, `sample_travel_step`,
  `make_py_rng`) and `sample_shock_arrivals` (replaced by `sample_shock_events`).

### Migration

- `ProbabilitySquare(config, seed)` takes an integer seed, and `step()` takes
  no config argument.
- Retune `shock_arrival_rate` (now per grid, per step), `mean_initial_intensity`
  and time windows; add `completion_reward` if the shift should earn surplus.
  `examples/urban_ruin_shift.yaml` is the calibrated reference.
- `jump_size_distribution` accepts only `exponential`.

### Known limits

- These keys are still accepted but unused: `simulation.dt` (leave at 1),
  `qdots.initial_distribution`, `probability_square.base_risk_intensity`,
  `congestion_memory`, `boundary_mode`, `snapshot_interval`,
  `order_pressure_weight`, `chaos_pressure_weight`,
  `disruption_field.shock_types`, the `risk.report_*` flags, and
  `visualization.*`.
- In the reference scenario every ruin is a surplus ruin; the SLA trigger is
  wired and tested but does not fire at the calibrated settings.
- The model is not yet validated against closed-form ruin theory.

---

## [0.3.0] - 2026-06

**AI narration**: optional Codex-powered research-report generation layered on top of `simulate`/`risk` results.

### Added

- New optional `ruin/ai/` package (`ruin.ai.narrator`) — reads and summarizes a finished `run_trajectory`/`run_monte_carlo` result, builds a complete prompt grounded in RUIN's own vocabulary (`D`, Q dots, D-state, chaos/order pressure, surplus, ruin — distilled from `RUIN.md`), and asks an OpenAI Codex thread (`Sandbox.read_only`, used purely as a text-generation engine) to write a structured markdown research report, which RUIN itself wraps with a disclosure header/footer and writes to disk
- `--explain PATH` flag on both `ruin simulate` and `ruin risk` — writes the AI-narrated report to `PATH` after the run completes; omitted by default so existing pipelines are unaffected
- `--explain-effort LEVEL` flag (`none`/`minimal`/`low`/`medium`/`high`/`xhigh`) on both subcommands — forwards Codex's `ReasoningEffort` straight through to the narration turn (as a plain string; the SDK's own validation coerces/rejects it), letting users trade narrative depth for speed and cost on any single `--explain` run
- **Reproducibility provenance for AI reports**: Codex's prose can't be made deterministic (the SDK exposes no seed/temperature knob), so each report instead brackets it with two things RUIN keeps fully traceable — a `CodexNarration`-backed header recording exactly which Codex turn produced it (requested model/effort, duration, token usage, thread/turn IDs), and an "Appendix: source data" section echoing, verbatim and byte-reproducibly, the same compressed `result`/`config` summary Codex was given, so every narrative claim can be checked against the real numbers in the same document
- New optional dependency extra `ai` (`pip install "ruin[ai]"`, `openai-codex>=0.1.0b3`); the `openai_codex` import is fully lazy so the rest of `ruin` stays importable without it
- `tests/test_ai_narrator.py` — pure-function tests for prompt-building/compression helpers (including the new `_build_*_facts` data-block helpers), mocked-Codex tests for `generate_report` (file output, disclosure wrapping + provenance header, source-data appendix, trajectory/risk auto-detection, `model=`/`effort=` forwarding), CLI wiring tests for `--explain`/`--explain-effort`, and a guarded test documenting the `ImportError` raised without the optional extra
- README "AI Research Reports (optional)" section documenting installation, Codex auth requirements, and both `--explain` examples

---

## [0.2.5] - 2026-06

**Maintenance release**: dependency tooling migration (uv), structured logging, and reproducibility fixes on top of the v0.2 Foundation Hardening base.

### Added

- `uv` support — lockfile-based, reproducible installs via `uv sync --dev` / `uv run`, documented as the recommended quickstart path in the README
- Structured logging in the CLI via `logging.getLogger("ruin.cli")`, streamed to stdout so existing JSON-output pipelines keep working unchanged
- Pre-commit skeleton (`.pre-commit-config.yaml`): ruff, ruff-format, mypy
- Hardened `.gitignore` (`uv.lock`, IDE/editor noise, `site/`)

### Changed

- Version bumped 0.2.0 → 0.2.5
- `config_ex.md` now notes that `ruin/config_models.py` (Pydantic) is the authoritative configuration source; the YAML shown there is illustrative only

### Fixed

- Resolved a `dev` dependency-group name collision between `[project.optional-dependencies]` and `[dependency-groups]` that shadowed packages and broke `uv sync --dev`
- `ruin simulate` / `ruin risk` JSON output now reliably stays on stdout (`logging.basicConfig(..., stream=sys.stdout)`) for pipeline and CI consumers
- `ProbabilitySquare` and `run_trajectory` now take a direct integer seed instead of deriving one by hashing `Random.getstate()` — restores true cross-environment RNG reproducibility

---

## [0.2.0] - 2026-05 (Foundation Hardening)

**Major milestone**: RUIN transitions from early conceptual prototype (v0.1) to a credible, production-grade research instrument with typed configuration, high-performance vectorized primitives, statistically sound risk analysis, and publication-quality visualization.

### Added

**Core Infrastructure**
- Full Pydantic v2 configuration system (`RuinConfig`, `QDotsConfig`, `DStateThresholds`, `RiskConfig`, etc.) with strict validation (`extra='forbid'`)
- `load_ruin_config()` — typed loader with graceful fallback to legacy `load_config()`
- `ConfigLike` union type + `to_dict()` bridge for 100% backward compatibility during migration
- Direct Pydantic attribute access pattern across all hot paths (no more unconditional dict conversion)

**Performance & Vectorization (NumPy)**
- Vectorized stochastic processes in `ruin/core/processes.py`:
  - `sample_gbm_displacements`
  - `sample_compound_poisson_jumps`
  - `sample_shock_arrivals`
  - `batch_qdot_travel_step()` — replaces per-QDot Python loops
- Major acceleration of the inner simulation loop in `ProbabilitySquare`

**Visualization**
- New Matplotlib-based visualization layer (`ruin/viz/square.py`)
- Real disruption field storage via `store_field_snapshots` flag
- `save_field_gif()` + `animate_probability_square()` with blitting, inferno colormap, Q-dot coloring by type/ruin state, and HUD overlays
- `--visualize` flag in CLI produces publication-quality animated GIFs

**Risk Analysis (Monte Carlo)**
- Bootstrap confidence intervals (`bootstrap_ci` with 2000 resamples)
- `run_monte_carlo()` returns `ruin_probability_ci`, `loss_std`, VaR, Expected Shortfall, mean time to ruin
- Parallel execution via `concurrent.futures.ProcessPoolExecutor`
- `--jobs` CLI flag with automatic core detection (`n_jobs = min(cpu_count, n_paths)`)

**CLI**
- Rich subcommand interface (`simulate`, `risk`)
- `--visualize`, `--jobs`, `--output`, `--max-steps` flags
- Direct passing of `RuinConfig` objects (typed path preferred)

**Testing & Quality**
- Expanded test suite (21+ tests) exercising both legacy dict and typed `RuinConfig` paths
- Golden config (`examples/urban_ruin_shift.yaml`) regression protection
- All core modules now prefer Pydantic attributes while preserving full v0.1 compatibility

### Changed

- `run_trajectory()`, `ProbabilitySquare`, `run_monte_carlo()` now accept `ConfigLike`
- `ProbabilitySquare.step()` and `__init__` avoid per-tick `to_dict()` cost when using typed config
- Visualization and risk layers updated to accept `RuinConfig` directly
- Test suite updated to exercise the typed path (removal of `.model_dump()` workarounds)

### Fixed

- Q-dot snapshot position extraction in visualization (handles both legacy and new `position` tuple format)
- Various edge cases in parallel MC worker and field snapshot handling
- **Devin PR #1 review findings (v0.2 Foundation Hardening)**
  - Restored population of `active_by_cell` before vectorized movement in `ProbabilitySquare.step()` — this was silently breaking `failed_ratio`, `local_chaos_pressure`, and per-QDot `order_pressure` calculations after the vectorization work.
  - Re-enabled `qdot_exposure_multiplier` (from both Pydantic `DisruptionFieldConfig` and legacy dict) when computing `field_exposure` on Q-dots.
- Added dedicated regression tests (`tests/test_pressure_and_exposure.py`) to prevent recurrence of the above two issues.

### Documentation

- Major README overhaul with v0.2 highlights, quickstart examples for GIF + parallel risk analysis
- Preserved original philosophy and vocabulary from v0.1

---

## [0.1.0] - Previous

Initial conceptual release (see git history and `RUIN.md` for original vision).

---

**Note**: v0.2 maintains 100% behavioral compatibility with all v0.1 configurations and simulation semantics. The Pydantic layer is additive and preferred for new work.
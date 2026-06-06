# RUIN Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-06

**AI narration**: optional Codex-powered research-report generation layered on top of `simulate`/`risk` results.

### Added

- New optional `ruin/ai/` package (`ruin.ai.narrator`) — reads and summarizes a finished `run_trajectory`/`run_monte_carlo` result, builds a complete prompt grounded in RUIN's own vocabulary (`D`, Q dots, D-state, chaos/order pressure, surplus, ruin — distilled from `RUIN.md`), and asks an OpenAI Codex thread (`Sandbox.read_only`, used purely as a text-generation engine) to write a structured markdown research report, which RUIN itself wraps with a disclosure header/footer and writes to disk
- `--explain PATH` flag on both `ruin simulate` and `ruin risk` — writes the AI-narrated report to `PATH` after the run completes; omitted by default so existing pipelines are unaffected
- `--explain-effort LEVEL` flag (`none`/`minimal`/`low`/`medium`/`high`/`xhigh`) on both subcommands — forwards Codex's `ReasoningEffort` straight through to the narration turn (as a plain string; the SDK's own validation coerces/rejects it), letting users trade narrative depth for speed and cost on any single `--explain` run
- New optional dependency extra `ai` (`pip install "ruin[ai]"`, `openai-codex>=0.1.0b3`); the `openai_codex` import is fully lazy so the rest of `ruin` stays importable without it
- `tests/test_ai_narrator.py` — pure-function tests for prompt-building/compression helpers, mocked-Codex tests for `generate_report` (file output, disclosure wrapping, trajectory/risk auto-detection, `model=`/`effort=` forwarding), CLI wiring tests for `--explain`/`--explain-effort`, and a guarded test documenting the `ImportError` raised without the optional extra
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
"""Parameter uncertainty: how much the ruin probability depends on what we assumed.

A Monte Carlo interval only says how well the simulation has converged for
one exact set of parameters. In practice the parameters themselves (how
often shocks arrive, how fast Q dots travel) are estimates. This module
measures what that does to the answer, in two ways:

- `run_sensitivity`: move one parameter at a time by a small relative step
  and see how far the ruin probability moves. Ranks the parameters.
- `run_parameter_uncertainty`: give each uncertain parameter a spread, draw
  many plausible parameter sets, and report the resulting distribution of
  ruin probabilities.

Parameters are named by dotted path, e.g. `disruption_field.shock_arrival_rate`.
`*` stands for every Q-dot type: `qdots.*.time_window`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ruin.config import ConfigLike, as_config
from ruin.config_models import QDOT_TYPE_NAMES, RuinConfig, expand_parameter
from ruin.risk.ruin_probability import run_monte_carlo, wilson_interval

DEFAULT_SENSITIVITY_PARAMETERS = (
    "disruption_field.shock_arrival_rate",
    "disruption_field.mean_initial_intensity",
    "processes.travel_drift",
    "processes.jump_intensity",
    "processes.demand_arrival_rate",
    "qdots.*.time_window",
    "qdots.*.delay_penalty",
    "qdots.*.completion_reward",
    "ruin.initial_buffer",
)

__all__ = [
    "DEFAULT_SENSITIVITY_PARAMETERS",
    "QDOT_TYPE_NAMES",
    "run_parameter_uncertainty",
    "run_sensitivity",
    "scale_parameters",
]


def scale_parameters(config: RuinConfig, factors: dict[str, float]) -> RuinConfig:
    """A validated copy of `config` with each named parameter multiplied by its factor."""
    data = config.model_dump()
    for parameter, factor in factors.items():
        for path in expand_parameter(parameter):
            node = data
            *parents, leaf = path.split(".")
            for key in parents:
                node = node[key]
            node[leaf] = node[leaf] * factor
    return RuinConfig.model_validate(data)


def _logit(p: float, n: int) -> float:
    # Keep 0 and 1 finite: an estimate from n paths cannot resolve below 1/(2n).
    p = min(max(p, 0.5 / n), 1.0 - 0.5 / n)
    return math.log(p / (1.0 - p))


def run_sensitivity(
    config: ConfigLike,
    parameters: tuple[str, ...] | list[str] | None = None,
    rel_step: float = 0.1,
    paths: int = 200,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Ruin probability with each parameter moved down and up by `rel_step`.

    Every run uses the same seeds, but a changed parameter still changes how
    the random stream is consumed, so two runs differ by Monte Carlo noise as
    well as by the parameter. `within_noise` marks parameters whose effect is
    smaller than about two standard errors: do not read a sign into those,
    and raise `paths` to resolve them. `elasticity` is the
    change in log-odds of ruin per 1% change in the parameter, times 100: an
    elasticity of 5 means a 1% increase multiplies the odds of ruin by about
    e^0.05, roughly +5%. Parameters are returned most influential first.
    """
    cfg = as_config(config)
    names = list(parameters or DEFAULT_SENSITIVITY_PARAMETERS)

    def ruin_probability(candidate: RuinConfig) -> float:
        result = run_monte_carlo(candidate, paths=paths, n_jobs=n_jobs, ci_boot=1)
        return float(result["ruin_probability"])

    baseline = ruin_probability(cfg)
    span = math.log(1.0 + rel_step) - math.log(1.0 - rel_step)
    # Standard error of the difference between two independent estimates near the baseline.
    spread = min(max(baseline, 0.5 / paths), 1.0 - 0.5 / paths)
    noise = math.sqrt(2.0 * spread * (1.0 - spread) / paths)
    rows: list[dict[str, Any]] = []
    for name in names:
        low = ruin_probability(scale_parameters(cfg, {name: 1.0 - rel_step}))
        high = ruin_probability(scale_parameters(cfg, {name: 1.0 + rel_step}))
        rows.append(
            {
                "parameter": name,
                "ruin_probability_low": low,
                "ruin_probability_high": high,
                "elasticity": round((_logit(high, paths) - _logit(low, paths)) / span, 2),
                "within_noise": abs(high - low) <= 2.0 * noise,
            }
        )
    rows.sort(key=lambda row: abs(row["elasticity"]), reverse=True)
    return {
        "paths": paths,
        "rel_step": rel_step,
        "baseline_ruin_probability": baseline,
        "baseline_ci": [round(v, 4) for v in wilson_interval(round(baseline * paths), paths)],
        "parameters": rows,
    }


def run_parameter_uncertainty(
    config: ConfigLike,
    spreads: dict[str, float] | None = None,
    draws: int = 40,
    paths: int = 100,
    seed: int | None = None,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Distribution of ruin probability when the parameters themselves are uncertain.

    `spreads` maps each uncertain parameter to its relative standard
    deviation (0.1 = "known to about 10%"); it defaults to the config's
    `risk.parameter_uncertainty`. Each draw multiplies every listed parameter
    by an independent lognormal factor with mean 1, then estimates the ruin
    probability for that parameter set from `paths` shifts.

    `predictive_ruin_probability` averages over the draws: it is the chance
    of ruin once parameter uncertainty is admitted, and it is usually higher
    than the baseline because ruin probability rises faster than it falls.
    """
    cfg = as_config(config)
    spreads = dict(cfg.risk.parameter_uncertainty if spreads is None else spreads)
    if not spreads:
        raise ValueError(
            "No uncertain parameters: set risk.parameter_uncertainty in the config "
            "or pass `spreads`"
        )
    rng = np.random.default_rng(cfg.simulation.seed if seed is None else seed)

    baseline = float(
        run_monte_carlo(cfg, paths=paths, n_jobs=n_jobs, ci_boot=1)["ruin_probability"]
    )
    probabilities = []
    for _ in range(draws):
        factors = {
            name: float(np.exp(rng.normal(-0.5 * sd * sd, sd))) for name, sd in spreads.items()
        }
        candidate = scale_parameters(cfg, factors)
        result = run_monte_carlo(candidate, paths=paths, n_jobs=n_jobs, ci_boot=1)
        probabilities.append(float(result["ruin_probability"]))

    p = np.array(probabilities)
    return {
        "draws": draws,
        "paths_per_draw": paths,
        "spreads": spreads,
        "baseline_ruin_probability": baseline,
        "predictive_ruin_probability": round(float(p.mean()), 4),
        "ruin_probability_quantiles": {
            "5%": round(float(np.quantile(p, 0.05)), 4),
            "25%": round(float(np.quantile(p, 0.25)), 4),
            "50%": round(float(np.quantile(p, 0.50)), 4),
            "75%": round(float(np.quantile(p, 0.75)), 4),
            "95%": round(float(np.quantile(p, 0.95)), 4),
        },
        "share_of_draws_above_50pct": round(float((p > 0.5).mean()), 4),
    }

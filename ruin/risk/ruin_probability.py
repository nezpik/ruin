"""RUIN risk analysis: Monte Carlo ruin probability and tail-loss statistics."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from statistics import mean, stdev
from typing import Any

import numpy as np

from ruin.config import ConfigLike, as_config
from ruin.config_models import RuinConfig
from ruin.simulation.agent_based import run_trajectory

_PATH_FIELDS = (
    "ruined",
    "ruin_time",
    "ruin_reason",
    "cumulative_loss",
    "final_surplus",
    "on_time_rate",
)


def _run_one_path(args: tuple[RuinConfig, int, int | None]) -> dict[str, Any]:
    """Top-level worker for parallel Monte Carlo (must be picklable).

    Returns only the per-path summary, so workers do not ship whole
    trajectories back to the parent process.
    """
    config, seed, max_steps = args
    result = run_trajectory(config, seed=seed, max_steps=max_steps, store_qdot_snapshots=False)
    return {key: result[key] for key in _PATH_FIELDS}


def value_at_risk(losses: list[float], confidence: float) -> float:
    if not losses:
        return 0.0
    ordered = sorted(losses)
    index = min(len(ordered) - 1, max(0, int(confidence * len(ordered)) - 1))
    return ordered[index]


def expected_shortfall(losses: list[float], confidence: float) -> float:
    if not losses:
        return 0.0
    var = value_at_risk(losses, confidence)
    tail = [loss for loss in losses if loss >= var]
    return mean(tail) if tail else var


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Simple bootstrap percentile CI for a scalar statistic."""
    if not values or len(values) < 2:
        m = mean(values) if values else 0.0
        return m, m

    rng = rng or np.random.default_rng(42)
    arr = np.array(values, dtype=float)
    n = len(arr)
    boot_means = []

    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    boot_arr = np.array(boot_means)
    alpha = (1.0 - confidence) / 2
    lower = float(np.percentile(boot_arr, 100 * alpha))
    upper = float(np.percentile(boot_arr, 100 * (1 - alpha)))
    return lower, upper


def run_monte_carlo(
    config: ConfigLike,
    paths: int | None = None,
    confidence: float | None = None,
    max_steps: int | None = None,
    ci_boot: int = 2000,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Run many destinies (path i uses seed + i) and return ruin-risk statistics."""
    cfg = as_config(config)
    n_paths = int(paths if paths is not None else cfg.risk.monte_carlo_paths)
    level = float(confidence if confidence is not None else cfg.risk.confidence_level)
    base_seed = int(cfg.simulation.seed)

    cpu = os.cpu_count() or 4
    if n_jobs is None:
        n_jobs = min(cpu, n_paths)
    n_jobs = max(1, min(n_jobs, n_paths))

    jobs = [(cfg, base_seed + i, max_steps) for i in range(n_paths)]
    results: list[dict[str, Any]]
    if n_jobs == 1:
        results = [_run_one_path(job) for job in jobs]
    else:
        # `map` keeps results in path order, so output never depends on
        # which worker happens to finish first.
        chunksize = max(1, n_paths // (n_jobs * 4))
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(_run_one_path, jobs, chunksize=chunksize))

    losses = [float(result["cumulative_loss"]) for result in results]
    final_surpluses = [float(result["final_surplus"]) for result in results]
    ruined = [result for result in results if result["ruined"]]
    ruin_times = [int(result["ruin_time"]) for result in ruined if result["ruin_time"] is not None]
    on_time = [float(r["on_time_rate"]) for r in results if r["on_time_rate"] is not None]

    ruin_prob = len(ruined) / max(n_paths, 1)
    ruin_ci = bootstrap_ci([1.0 if r["ruined"] else 0.0 for r in results], level, ci_boot)
    loss_std = stdev(losses) if len(losses) > 1 else 0.0
    mean_time = mean(ruin_times) if ruin_times else None

    return {
        "paths": n_paths,
        "ruin_probability": ruin_prob,
        "ruin_probability_ci": [round(ruin_ci[0], 4), round(ruin_ci[1], 4)],
        "ruin_by_surplus": sum(1 for r in ruined if r["ruin_reason"] == "surplus"),
        "ruin_by_sla": sum(1 for r in ruined if r["ruin_reason"] == "sla"),
        "mean_loss": round(mean(losses), 4) if losses else 0.0,
        "loss_std": round(loss_std, 4),
        "value_at_risk": round(value_at_risk(losses, level), 4),
        "expected_shortfall": round(expected_shortfall(losses, level), 4),
        "mean_final_surplus": round(mean(final_surpluses), 4) if final_surpluses else 0.0,
        "mean_on_time_rate": round(mean(on_time), 4) if on_time else None,
        "time_to_ruin": ruin_times[:100],  # cap for output size
        "mean_time_to_ruin": round(mean_time, 2) if mean_time is not None else None,
        "confidence_level": level,
        "n_jobs": n_jobs,
    }

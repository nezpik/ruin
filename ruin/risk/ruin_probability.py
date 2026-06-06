"""RUIN risk analysis (v0.2).

Monte Carlo ruin probability estimation with improved statistics and parallel execution.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from statistics import mean, stdev
from typing import Any, Union

import numpy as np

from ruin.config import to_dict
from ruin.config_models import RuinConfig
from ruin.simulation.agent_based import run_trajectory


ConfigLike = Union[dict[str, Any], RuinConfig]


def _run_one_path(config: ConfigLike, seed: int, max_steps: int | None) -> dict[str, Any]:
    """Top-level worker for parallel Monte Carlo (must be picklable).

    v0.2: Pass original ConfigLike (Pydantic model or dict) directly downstream.
    """
    return run_trajectory(config, seed=seed, max_steps=max_steps)


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
    """Run Monte Carlo simulation (parallelized) and return enriched ruin-risk statistics.

    v0.2: Prefers direct Pydantic attributes when RuinConfig is passed.
    """
    # v0.2: prefer Pydantic attributes
    if isinstance(config, RuinConfig):
        risk_cfg = config.risk
        sim = config.simulation

        n_paths = int(paths if paths is not None else risk_cfg.monte_carlo_paths)
        level = float(confidence if confidence is not None else risk_cfg.confidence_level)
        base_seed = int(sim.seed)
    else:
        cfg = to_dict(config)
        risk_config = cfg.get("risk", {})
        simulation = cfg["simulation"]

        n_paths = int(paths or risk_config.get("monte_carlo_paths", 1000))
        level = float(confidence or risk_config.get("confidence_level", 0.95))
        base_seed = int(simulation.get("seed", 42))

    # Determine parallelism - use all available cores by default
    cpu = os.cpu_count() or 4
    if n_jobs is None:
        n_jobs = min(cpu, n_paths)
    n_jobs = max(1, min(n_jobs, n_paths))

    # Parallel execution
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(_run_one_path, config, base_seed + i, max_steps): i
            for i in range(n_paths)
        }

        for future in as_completed(futures):
            results.append(future.result())

    losses = [float(result["cumulative_loss"]) for result in results]
    ruined = [result for result in results if result["ruined"]]
    ruin_times = [
        int(result["ruin_time"]) for result in ruined if result.get("ruin_time") is not None
    ]

    ruin_prob = len(ruined) / max(n_paths, 1)
    mean_loss = mean(losses) if losses else 0.0

    # Bootstrap CI for ruin probability (Bernoulli)
    ruin_ci = bootstrap_ci([1.0 if r["ruined"] else 0.0 for r in results], level, ci_boot)

    # Basic stats
    loss_std = stdev(losses) if len(losses) > 1 else 0.0
    mean_time = mean(ruin_times) if ruin_times else None

    return {
        "paths": n_paths,
        "ruin_probability": ruin_prob,
        "ruin_probability_ci": [round(ruin_ci[0], 4), round(ruin_ci[1], 4)],
        "mean_loss": round(mean_loss, 4),
        "loss_std": round(loss_std, 4),
        "value_at_risk": round(value_at_risk(losses, level), 4),
        "expected_shortfall": round(expected_shortfall(losses, level), 4),
        "time_to_ruin": ruin_times[:100],  # cap for output size
        "mean_time_to_ruin": round(mean_time, 2) if mean_time is not None else None,
        "confidence_level": level,
        "n_jobs": n_jobs,
    }

"""RUIN risk analysis: Monte Carlo ruin probability and tail-loss statistics."""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from statistics import NormalDist, mean, stdev
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
    "min_surplus",
    "on_time_rate",
    "loss_by_type",
    "loss_concentration",
    "regime_occupancy",
)


def _run_one_path(args: tuple[RuinConfig, int, int | None]) -> dict[str, Any]:
    """Top-level worker for parallel Monte Carlo (must be picklable).

    Each path runs to the horizon even after ruin, so its loss is the loss of
    the whole shift rather than a figure cut off at the moment of ruin.
    Returns only the per-path summary, so workers do not ship whole
    trajectories back to the parent process.
    """
    config, seed, max_steps = args
    result = run_trajectory(
        config, seed=seed, max_steps=max_steps, store_qdot_snapshots=False, stop_at_ruin=False
    )
    return {key: result[key] for key in _PATH_FIELDS}


def value_at_risk(losses: list[float], confidence: float) -> float:
    """Empirical VaR: the smallest loss x such that P(loss <= x) >= confidence."""
    if not losses:
        return 0.0
    ordered = sorted(losses)
    index = math.ceil(confidence * len(ordered) - 1e-9) - 1
    return ordered[min(len(ordered) - 1, max(0, index))]


def expected_shortfall(losses: list[float], confidence: float) -> float:
    """Mean of the losses at or beyond the VaR."""
    if not losses:
        return 0.0
    var = value_at_risk(losses, confidence)
    tail = [loss for loss in losses if loss >= var]
    return mean(tail) if tail else var


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Unlike a bootstrap or normal-approximation interval it stays informative
    when no path (or every path) ruins: 0 ruins in 200 paths gives an upper
    bound of about 1.9%, not a zero-width interval.
    """
    if n <= 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(
    values: list[float],
    confidence: float = 0.95,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Bootstrap percentile CI for the mean of `values`."""
    if not values or len(values) < 2:
        m = mean(values) if values else 0.0
        return m, m
    rng = rng or np.random.default_rng(42)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    alpha = (1.0 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def bootstrap_tail_ci(
    losses: list[float],
    level: float,
    confidence: float = 0.95,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bootstrap percentile CIs for VaR and Expected Shortfall at `level`.

    Returns ((var_low, var_high), (es_low, es_high)). Tail estimates rest on
    few observations (5% of the paths at level 0.95), so these intervals are
    usually much wider than the one for the mean loss.
    """
    if len(losses) < 2:
        v = value_at_risk(losses, level)
        return (v, v), (v, v)
    rng = rng or np.random.default_rng(42)
    arr = np.asarray(losses, dtype=float)
    n = len(arr)
    samples = np.sort(arr[rng.integers(0, n, size=(n_boot, n))], axis=1)
    index = min(n - 1, max(0, math.ceil(level * n - 1e-9) - 1))
    var = samples[:, index]
    es = samples[:, index:].mean(axis=1)
    alpha = (1.0 - confidence) / 2

    def band(x: np.ndarray) -> tuple[float, float]:
        return float(np.quantile(x, alpha)), float(np.quantile(x, 1 - alpha))

    return band(var), band(es)


def _rounded(pair: tuple[float, float]) -> list[float]:
    return [round(pair[0], 4), round(pair[1], 4)]


def run_monte_carlo(
    config: ConfigLike,
    paths: int | None = None,
    confidence: float | None = None,
    max_steps: int | None = None,
    ci_boot: int = 2000,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Run many destinies (path i uses seed + i) and return ruin-risk statistics.

    `confidence` is used both as the VaR / Expected Shortfall level and as the
    coverage of every reported confidence interval.
    """
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
    min_surpluses = [float(r["min_surplus"]) for r in results if r["min_surplus"] is not None]
    ruined = [result for result in results if result["ruined"]]
    ruin_times = [int(result["ruin_time"]) for result in ruined if result["ruin_time"] is not None]
    on_time = [float(r["on_time_rate"]) for r in results if r["on_time_rate"] is not None]

    total_loss = sum(losses)
    loss_share_by_type = {
        name: round(sum(float(r["loss_by_type"][name]) for r in results) / total_loss, 4)
        if total_loss > 0
        else 0.0
        for name in results[0]["loss_by_type"]
    }

    # Euler allocation of Expected Shortfall: each type's mean loss over the
    # tail paths. The parts add up to the Expected Shortfall itself.
    var = value_at_risk(losses, level)
    tail = [r for r in results if float(r["cumulative_loss"]) >= var]
    es_by_type = {
        name: round(mean(float(r["loss_by_type"][name]) for r in tail), 4) if tail else 0.0
        for name in results[0]["loss_by_type"]
    }
    top_shares = [
        float(r["loss_concentration"]["top_1pct_share"])
        for r in results
        if float(r["cumulative_loss"]) > 0
    ]
    with_loss = [
        r["loss_concentration"]["qdots_with_loss"] / max(r["loss_concentration"]["qdots"], 1)
        for r in results
    ]

    occupancies = [r["regime_occupancy"] for r in results if r["regime_occupancy"] is not None]
    mean_occupancy = (
        {name: round(mean(o[name] for o in occupancies), 4) for name in occupancies[0]}
        if occupancies
        else None
    )

    rng = np.random.default_rng(base_seed)
    var_ci, es_ci = bootstrap_tail_ci(losses, level, level, ci_boot, rng)
    loss_std = stdev(losses) if len(losses) > 1 else 0.0
    mean_time = mean(ruin_times) if ruin_times else None

    return {
        "paths": n_paths,
        "ruin_probability": len(ruined) / max(n_paths, 1),
        "ruin_probability_ci": _rounded(wilson_interval(len(ruined), n_paths, level)),
        "ruin_by_surplus": sum(1 for r in ruined if r["ruin_reason"] == "surplus"),
        "ruin_by_sla": sum(1 for r in ruined if r["ruin_reason"] == "sla"),
        "mean_loss": round(mean(losses), 4) if losses else 0.0,
        "mean_loss_ci": _rounded(bootstrap_ci(losses, level, ci_boot, rng)),
        "loss_std": round(loss_std, 4),
        "value_at_risk": round(value_at_risk(losses, level), 4),
        "value_at_risk_ci": _rounded(var_ci),
        "expected_shortfall": round(expected_shortfall(losses, level), 4),
        "expected_shortfall_ci": _rounded(es_ci),
        "loss_share_by_type": loss_share_by_type,
        "expected_shortfall_by_type": es_by_type,
        "share_of_qdots_with_loss": round(mean(with_loss), 6) if with_loss else 0.0,
        "loss_share_of_top_1pct_qdots": round(mean(top_shares), 4) if top_shares else None,
        "mean_regime_occupancy": mean_occupancy,
        "mean_final_surplus": round(mean(final_surpluses), 4) if final_surpluses else 0.0,
        "mean_min_surplus": round(mean(min_surpluses), 4) if min_surpluses else None,
        "mean_on_time_rate": round(mean(on_time), 4) if on_time else None,
        "time_to_ruin": ruin_times[:100],  # cap for output size
        "mean_time_to_ruin": round(mean_time, 2) if mean_time is not None else None,
        "confidence_level": level,
        "n_jobs": n_jobs,
    }

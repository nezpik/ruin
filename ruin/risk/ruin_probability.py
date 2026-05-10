from __future__ import annotations

from statistics import mean
from typing import Any

from ruin.simulation.agent_based import run_trajectory


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


def run_monte_carlo(config: dict[str, Any], paths: int | None = None, confidence: float | None = None, max_steps: int | None = None) -> dict[str, Any]:
    risk_config = config.get("risk", {})
    simulation = config["simulation"]
    n_paths = int(paths or risk_config.get("monte_carlo_paths", 1000))
    level = float(confidence or risk_config.get("confidence_level", 0.95))
    base_seed = int(simulation.get("seed", 42))
    results = [run_trajectory(config, seed=base_seed + i, max_steps=max_steps) for i in range(n_paths)]
    losses = [float(result["cumulative_loss"]) for result in results]
    ruined = [result for result in results if result["ruined"]]
    ruin_times = [int(result["ruin_time"]) for result in ruined if result["ruin_time"] is not None]
    return {
        "paths": n_paths,
        "ruin_probability": len(ruined) / max(n_paths, 1),
        "mean_loss": mean(losses) if losses else 0.0,
        "value_at_risk": value_at_risk(losses, level),
        "expected_shortfall": expected_shortfall(losses, level),
        "time_to_ruin": ruin_times,
        "mean_time_to_ruin": mean(ruin_times) if ruin_times else None,
    }

"""Structural uncertainty: how much the answer depends on the *shape* of disruption.

Parameter uncertainty asks "what if the shock rate is 10% off?". Structural
uncertainty asks a harder question: "what if disruption is not the kind of
thing the model says it is?". At the same average disruption, memoryless
light-tailed shocks and clustered heavy-tailed ones can give very different
ruin probabilities, because the response of a logistics system to disruption
is convex: bad stretches cost more than calm stretches save.

`run_structure_ensemble` takes a scenario with a hidden regime and runs it
under four structures that all deliver the same realised mean disruption:

- `thin`: constant-rate Poisson shocks with exponential sizes (memoryless)
- `heavy_tails`: constant rate, lognormal sizes
- `clustered`: the scenario's hidden regime, exponential sizes
- `full`: the scenario as written

The spread of ruin probability across them is the structural uncertainty. A
conclusion that only holds under one of them should not be trusted.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from ruin.config import ConfigLike, as_config
from ruin.config_models import RuinConfig
from ruin.risk.ruin_probability import run_monte_carlo
from ruin.simulation.agent_based import run_trajectory

# Used for the `heavy_tails` variant when the scenario itself has no heavy tails.
DEFAULT_SIGMA = 1.0


def _mean_shock_mass(args: tuple[RuinConfig, int]) -> float:
    cfg, seed = args
    result = run_trajectory(cfg, seed=seed, store_qdot_snapshots=False, stop_at_ruin=False)
    return float(np.mean([s["shock_mass"] for s in result["snapshots"]]))


def realised_mean_shock_mass(
    config: ConfigLike, shifts: int = 60, n_jobs: int | None = None
) -> float:
    """Average disruption mass injected per step, measured over whole shifts.

    Measured rather than derived because pressure feedback changes how much
    time the hidden regime spends in each level.
    """
    cfg = as_config(config)
    base = int(cfg.simulation.seed)
    jobs = [(cfg, base + 10_000 + i) for i in range(shifts)]
    workers = max(1, min(n_jobs or os.cpu_count() or 4, shifts))
    if workers == 1:
        return float(np.mean([_mean_shock_mass(job) for job in jobs]))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return float(np.mean(list(executor.map(_mean_shock_mass, jobs))))


def structure_variants(config: ConfigLike, mean_mass: float) -> dict[str, RuinConfig]:
    """The four structures, mean-matched to `mean_mass` injected per step."""
    cfg = as_config(config)
    sigma = cfg.disruption_field.shock_size_sigma or DEFAULT_SIGMA

    def thin(heavy: bool) -> RuinConfig:
        v = cfg.model_copy(deep=True)
        v.probability_square.regime.enabled = False
        v.disruption_field.congestion_persistence = 0.0
        v.disruption_field.shock_size_sigma = sigma if heavy else 0.0
        v.disruption_field.shock_arrival_rate = mean_mass / max(
            v.disruption_field.mean_initial_intensity, 1e-12
        )
        return v

    clustered = cfg.model_copy(deep=True)
    clustered.disruption_field.shock_size_sigma = 0.0
    clustered.disruption_field.congestion_persistence = 0.0
    return {
        "thin": thin(heavy=False),
        "heavy_tails": thin(heavy=True),
        "clustered": clustered,
        "full": cfg,
    }


def run_structure_ensemble(
    config: ConfigLike,
    paths: int = 300,
    n_jobs: int | None = None,
    shifts: int = 60,
) -> dict[str, Any]:
    """Ruin risk under each structure, all with the same realised mean disruption."""
    cfg = as_config(config)
    if not cfg.probability_square.regime.enabled:
        raise ValueError(
            "The structure ensemble needs a scenario with probability_square.regime.enabled"
        )
    mean_mass = realised_mean_shock_mass(cfg, shifts=shifts, n_jobs=n_jobs)
    variants: dict[str, Any] = {}
    for name, variant in structure_variants(cfg, mean_mass).items():
        result = run_monte_carlo(variant, paths=paths, n_jobs=n_jobs, ci_boot=200)
        variants[name] = {
            key: result[key]
            for key in (
                "ruin_probability",
                "ruin_probability_ci",
                "ruin_by_surplus",
                "ruin_by_sla",
                "mean_loss",
                "value_at_risk",
                "expected_shortfall",
                "mean_on_time_rate",
            )
        }
    probabilities = [v["ruin_probability"] for v in variants.values()]
    return {
        "paths": paths,
        "matched_mean_shock_mass_per_step": round(mean_mass, 4),
        "variants": variants,
        "ruin_probability_range": [min(probabilities), max(probabilities)],
    }

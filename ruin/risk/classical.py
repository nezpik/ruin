"""Classical Cramér-Lundberg ruin theory, used to validate RUIN's risk layer.

The spatial model has no closed-form ruin probability, so it cannot be
checked against theory directly. Its surplus accounting, ruin detection, and
Monte Carlo estimator can: strip away the grid and feed the same
`NetworkSurplus` a constant premium and compound Poisson claims with
exponential sizes, and the ruin probability is known exactly.

    U(t) = u + c*t - sum of claims up to t
    psi(u) = (lam*mu / c) * exp(-(1/mu - lam/c) * u)      for c > lam*mu

where `lam` is the claim rate, `mu` the mean claim, and `c` the premium rate.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ruin.core.network import NetworkSurplus
from ruin.core.processes import make_rng, sample_compound_poisson_jumps
from ruin.risk.ruin_probability import wilson_interval


def cramer_lundberg_ruin_probability(
    initial_surplus: float, premium_rate: float, claim_rate: float, mean_claim: float
) -> float:
    """Exact infinite-horizon ruin probability for exponential claims."""
    if initial_surplus < 0:
        return 1.0
    if claim_rate <= 0 or mean_claim <= 0:
        return 0.0
    if premium_rate <= claim_rate * mean_claim:
        return 1.0  # no safety loading: ruin is certain
    adjustment = 1.0 / mean_claim - claim_rate / premium_rate
    return (claim_rate * mean_claim / premium_rate) * math.exp(-adjustment * initial_surplus)


def simulate_classical_ruin(
    initial_surplus: float,
    premium_rate: float,
    claim_rate: float,
    mean_claim: float,
    horizon: float,
    paths: int,
    seed: int | None = None,
    batch: int = 5000,
) -> int:
    """Count ruined paths with an exact, event-driven simulation.

    Ruin can only happen at a claim, so each path is simulated claim by claim
    (exponential waiting times, exponential sizes) with no time-step error.
    """
    rng = make_rng(seed)
    expected = claim_rate * horizon
    n_claims = int(expected + 8.0 * math.sqrt(expected) + 20)
    ruined = 0
    remaining = paths
    while remaining > 0:
        n = min(batch, remaining)
        times = np.cumsum(rng.exponential(1.0 / claim_rate, size=(n, n_claims)), axis=1)
        claims = np.cumsum(rng.exponential(mean_claim, size=(n, n_claims)), axis=1)
        surplus = initial_surplus + premium_rate * times - claims
        ruined += int(((surplus < 0) & (times <= horizon)).any(axis=1).sum())
        remaining -= n
    return ruined


def simulate_classical_ruin_stepped(
    initial_surplus: float,
    premium_rate: float,
    claim_rate: float,
    mean_claim: float,
    horizon: float,
    paths: int,
    dt: float = 0.1,
    seed: int | None = None,
) -> int:
    """Count ruined paths by stepping RUIN's own `NetworkSurplus` in time.

    This is the same surplus object and compound Poisson sampler the spatial
    engine uses. Ruin is only checked at the end of each step, after the
    step's premium has been credited, so the count is slightly low; the gap
    closes as `dt` shrinks.
    """
    rng = make_rng(seed)
    n_steps = round(horizon / dt)
    ruined = 0
    for _ in range(paths):
        claims = sample_compound_poisson_jumps(n_steps, claim_rate, mean_claim, dt=dt, rng=rng)
        network = NetworkSurplus(
            initial_buffer=initial_surplus,
            barrier=0.0,
            sla_threshold=1.0,
            recovery_rate=premium_rate * dt,
            rule="surplus",
        )
        for step, claim in enumerate(claims, start=1):
            network.update(step, float(claim), late_count=0, active_count=1)
            if network.ruined:
                ruined += 1
                break
    return ruined


def validate_against_cramer_lundberg(
    paths: int = 20000,
    seed: int = 42,
    confidence: float = 0.99,
    premium_rate: float = 1.5,
    claim_rate: float = 1.0,
    mean_claim: float = 1.0,
    initial_surpluses: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0),
    horizon: float = 300.0,
) -> dict[str, Any]:
    """Compare Monte Carlo ruin probabilities with the closed form.

    The horizon is long enough that finite-horizon and infinite-horizon ruin
    probabilities agree to well below Monte Carlo error. Each row reports
    whether the exact value falls inside the Wilson interval of the estimate.
    """
    rows = []
    for i, u in enumerate(initial_surpluses):
        exact = cramer_lundberg_ruin_probability(u, premium_rate, claim_rate, mean_claim)
        ruined = simulate_classical_ruin(
            u, premium_rate, claim_rate, mean_claim, horizon, paths, seed=seed + i
        )
        low, high = wilson_interval(ruined, paths, confidence)
        rows.append(
            {
                "initial_surplus": u,
                "exact": round(exact, 6),
                "estimate": round(ruined / paths, 6),
                "ci": [round(low, 6), round(high, 6)],
                "exact_within_ci": low <= exact <= high,
            }
        )
    return {
        "model": "Cramer-Lundberg, exponential claims",
        "premium_rate": premium_rate,
        "claim_rate": claim_rate,
        "mean_claim": mean_claim,
        "horizon": horizon,
        "paths": paths,
        "confidence_level": confidence,
        "rows": rows,
        "passed": all(row["exact_within_ci"] for row in rows),
    }

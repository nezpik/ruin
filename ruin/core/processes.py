"""Stochastic processes for RUIN v0.2.

Provides both vectorized (NumPy) primitives for Monte Carlo / bulk simulation
and lightweight scalar helpers for the per-QDot agent loop.

All sampling is reproducible via np.random.Generator.
"""

from __future__ import annotations

from random import Random
from typing import Optional

import numpy as np


# =============================================================================
# RNG helpers
# =============================================================================

def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create a reproducible NumPy Generator (preferred in v0.2+)."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)


def make_py_rng(seed: Optional[int] = None) -> Random:
    """Legacy Python random.Random for scalar/agent loops that still need it."""
    return Random(seed) if seed is not None else Random()


# =============================================================================
# Vectorized core processes (primary v0.2 API)
# =============================================================================

def sample_gbm_displacements(
    n: int,
    drift: float,
    volatility: float,
    dt: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample n independent GBM increments for travel (log-return style).

    Returns array of shape (n,) — additive displacements in grid units.
    This is the vectorized replacement for per-QDot travel noise.
    """
    rng = rng or make_rng()
    # Standard Wiener increment: N(0, sqrt(dt))
    z = rng.standard_normal(n)
    # GBM-style: mu*dt + sigma*sqrt(dt)*Z
    return drift * dt + volatility * np.sqrt(dt) * z


def sample_compound_poisson_jumps(
    n: int,
    intensity: float,
    mean_jump_size: float,
    dt: float = 1.0,
    distribution: str = "exponential",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Vectorized compound Poisson jump sizes for n agents over dt.

    Returns array of shape (n,). Most entries are 0; occasional large jumps.
    """
    rng = rng or make_rng()
    # Number of jumps per agent ~ Poisson(lambda = intensity * dt)
    lam = max(0.0, intensity * dt)
    n_jumps = rng.poisson(lam, size=n)

    jumps = np.zeros(n, dtype=float)

    if distribution.lower() == "exponential":
        # For each agent that has jumps, sample sum of exponentials
        has_jumps = n_jumps > 0
        if np.any(has_jumps):
            # Sample all required exponential rvs at once
            total_jumps = int(np.sum(n_jumps[has_jumps]))
            if total_jumps > 0:
                exp_samples = rng.exponential(
                    scale=max(mean_jump_size, 1e-9), size=total_jumps
                )
                # Assign them back (simple sequential fill — good enough for v0.2)
                idx = 0
                for i in np.where(has_jumps)[0]:
                    nj = n_jumps[i]
                    jumps[i] = np.sum(exp_samples[idx : idx + nj])
                    idx += nj
    else:
        raise ValueError(f"Unsupported jump distribution: {distribution}")

    return jumps


def sample_shock_arrivals(
    n_locations: int,
    rate: float,
    dt: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Bernoulli mask for shock arrivals (used by DisruptionField).

    Returns boolean array of length n_locations.
    """
    rng = rng or make_rng()
    p = min(1.0, max(0.0, rate * dt))
    return rng.random(n_locations) < p


# =============================================================================
# Legacy scalar classes (kept for compatibility during transition)
# =============================================================================

class PoissonArrivalProcess:
    """Legacy scalar Poisson process (still used by some agent code)."""

    def __init__(self, rate: float, rng: Random) -> None:
        self.rate = max(0.0, rate)
        self.rng = rng

    def event_occurs(self, dt: float = 1.0) -> bool:
        probability = min(1.0, self.rate * dt)
        return self.rng.random() < probability


class JumpDiffusionProcess:
    """Legacy scalar jump-diffusion sampler."""

    def __init__(
        self,
        drift: float,
        volatility: float,
        jump_intensity: float,
        jump_mean_size: float,
        rng: Random,
    ) -> None:
        self.drift = drift
        self.volatility = volatility
        self.jump_intensity = jump_intensity
        self.jump_mean_size = jump_mean_size
        self.rng = rng

    def sample_jump(self) -> float:
        if self.rng.random() < self.jump_intensity:
            return self.rng.expovariate(1.0 / max(self.jump_mean_size, 0.0001))
        return 0.0


# =============================================================================
# Convenience: combined travel + jump step for one agent (scalar)
# =============================================================================

def sample_travel_step(
    drift: float,
    volatility: float,
    jump_intensity: float,
    jump_mean_size: float,
    dt: float = 1.0,
    rng: Optional[Random] = None,
) -> float:
    """Scalar helper used by QDot / agent loop during transition."""
    py_rng = rng or make_py_rng()
    # GBM component
    z = py_rng.gauss(0.0, 1.0)
    gbm = drift * dt + volatility * np.sqrt(dt) * z
    # Jump component
    jump = 0.0
    if py_rng.random() < jump_intensity:
        jump = py_rng.expovariate(1.0 / max(jump_mean_size, 0.0001))
    return gbm + jump

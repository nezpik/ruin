"""Stochastic processes for RUIN.

Vectorized NumPy primitives. All sampling is reproducible through an explicit
`np.random.Generator`.
"""

from __future__ import annotations

import math

import numpy as np

# Field exposure at which a cell is fully blocked: drag bottoms out and the
# cell's contribution to chaos pressure saturates at 1.
EXPOSURE_SATURATION = 5.0
# Fraction of normal speed that survives in a fully blocked cell.
MIN_DRAG = 0.05


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Create a reproducible NumPy Generator."""
    return np.random.default_rng(seed)


def sample_gbm_displacements(
    n: int,
    drift: float | np.ndarray,
    volatility: float | np.ndarray,
    dt: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample n diffusion increments: drift*dt + volatility*sqrt(dt)*Z.

    This is the increment of the log of a geometric Brownian motion, used
    here as an additive travel displacement in grid cells.
    """
    rng = rng or make_rng()
    z: np.ndarray = rng.standard_normal(n)
    return np.asarray(drift * dt + volatility * math.sqrt(dt) * z, dtype=float)


def sample_compound_poisson_jumps(
    n: int,
    intensity: float,
    mean_jump_size: float,
    dt: float = 1.0,
    distribution: str = "exponential",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compound Poisson jump totals for n agents over dt.

    Returns an array of shape (n,). Most entries are 0; a few carry the sum
    of one or more exponential jumps.
    """
    if distribution.lower() != "exponential":
        raise ValueError(f"Unsupported jump distribution: {distribution}")
    rng = rng or make_rng()
    n_jumps = rng.poisson(max(0.0, intensity * dt), size=n)
    total = int(n_jumps.sum())
    if total == 0 or mean_jump_size <= 0:
        return np.zeros(n, dtype=float)
    sizes = rng.exponential(scale=mean_jump_size, size=total)
    owners = np.repeat(np.arange(n), n_jumps)
    return np.bincount(owners, weights=sizes, minlength=n).astype(float)


def sample_shock_events(
    rate: float,
    mean_intensity: float,
    width: int,
    height: int,
    dt: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample this step's disruption shocks for the whole grid.

    The number of shocks is Poisson(rate * dt); each lands on a uniformly
    random cell with exponential intensity. Returns (xs, ys, intensities).
    """
    rng = rng or make_rng()
    k = int(rng.poisson(max(0.0, rate * dt)))
    xs = rng.integers(0, width, size=k)
    ys = rng.integers(0, height, size=k)
    intensities = rng.exponential(scale=max(mean_intensity, 0.0), size=k)
    return xs, ys, intensities


def field_drag(exposures: np.ndarray) -> np.ndarray:
    """Speed multiplier in [MIN_DRAG, 1] from local field exposure."""
    blocked = np.clip(exposures, 0.0, EXPOSURE_SATURATION) / EXPOSURE_SATURATION
    return 1.0 - (1.0 - MIN_DRAG) * blocked


def batch_qdot_travel_step(
    x: np.ndarray,
    y: np.ndarray,
    dest_x: np.ndarray,
    dest_y: np.ndarray,
    carry: np.ndarray,
    field_exposures: np.ndarray,
    drift_mults: np.ndarray,
    vol_mults: np.ndarray,
    base_drift: float,
    base_vol: float,
    jump_intensity: float,
    jump_mean_size: float,
    rng: np.random.Generator,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Move many Q dots one step toward their destinations.

    Each dot's progress this step is a jump-diffusion: a diffusion increment
    whose drift is slowed by local field drag, minus a compound Poisson
    setback. Negative progress is a stall. Fractional progress is carried to
    the next step. Dots move along the grid (Manhattan paths), splitting
    their steps between the two axes at random in proportion to the distance
    left on each, so they can never leave the destiny frame.

    Returns (new_x, new_y, new_carry, moved, arrived).
    """
    n = len(x)
    if n == 0:
        empty = np.zeros(0, dtype=bool)
        return x, y, carry, empty, empty

    drift = base_drift * drift_mults * field_drag(field_exposures)
    diffusion = sample_gbm_displacements(n, drift, base_vol * vol_mults, dt=dt, rng=rng)
    setbacks = sample_compound_poisson_jumps(n, jump_intensity, jump_mean_size, dt=dt, rng=rng)
    progress = np.maximum(0.0, diffusion - setbacks) + carry

    rem_x = np.abs(dest_x - x)
    rem_y = np.abs(dest_y - y)
    remaining = rem_x + rem_y

    steps = np.minimum(np.floor(progress).astype(int), remaining)
    new_carry = progress - np.floor(progress)

    share_x = np.divide(rem_x, remaining, out=np.zeros(n, dtype=float), where=remaining > 0)
    steps_x = np.minimum(rng.binomial(steps, share_x), rem_x)
    steps_y = np.minimum(steps - steps_x, rem_y)
    steps_x = np.minimum(steps - steps_y, rem_x)

    new_x = x + np.sign(dest_x - x) * steps_x
    new_y = y + np.sign(dest_y - y) * steps_y

    moved = steps > 0
    arrived = (new_x == dest_x) & (new_y == dest_y)
    new_carry[arrived] = 0.0
    return new_x, new_y, new_carry, moved, arrived

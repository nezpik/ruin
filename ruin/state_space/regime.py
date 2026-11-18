"""The hidden regime of D.

D is not only a grid with a disruption field on it: it is in a *regime*, and
the regime is what generates disruption. Regimes persist (a bad day lasts),
they change one level at a time, and they respond to D's own pressures, so a
system that is already late tends to slide into a worse regime and tends to
stay there. What an observer sees is the regime's footprint in the pressures
(the D-state label), never the regime itself.

The chain has four ordered levels, named after the D-states they most
resemble. Per step, from level k:

    up    = escalate[k]  * pull(k, L) * (1 - order_support * o)
    down  = relax[k-1]   * pull(k, L) * (1 + order_support * o)

where L is the level that chaos pressure points to and `o` is order pressure
clipped to [0, 1]. The pull moves the regime toward L: a move toward L is
`pressure_pull` times likelier, a move away from L `pressure_pull` times
less likely, and nothing changes while L equals the current level. That
neutral zone is a dead band: pressure inside a regime's own band leaves it
alone, so a regime is entered by pressure above its floor and held until
pressure falls below it. With `pressure_pull = 1` and `order_support = 0` the
chain is exogenous and has the exact stationary distribution given by
`RegimeConfig.stationary()`.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from ruin.config_models import DStateThresholds, RegimeConfig


class Regime(StrEnum):
    STABLE = "STABLE"
    STRESSED = "STRESSED"
    DISRUPTED = "DISRUPTED"
    CHAOTIC = "CHAOTIC"


REGIMES = tuple(Regime)


class RegimeProcess:
    """Hazards and transitions of the regime chain."""

    def __init__(self, config: RegimeConfig, thresholds: DStateThresholds) -> None:
        self.config = config
        # Chaos pressure at which each level begins; level 0 begins at 0.
        self.floors = (
            0.0,
            thresholds.stressed_chaos_pressure,
            thresholds.disrupted_chaos_pressure,
            thresholds.chaotic_chaos_pressure,
        )

    def level_from_pressure(self, chaos: float) -> int:
        """The level that a chaos pressure points to (how many floors it has reached)."""
        return sum(1 for floor in self.floors[1:] if chaos >= floor)

    def hazards(self, level: int, chaos: float, order: float) -> tuple[float, float]:
        """Probabilities of moving up and down from `level` this step."""
        cfg = self.config
        o = min(max(order, 0.0), 1.0)
        pointed = self.level_from_pressure(chaos)
        up = down = 0.0
        if level < 3:
            pull = cfg.pressure_pull if pointed > level else 1.0
            pull = 1.0 / cfg.pressure_pull if pointed < level else pull
            up = cfg.escalate[level] * pull * (1.0 - cfg.order_support * o)
        if level > 0:
            pull = cfg.pressure_pull if pointed < level else 1.0
            pull = 1.0 / cfg.pressure_pull if pointed > level else pull
            down = cfg.relax[level - 1] * pull * (1.0 + cfg.order_support * o)
        up = min(1.0, up)
        down = min(down, 1.0 - up)
        return up, down

    def step(self, level: int, chaos: float, order: float, rng: np.random.Generator) -> int:
        """Draw the next level."""
        up, down = self.hazards(level, chaos, order)
        u = rng.random()
        if u < up:
            return level + 1
        if u < up + down:
            return level - 1
        return level

    def stationary_level(self, rng: np.random.Generator) -> int:
        """A level drawn from the long-run mix of the exogenous chain: what you
        would assume about D if you knew nothing about its present regime."""
        return int(rng.choice(4, p=self.config.stationary()))

    def initial_level(self, rng: np.random.Generator) -> int:
        """Level at the start of a shift."""
        start = self.config.start
        if start == "stationary":
            return self.stationary_level(rng)
        return ("stable", "stressed", "disrupted", "chaotic").index(start)

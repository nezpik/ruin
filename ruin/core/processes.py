from __future__ import annotations

from random import Random


class PoissonArrivalProcess:
    def __init__(self, rate: float, rng: Random) -> None:
        self.rate = max(0.0, rate)
        self.rng = rng

    def event_occurs(self, dt: float = 1.0) -> bool:
        probability = min(1.0, self.rate * dt)
        return self.rng.random() < probability


class JumpDiffusionProcess:
    def __init__(self, drift: float, volatility: float, jump_intensity: float, jump_mean_size: float, rng: Random) -> None:
        self.drift = drift
        self.volatility = volatility
        self.jump_intensity = jump_intensity
        self.jump_mean_size = jump_mean_size
        self.rng = rng

    def sample_jump(self) -> float:
        if self.rng.random() < self.jump_intensity:
            return self.rng.expovariate(1.0 / max(self.jump_mean_size, 0.0001))
        return 0.0

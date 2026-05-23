from __future__ import annotations

from random import Random
from typing import Optional

import numpy as np

from ruin.core.processes import make_rng


class DisruptionField:
    def __init__(
        self,
        width: int,
        height: int,
        memory_weight: float,
        propagation_radius: int,
        propagation_decay: float,
        temporal_decay: float,
        recovery_rate: float,
        rng: Random,
        np_rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.memory_weight = memory_weight
        self.propagation_radius = propagation_radius
        self.propagation_decay = propagation_decay
        self.temporal_decay = temporal_decay
        self.recovery_rate = recovery_rate
        self.rng = rng
        self.np_rng = np_rng or make_rng()
        self.grid = [[0.0 for _ in range(width)] for _ in range(height)]

    def inject(self, x: int, y: int, intensity: float) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] += max(0.0, intensity)

    def random_shock(self, mean_initial_intensity: float) -> dict[str, float | int]:
        x = self.rng.randrange(self.width)
        y = self.rng.randrange(self.height)
        intensity = self.rng.expovariate(1.0 / max(mean_initial_intensity, 0.0001))
        self.inject(x, y, intensity)
        return {"x": x, "y": y, "intensity": intensity}

    def step(self) -> None:
        next_grid = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                value = self.grid[y][x]
                if value <= 0:
                    continue
                retained = value * self.memory_weight * max(
                    0.0, 1.0 - self.temporal_decay - self.recovery_rate
                )
                next_grid[y][x] += retained
                for yy in range(
                    max(0, y - self.propagation_radius),
                    min(self.height, y + self.propagation_radius + 1),
                ):
                    for xx in range(
                        max(0, x - self.propagation_radius),
                        min(self.width, x + self.propagation_radius + 1),
                    ):
                        distance = abs(xx - x) + abs(yy - y)
                        if distance == 0 or distance > self.propagation_radius:
                            continue
                        next_grid[yy][xx] += (
                            value * (self.propagation_decay ** distance) * 0.1
                        )
        self.grid = next_grid

    def exposure_at(self, x: int, y: int) -> float:
        return self.grid[y][x]

    def flat(self) -> list[float]:
        return [value for row in self.grid for value in row]

    def snapshot(self) -> list[list[float]]:
        return [[round(value, 4) for value in row] for row in self.grid]

    # v0.2 helper for future vectorization
    def as_array(self) -> np.ndarray:
        return np.array(self.grid, dtype=float)

from __future__ import annotations

import numpy as np

BOUNDARY_MODES = ("reflect", "wrap", "absorb")


class DisruptionField:
    """Spatial memory of disruption on the grid.

    Each step a cell keeps `gain = memory_weight * (1 - temporal_decay -
    recovery_rate)` of its intensity. Of what survives, `propagation_share`
    moves to neighbouring cells within `propagation_radius` (Manhattan
    distance), weighted by `propagation_decay ** distance` and normalised to
    sum to one. Total intensity therefore never grows between shocks: a
    shock spreads and fades, it cannot amplify itself.

    `boundary_mode` decides what happens to intensity that would spread past
    the edge of the grid:

    - `reflect`: it folds back inside. Disruption, like Q dots, cannot leave
      the destiny frame. Total intensity shrinks by exactly `gain` per step.
    - `wrap`: it re-enters from the opposite edge (a torus). Also exact.
    - `absorb`: it is lost, so the edges act as a sink.
    """

    def __init__(
        self,
        width: int,
        height: int,
        memory_weight: float,
        propagation_radius: int,
        propagation_decay: float,
        temporal_decay: float,
        recovery_rate: float,
        propagation_share: float = 0.3,
        boundary_mode: str = "reflect",
    ) -> None:
        if boundary_mode not in BOUNDARY_MODES:
            raise ValueError(f"Unknown boundary_mode: {boundary_mode}")
        if propagation_radius > min(width, height):
            raise ValueError("propagation_radius must not exceed the smaller grid dimension")
        self.width = width
        self.height = height
        self.radius = propagation_radius
        self.boundary_mode = boundary_mode
        self.gain = memory_weight * max(0.0, 1.0 - temporal_decay - recovery_rate)
        if self.gain >= 1.0:
            raise ValueError("Disruption field gain must be < 1 so shocks decay")
        self.grid = np.zeros((height, width), dtype=float)

        kernel: list[tuple[int, int, float]] = []
        for dy in range(-propagation_radius, propagation_radius + 1):
            for dx in range(-propagation_radius, propagation_radius + 1):
                distance = abs(dx) + abs(dy)
                if distance == 0 or distance > propagation_radius:
                    continue
                kernel.append((dy, dx, propagation_decay**distance))
        total = sum(weight for _, _, weight in kernel)
        if total > 0:
            self.kernel = [(dy, dx, weight / total) for dy, dx, weight in kernel]
            self.propagation_share = propagation_share
        else:
            self.kernel = []
            self.propagation_share = 0.0

    def inject(self, x: int, y: int, intensity: float) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] += max(0.0, intensity)

    def inject_many(self, xs: np.ndarray, ys: np.ndarray, intensities: np.ndarray | float) -> None:
        if len(xs):
            np.add.at(self.grid, (ys, xs), intensities)

    def step(self) -> None:
        surviving = self.grid * self.gain
        nxt = surviving * (1.0 - self.propagation_share)
        if self.kernel and self.propagation_share > 0:
            nxt += self._spread(surviving * self.propagation_share)
        self.grid = nxt

    def _spread(self, spreading: np.ndarray) -> np.ndarray:
        """Redistribute `spreading` through the kernel, applying the boundary mode."""
        h, w, r = self.height, self.width, self.radius
        padded = np.zeros((h + 2 * r, w + 2 * r), dtype=float)
        for dy, dx, weight in self.kernel:
            padded[r + dy : r + dy + h, r + dx : r + dx + w] += weight * spreading

        if self.boundary_mode == "reflect":
            padded[r : 2 * r, :] += padded[:r, :][::-1, :]
            padded[h : h + r, :] += padded[h + r :, :][::-1, :]
            padded[:, r : 2 * r] += padded[:, :r][:, ::-1]
            padded[:, w : w + r] += padded[:, w + r :][:, ::-1]
        elif self.boundary_mode == "wrap":
            padded[h : h + r, :] += padded[:r, :]
            padded[r : 2 * r, :] += padded[h + r :, :]
            padded[:, w : w + r] += padded[:, :r]
            padded[:, r : 2 * r] += padded[:, w + r :]
        return padded[r : r + h, r : r + w]

    def exposure_at(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return np.asarray(self.grid[ys, xs], dtype=float)

    def total(self) -> float:
        return float(self.grid.sum())

    def snapshot(self) -> list[list[float]]:
        rounded: list[list[float]] = np.round(self.grid, 4).tolist()
        return rounded

    def as_array(self) -> np.ndarray:
        return self.grid.copy()

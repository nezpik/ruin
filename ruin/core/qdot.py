from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import Random


class QDotType(str, Enum):
    STANDARD = "STANDARD"
    EXPRESS = "EXPRESS"
    BULKY = "BULKY"


@dataclass
class QDot:
    id: int
    type: QDotType
    x: int
    y: int
    time_window: float
    delay_penalty: float
    drift_multiplier: float
    volatility_multiplier: float
    is_ruined: bool = False
    field_exposure: float = 0.0
    order_pressure: float = 1.0
    d_state_exposure: str = "STABLE"
    late: bool = False

    @property
    def position(self) -> tuple[int, int]:
        return self.x, self.y

    def step(self, width: int, height: int, base_drift: float, volatility: float, rng: Random) -> float:
        if self.is_ruined:
            return 0.0

        drag = max(0.05, 1.0 - min(self.field_exposure, 5.0) * 0.08)
        effective_drift = base_drift * self.drift_multiplier * drag
        noise = rng.gauss(0.0, volatility * self.volatility_multiplier)
        dx = int(round(effective_drift + noise))
        dy = int(round(rng.choice([-1, 0, 1]) * max(0.0, volatility * self.volatility_multiplier)))

        self.x = min(width - 1, max(0, self.x + dx))
        self.y = min(height - 1, max(0, self.y + dy))

        slack_loss = 1.0 + max(0.0, self.field_exposure) * 0.25
        self.time_window -= slack_loss
        self.late = self.time_window <= 0
        if self.late:
            self.is_ruined = True
            return self.delay_penalty * (1.0 + self.field_exposure)
        return 0.0

    def snapshot(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": self.type.value,
            "position": self.position,
            "slack": round(self.time_window, 4),
            "penalty": self.delay_penalty,
            "field_exposure": round(self.field_exposure, 4),
            "order_pressure": round(self.order_pressure, 4),
            "d_state_exposure": self.d_state_exposure,
            "ruined": self.is_ruined,
            "late": self.late,
        }

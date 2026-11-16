from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QDotType(str, Enum):
    STANDARD = "STANDARD"
    EXPRESS = "EXPRESS"
    BULKY = "BULKY"


class QDotStatus(str, Enum):
    ACTIVE = "ACTIVE"  # travelling, still inside its time window
    LATE = "LATE"  # travelling, time window exhausted, accruing penalty
    DELIVERED = "DELIVERED"  # reached its destination on time
    DELIVERED_LATE = "DELIVERED_LATE"  # reached its destination after the window
    FAILED = "FAILED"  # abandoned after a full extra window of lateness


@dataclass
class QDot:
    """One Q dot as a record.

    The engine keeps Q dots as arrays (see `ProbabilitySquare`); this class is
    the per-dot view used for inspection and snapshots.
    """

    id: int
    type: QDotType
    x: int
    y: int
    dest_x: int
    dest_y: int
    time_window: float
    delay_penalty: float
    status: QDotStatus = QDotStatus.ACTIVE
    field_exposure: float = 0.0
    order_pressure: float = 0.0
    d_state_exposure: str = "STABLE"

    @property
    def position(self) -> tuple[int, int]:
        return self.x, self.y

    @property
    def destination(self) -> tuple[int, int]:
        return self.dest_x, self.dest_y

    @property
    def late(self) -> bool:
        return self.status in (QDotStatus.LATE, QDotStatus.DELIVERED_LATE, QDotStatus.FAILED)

    @property
    def is_ruined(self) -> bool:
        return self.status is QDotStatus.FAILED

    def snapshot(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": self.type.value,
            "position": self.position,
            "destination": self.destination,
            "slack": round(self.time_window, 4),
            "penalty": self.delay_penalty,
            "field_exposure": round(self.field_exposure, 4),
            "order_pressure": round(self.order_pressure, 4),
            "d_state_exposure": self.d_state_exposure,
            "status": self.status.value,
            "ruined": self.is_ruined,
            "late": self.late,
        }

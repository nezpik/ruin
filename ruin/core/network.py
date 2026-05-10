from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NetworkSurplus:
    initial_buffer: float
    barrier: float
    sla_threshold: float
    recovery_rate: float = 0.0
    cumulative_loss: float = 0.0
    surplus_path: list[float] = field(default_factory=list)
    late_ratio_path: list[float] = field(default_factory=list)
    ruined: bool = False
    ruin_time: int | None = None
    ruin_reason: str | None = None

    @property
    def surplus(self) -> float:
        return self.initial_buffer + self.recovery_rate * len(self.surplus_path) - self.cumulative_loss

    def update(self, time: int, penalty: float, late_count: int, active_count: int) -> None:
        self.cumulative_loss += penalty
        late_ratio = late_count / max(active_count, 1)
        self.surplus_path.append(self.surplus)
        self.late_ratio_path.append(late_ratio)

        if self.ruined:
            return
        if self.surplus <= self.barrier:
            self.ruined = True
            self.ruin_time = time
            self.ruin_reason = "surplus"
        elif late_ratio >= self.sla_threshold:
            self.ruined = True
            self.ruin_time = time
            self.ruin_reason = "sla"

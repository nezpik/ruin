from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field


@dataclass
class NetworkSurplus:
    """Surplus process U(t) = U(0) + C(t) - L(t) with the configured ruin rule.

    C(t) is planned recovery (`recovery_rate` per step) plus income earned from
    on-time completions. L(t) is the cumulative delay penalty.

    The SLA side watches deadlines: over the last `sla_window` steps, the
    miss rate is deadlines missed / (deadlines missed + on-time deliveries).
    The SLA is breached when that rate reaches `sla_threshold`. It is only
    judged once a full window has been observed.
    """

    initial_buffer: float
    barrier: float
    sla_threshold: float
    recovery_rate: float = 0.0
    rule: str = "surplus_or_sla"
    sla_window: int = 60
    elapsed: int = 0
    cumulative_loss: float = 0.0
    cumulative_income: float = 0.0
    surplus_path: list[float] = field(default_factory=list)
    late_ratio_path: list[float] = field(default_factory=list)
    sla_miss_rate_path: list[float] = field(default_factory=list)
    _recent: deque[tuple[int, int]] = field(default_factory=deque, repr=False)
    _recent_missed: int = 0
    _recent_on_time: int = 0
    ruined: bool = False
    ruin_time: int | None = None
    ruin_reason: str | None = None

    @property
    def surplus(self) -> float:
        return (
            self.initial_buffer
            + self.recovery_rate * self.elapsed
            + self.cumulative_income
            - self.cumulative_loss
        )

    def update(
        self,
        time: int,
        penalty: float,
        late_count: int,
        active_count: int,
        income: float = 0.0,
        missed: int = 0,
        on_time: int = 0,
    ) -> None:
        self.elapsed = time
        self.cumulative_loss += penalty
        self.cumulative_income += income
        late_ratio = late_count / max(active_count, 1)
        self.surplus_path.append(self.surplus)
        self.late_ratio_path.append(late_ratio)

        self._recent.append((missed, on_time))
        self._recent_missed += missed
        self._recent_on_time += on_time
        if len(self._recent) > self.sla_window:
            old_missed, old_on_time = self._recent.popleft()
            self._recent_missed -= old_missed
            self._recent_on_time -= old_on_time
        judged = self._recent_missed + self._recent_on_time
        miss_rate = self._recent_missed / judged if judged else 0.0
        self.sla_miss_rate_path.append(miss_rate)
        sla_breached = len(self._recent) >= self.sla_window and miss_rate >= self.sla_threshold

        if self.ruined:
            return
        if self.rule in ("surplus", "surplus_or_sla") and self.surplus <= self.barrier:
            self.ruined = True
            self.ruin_time = time
            self.ruin_reason = "surplus"
        elif self.rule in ("sla", "surplus_or_sla") and sla_breached:
            self.ruined = True
            self.ruin_time = time
            self.ruin_reason = "sla"

    def fork(self) -> NetworkSurplus:
        """An independent copy of the surplus state, for branching futures."""
        other = copy.copy(self)
        other.surplus_path = list(self.surplus_path)
        other.late_ratio_path = list(self.late_ratio_path)
        other.sla_miss_rate_path = list(self.sla_miss_rate_path)
        other._recent = deque(self._recent)
        return other

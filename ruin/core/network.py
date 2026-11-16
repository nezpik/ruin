from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NetworkSurplus:
    """Surplus process U(t) = U(0) + C(t) - L(t) with the configured ruin rule.

    C(t) is planned recovery (`recovery_rate` per step) plus income earned from
    on-time completions. L(t) is the cumulative delay penalty.
    """

    initial_buffer: float
    barrier: float
    sla_threshold: float
    recovery_rate: float = 0.0
    rule: str = "surplus_or_sla"
    elapsed: int = 0
    cumulative_loss: float = 0.0
    cumulative_income: float = 0.0
    surplus_path: list[float] = field(default_factory=list)
    late_ratio_path: list[float] = field(default_factory=list)
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
    ) -> None:
        self.elapsed = time
        self.cumulative_loss += penalty
        self.cumulative_income += income
        late_ratio = late_count / max(active_count, 1)
        self.surplus_path.append(self.surplus)
        self.late_ratio_path.append(late_ratio)

        if self.ruined:
            return
        if self.rule in ("surplus", "surplus_or_sla") and self.surplus <= self.barrier:
            self.ruined = True
            self.ruin_time = time
            self.ruin_reason = "surplus"
        elif self.rule in ("sla", "surplus_or_sla") and late_ratio >= self.sla_threshold:
            self.ruined = True
            self.ruin_time = time
            self.ruin_reason = "sla"

from __future__ import annotations

from typing import Any

from ruin.config import ConfigLike, as_config
from ruin.core.network import NetworkSurplus
from ruin.metrics.pressure import DState
from ruin.state_space.probability_square import ProbabilitySquare


def run_trajectory(
    config: ConfigLike,
    seed: int | None = None,
    max_steps: int | None = None,
    store_field_snapshots: bool = False,
    store_qdot_snapshots: bool = True,
) -> dict[str, Any]:
    """Run one destiny: a single shift, stopping at ruin or the horizon."""
    cfg = as_config(config)
    ruin_cfg = cfg.ruin
    horizon = int(max_steps if max_steps is not None else cfg.simulation.shift_duration)

    square = ProbabilitySquare(
        cfg,
        seed,
        store_field_snapshots=store_field_snapshots,
        store_qdot_snapshots=store_qdot_snapshots,
    )
    network = NetworkSurplus(
        initial_buffer=float(ruin_cfg.initial_buffer),
        barrier=float(ruin_cfg.barrier),
        sla_threshold=float(ruin_cfg.sla_threshold),
        recovery_rate=float(ruin_cfg.recovery_rate),
        rule=ruin_cfg.rule,
    )

    for time in range(1, horizon + 1):
        snapshot = square.step()
        network.update(
            time,
            float(snapshot["penalty"]),
            int(snapshot["late_count"]),
            int(snapshot["active_count"]),
            income=float(snapshot["income"]),
        )
        if network.ruined:
            square.d_state = DState.RUINED
            snapshot["d_state"] = DState.RUINED.value
            break

    terminated = square.delivered + square.delivered_late + square.failed
    return {
        "ruined": network.ruined,
        "ruin_time": network.ruin_time,
        "ruin_reason": network.ruin_reason,
        "cumulative_loss": round(network.cumulative_loss, 6),
        "cumulative_income": round(network.cumulative_income, 6),
        "final_surplus": round(network.surplus, 6),
        "final_d_state": square.d_state.value,
        "spawned": square.spawned,
        "delivered": square.delivered,
        "delivered_late": square.delivered_late,
        "failed": square.failed,
        "on_time_rate": round(square.delivered / terminated, 6) if terminated else None,
        "surplus_path": network.surplus_path,
        "late_ratio_path": network.late_ratio_path,
        "d_state_path": [snapshot["d_state"] for snapshot in square.snapshots],
        "chaos_pressure_path": [snapshot["chaos_pressure"] for snapshot in square.snapshots],
        "order_pressure_path": [snapshot["order_pressure"] for snapshot in square.snapshots],
        "snapshots": square.snapshots,
        "final_snapshot": square.snapshots[-1] if square.snapshots else {},
    }

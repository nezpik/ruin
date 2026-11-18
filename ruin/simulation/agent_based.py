from __future__ import annotations

from typing import Any

import numpy as np

from ruin.analysis.organization import StructureRecorder
from ruin.config import ConfigLike, as_config
from ruin.core.network import NetworkSurplus
from ruin.metrics.pressure import DState
from ruin.state_space.probability_square import QDOT_TYPES, ProbabilitySquare
from ruin.state_space.regime import REGIMES


def make_network(config: ConfigLike) -> NetworkSurplus:
    """The surplus process for a config, at the start of a shift."""
    ruin_cfg = as_config(config).ruin
    return NetworkSurplus(
        initial_buffer=float(ruin_cfg.initial_buffer),
        barrier=float(ruin_cfg.barrier),
        sla_threshold=float(ruin_cfg.sla_threshold),
        recovery_rate=float(ruin_cfg.recovery_rate),
        rule=ruin_cfg.rule,
        sla_window=int(ruin_cfg.sla_window),
    )


def advance_shift(
    square: ProbabilitySquare,
    network: NetworkSurplus,
    until: int,
    stop_at_ruin: bool = True,
) -> list[float]:
    """Step D and its surplus forward to time `until` (or to ruin).

    Returns the loss accrued over these steps, by Q-dot type.
    """
    loss_by_type = [0.0, 0.0, 0.0]
    while square.time < until:
        was_ruined = network.ruined
        if was_ruined and stop_at_ruin:
            break
        snapshot = square.step(system_ruined=was_ruined)
        network.update(
            square.time,
            float(snapshot["penalty"]),
            int(snapshot["late_count"]),
            int(snapshot["active_count"]),
            income=float(snapshot["income"]),
            missed=int(snapshot["missed"]),
            on_time=int(snapshot["delivered"]),
        )
        for i, value in enumerate(snapshot["penalty_by_type"]):
            loss_by_type[i] += value
        if network.ruined and not was_ruined:
            square.d_state = DState.RUINED
            snapshot["d_state"] = DState.RUINED.value
            if square.recorder is not None:
                square.recorder.mark_ruin(square.time)
            if stop_at_ruin:
                break
    return loss_by_type


def _by_name(names: list[str], values: list[float] | None) -> dict[str, float] | None:
    if values is None:
        return None
    return {name: round(v, 6) for name, v in zip(names, values, strict=True)}


def _delay_attribution(square: ProbabilitySquare) -> dict[str, Any] | None:
    """Field-induced delay (dot-steps) by origin of the disruption, when tracked."""
    if square.sources is None:
        return None
    total = square.delay_total
    return {
        "total_delay_steps": round(total, 3),
        "by_source": {
            name: round(v, 3)
            for name, v in zip(square.source_names, square.delay_by_source, strict=True)
        },
        "share_by_source": {
            name: round(v / total, 4) if total > 0 else 0.0
            for name, v in zip(square.source_names, square.delay_by_source, strict=True)
        },
        "unattributed_share": max(0.0, round(1.0 - sum(square.delay_by_source) / total, 4))
        if total > 0
        else 0.0,
    }


def _occupancy(snapshots: list[dict[str, Any]]) -> dict[str, float] | None:
    """Share of steps spent in each hidden regime (None when regimes are off)."""
    path = [snapshot["regime"] for snapshot in snapshots]
    if not path or path[0] is None:
        return None
    return {name.value: round(path.count(name.value) / len(path), 4) for name in REGIMES}


def loss_contributions(square: ProbabilitySquare, top: int = 10) -> dict[str, Any]:
    """Who carried the loss: the Q dots that cost the most, and how concentrated loss is."""
    ledger = square.ledger()
    total = sum(row["loss"] for row in ledger)
    ranked = sorted(ledger, key=lambda row: row["loss"], reverse=True)
    one_percent = max(1, len(ranked) // 100)
    contributors = [
        {**row, "share_of_loss": round(row["loss"] / total, 4)}
        for row in ranked[:top]
        if row["loss"] > 0
    ]
    return {
        "top_contributors": contributors,
        "loss_concentration": {
            "qdots": len(ledger),
            "qdots_with_loss": sum(1 for row in ledger if row["loss"] > 0),
            "top_1pct_share": round(sum(r["loss"] for r in ranked[:one_percent]) / total, 4)
            if total > 0
            else 0.0,
        },
    }


def run_trajectory(
    config: ConfigLike,
    seed: int | None = None,
    max_steps: int | None = None,
    store_field_snapshots: bool = False,
    store_qdot_snapshots: bool = True,
    stop_at_ruin: bool = True,
    recorder: StructureRecorder | None = None,
    attribute_sources: bool = False,
) -> dict[str, Any]:
    """Run one destiny: a single shift.

    By default the shift stops at ruin. With `stop_at_ruin=False` it runs to
    the horizon regardless: `ruin_time` still records the first barrier
    breach, the D-state stays RUINED from then on, and the loss figures cover
    the whole shift instead of being cut off at the moment of ruin.
    """
    cfg = as_config(config)
    horizon = int(max_steps if max_steps is not None else cfg.simulation.shift_duration)

    square = ProbabilitySquare(
        cfg,
        seed,
        store_field_snapshots=store_field_snapshots,
        store_qdot_snapshots=store_qdot_snapshots,
        recorder=recorder,
        attribute_sources=attribute_sources,
    )
    network = make_network(cfg)
    if recorder is not None:
        recorder.set_horizon(horizon)
    loss_by_type = advance_shift(square, network, horizon, stop_at_ruin=stop_at_ruin)

    terminated = square.delivered + square.delivered_late + square.failed
    return {
        "ruined": network.ruined,
        "ruin_time": network.ruin_time,
        "ruin_reason": network.ruin_reason,
        "cumulative_loss": round(network.cumulative_loss, 6),
        "cumulative_income": round(network.cumulative_income, 6),
        "final_surplus": round(network.surplus, 6),
        "min_surplus": round(min(network.surplus_path), 6) if network.surplus_path else None,
        "loss_by_type": {
            qtype.value: round(loss, 6)
            for qtype, loss in zip(QDOT_TYPES, loss_by_type, strict=True)
        },
        "final_d_state": square.d_state.value,
        "spawned": square.spawned,
        "delivered": square.delivered,
        "delivered_late": square.delivered_late,
        "failed": square.failed,
        "on_time_rate": round(square.delivered / terminated, 6) if terminated else None,
        "loss_by_regime": _by_name([r.value for r in REGIMES], square.loss_by_regime),
        "delay_by_source": _delay_attribution(square),
        "structure": recorder.summary() if recorder is not None else None,
        **loss_contributions(square),
        "loss_field": np.round(square.loss_field.reshape(square.height, square.width), 4).tolist(),
        "surplus_path": network.surplus_path,
        "late_ratio_path": network.late_ratio_path,
        "sla_miss_rate_path": network.sla_miss_rate_path,
        "regime_path": [snapshot["regime"] for snapshot in square.snapshots],
        "regime_occupancy": _occupancy(square.snapshots),
        "shock_count_path": [snapshot["shock_count"] for snapshot in square.snapshots],
        "field_retention_path": [snapshot["field_retention"] for snapshot in square.snapshots],
        "d_state_path": [snapshot["d_state"] for snapshot in square.snapshots],
        "chaos_pressure_path": [snapshot["chaos_pressure"] for snapshot in square.snapshots],
        "order_pressure_path": [snapshot["order_pressure"] for snapshot in square.snapshots],
        "snapshots": square.snapshots,
        "final_snapshot": square.snapshots[-1] if square.snapshots else {},
    }

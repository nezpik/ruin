from __future__ import annotations

from random import Random
from typing import Any, Union

from ruin.config import to_dict
from ruin.config_models import RuinConfig
from ruin.core.network import NetworkSurplus
from ruin.state_space.probability_square import ProbabilitySquare


ConfigLike = Union[dict[str, Any], RuinConfig]


def run_trajectory(
    config: ConfigLike,
    seed: int | None = None,
    max_steps: int | None = None,
    store_field_snapshots: bool = False,
) -> dict[str, Any]:
    # v0.2: prefer direct Pydantic attributes when a validated RuinConfig is passed
    if isinstance(config, RuinConfig):
        sim = config.simulation
        ruin_cfg = config.ruin
        effective_seed = seed if seed is not None else getattr(sim, "seed", 42)
        horizon = int(max_steps or getattr(sim, "shift_duration", 480))

        square = ProbabilitySquare(config, int(effective_seed), store_field_snapshots=store_field_snapshots)

        network = NetworkSurplus(
            initial_buffer=float(ruin_cfg.initial_buffer),
            barrier=float(getattr(ruin_cfg, "barrier", 0.0)),
            sla_threshold=float(getattr(ruin_cfg, "sla_threshold", 0.12)),
            recovery_rate=float(getattr(ruin_cfg, "recovery_rate", 0.0)),
        )
    else:
        cfg = to_dict(config)
        simulation = cfg["simulation"]
        ruin_config = cfg["ruin"]
        effective_seed = seed if seed is not None else simulation.get("seed", 42)
        horizon = int(max_steps or simulation.get("shift_duration", 480))

        square = ProbabilitySquare(config, int(effective_seed), store_field_snapshots=store_field_snapshots)

        network = NetworkSurplus(
            initial_buffer=float(ruin_config["initial_buffer"]),
            barrier=float(ruin_config["barrier"]),
            sla_threshold=float(ruin_config.get("sla_threshold", 0.12)),
            recovery_rate=float(ruin_config.get("recovery_rate", 0.0)),
        )

    for time in range(1, horizon + 1):
        snapshot = square.step(config, network.ruined)  # pass original (now supports Pydantic)
        network.update(time, float(snapshot["penalty"]), int(snapshot["late_count"]), len(square.qdots))
        if network.ruined:
            square.d_state = square.d_state.RUINED
            break

    final_snapshot = square.snapshots[-1] if square.snapshots else {}
    return {
        "ruined": network.ruined,
        "ruin_time": network.ruin_time,
        "ruin_reason": network.ruin_reason,
        "cumulative_loss": round(network.cumulative_loss, 6),
        "final_surplus": round(network.surplus, 6),
        "final_d_state": square.d_state.value,
        "surplus_path": network.surplus_path,
        "late_ratio_path": network.late_ratio_path,
        "d_state_path": [snapshot["d_state"] for snapshot in square.snapshots],
        "chaos_pressure_path": [snapshot["chaos_pressure"] for snapshot in square.snapshots],
        "order_pressure_path": [snapshot["order_pressure"] for snapshot in square.snapshots],
        "snapshots": square.snapshots,
        "final_snapshot": final_snapshot,
    }

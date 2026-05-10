from __future__ import annotations

from random import Random
from typing import Any

from ruin.core.qdot import QDot, QDotType
from ruin.metrics.pressure import DState, choose_d_state, global_chaos_pressure, global_order_pressure, local_chaos_pressure, local_order_pressure
from ruin.state_space.disruption_field import DisruptionField


class ProbabilitySquare:
    def __init__(self, config: dict[str, Any], rng: Random) -> None:
        simulation = config["simulation"]
        square = config["probability_square"]
        field_config = config["disruption_field"]
        self.width = int(simulation["grid_width"])
        self.height = int(simulation["grid_height"])
        self.rng = rng
        self.time = 0
        self.d_state = DState(str(square.get("initial_d_state", "stable")).upper())
        self.previous_chaos_pressure = 0.0
        self.global_chaos_pressure = 0.0
        self.global_order_pressure = 0.0
        self.snapshots: list[dict[str, Any]] = []
        self.thresholds = square.get("d_state_thresholds", {})
        self.qdot_feedback_weight = float(square.get("qdot_feedback_weight", 0.5))
        self.max_capacity = self.width * self.height
        self.field = DisruptionField(
            self.width,
            self.height,
            float(field_config.get("memory_weight", 0.85)),
            int(field_config.get("propagation_radius", 2)),
            float(field_config.get("propagation_decay", 0.55)),
            float(field_config.get("temporal_decay", 0.12)),
            float(field_config.get("recovery_rate", 0.08)),
            rng,
        )
        self.qdots = self._spawn_qdots(config)
        self.initial_qdots = len(self.qdots)

    def _spawn_qdots(self, config: dict[str, Any]) -> list[QDot]:
        qcfg = config["qdots"]
        qdots: list[QDot] = []
        specs = [
            (QDotType.STANDARD, int(qcfg.get("n_standard", 0)), qcfg["standard"]),
            (QDotType.EXPRESS, int(qcfg.get("n_express", 0)), qcfg["express"]),
            (QDotType.BULKY, int(qcfg.get("n_bulky", 0)), qcfg["bulky"]),
        ]
        qid = 0
        for qtype, count, spec in specs:
            for _ in range(count):
                qdots.append(
                    QDot(
                        id=qid,
                        type=qtype,
                        x=self.rng.randrange(max(1, self.width // 4)),
                        y=self.rng.randrange(self.height),
                        time_window=float(spec["time_window"]),
                        delay_penalty=float(spec["delay_penalty"]),
                        drift_multiplier=float(spec["drift_multiplier"]),
                        volatility_multiplier=float(spec["volatility_multiplier"]),
                    )
                )
                qid += 1
        return qdots

    def step(self, config: dict[str, Any], system_ruined: bool) -> dict[str, Any]:
        self.time += 1
        processes = config["processes"]
        field_config = config["disruption_field"]
        if bool(field_config.get("enabled", True)) and self.rng.random() < float(field_config.get("shock_arrival_rate", 0.05)):
            self.field.random_shock(float(field_config.get("mean_initial_intensity", 2.0)))

        jump_probability = float(processes.get("jump_intensity", 0.08))
        if self.rng.random() < jump_probability:
            self.field.random_shock(float(processes.get("jump_mean_size", 3.0)))

        self.field.step()

        penalty = 0.0
        failed_by_cell: dict[tuple[int, int], int] = {}
        active_by_cell: dict[tuple[int, int], int] = {}
        moving_by_cell: dict[tuple[int, int], int] = {}

        for qdot in self.qdots:
            active_by_cell[qdot.position] = active_by_cell.get(qdot.position, 0) + int(not qdot.is_ruined)
            qdot.field_exposure = self.field.exposure_at(qdot.x, qdot.y) * float(field_config.get("qdot_exposure_multiplier", 1.0))
            qdot.d_state_exposure = self.d_state.value
            before = qdot.position
            penalty += qdot.step(
                self.width,
                self.height,
                float(processes.get("travel_drift", 1.2)),
                float(processes.get("travel_volatility", 0.4)),
                self.rng,
            )
            if qdot.position != before:
                moving_by_cell[qdot.position] = moving_by_cell.get(qdot.position, 0) + 1
            if qdot.is_ruined:
                failed_by_cell[qdot.position] = failed_by_cell.get(qdot.position, 0) + 1

        for (x, y), failed in failed_by_cell.items():
            self.field.inject(x, y, failed * self.qdot_feedback_weight)

        local_pressures: list[float] = []
        for y in range(self.height):
            for x in range(self.width):
                active = active_by_cell.get((x, y), 0)
                failed = failed_by_cell.get((x, y), 0)
                failed_ratio = failed / max(active + failed, 1)
                local_pressures.append(local_chaos_pressure(self.field.exposure_at(x, y), 0.0, failed_ratio))

        late_count = sum(1 for qdot in self.qdots if qdot.late)
        active_count = sum(1 for qdot in self.qdots if not qdot.is_ruined)
        late_ratio = late_count / max(active_count, 1)
        self.global_chaos_pressure = global_chaos_pressure(local_pressures, late_ratio)
        self.global_order_pressure = global_order_pressure(active_count, late_count, self.max_capacity, self.initial_qdots)

        for qdot in self.qdots:
            active = active_by_cell.get(qdot.position, 0)
            moving = moving_by_cell.get(qdot.position, 0)
            failed = failed_by_cell.get(qdot.position, 0)
            qdot.order_pressure = local_order_pressure(active, moving, failed)

        self.d_state = choose_d_state(
            system_ruined,
            self.global_chaos_pressure,
            self.previous_chaos_pressure,
            float(self.thresholds.get("stressed_chaos_pressure", 0.35)),
            float(self.thresholds.get("disrupted_chaos_pressure", 0.55)),
            float(self.thresholds.get("chaotic_chaos_pressure", 0.75)),
            float(self.thresholds.get("recovering_decay_ratio", 0.30)),
        )
        self.previous_chaos_pressure = self.global_chaos_pressure

        snapshot = self.snapshot(penalty, late_count, active_count)
        self.snapshots.append(snapshot)
        return snapshot

    def snapshot(self, penalty: float, late_count: int, active_count: int) -> dict[str, Any]:
        return {
            "time": self.time,
            "d_state": self.d_state.value,
            "chaos_pressure": round(self.global_chaos_pressure, 6),
            "order_pressure": round(self.global_order_pressure, 6),
            "penalty": round(penalty, 6),
            "late_count": late_count,
            "active_count": active_count,
            "qdots": [qdot.snapshot() for qdot in self.qdots],
        }

from __future__ import annotations

from random import Random
from typing import Any, Union

import numpy as np

from ruin.config import to_dict
from ruin.config_models import RuinConfig
from ruin.core.processes import batch_qdot_travel_step, make_rng, sample_shock_arrivals
from ruin.core.qdot import QDot, QDotType
from ruin.metrics.pressure import (
    DState,
    choose_d_state,
    global_chaos_pressure,
    global_order_pressure,
    local_chaos_pressure,
    local_order_pressure,
)
from ruin.state_space.disruption_field import DisruptionField


ConfigLike = Union[dict[str, Any], RuinConfig]


class ProbabilitySquare:
    def __init__(self, config: ConfigLike, rng: Random, store_field_snapshots: bool = False) -> None:
        cfg = to_dict(config)  # support both dict and RuinConfig Pydantic models
        self._raw_config: ConfigLike = config  # keep original for typed access during v0.2 migration

        # v0.2: prefer Pydantic attributes when a validated RuinConfig is passed
        if isinstance(config, RuinConfig):
            sim = config.simulation
            self.width = int(sim.grid_width)
            self.height = int(sim.grid_height)
        else:
            simulation = cfg["simulation"]
            self.width = int(simulation["grid_width"])
            self.height = int(simulation["grid_height"])

        self.rng = rng

        # v0.2: independent NumPy RNG for vectorized sampling
        seed = 42
        try:
            if hasattr(rng, "getstate"):
                state = rng.getstate()
                if isinstance(state, tuple) and len(state) > 1:
                    seed = abs(hash(str(state[1]))) % (2**31)
        except Exception:
            seed = 42
        self.np_rng = make_rng(seed)

        self.time = 0

        # v0.2: extract sections with Pydantic attribute preference
        if isinstance(self._raw_config, RuinConfig):
            square_cfg = self._raw_config.probability_square
            field_cfg = self._raw_config.disruption_field
            self.d_state = DState(str(getattr(square_cfg, "initial_d_state", "stable")).upper())
            # Normalize to dict so .get() works in choose_d_state call
            d_thresh = getattr(square_cfg, "d_state_thresholds", None)
            self.thresholds = d_thresh.model_dump() if d_thresh is not None else {}
            self.qdot_feedback_weight = float(getattr(square_cfg, "qdot_feedback_weight", 0.5))

            mem_w = float(getattr(field_cfg, "memory_weight", 0.85))
            prop_r = int(getattr(field_cfg, "propagation_radius", 2))
            prop_d = float(getattr(field_cfg, "propagation_decay", 0.55))
            temp_d = float(getattr(field_cfg, "temporal_decay", 0.12))
            rec_r = float(getattr(field_cfg, "recovery_rate", 0.08))
        else:
            square = cfg["probability_square"]
            field_config = cfg["disruption_field"]
            self.d_state = DState(str(square.get("initial_d_state", "stable")).upper())
            self.thresholds = square.get("d_state_thresholds", {})
            self.qdot_feedback_weight = float(square.get("qdot_feedback_weight", 0.5))

            mem_w = float(field_config.get("memory_weight", 0.85))
            prop_r = int(field_config.get("propagation_radius", 2))
            prop_d = float(field_config.get("propagation_decay", 0.55))
            temp_d = float(field_config.get("temporal_decay", 0.12))
            rec_r = float(field_config.get("recovery_rate", 0.08))

        self.previous_chaos_pressure = 0.0
        self.global_chaos_pressure = 0.0
        self.global_order_pressure = 0.0
        self.snapshots: list[dict[str, Any]] = []
        self.max_capacity = self.width * self.height
        self.store_field_snapshots = store_field_snapshots

        self.field = DisruptionField(
            self.width,
            self.height,
            mem_w,
            prop_r,
            prop_d,
            temp_d,
            rec_r,
            self.rng,
            self.np_rng,
        )

        qdots_cfg = cfg.get("qdots", {})
        self.qdots: list[QDot] = []
        qid = 0
        for qtype, count in [
            (QDotType.STANDARD, int(qdots_cfg.get("n_standard", 120))),
            (QDotType.EXPRESS, int(qdots_cfg.get("n_express", 40))),
            (QDotType.BULKY, int(qdots_cfg.get("n_bulky", 40))),
        ]:
            for _ in range(count):
                self.qdots.append(
                    QDot(
                        id=qid,
                        type=qtype,
                        x=self.rng.randrange(self.width),
                        y=self.rng.randrange(self.height),
                        time_window=float(
                            qdots_cfg.get(f"{qtype.value.lower()}_time_window", 120 if qtype == QDotType.STANDARD else (60 if qtype == QDotType.EXPRESS else 180))
                        ),
                        delay_penalty=float(
                            qdots_cfg.get(f"{qtype.value.lower()}_delay_penalty", 1.0 if qtype == QDotType.STANDARD else (2.5 if qtype == QDotType.EXPRESS else 2.0))
                        ),
                        drift_multiplier=float(
                            qdots_cfg.get(f"{qtype.value.lower()}_drift_multiplier", 1.0 if qtype == QDotType.STANDARD else (1.25 if qtype == QDotType.EXPRESS else 0.7))
                        ),
                        volatility_multiplier=float(
                            qdots_cfg.get(f"{qtype.value.lower()}_volatility_multiplier", 1.0 if qtype == QDotType.STANDARD else (1.2 if qtype == QDotType.EXPRESS else 0.9))
                        ),
                    )
                )
                qid += 1
        self.initial_qdots = len(self.qdots)

    def step(self, config: ConfigLike, system_ruined: bool = False) -> dict[str, Any]:
        cfg = to_dict(config)
        self.time += 1

        # v0.2: prefer direct Pydantic attributes when available
        if isinstance(self._raw_config, RuinConfig):
            proc = self._raw_config.processes
            field = self._raw_config.disruption_field
            shock_rate = float(getattr(field, "shock_arrival_rate", 0.05))
            shock_mean = float(getattr(field, "mean_initial_intensity", 1.0))
            jump_rate = float(getattr(proc, "jump_intensity", 0.08))
            jump_mean = float(getattr(proc, "jump_mean_size", 3.0))
            travel_drift = float(getattr(proc, "travel_drift", 1.2))
            travel_vol = float(getattr(proc, "travel_volatility", 0.4))
        else:
            processes = cfg.get("processes", {})
            field_cfg = cfg.get("disruption_field", {})
            shock_rate = float(field_cfg.get("shock_arrival_rate", 0.05))
            shock_mean = float(field_cfg.get("shock_mean_intensity", 1.0))
            jump_rate = float(processes.get("jump_intensity", 0.08))
            jump_mean = float(processes.get("jump_mean_size", 3.0))
            travel_drift = float(processes.get("travel_drift", 1.2))
            travel_vol = float(processes.get("travel_volatility", 0.4))

        # v0.2 vectorized shock arrivals (using 1D sampler + reshape)
        n_cells = self.width * self.height
        shock_mask_1d = sample_shock_arrivals(
            n_cells,
            shock_rate,
            dt=1.0,
            rng=self.np_rng,
        )
        shock_mask = shock_mask_1d.reshape(self.height, self.width)

        for y in range(self.height):
            for x in range(self.width):
                if shock_mask[y, x]:
                    intensity = float(self.np_rng.exponential(shock_mean))
                    self.field.inject(x, y, intensity)

        self.field.step()

        # Jump-driven shocks (also vectorized)
        jump_mask_1d = sample_shock_arrivals(
            n_cells,
            jump_rate,
            dt=1.0,
            rng=self.np_rng,
        )
        jump_mask = jump_mask_1d.reshape(self.height, self.width)

        for y in range(self.height):
            for x in range(self.width):
                if jump_mask[y, x]:
                    intensity = float(self.np_rng.exponential(jump_mean))
                    self.field.inject(x, y, intensity * 0.6)

        penalty = 0.0
        active_by_cell: dict[tuple[int, int], int] = {}
        moving_by_cell: dict[tuple[int, int], int] = {}
        failed_by_cell: dict[tuple[int, int], int] = {}

        # v0.2: collect non-ruined QDots for vectorized batch step
        active_qdots = [q for q in self.qdots if not q.is_ruined]
        n_active = len(active_qdots)

        if n_active > 0:
            xs = np.array([q.x for q in active_qdots], dtype=int)
            ys = np.array([q.y for q in active_qdots], dtype=int)
            tws = np.array([q.time_window for q in active_qdots], dtype=float)
            exposures = np.array([self.field.exposure_at(q.x, q.y) for q in active_qdots], dtype=float)
            drift_m = np.array([q.drift_multiplier for q in active_qdots], dtype=float)
            vol_m = np.array([q.volatility_multiplier for q in active_qdots], dtype=float)

            new_xs, new_ys, new_tws, raw_penalties, late_mask = batch_qdot_travel_step(
                xs, ys, tws, exposures, drift_m, vol_m,
                travel_drift, travel_vol, self.width, self.height, self.np_rng
            )

            # Write back + accumulate scaled penalties + track movement/ruin
            for i, qdot in enumerate(active_qdots):
                before = (qdot.x, qdot.y)
                qdot.x = int(new_xs[i])
                qdot.y = int(new_ys[i])
                qdot.time_window = float(new_tws[i])
                qdot.field_exposure = exposures[i]

                if (qdot.x, qdot.y) != before:
                    moving_by_cell[(qdot.x, qdot.y)] = moving_by_cell.get((qdot.x, qdot.y), 0) + 1

                if late_mask[i] and not qdot.late:
                    qdot.late = True
                    qdot.is_ruined = True
                    scaled_penalty = raw_penalties[i] * qdot.delay_penalty
                    penalty += scaled_penalty
                    failed_by_cell[(qdot.x, qdot.y)] = failed_by_cell.get((qdot.x, qdot.y), 0) + 1
                elif late_mask[i]:
                    qdot.late = True
                    qdot.is_ruined = True
                    failed_by_cell[(qdot.x, qdot.y)] = failed_by_cell.get((qdot.x, qdot.y), 0) + 1

        for (x, y), failed in failed_by_cell.items():
            self.field.inject(x, y, failed * self.qdot_feedback_weight)

        local_pressures: list[float] = []
        for y in range(self.height):
            for x in range(self.width):
                active = active_by_cell.get((x, y), 0)
                failed = failed_by_cell.get((x, y), 0)
                failed_ratio = failed / max(active + failed, 1)
                local_pressures.append(
                    local_chaos_pressure(self.field.exposure_at(x, y), 0.0, failed_ratio)
                )

        late_count = sum(1 for qdot in self.qdots if qdot.late)
        active_count = sum(1 for qdot in self.qdots if not qdot.is_ruined)
        late_ratio = late_count / max(active_count, 1)

        self.global_chaos_pressure = global_chaos_pressure(local_pressures, late_ratio)
        self.global_order_pressure = global_order_pressure(
            active_count, late_count, self.max_capacity, self.initial_qdots
        )

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
        snap = {
            "time": self.time,
            "d_state": self.d_state.value,
            "chaos_pressure": round(self.global_chaos_pressure, 6),
            "order_pressure": round(self.global_order_pressure, 6),
            "penalty": round(penalty, 6),
            "late_count": late_count,
            "active_count": active_count,
            "qdots": [qdot.snapshot() for qdot in self.qdots],
        }
        if self.store_field_snapshots:
            snap["field"] = self.field.snapshot()  # list of lists, serializable
        return snap

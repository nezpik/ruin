from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ruin.config import ConfigLike, as_config
from ruin.core.processes import (
    batch_qdot_travel_step,
    blocked_fraction,
    make_rng,
    sample_shock_events,
)
from ruin.core.qdot import QDot, QDotStatus, QDotType
from ruin.metrics.pressure import (
    DState,
    choose_d_state,
    global_chaos_pressure,
    global_order_pressure,
)
from ruin.state_space.disruption_field import DisruptionField
from ruin.state_space.regime import REGIMES, Regime, RegimeProcess

QDOT_TYPES = (QDotType.STANDARD, QDotType.EXPRESS, QDotType.BULKY)

_ACTIVE, _LATE, _DELIVERED, _DELIVERED_LATE, _FAILED = range(5)
_STATUS_NAMES = (
    QDotStatus.ACTIVE.value,
    QDotStatus.LATE.value,
    QDotStatus.DELIVERED.value,
    QDotStatus.DELIVERED_LATE.value,
    QDotStatus.FAILED.value,
)


class ProbabilitySquare:
    """The destiny frame D: a grid, its disruption field, and the Q dots in it.

    Q dots are held as parallel arrays. Each one enters D at an origin cell
    with a destination and a time window, travels toward the destination,
    and leaves D only by terminating: delivered on time, delivered late, or
    failed (abandoned after a full extra window of lateness).
    """

    def __init__(
        self,
        config: ConfigLike,
        seed: int | None = None,
        store_field_snapshots: bool = False,
        store_qdot_snapshots: bool = True,
    ) -> None:
        cfg = as_config(config)
        self.config = cfg
        self.width = int(cfg.simulation.grid_width)
        self.height = int(cfg.simulation.grid_height)
        self.dt = float(cfg.simulation.dt)
        self.rng = make_rng(cfg.simulation.seed if seed is None else seed)
        self.time = 0

        square = cfg.probability_square
        self.d_state = DState(square.initial_d_state.upper())
        self.thresholds = square.d_state_thresholds
        self.qdot_feedback_weight = float(square.qdot_feedback_weight)
        self.base_risk = float(square.base_risk_intensity)
        self.congestion_memory = float(square.congestion_memory)
        self.snapshot_interval = int(square.snapshot_interval)
        self.regime_process = (
            RegimeProcess(square.regime, square.d_state_thresholds)
            if square.regime.enabled
            else None
        )
        self.regime_level = 0
        # Per-cell memory of where Q dots have recently stalled, in [0, 1].
        self.congestion = np.zeros(self.width * self.height, dtype=float)

        field_cfg = cfg.disruption_field
        self.field = DisruptionField(
            self.width,
            self.height,
            field_cfg.memory_weight,
            field_cfg.propagation_radius,
            field_cfg.propagation_decay,
            field_cfg.temporal_decay,
            field_cfg.recovery_rate,
            field_cfg.propagation_share,
            square.boundary_mode,
        )

        qdots = cfg.qdots
        params = (qdots.standard, qdots.express, qdots.bulky)
        self._time_window = np.array([p.time_window for p in params], dtype=float)
        self._delay_penalty = np.array([p.delay_penalty for p in params], dtype=float)
        self._drift_mult = np.array([p.drift_multiplier for p in params], dtype=float)
        self._vol_mult = np.array([p.volatility_multiplier for p in params], dtype=float)
        self._reward = np.array([p.completion_reward for p in params], dtype=float)

        counts = np.array([qdots.n_standard, qdots.n_express, qdots.n_bulky], dtype=int)
        self._type_mix = counts / counts.sum() if counts.sum() > 0 else np.full(3, 1.0 / 3.0)

        self.previous_chaos_pressure = 0.0
        self.global_chaos_pressure = 0.0
        self.global_order_pressure = 0.0
        self.snapshots: list[dict[str, Any]] = []
        self.store_field_snapshots = store_field_snapshots
        self.store_qdot_snapshots = store_qdot_snapshots

        # Lifetime counters.
        self.spawned = 0
        self.delivered = 0
        self.delivered_late = 0
        self.failed = 0

        # Q-dot state (one entry per Q dot currently inside D).
        self._ids = np.zeros(0, dtype=int)
        self._type = np.zeros(0, dtype=int)
        self._x = np.zeros(0, dtype=int)
        self._y = np.zeros(0, dtype=int)
        self._dest_x = np.zeros(0, dtype=int)
        self._dest_y = np.zeros(0, dtype=int)
        self._slack = np.zeros(0, dtype=float)
        self._carry = np.zeros(0, dtype=float)
        self._late = np.zeros(0, dtype=bool)
        self._exposure = np.zeros(0, dtype=float)
        self._order = np.zeros(0, dtype=float)
        self._origin_x = np.zeros(0, dtype=int)
        self._origin_y = np.zeros(0, dtype=int)
        self._spawn_time = np.zeros(0, dtype=int)
        self._cost = np.zeros(0, dtype=float)  # delay penalty accrued so far

        # Loss ledger: where loss accrued, and one row per terminated Q dot.
        self.loss_field = np.zeros(self.width * self.height, dtype=float)
        self._ledger: list[np.ndarray] = []

        self._spawn(np.repeat(np.arange(3), counts))
        self.initial_qdots = int(counts.sum())
        if self.regime_process is not None:
            self.regime_level = self.regime_process.initial_level(self.rng)

    def redraw_regime(self, probabilities: np.ndarray) -> None:
        """Replace the hidden regime with a draw from `probabilities` (one per level).

        Used to ask what the future looks like to someone who knows the whole
        state of D except which regime it is in, and who holds those beliefs
        about it."""
        if self.regime_process is not None:
            p = np.asarray(probabilities, dtype=float)
            self.regime_level = int(self.rng.choice(4, p=p / p.sum()))

    def forget_regime(self) -> None:
        """Redraw the hidden regime from its long-run mix: someone who knows
        nothing about which regime D is in."""
        if self.regime_process is not None:
            self.redraw_regime(np.asarray(self.regime_process.config.stationary()))

    @property
    def regime(self) -> Regime | None:
        """The hidden regime D is in right now (None when regimes are off)."""
        return REGIMES[self.regime_level] if self.regime_process is not None else None

    # ------------------------------------------------------------------ Q dots

    @property
    def active_count(self) -> int:
        return len(self._ids)

    @property
    def qdots(self) -> list[QDot]:
        """Q dots currently inside D, as records."""
        return self._records(
            np.zeros(self.active_count, dtype=bool), np.zeros(self.active_count, dtype=bool)
        )

    def _spawn(self, types: np.ndarray) -> None:
        n = len(types)
        if n == 0:
            return
        x = self.rng.integers(0, self.width, size=n)
        y = self.rng.integers(0, self.height, size=n)
        dest_x = self.rng.integers(0, self.width, size=n)
        dest_y = self.rng.integers(0, self.height, size=n)
        if self.width > 1:
            # A commitment must have somewhere to go.
            same = (dest_x == x) & (dest_y == y)
            dest_x[same] = (dest_x[same] + 1) % self.width

        self._ids = np.concatenate([self._ids, np.arange(self.spawned, self.spawned + n)])
        self._type = np.concatenate([self._type, types.astype(int)])
        self._x = np.concatenate([self._x, x])
        self._y = np.concatenate([self._y, y])
        self._dest_x = np.concatenate([self._dest_x, dest_x])
        self._dest_y = np.concatenate([self._dest_y, dest_y])
        self._slack = np.concatenate([self._slack, self._time_window[types]])
        self._carry = np.concatenate([self._carry, np.zeros(n)])
        self._late = np.concatenate([self._late, np.zeros(n, dtype=bool)])
        self._exposure = np.concatenate([self._exposure, np.zeros(n)])
        self._order = np.concatenate([self._order, np.zeros(n)])
        self._origin_x = np.concatenate([self._origin_x, x])
        self._origin_y = np.concatenate([self._origin_y, y])
        self._spawn_time = np.concatenate([self._spawn_time, np.full(n, self.time)])
        self._cost = np.concatenate([self._cost, np.zeros(n)])
        self.spawned += n

    _STATE_ARRAYS = (
        "_ids",
        "_type",
        "_x",
        "_y",
        "_dest_x",
        "_dest_y",
        "_slack",
        "_carry",
        "_late",
        "_exposure",
        "_order",
        "_origin_x",
        "_origin_y",
        "_spawn_time",
        "_cost",
    )

    def _keep(self, mask: np.ndarray) -> None:
        for name in self._STATE_ARRAYS:
            setattr(self, name, getattr(self, name)[mask])

    def fork(self, seed: int | np.random.SeedSequence) -> ProbabilitySquare:
        """An independent copy of D as it stands now, with a fresh random stream.

        The fork shares the past and nothing of the future: stepping it does
        not touch this square. Used to branch many possible futures from one
        present. The copy starts with empty snapshot and ledger histories.
        """
        other = copy.copy(self)
        for name in self._STATE_ARRAYS:
            setattr(other, name, getattr(self, name).copy())
        other.field = copy.copy(self.field)
        other.field.grid = self.field.grid.copy()
        other.congestion = self.congestion.copy()
        other.loss_field = self.loss_field.copy()
        other.snapshots = []
        other._ledger = []
        other.rng = np.random.default_rng(seed)
        return other

    def ledger(self) -> list[dict[str, Any]]:
        """One row per Q dot that has left D or is still in it, with the loss it carried."""
        columns = [
            np.column_stack(
                [
                    self._ids,
                    self._type,
                    self._origin_x,
                    self._origin_y,
                    self._dest_x,
                    self._dest_y,
                    self._spawn_time,
                    np.full(len(self._ids), -1),
                    np.where(self._late, _LATE, _ACTIVE),
                    self._cost,
                ]
            ),
            *self._ledger,
        ]
        rows = np.concatenate(columns) if columns else np.zeros((0, 10))
        return [
            {
                "id": int(r[0]),
                "type": QDOT_TYPES[int(r[1])].value,
                "origin": (int(r[2]), int(r[3])),
                "destination": (int(r[4]), int(r[5])),
                "spawn_time": int(r[6]),
                "end_time": int(r[7]) if r[7] >= 0 else None,
                "status": _STATUS_NAMES[int(r[8])],
                "loss": round(float(r[9]), 6),
            }
            for r in rows
        ]

    def _records(self, arrived: np.ndarray, failed: np.ndarray) -> list[QDot]:
        records: list[QDot] = []
        for i in range(len(self._ids)):
            late = bool(self._late[i])
            if failed[i]:
                status = QDotStatus.FAILED
            elif arrived[i]:
                status = QDotStatus.DELIVERED_LATE if late else QDotStatus.DELIVERED
            else:
                status = QDotStatus.LATE if late else QDotStatus.ACTIVE
            kind = int(self._type[i])
            records.append(
                QDot(
                    id=int(self._ids[i]),
                    type=QDOT_TYPES[kind],
                    x=int(self._x[i]),
                    y=int(self._y[i]),
                    dest_x=int(self._dest_x[i]),
                    dest_y=int(self._dest_y[i]),
                    time_window=float(self._slack[i]),
                    delay_penalty=float(self._delay_penalty[kind]),
                    status=status,
                    field_exposure=float(self._exposure[i]),
                    order_pressure=float(self._order[i]),
                    d_state_exposure=self.d_state.value,
                )
            )
        return records

    def _cell_counts(self, mask: np.ndarray) -> np.ndarray:
        cells = self._y[mask] * self.width + self._x[mask]
        return np.bincount(cells, minlength=self.width * self.height)

    # -------------------------------------------------------------------- step

    def step(self, system_ruined: bool = False) -> dict[str, Any]:
        cfg = self.config
        proc = cfg.processes
        field_cfg = cfg.disruption_field
        dt = self.dt
        self.time += 1

        # 0. The regime of D moves, pulled by last step's pressures.
        rate_mult = size_mult = noise_mult = 1.0
        if self.regime_process is not None:
            regime_cfg = self.regime_process.config
            self.regime_level = self.regime_process.step(
                self.regime_level,
                self.global_chaos_pressure,
                self.global_order_pressure,
                self.rng,
            )
            rate_mult = regime_cfg.shock_rate[self.regime_level]
            size_mult = regime_cfg.shock_size[self.regime_level]
            noise_mult = regime_cfg.travel_noise[self.regime_level]

        # 1. New uncertainty enters: demand arrivals and disruption shocks.
        arrivals = int(self.rng.poisson(proc.demand_arrival_rate * dt))
        self._spawn(self.rng.choice(3, size=arrivals, p=self._type_mix))

        shock_count, shock_mass = 0, 0.0
        if field_cfg.enabled:
            xs, ys, intensities = sample_shock_events(
                field_cfg.shock_arrival_rate * rate_mult,
                field_cfg.mean_initial_intensity * size_mult,
                self.width,
                self.height,
                dt=dt,
                rng=self.rng,
                size_sigma=field_cfg.shock_size_sigma,
            )
            self.field.inject_many(xs, ys, intensities)
            shock_count, shock_mass = len(xs), float(intensities.sum())

        # 2. The field spreads and decays; where Q dots are stalled it lingers.
        gain_map = None
        if field_cfg.congestion_persistence > 0:
            base = self.field.gain
            load = self.congestion.reshape(self.height, self.width)
            gain_map = base + field_cfg.congestion_persistence * load * (field_cfg.max_gain - base)
        field_retention = self.field.retention(gain_map)
        self.field.step(gain_map)

        # 3. Q dots move toward their destinations through the field.
        self._exposure = (
            self.field.exposure_at(self._x, self._y) + self.base_risk
        ) * field_cfg.qdot_exposure_multiplier
        was_late = self._late.copy()
        self._x, self._y, self._carry, moved, arrived = batch_qdot_travel_step(
            self._x,
            self._y,
            self._dest_x,
            self._dest_y,
            self._carry,
            self._exposure,
            self._drift_mult[self._type],
            self._vol_mult[self._type],
            proc.travel_drift,
            proc.travel_volatility * noise_mult,
            proc.jump_intensity * noise_mult,
            proc.jump_mean_size,
            self.rng,
            dt=dt,
        )

        # 4. Time windows run down; on-time arrivals earn, late dots cost.
        self._slack = self._slack - dt
        travelling = ~arrived
        self._late = was_late | (travelling & (self._slack <= 0))
        newly_late = self._late & ~was_late
        failed = travelling & (self._slack <= -self._time_window[self._type])

        on_time = arrived & ~was_late
        income = float(self._reward[self._type[on_time]].sum())
        accruing = self._late & travelling
        # A late dot costs its delay penalty each step, up to double in a fully blocked cell.
        accrued = (
            self._delay_penalty[self._type[accruing]]
            * dt
            * (1.0 + blocked_fraction(self._exposure[accruing]))
        )
        penalty = float(accrued.sum())
        self._cost[accruing] += accrued
        np.add.at(self.loss_field, self._y[accruing] * self.width + self._x[accruing], accrued)
        penalty_by_type = np.bincount(self._type[accruing], weights=accrued, minlength=3)

        # 5. Q dots reshape D: lateness and failure add local disruption.
        feedback = newly_late | failed
        self.field.inject_many(self._x[feedback], self._y[feedback], self.qdot_feedback_weight)

        # 6. Pressures and D-state, over the Q dots still travelling.
        in_flight = travelling & ~failed
        active_count = int(in_flight.sum())
        late_count = int((self._late & in_flight).sum())
        moving_count = int((moved & in_flight).sum())
        late_ratio = late_count / max(active_count, 1)

        active_cells = self._cell_counts(in_flight)
        moving_cells = self._cell_counts(moved & in_flight)
        late_cells = self._cell_counts(self._late & in_flight)

        # Congestion memory: an exponentially weighted record of the share of
        # Q dots in each cell that failed to move.
        stalled_share = (active_cells - moving_cells) / np.maximum(active_cells, 1)
        self.congestion = (
            self.congestion_memory * self.congestion
            + (1.0 - self.congestion_memory) * stalled_share
        )

        blocked = blocked_fraction(self.field.grid.ravel() + self.base_risk)
        local_chaos = blocked + self.congestion + late_cells / np.maximum(active_cells, 1)
        local_order = np.maximum(0, active_cells + moving_cells - late_cells)
        self._order = local_order[self._y * self.width + self._x].astype(float)

        self.global_chaos_pressure = global_chaos_pressure(local_chaos, late_ratio)
        self.global_order_pressure = global_order_pressure(active_count, moving_count, late_count)
        self.d_state = choose_d_state(
            system_ruined,
            self.global_chaos_pressure,
            self.previous_chaos_pressure,
            self.thresholds.stressed_chaos_pressure,
            self.thresholds.disrupted_chaos_pressure,
            self.thresholds.chaotic_chaos_pressure,
            self.thresholds.recovering_decay_ratio,
        )
        self.previous_chaos_pressure = self.global_chaos_pressure

        # 7. Record, then let terminated Q dots leave D.
        n_delivered = int(on_time.sum())
        n_delivered_late = int((arrived & was_late).sum())
        n_failed = int(failed.sum())
        self.delivered += n_delivered
        self.delivered_late += n_delivered_late
        self.failed += n_failed

        snapshot: dict[str, Any] = {
            "time": self.time,
            "d_state": self.d_state.value,
            "chaos_pressure": round(self.global_chaos_pressure, 6),
            "order_pressure": round(self.global_order_pressure, 6),
            "penalty": round(penalty, 6),
            "penalty_by_type": [round(float(v), 6) for v in penalty_by_type],
            "income": round(income, 6),
            "regime": self.regime.value if self.regime is not None else None,
            "shock_count": shock_count,
            "shock_mass": round(shock_mass, 6),
            "field_retention": round(field_retention, 6),
            "late_count": late_count,
            "missed": int(newly_late.sum()),
            "active_count": active_count,
            "delivered": n_delivered,
            "delivered_late": n_delivered_late,
            "failed": n_failed,
            "field_mean": round(float(self.field.grid.mean()), 6),
        }
        if self.time % self.snapshot_interval == 0:
            if self.store_qdot_snapshots:
                snapshot["qdots"] = [q.snapshot() for q in self._records(arrived, failed)]
            if self.store_field_snapshots:
                shape = (self.height, self.width)
                snapshot["field"] = self.field.snapshot()
                snapshot["chaos_field"] = np.round(local_chaos.reshape(shape), 4).tolist()
                snapshot["order_field"] = local_order.reshape(shape).tolist()
                snapshot["loss_field"] = np.round(self.loss_field.reshape(shape), 4).tolist()
        self.snapshots.append(snapshot)

        leaving = ~in_flight
        if leaving.any():
            status = np.where(failed, _FAILED, np.where(was_late, _DELIVERED_LATE, _DELIVERED))[
                leaving
            ]
            self._ledger.append(
                np.column_stack(
                    [
                        self._ids[leaving],
                        self._type[leaving],
                        self._origin_x[leaving],
                        self._origin_y[leaving],
                        self._dest_x[leaving],
                        self._dest_y[leaving],
                        self._spawn_time[leaving],
                        np.full(int(leaving.sum()), self.time),
                        status,
                        self._cost[leaving],
                    ]
                )
            )
        self._keep(in_flight)
        return snapshot

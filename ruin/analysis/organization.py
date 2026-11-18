"""Chaotic organization: the structure that forms when Q dots try to keep order in D.

Disruption is not only disorder. Blocked regions form, persist, move, and clear;
Q dots pile up in them or steer around them; pressure concentrates in some
places; loss concentrates in fewer still. `StructureRecorder` watches the
engine step by step and measures that structure:

- **Jams**: connected regions where disruption blocks at least half a cell
  (intensity at least `JAM_THRESHOLD`). They are tracked from step to step, so
  each has a birth, a lifetime, a peak, a death, and a record of the Q dots and
  loss that passed through it.
- **Pile-up**: how many more Q dots sit inside jam cells than the same cells
  hold when no jam is near. Above 1, jams trap; near 1, Q dots pass through.
- **Diversion**: how much more *flow* (Q dots moving into a cell per step) the
  ring of cells just outside a jam carries than the same cells do when no jam is
  near. Above 1, traffic is being funnelled beside jams. It is flow, not density,
  because Q dots also queue beside a jam, slowed by its spread, and queueing is
  not diversion.
- **Corridors**: connected sets of ring cells that carry persistently more flow
  than they do away from jams: forced routes through low-disruption cells. They
  can only form if Q dots steer (`processes.avoidance` > 0).
- **Pressure zones**: where chaos pressure concentrates over the shift, and
  whether those places stay put.
- **Recovery**: how long jams take to clear, and whether Q dots flow back into
  cleared cells.
- **Ruin zones**: where loss concentrates, how much of it accrues inside the
  structures above, and which jams carried the loss in the steps before ruin.

Every density comparison uses a per-cell baseline (the cell's own mean when no
jam is within `RING` cells), so the geometry of D, where random trips cross more
often in the middle, is not mistaken for organization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ruin.core.processes import EXPOSURE_SATURATION

# A cell is part of a jam when disruption blocks at least half of it.
JAM_THRESHOLD = EXPOSURE_SATURATION / 2.0
# Smallest region that counts as a jam, in cells.
MIN_JAM_AREA = 3
# Width of the ring around a jam, in cells (Manhattan distance).
RING = 2
# A cell counts as clear again once intensity falls below this share of the jam threshold.
CLEAR_FRACTION = 0.5
# Steps a cell is "recovering" after last being part of a jam.
RECOVERY_WINDOW = 8
# Fewest observations of a cell, in a given state and away from jams, for a baseline.
MIN_BASELINE_STEPS = 20
# A ring cell is a corridor cell when it carries this many times its baseline flow ...
CORRIDOR_EXCESS = 1.5
# ... and that excess is at least this many standard errors (Poisson) above the baseline,
# so that chance fluctuations in sparsely visited cells are not read as routes.
CORRIDOR_Z = 3.0
# A corridor is a route, not a cell: fewest connected cells that count.
MIN_CORRIDOR_CELLS = 3
# Steps before ruin over which loss is attributed to jams.
RUIN_WINDOW = 24
# Share of cells (by mean chaos pressure) that make up the pressure zones.
ZONE_QUANTILE = 0.9


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label the 4-connected regions of a boolean grid.

    Returns (labels, count): labels are 1..count, background is 0."""
    labels = np.zeros(mask.shape, dtype=int)
    height, width = mask.shape
    count = 0
    for y0, x0 in zip(*np.nonzero(mask), strict=True):
        if labels[y0, x0]:
            continue
        count += 1
        stack = [(int(y0), int(x0))]
        labels[y0, x0] = count
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = count
                    stack.append((ny, nx))
    return labels, count


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a boolean grid by `radius` cells in Manhattan distance."""
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


@dataclass(frozen=True)
class StepFields:
    """What the engine hands the recorder after each step (all shaped (height, width))."""

    time: int
    regime: str | None
    intensity: np.ndarray  # disruption intensity
    density: np.ndarray  # Q dots in flight per cell
    flow: np.ndarray  # Q dots that moved into the cell this step
    chaos: np.ndarray  # local chaos pressure
    accrued: np.ndarray  # loss accrued in the cell this step
    loss: np.ndarray  # loss accrued in the cell so far


@dataclass
class Jam:
    """One tracked jam."""

    id: int
    born: int
    born_regime: str | None
    parent: int | None
    last_seen: int
    peak_area: int = 0
    peak_intensity: float = 0.0
    peak_mass: float = 0.0
    peak_time: int = 0
    area_steps: int = 0
    dot_steps: float = 0.0
    loss: float = 0.0
    centroid: tuple[float, float] = (0.0, 0.0)
    peak_cells: np.ndarray | None = None  # boolean mask of the jam when it was largest
    alive: bool = True

    @property
    def lifetime(self) -> int:
        return self.last_seen - self.born + 1


@dataclass
class StructureRecorder:
    """Measures chaotic organization as the engine runs. Pass it to `run_trajectory`."""

    jam_threshold: float = JAM_THRESHOLD
    min_jam_area: int = MIN_JAM_AREA
    ring: int = RING
    jams: dict[int, Jam] = field(default_factory=dict)
    steps: int = 0
    ruin_time: int | None = None
    horizon: int = 0
    _next_id: int = 1
    _id_map: np.ndarray | None = None
    _shape: tuple[int, int] = (0, 0)
    _arrays: dict[str, np.ndarray] = field(default_factory=dict)
    # Where loss accrued: inside jams, in the ring around them, in the aftermath of a jam
    # (cells that were jammed within the last RECOVERY_WINDOW steps), and everywhere else.
    _loss_in: dict[str, float] = field(
        default_factory=lambda: {"jam": 0.0, "ring": 0.0, "aftermath": 0.0, "far": 0.0}
    )
    _window: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    _ruin_window: list[tuple[np.ndarray, np.ndarray]] | None = None
    _recovering_steps: int = 0
    _recovering_density: float = 0.0
    _recovering_expected: float = 0.0
    _final: StepFields | None = None

    # ------------------------------------------------------------------ recording

    def _setup(self, shape: tuple[int, int]) -> None:
        self._shape = shape
        cells = shape[0] * shape[1]
        names = (
            "far_sum", "far_n", "jam_sum", "jam_n", "ring_sum", "ring_n",
            "far_flow", "ring_flow",
            "chaos_sum", "chaos_first", "chaos_second", "recover_sum", "recover_n",
        )  # fmt: skip
        self._arrays = {name: np.zeros(cells) for name in names}
        self._arrays["last_jam"] = np.full(cells, -10_000.0)
        self._id_map = np.zeros(shape, dtype=int)

    def observe(self, fields: StepFields) -> None:
        """Take in one step."""
        if self._id_map is None:
            self._setup(fields.intensity.shape)
        assert self._id_map is not None
        self.steps += 1
        self._final = fields
        a = self._arrays
        t = fields.time

        id_map = self._track(fields)
        jam = id_map > 0
        ring = dilate(jam, self.ring) & ~jam
        # A cell's baseline uses only quiet steps: no jam within the ring, and not in the
        # aftermath of one (when traffic surges back in and would inflate the baseline).
        quiet = (t - a["last_jam"] > RECOVERY_WINDOW).reshape(jam.shape)
        far = ~dilate(jam, self.ring) & quiet
        density = fields.density.ravel().astype(float)

        flow = fields.flow.ravel().astype(float)
        a["ring_flow"][ring.ravel()] += flow[ring.ravel()]
        a["far_flow"][far.ravel()] += flow[far.ravel()]
        aftermath = ~jam & ~ring & ~far
        self._loss_in["aftermath"] += float(fields.accrued[aftermath].sum())
        for key, mask in (("jam", jam), ("ring", ring), ("far", far)):
            flat = mask.ravel()
            a[f"{key}_sum"][flat] += density[flat]
            a[f"{key}_n"][flat] += 1.0
            self._loss_in[key] += float(fields.accrued[mask].sum())

        # Cells that were part of a jam a moment ago and have since cleared.
        recovering = (
            (t - a["last_jam"] <= RECOVERY_WINDOW)
            & (fields.intensity.ravel() < self.jam_threshold * CLEAR_FRACTION)
            & ~jam.ravel()
        )
        a["recover_sum"][recovering] += density[recovering]
        a["recover_n"][recovering] += 1.0
        a["last_jam"][jam.ravel()] = t

        a["chaos_sum"] += fields.chaos.ravel()
        half = "chaos_first" if t <= self.horizon // 2 else "chaos_second"
        a[half] += fields.chaos.ravel()

        self._window.append((fields.accrued.copy(), id_map.copy()))
        if len(self._window) > RUIN_WINDOW:
            self._window.pop(0)

    def set_horizon(self, steps: int) -> None:
        """Tell the recorder how long the shift is (used to split it in halves)."""
        self.horizon = steps

    def mark_ruin(self, time: int) -> None:
        """Note the moment of ruin, keeping the recent past for attribution."""
        if self.ruin_time is None:
            self.ruin_time = time
            self._ruin_window = list(self._window)

    # ------------------------------------------------------------------- tracking

    def _track(self, fields: StepFields) -> np.ndarray:
        assert self._id_map is not None
        previous = self._id_map
        mask = fields.intensity >= self.jam_threshold
        labels, count = label_components(mask)
        regions = [(int((labels == k).sum()), k) for k in range(1, count + 1)]
        regions = [r for r in regions if r[0] >= self.min_jam_area]
        regions.sort(reverse=True)  # largest first: it keeps an identity ahead of smaller ones

        id_map = np.zeros_like(previous)
        claimed: set[int] = set()
        for area, k in regions:
            cells = labels == k
            overlap = previous[cells]
            candidates = [int(i) for i in np.unique(overlap[overlap > 0])]
            candidates.sort(key=lambda i: -int((overlap == i).sum()))
            heir = next((i for i in candidates if i not in claimed), None)
            if heir is None:
                heir = self._next_id
                self._next_id += 1
                self.jams[heir] = Jam(
                    id=heir,
                    born=fields.time,
                    born_regime=fields.regime,
                    parent=candidates[0] if candidates else None,
                    last_seen=fields.time,
                )
            claimed.add(heir)
            id_map[cells] = heir
            jam = self.jams[heir]
            jam.last_seen = fields.time
            jam.area_steps += area
            mass = float(fields.intensity[cells].sum())
            jam.dot_steps += float(fields.density[cells].sum())
            jam.loss += float(fields.accrued[cells].sum())
            peak = float(fields.intensity[cells].max())
            if area >= jam.peak_area:
                ys, xs = np.nonzero(cells)
                jam.centroid = (float(ys.mean()), float(xs.mean()))
                jam.peak_cells = cells.copy()
            jam.peak_area = max(jam.peak_area, area)
            jam.peak_intensity = max(jam.peak_intensity, peak)
            if mass >= jam.peak_mass:
                jam.peak_mass, jam.peak_time = mass, fields.time
        for jam in self.jams.values():
            if jam.alive and jam.id not in claimed:
                jam.alive = False
        self._id_map = id_map
        return id_map

    # -------------------------------------------------------------------- summary

    def _baseline(self) -> np.ndarray:
        """Per-cell mean density away from jams (NaN where too few observations)."""
        a = self._arrays
        mean = np.full_like(a["far_sum"], np.nan)
        ok = a["far_n"] >= MIN_BASELINE_STEPS
        mean[ok] = a["far_sum"][ok] / a["far_n"][ok]
        return mean

    def _ratio(self, key: str) -> float | None:
        """Observed over expected Q-dot density in `key` cells, using each cell's own baseline."""
        a = self._arrays
        base = self._baseline()
        ok = ~np.isnan(base) & (a[f"{key}_n"] > 0) & (base > 0)
        expected = float((a[f"{key}_n"][ok] * base[ok]).sum())
        if expected <= 0:
            return None
        return float(a[f"{key}_sum"][ok].sum() / expected)

    def _flow_baseline(self) -> np.ndarray:
        a = self._arrays
        mean = np.full_like(a["far_flow"], np.nan)
        ok = a["far_n"] >= MIN_BASELINE_STEPS
        mean[ok] = a["far_flow"][ok] / a["far_n"][ok]
        return mean

    def _diversion(self) -> float | None:
        """Observed over expected flow into ring cells, each against its own baseline."""
        a = self._arrays
        base = self._flow_baseline()
        ok = ~np.isnan(base) & (a["ring_n"] > 0) & (base > 0)
        expected = float((a["ring_n"][ok] * base[ok]).sum())
        return float(a["ring_flow"][ok].sum() / expected) if expected > 0 else None

    def _corridors(self) -> dict[str, Any]:
        a = self._arrays
        base = self._flow_baseline()
        ok = (a["ring_n"] >= MIN_BASELINE_STEPS) & ~np.isnan(base) & (base > 0)
        excess = np.zeros_like(base)
        z = np.zeros_like(base)
        expected = np.zeros_like(base)
        expected[ok] = a["ring_n"][ok] * base[ok]
        excess[ok] = a["ring_flow"][ok] / expected[ok]
        z[ok] = (a["ring_flow"][ok] - expected[ok]) / np.sqrt(np.maximum(expected[ok], 1.0))
        cells = (ok & (excess >= CORRIDOR_EXCESS) & (z >= CORRIDOR_Z)).reshape(self._shape)
        labels, count = label_components(cells)
        sizes = [int((labels == k).sum()) for k in range(1, count + 1)]
        for k, size in enumerate(sizes, start=1):
            if size < MIN_CORRIDOR_CELLS:
                cells[labels == k] = False
        sizes = [size for size in sizes if size >= MIN_CORRIDOR_CELLS]
        total_ring = float(a["ring_flow"].sum())
        carried = float(a["ring_flow"][cells.ravel()].sum())
        return {
            "corridors": len(sizes),
            "corridor_cells": int(cells.sum()),
            "largest_corridor_cells": max(sizes, default=0),
            "mean_excess": round(float(excess[cells.ravel()].mean()), 3) if cells.any() else None,
            "share_of_ring_flow": round(carried / total_ring, 3) if total_ring > 0 else None,
        }

    def _pressure(self) -> dict[str, Any]:
        a = self._arrays
        mean = (a["chaos_sum"] / max(self.steps, 1)).reshape(self._shape)
        flat = mean.ravel()
        cut = float(np.quantile(flat, ZONE_QUANTILE))
        hot = mean >= cut
        labels, count = label_components(hot)
        zones: list[dict[str, Any]] = []
        for k in range(1, count + 1):
            cells = labels == k
            ys, xs = np.nonzero(cells)
            zones.append(
                {
                    "cells": int(cells.sum()),
                    "mean_chaos": round(float(mean[cells].mean()), 4),
                    "centroid": (round(float(ys.mean()), 1), round(float(xs.mean()), 1)),
                }
            )
        zones.sort(key=lambda z: -int(z["cells"]))
        first, second = a["chaos_first"], a["chaos_second"]
        persistence = (
            float(np.corrcoef(first, second)[0, 1])
            if first.std() > 0 and second.std() > 0
            else None
        )
        return {
            "top_decile_share_of_pressure": round(float(flat[flat >= cut].sum() / flat.sum()), 3)
            if flat.sum() > 0
            else None,
            "zones": zones[:5],
            "map_correlation_between_halves": None
            if persistence is None
            else round(persistence, 3),
        }

    def _recovery(self) -> dict[str, Any]:
        a = self._arrays
        base = self._baseline()
        ok = ~np.isnan(base) & (a["recover_n"] > 0) & (base > 0)
        expected = float((a["recover_n"][ok] * base[ok]).sum())
        flow = float(a["recover_sum"][ok].sum() / expected) if expected > 0 else None
        cleared = [j for j in self.jams.values() if not j.alive]
        decay = [j.last_seen - j.peak_time for j in cleared]
        return {
            "jams_cleared": len(cleared),
            "median_steps_peak_to_clear": float(np.median(decay)) if decay else None,
            "recovering_cell_steps": int(a["recover_n"].sum()),
            "flow_into_recovering_cells": None if flow is None else round(flow, 3),
        }

    def _loss(self) -> dict[str, Any]:
        assert self._final is not None
        a = self._arrays
        total_loss = float(self._final.loss.sum())
        flat = np.sort(self._final.loss.ravel())[::-1]
        cumulative = np.cumsum(flat)
        half = int(np.searchsorted(cumulative, 0.5 * total_loss) + 1) if total_loss > 0 else 0
        share_jam = self._loss_in["jam"] / total_loss if total_loss > 0 else None
        cell_steps = max(float(self.steps * self._shape[0] * self._shape[1]), 1.0)
        out: dict[str, Any] = {
            "total_loss": round(total_loss, 3),
            "cells_carrying_half_the_loss": half,
            "share_of_cells_carrying_half_the_loss": round(half / flat.size, 4) if half else None,
            "loss_share_inside_jams": None if share_jam is None else round(share_jam, 3),
            "loss_share_in_ring": None
            if total_loss <= 0
            else round(self._loss_in["ring"] / total_loss, 3),
            "jam_share_of_cell_steps": round(float(a["jam_n"].sum()) / cell_steps, 4),
        }
        out["loss_enrichment_inside_jams"] = (
            None
            if share_jam is None or out["jam_share_of_cell_steps"] <= 0
            else round(share_jam / out["jam_share_of_cell_steps"], 2)
        )
        return out

    def _ruin_zone(self) -> dict[str, Any] | None:
        if self._ruin_window is None:
            return None
        per_jam: dict[int, float] = {}
        outside = 0.0
        for accrued, id_map in self._ruin_window:
            outside += float(accrued[id_map == 0].sum())
            for jam_id in np.unique(id_map[id_map > 0]):
                per_jam[int(jam_id)] = per_jam.get(int(jam_id), 0.0) + float(
                    accrued[id_map == jam_id].sum()
                )
        total = outside + sum(per_jam.values())
        ranked = sorted(per_jam.items(), key=lambda kv: -kv[1])
        return {
            "ruin_time": self.ruin_time,
            "window_steps": len(self._ruin_window),
            "loss_in_window": round(total, 3),
            "share_outside_any_jam": None if total <= 0 else round(outside / total, 3),
            "top_jams": [
                {
                    "id": jam_id,
                    "share": round(loss / total, 3),
                    "centroid": tuple(round(c, 1) for c in self.jams[jam_id].centroid),
                    "peak_area": self.jams[jam_id].peak_area,
                    "born": self.jams[jam_id].born,
                    "born_regime": self.jams[jam_id].born_regime,
                }
                for jam_id, loss in ranked[:3]
                if total > 0
            ],
        }

    def summary(self) -> dict[str, Any]:
        """Everything measured, as plain numbers."""
        if self._id_map is None:
            raise ValueError("The recorder has not observed any steps")
        lifetimes = [j.lifetime for j in self.jams.values()]
        by_regime: dict[str, int] = {}
        for j in self.jams.values():
            key = j.born_regime or "NONE"
            by_regime[key] = by_regime.get(key, 0) + 1
        biggest = max(self.jams.values(), key=lambda j: j.loss, default=None)
        return {
            "steps": self.steps,
            "jams": {
                "count": len(self.jams),
                "born_per_100_steps": round(100.0 * len(self.jams) / max(self.steps, 1), 2),
                "median_lifetime": float(np.median(lifetimes)) if lifetimes else None,
                "max_lifetime": max(lifetimes, default=0),
                "max_area": max((j.peak_area for j in self.jams.values()), default=0),
                "splits": sum(1 for j in self.jams.values() if j.parent is not None),
                "born_by_regime": by_regime,
                "largest_loss_share": None
                if biggest is None or self._final is None or self._final.loss.sum() <= 0
                else round(biggest.loss / float(self._final.loss.sum()), 3),
            },
            "pile_up": self._ratio("jam"),
            "diversion": self._diversion(),
            "corridors": self._corridors(),
            "pressure": self._pressure(),
            "recovery": self._recovery(),
            "loss": self._loss(),
            "ruin_zone": self._ruin_zone(),
        }

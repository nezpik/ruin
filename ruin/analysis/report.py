"""Chaotic organization over many shifts, and a picture of one.

`run_organization` runs shifts with a `StructureRecorder` and attribution
switched on and reports what is typical, with standard errors across shifts for
the headline ratios. `plot_organization` draws one shift: when each jam lived,
where loss accrued, and where the delay came from.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ruin.analysis.organization import StructureRecorder
from ruin.config import ConfigLike, as_config
from ruin.config_models import RuinConfig
from ruin.simulation.agent_based import run_trajectory


def _shift(args: tuple[RuinConfig, int]) -> dict[str, Any]:
    cfg, seed = args
    recorder = StructureRecorder()
    result = run_trajectory(
        cfg,
        seed=seed,
        store_qdot_snapshots=False,
        stop_at_ruin=False,
        recorder=recorder,
        attribute_sources=True,
    )
    return {
        "ruined": result["ruined"],
        "ruin_time": result["ruin_time"],
        "loss": result["cumulative_loss"],
        "on_time_rate": result["on_time_rate"],
        "structure": result["structure"],
        "loss_by_regime": result["loss_by_regime"],
        "delay": result["delay_by_source"],
    }


def _stat(values: list[float | None]) -> dict[str, float | None]:
    """Mean and standard error across shifts, ignoring shifts where it is undefined."""
    v = np.array([x for x in values if x is not None], dtype=float)
    if len(v) == 0:
        return {"mean": None, "se": None, "n": 0}
    se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None
    return {
        "mean": round(float(v.mean()), 3),
        "se": None if se is None else round(se, 3),
        "n": len(v),
    }


def run_organization(
    config: ConfigLike,
    shifts: int = 60,
    seed: int | None = None,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Measure chaotic organization over `shifts` whole shifts of the scenario."""
    cfg = as_config(config)
    base = int(cfg.simulation.seed if seed is None else seed)
    jobs = [(cfg, base + i) for i in range(shifts)]
    workers = max(1, min(n_jobs or os.cpu_count() or 4, shifts))
    if workers == 1:
        runs = [_shift(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            runs = list(executor.map(_shift, jobs, chunksize=max(1, shifts // (workers * 4))))
    return aggregate(runs, cfg)


def aggregate(runs: list[dict[str, Any]], cfg: RuinConfig) -> dict[str, Any]:
    """Combine per-shift results into one report."""
    s = [r["structure"] for r in runs]
    ruined = [r for r in runs if r["ruined"] and r["structure"]["ruin_zone"] is not None]
    has_regime = runs[0]["loss_by_regime"] is not None

    out: dict[str, Any] = {
        "shifts": len(runs),
        "avoidance": cfg.processes.avoidance,
        "ruin_probability": round(float(np.mean([r["ruined"] for r in runs])), 3),
        "mean_loss": round(float(np.mean([r["loss"] for r in runs])), 1),
        "jams": {
            "born_per_100_steps": _stat([x["jams"]["born_per_100_steps"] for x in s]),
            "median_lifetime": _stat([x["jams"]["median_lifetime"] for x in s]),
            "largest_area_cells": _stat([x["jams"]["max_area"] for x in s]),
            "share_of_grid_in_largest_jam": _stat(
                [
                    x["jams"]["max_area"] / (cfg.simulation.grid_width * cfg.simulation.grid_height)
                    for x in s
                ]
            ),
        },
        "pile_up": _stat([x["pile_up"] for x in s]),
        "diversion": _stat([x["diversion"] for x in s]),
        "corridors": {
            "per_shift": _stat([x["corridors"]["corridors"] for x in s]),
            "cells_per_shift": _stat([x["corridors"]["corridor_cells"] for x in s]),
        },
        "pressure": {
            "top_decile_share": _stat([x["pressure"]["top_decile_share_of_pressure"] for x in s]),
            "hotspots_persist_between_halves": _stat(
                [x["pressure"]["map_correlation_between_halves"] for x in s]
            ),
        },
        "recovery": {
            "median_steps_peak_to_clear": _stat(
                [x["recovery"]["median_steps_peak_to_clear"] for x in s]
            ),
            "flow_into_recovering_cells": _stat(
                [x["recovery"]["flow_into_recovering_cells"] for x in s]
            ),
        },
        "loss": {
            "share_inside_jams": _stat([x["loss"]["loss_share_inside_jams"] for x in s]),
            "enrichment_inside_jams": _stat([x["loss"]["loss_enrichment_inside_jams"] for x in s]),
            "share_of_cells_carrying_half_the_loss": _stat(
                [x["loss"]["share_of_cells_carrying_half_the_loss"] for x in s]
            ),
        },
    }
    if ruined:
        zones = [r["structure"]["ruin_zone"] for r in ruined]
        top = [z["top_jams"][0] for z in zones if z["top_jams"]]
        out["ruin_zone"] = {
            "ruined_shifts": len(ruined),
            "loss_share_outside_any_jam": _stat([z["share_outside_any_jam"] for z in zones]),
            "top_jam_share_of_loss": _stat([t["share"] for t in top]),
            "top_jam_peak_area_cells": _stat([t["peak_area"] for t in top]),
            "top_jam_born_in_disrupted_or_chaotic": round(
                float(np.mean([t["born_regime"] in ("DISRUPTED", "CHAOTIC") for t in top])), 3
            )
            if has_regime and top
            else None,
        }
    delay = [r["delay"] for r in runs if r["delay"] is not None]
    if delay:
        names = list(delay[0]["share_by_source"])
        total_delay = {n: sum(d["by_source"][n] for d in delay) for n in names}
        grand = sum(total_delay.values())
        out["attribution"] = {
            "delay_share_by_origin": {n: round(v / grand, 4) for n, v in total_delay.items()}
            if grand > 0
            else None,
        }
        if has_regime:
            loss = {k: sum(r["loss_by_regime"][k] for r in runs) for k in runs[0]["loss_by_regime"]}
            total_loss = sum(loss.values())
            out["attribution"]["loss_share_by_regime_when_accrued"] = (
                {k: round(v / total_loss, 4) for k, v in loss.items()} if total_loss > 0 else None
            )
    return out


# Dark chart surface and ink, as in the destiny chart.
_SURFACE, _INK, _INK_2, _MUTED, _GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a"
_BLUE, _ORANGE, _CRITICAL = "#3987e5", "#d95926", "#d03b3b"
_REGIME_GRAY = {"STABLE": 0.18, "STRESSED": 0.34, "DISRUPTED": 0.56, "CHAOTIC": 0.80}


def plot_organization(
    result: dict[str, Any], recorder: StructureRecorder, output: str | Path
) -> Path:
    """One shift: when each jam lived, where loss accrued, and where the delay came from."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    output_path = Path(output)
    jams = sorted(recorder.jams.values(), key=lambda j: j.born)
    horizon = max(recorder.horizon, recorder.steps)
    loss_map = recorder._final.loss if recorder._final is not None else np.zeros(recorder._shape)
    ramp = LinearSegmentedColormap.from_list("loss", [_SURFACE, "#1f4a85", _BLUE, "#bcd6f7"])

    fig = plt.figure(figsize=(11, 6.6), facecolor=_SURFACE)
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1], hspace=0.45, wspace=0.28
    )
    gantt = fig.add_subplot(grid[:, 0])
    heat = fig.add_subplot(grid[0, 1])
    bars = fig.add_subplot(grid[1, 1])
    for ax in (gantt, heat, bars):
        ax.set_facecolor(_SURFACE)
        ax.tick_params(colors=_MUTED, length=0, labelsize=8)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

    # --- jam timeline, with the hidden regime above it
    peak = max((j.peak_area for j in jams), default=1)
    zone = recorder.summary()["ruin_zone"]
    ruin_ids = {t["id"] for t in zone["top_jams"][:1]} if zone else set()
    for row, jam in enumerate(jams):
        shade = 0.18 + 0.82 * np.log1p(jam.peak_area) / np.log1p(peak)
        gantt.barh(
            row,
            jam.lifetime,
            left=jam.born,
            height=1.0,
            color=_BLUE,
            alpha=float(shade),
            linewidth=1.4 if jam.id in ruin_ids else 0,
            edgecolor=_CRITICAL if jam.id in ruin_ids else "none",
        )
    regimes = [s["regime"] for s in result["snapshots"]]
    if regimes and regimes[0] is not None:
        gray = np.array([[_REGIME_GRAY[r] for r in regimes]])
        gantt.imshow(
            gray, aspect="auto", cmap="gray", vmin=0, vmax=1, extent=(0, len(regimes), -4.5, -1.5),
            interpolation="nearest",
        )  # fmt: skip
    gantt.set_xlim(0, horizon)
    gantt.set_ylim(len(jams) + 1, -5)
    gantt.set_xlabel("time (steps)", color=_INK_2, fontsize=9)
    gantt.set_ylabel("jams, oldest first (brighter = larger at its peak)", color=_INK_2, fontsize=9)
    gantt.set_title(
        f"{len(jams)} jams over the shift, with the hidden regime above (brighter is worse)",
        color=_INK, fontsize=10, loc="left",
    )  # fmt: skip
    if result["ruined"]:
        gantt.axvline(result["ruin_time"], color=_CRITICAL, linewidth=1.2, linestyle="--")

    # --- where loss accrued
    heat.imshow(loss_map, cmap=ramp, origin="lower", interpolation="nearest")
    heat.set_xticks([])
    heat.set_yticks([])
    heat.set_title("Where loss accrued", color=_INK, fontsize=10, loc="left")
    if ruin_ids:
        top = recorder.jams[next(iter(ruin_ids))]
        if top.peak_cells is not None and not top.peak_cells.all():
            heat.contour(
                top.peak_cells.astype(float), levels=[0.5], colors=_CRITICAL, linewidths=1.5
            )
        heat.set_title(
            "Where loss accrued (red: the jam that took the shift)",
            color=_INK,
            fontsize=10,
            loc="left",
        )

    # --- cause and effect
    attribution = result["delay_by_source"]
    names = list(attribution["share_by_source"]) if attribution else []
    origin = [attribution["share_by_source"][n] for n in names] if attribution else []
    when_loss = result["loss_by_regime"]
    total_loss = sum(when_loss.values()) if when_loss else 0.0
    accrued = [(when_loss.get(n, 0.0) / total_loss) if total_loss > 0 else 0.0 for n in names]
    y = np.arange(len(names))
    bars.barh(y - 0.2, origin, height=0.38, color=_BLUE, label="Delay by origin of the disruption")
    bars.barh(
        y + 0.2, accrued, height=0.38, color=_ORANGE,
        hatch="//", edgecolor=_SURFACE, linewidth=0, label="Loss by regime when it accrued",
    )  # fmt: skip
    bars.set_yticks(y)
    bars.set_yticklabels([n.title() for n in names], color=_INK_2, fontsize=8)
    bars.invert_yaxis()
    bars.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    bars.set_title("Cause and effect", color=_INK, fontsize=10, loc="left")
    legend = bars.legend(loc="lower right", frameon=False, fontsize=7.5)
    for text in legend.get_texts():
        text.set_color(_INK_2)

    fig.savefig(output_path, dpi=140, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return output_path

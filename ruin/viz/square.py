"""RUIN visualization.

- Legacy text frame exporter
- Matplotlib animation of the Probability Square (GIF export), driven by the
  `visualization` section of the config
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from ruin.analysis.organization import JAM_THRESHOLD, MIN_JAM_AREA, label_components
from ruin.config import ConfigLike, as_config
from ruin.core.processes import EXPOSURE_SATURATION


def save_text_frames(result: dict[str, Any], output: str | Path, limit: int = 10) -> None:
    """Legacy text-only frame exporter."""
    path = Path(output)
    lines: list[str] = []
    snapshots = result.get("snapshots", [])
    if len(snapshots) <= limit:
        selected = snapshots
    else:
        step = max(1, len(snapshots) // max(limit, 1))
        selected = snapshots[::step][:limit]
        if snapshots[-1] not in selected:
            selected[-1] = snapshots[-1]

    for snapshot in selected:
        lines.append(
            f"t={snapshot['time']} D={snapshot['d_state']} chaos={snapshot['chaos_pressure']} "
            f"order={snapshot['order_pressure']} penalty={snapshot['penalty']}"
        )
        lines.append(
            f"  active={snapshot['active_count']} late={snapshot['late_count']} "
            f"delivered={snapshot.get('delivered', 0)} "
            f"delivered_late={snapshot.get('delivered_late', 0)} "
            f"failed={snapshot.get('failed', 0)} income={snapshot.get('income', 0.0)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_grid_dims(config: ConfigLike) -> tuple[int, int]:
    sim = as_config(config).simulation
    return int(sim.grid_width), int(sim.grid_height)


# What each `visualization.heatmap_field` option draws: snapshot key, label, colour scale.
_HEATMAPS: dict[str, tuple[str, str, float | None]] = {
    "disruption_intensity": ("field", "Disruption intensity", EXPOSURE_SATURATION),
    "chaos_pressure": ("chaos_field", "Local chaos pressure", 2.0),
    "order_pressure": ("order_field", "Local order pressure", None),
    "accumulated_loss": ("loss_field", "Loss accrued in cell so far", None),
}

_TYPE_COLORS = {"STANDARD": "white", "EXPRESS": "cyan", "BULKY": "orange"}


def _qdot_color(q: dict[str, Any], show_types: bool) -> str:
    status = q.get("status", "ACTIVE")
    if q.get("ruined") or status == "FAILED":
        return "red"
    if status in ("DELIVERED", "DELIVERED_LATE"):
        return "lime"
    if q.get("late"):
        return "magenta"
    if not show_types:
        return "white"
    return _TYPE_COLORS.get(str(q.get("type", "STANDARD")).upper(), "white")


def _select_frames(snapshots: list[dict[str, Any]], max_frames: int) -> list[dict[str, Any]]:
    """Pick at most `max_frames` snapshots spread evenly over the whole trajectory."""
    drawable = [s for s in snapshots if "qdots" in s or "field" in s] or snapshots
    if len(drawable) <= max_frames:
        return drawable
    picks = np.linspace(0, len(drawable) - 1, max_frames).round().astype(int)
    return [drawable[i] for i in picks]


def animate_probability_square(
    result: dict[str, Any],
    config: ConfigLike,
    output: str | Path,
    fps: int = 8,
    dpi: int = 90,
    max_frames: int | None = None,
) -> Path:
    """Create an animated GIF of the Probability Square.

    What is drawn follows the `visualization` section of the config: which
    per-cell quantity the heatmap shows, how many frames are rendered (spread
    evenly over the trajectory), and whether Q-dot types, the D-state, the
    pressures, and the ruin event are shown. `max_frames` overrides
    `visualization.frames`. The heatmap needs a trajectory run with
    `store_field_snapshots=True`.
    """
    output_path = Path(output)
    cfg = as_config(config)
    viz = cfg.visualization
    snapshots = result.get("snapshots", [])
    if not snapshots:
        raise ValueError("No snapshots found in trajectory result")
    frames = _select_frames(snapshots, max_frames if max_frames is not None else viz.frames)

    width, height = _extract_grid_dims(cfg)
    heat_key, heat_label, heat_max = _HEATMAPS[viz.heatmap_field]
    show_heatmap = viz.show_disruption_field and any(heat_key in s for s in frames)
    if show_heatmap and heat_max is None:
        heat_max = max(1.0, max(np.max(s[heat_key]) for s in frames if heat_key in s))
    ruin_time = result.get("ruin_time")

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#111111")
    ax.set_facecolor("#111111")
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect("equal")
    ax.set_title("RUIN — Probability Square (Destiny Frame D)", color="white", fontsize=12)
    ax.set_xlabel("x", color="#888888")
    ax.set_ylabel("y", color="#888888")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.tick_params(colors="#888888")

    blank = np.zeros((height, width))
    field_img = ax.imshow(
        blank, cmap="inferno", origin="lower", vmin=0, vmax=heat_max or 1.0, interpolation="nearest"
    )
    if show_heatmap:
        colorbar = plt.colorbar(field_img, ax=ax, shrink=0.8)
        colorbar.set_label(heat_label, color="#cccccc")
        colorbar.ax.tick_params(colors="#888888")

    scatter = ax.scatter([], [], s=22, edgecolors="#222222", linewidths=0.4)
    outlines: list[Any] = []  # the jam outlines drawn for the current frame
    jam_text = ax.text(
        0.98, 0.02, "", transform=ax.transAxes, color="white", fontsize=9, va="bottom", ha="right",
        fontfamily="monospace",
    )  # fmt: skip

    text_kw: dict[str, Any] = {"transform": ax.transAxes, "fontfamily": "monospace"}
    time_text = ax.text(0.02, 0.98, "", color="white", fontsize=10, va="top", **text_kw)
    state_text = ax.text(
        0.98,
        0.98,
        "",
        color="#ffcc00",
        fontsize=11,
        va="top",
        ha="right",
        fontweight="bold",
        **text_kw,
    )
    pressure_text = ax.text(0.02, 0.02, "", color="#88ff88", fontsize=9, va="bottom", **text_kw)
    ruin_text = ax.text(
        0.5,
        0.5,
        "",
        color="#ff3333",
        fontsize=20,
        va="center",
        ha="center",
        fontweight="bold",
        **text_kw,
    )
    artists = (field_img, scatter, time_text, state_text, pressure_text, ruin_text, jam_text)

    def update(frame_idx: int) -> tuple[Any, ...]:
        snap = frames[frame_idx]

        if show_heatmap and heat_key in snap:
            field_img.set_data(np.array(snap[heat_key], dtype=float))
        for outline in outlines:
            outline.remove()
        outlines.clear()
        if viz.show_jams and "field" in snap:
            blocked = np.array(snap["field"], dtype=float) >= JAM_THRESHOLD
            if blocked.any() and not blocked.all():
                outlines.append(
                    ax.contour(blocked.astype(float), levels=[0.5], colors="white", linewidths=0.8)
                )
            labels, count = label_components(blocked)
            jams = sum(1 for k in range(1, count + 1) if (labels == k).sum() >= MIN_JAM_AREA)
            jam_text.set_text(f"jams: {jams}")

        qdots = snap.get("qdots", [])
        if qdots:
            scatter.set_offsets([q["position"] for q in qdots])
            scatter.set_facecolor([_qdot_color(q, viz.show_qdot_types) for q in qdots])
        else:
            scatter.set_offsets(np.empty((0, 2)))

        time_text.set_text(f"t = {snap['time']}")
        if viz.show_d_state:
            regime = snap.get("regime")
            hidden = f"\nregime: {regime}" if regime else ""
            state_text.set_text(f"D = {snap['d_state']}{hidden}")
        if viz.show_order_chaos_pressure:
            pressure_text.set_text(
                f"chaos={snap.get('chaos_pressure', 0.0):.2f}  "
                f"order={snap.get('order_pressure', 0.0):.2f}"
            )
        if viz.show_ruin_event and ruin_time is not None and snap["time"] >= ruin_time:
            ruin_text.set_text(f"RUIN at t = {ruin_time}\n({result.get('ruin_reason')})")
        return artists

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return output_path


def save_field_gif(
    result: dict[str, Any],
    config: ConfigLike,
    output: str | Path,
    fps: int = 8,
    dpi: int = 90,
    max_frames: int | None = None,
) -> Path:
    """Convenience wrapper around `animate_probability_square`."""
    return animate_probability_square(
        result, config, output, fps=fps, dpi=dpi, max_frames=max_frames
    )

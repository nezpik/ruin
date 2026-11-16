"""RUIN visualization.

- Legacy text frame exporter
- Matplotlib animation of Probability Square + Disruption Field (GIF export)
  Now supports real field grids when store_field_snapshots=True
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

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
            f"t={snapshot['time']} D={snapshot['d_state']} chaos={snapshot['chaos_pressure']} order={snapshot['order_pressure']} penalty={snapshot['penalty']}"
        )
        lines.append(
            f"  active={snapshot['active_count']} late={snapshot['late_count']} "
            f"delivered={snapshot.get('delivered', 0)} delivered_late={snapshot.get('delivered_late', 0)} "
            f"failed={snapshot.get('failed', 0)} income={snapshot.get('income', 0.0)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_grid_dims(config: ConfigLike) -> tuple[int, int]:
    sim = as_config(config).simulation
    return int(sim.grid_width), int(sim.grid_height)


def animate_probability_square(
    result: dict[str, Any],
    config: ConfigLike,
    output: str | Path,
    fps: int = 8,
    dpi: int = 90,
    max_frames: int | None = None,
) -> Path:
    """
    Create an animated GIF of the Probability Square + real Disruption Field.

    If snapshots contain "field" (list of lists), it uses the actual data.
    Otherwise falls back to synthetic field.
    """
    output_path = Path(output)
    snapshots = result.get("snapshots", [])
    if not snapshots:
        raise ValueError("No snapshots found in trajectory result")

    if max_frames is not None:
        snapshots = snapshots[:max_frames]

    # v0.2: support both dict and RuinConfig
    width, height = _extract_grid_dims(config)

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

    field_img = ax.imshow(
        np.zeros((height, width)),
        cmap="inferno",
        origin="lower",
        vmin=0,
        vmax=EXPOSURE_SATURATION,
        interpolation="nearest",
    )
    plt.colorbar(field_img, ax=ax, label="Disruption Intensity", shrink=0.8)

    scatter = ax.scatter([], [], s=22, edgecolors="#222222", linewidths=0.4)

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="white",
                        fontsize=10, va="top", fontfamily="monospace")
    state_text = ax.text(0.98, 0.98, "", transform=ax.transAxes, color="#ffcc00",
                         fontsize=11, va="top", ha="right", fontweight="bold")
    pressure_text = ax.text(0.02, 0.02, "", transform=ax.transAxes, color="#88ff88",
                            fontsize=9, va="bottom", fontfamily="monospace")

    def init() -> tuple[Any, ...]:
        scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text("")
        state_text.set_text("")
        pressure_text.set_text("")
        field_img.set_data(np.zeros((height, width)))
        return field_img, scatter, time_text, state_text, pressure_text

    def update(frame_idx: int) -> tuple[Any, ...]:
        snap = snapshots[frame_idx]
        t = snap["time"]
        d_state = snap["d_state"]
        chaos = snap.get("chaos_pressure", 0.0)
        order = snap.get("order_pressure", 0.0)

        # Use real field if stored, else synthetic
        if snap.get("field"):
            field = np.array(snap["field"], dtype=float)
        else:
            mean_int = snap.get("field_intensity_mean", 0.5)
            field = np.random.default_rng(42).normal(mean_int, 0.8, (height, width))
            field = np.clip(field, 0, None)

        field_img.set_data(field)

        qdots = snap.get("qdots", [])
        if qdots:
            xs = []
            ys = []
            for q in qdots:
                pos = q.get("position")
                if isinstance(pos, (list, tuple)) and len(pos) == 2:
                    xs.append(pos[0])
                    ys.append(pos[1])
                else:
                    xs.append(q.get("x", 0))
                    ys.append(q.get("y", 0))
            colors = []
            for q in qdots:
                status = q.get("status", "ACTIVE")
                if q.get("ruined") or status == "FAILED":
                    colors.append("red")
                elif status in ("DELIVERED", "DELIVERED_LATE"):
                    colors.append("lime")
                elif q.get("late"):
                    colors.append("magenta")
                else:
                    qtype = str(q.get("type", "STANDARD")).upper()
                    if qtype == "EXPRESS":
                        colors.append("cyan")
                    elif qtype == "BULKY":
                        colors.append("orange")
                    else:
                        colors.append("white")
            scatter.set_offsets(list(zip(xs, ys)))
            scatter.set_facecolor(colors)
        else:
            scatter.set_offsets(np.empty((0, 2)))

        time_text.set_text(f"t = {t}")
        state_text.set_text(f"D = {d_state}")
        pressure_text.set_text(f"chaos={chaos:.2f}  order={order:.2f}")

        return field_img, scatter, time_text, state_text, pressure_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        init_func=init,
        interval=1000 / fps,
        blit=False,
    )

    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
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
    """Convenience wrapper around animate_probability_square for real-field GIFs."""
    return animate_probability_square(result, config, output, fps=fps, dpi=dpi, max_frames=max_frames)

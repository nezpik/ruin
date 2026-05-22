"""RUIN visualization (v0.2).

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
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


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
        ruined = [q for q in snapshot.get("qdots", []) if q.get("ruined")]
        lines.append(f"  active={snapshot['active_count']} late={snapshot['late_count']} ruined_sample={len(ruined)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def animate_probability_square(
    result: dict[str, Any],
    config: dict[str, Any],
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

    sim = config.get("simulation", {})
    width = int(sim.get("grid_width", 30))
    height = int(sim.get("grid_height", 30))

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
        vmax=4.0,
        interpolation="nearest",
    )
    plt.colorbar(field_img, ax=ax, label="Disruption Intensity", shrink=0.8)

    scatter = ax.scatter([], [], c=[], s=25, cmap="cool", edgecolors="white", linewidths=0.5)

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="white",
                        fontsize=10, va="top", fontfamily="monospace")
    state_text = ax.text(0.98, 0.98, "", transform=ax.transAxes, color="#ffcc00",
                         fontsize=11, va="top", ha="right", fontweight="bold")
    pressure_text = ax.text(0.02, 0.02, "", transform=ax.transAxes, color="#88ff88",
                            fontsize=9, va="bottom", fontfamily="monospace")

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_array(np.array([]))
        time_text.set_text("")
        state_text.set_text("")
        pressure_text.set_text("")
        field_img.set_data(np.zeros((height, width)))
        return field_img, scatter, time_text, state_text, pressure_text

    def update(frame_idx: int):
        snap = snapshots[frame_idx]
        t = snap["time"]
        d_state = snap["d_state"]
        chaos = snap.get("chaos_pressure", 0.0)
        order = snap.get("order_pressure", 0.0)

        # Use real field if stored, else synthetic
        if "field" in snap and snap["field"]:
            field = np.array(snap["field"], dtype=float)
        else:
            mean_int = snap.get("field_intensity_mean", 0.5)
            field = np.random.RandomState(t).rand(height, width) * (mean_int + 0.5)
            field = np.clip(field, 0, 5)

        field_img.set_data(field)
        field_img.set_clim(0, max(2.0, float(field.max()) * 1.1))

        # Q-dots
        qdots = snap.get("qdots", [])
        if qdots:
            positions = [q.get("position", (0, 0)) for q in qdots]
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]

            colors = []
            sizes = []
            for q in qdots:
                if q.get("ruined"):
                    colors.append(0.0)
                    sizes.append(18)
                elif q.get("late"):
                    colors.append(0.3)
                    sizes.append(28)
                else:
                    qtype = q.get("type", "STANDARD")
                    if qtype == "EXPRESS":
                        colors.append(0.9)
                    elif qtype == "BULKY":
                        colors.append(0.6)
                    else:
                        colors.append(0.75)
                    sizes.append(22)

            scatter.set_offsets(np.c_[xs, ys])
            scatter.set_array(np.array(colors))
            scatter.set_sizes(sizes)
        else:
            scatter.set_offsets(np.empty((0, 2)))

        time_text.set_text(f"t = {t:03d}")
        state_text.set_text(f"D = {d_state}")
        pressure_text.set_text(f"chaos={chaos:.3f}  order={order:.3f}")

        return field_img, scatter, time_text, state_text, pressure_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        init_func=init,
        interval=1000 / fps,
        blit=True,
        repeat=False,
    )

    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    return output_path


def save_field_gif(
    result: dict[str, Any],
    config: dict[str, Any],
    output: str | Path = "ruin_field.gif",
    fps: int = 8,
    dpi: int = 90,
    max_frames: int | None = 120,
) -> Path:
    return animate_probability_square(result, config, output, fps=fps, dpi=dpi, max_frames=max_frames)

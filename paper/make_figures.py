"""Figures for the paper, in a print-friendly light theme.

Run from the repository root:

    python paper/make_figures.py            # uses the cached results in paper/data
    python paper/make_figures.py --refresh  # recompute them (a few minutes)

Figure 1 simulates one shift of the reference scenario (seed 81) and draws it.
Figures 2 and 3 are drawn from cached analysis results: the structure ensemble
(`ruin structure`), the three destiny views (`ruin destiny --blind --observer
--incidents`), and the organization analysis (`ruin organize`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

from ruin.analysis.organization import JAM_THRESHOLD, StructureRecorder, label_components
from ruin.config import load_ruin_config
from ruin.core.processes import EXPOSURE_SATURATION
from ruin.simulation.agent_based import run_trajectory

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
SEED = 81

# Light surface and ink; categorical slots 1-3 (validated for colour-vision deficiency
# on white: all pass; aqua is just under 3:1 contrast, so every line is also labelled
# and distinguished by dash and marker).
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, CRITICAL = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b"
# Disruption: one hue, light to dark. Regime strip: neutral gray, darker is worse.
RAMP = LinearSegmentedColormap.from_list("disruption", ["#f3f7fd", "#9cc0ee", BLUE, "#0d2f5c"])
REGIME_GRAY = {"STABLE": 0.95, "STRESSED": 0.78, "DISRUPTED": 0.52, "CHAOTIC": 0.22}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.formatter.use_mathtext": True,
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "pdf.fonttype": 42,
    }
)


def _tidy(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, width=0.5)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)


def _load(name):
    return json.loads((DATA / f"{name}.json").read_text())


def refresh_cache():
    """Recompute the cached analyses (slow)."""
    from ruin.inference import fit_regime_observer
    from ruin.risk.destiny import run_destiny
    from ruin.risk.structure import run_structure_ensemble

    cfg = load_ruin_config("examples/urban_ruin_shift.yaml")
    observer = fit_regime_observer(cfg, shifts=150)
    destiny = run_destiny(
        cfg,
        seed=SEED,
        blind=True,
        observer=observer,
        observer_incidents=True,
        branches=64,
        every=24,
    )
    for key in ("surplus_path", "regime_path"):
        destiny.pop(key, None)
    (DATA / "destiny_seed81.json").write_text(json.dumps(destiny, indent=1))
    (DATA / "structure.json").write_text(
        json.dumps(run_structure_ensemble(cfg, paths=400), indent=1)
    )


# --------------------------------------------------------------------- figure 1


def figure_anatomy(cfg):
    """One shift: the square at its worst moment, the surplus, and what destiny said."""
    result = run_trajectory(cfg, seed=SEED, store_field_snapshots=True, stop_at_ruin=True)
    snaps = result["snapshots"]
    largest = []
    for s in snaps:
        labels, count = label_components(np.array(s["field"]) >= JAM_THRESHOLD)
        largest.append(max((int((labels == k).sum()) for k in range(1, count + 1)), default=0))
    pick = int(np.argmax(largest))
    snap = snaps[pick]
    destiny = _load("destiny_seed81")
    ruin_time = result["ruin_time"]

    fig = plt.figure(figsize=(7.0, 2.45))
    grid = fig.add_gridspec(
        2, 3, width_ratios=[1.0, 1.25, 1.25], height_ratios=[0.07, 1.0], wspace=0.34, hspace=0.05
    )

    # (a) the Probability Square
    ax = fig.add_subplot(grid[:, 0])
    field = np.array(snap["field"])
    ax.imshow(
        field, cmap=RAMP, origin="lower", vmin=0, vmax=EXPOSURE_SATURATION, interpolation="nearest"
    )
    ax.contour(field >= JAM_THRESHOLD, levels=[0.5], colors=INK, linewidths=0.9)
    dots = snap["qdots"]
    on_time = np.array([q["position"] for q in dots if not q["late"]])
    late = np.array([q["position"] for q in dots if q["late"]])
    # White dots with a thin dark edge stay visible on both the pale and the dark parts of the map.
    ax.scatter(
        on_time[:, 0],
        on_time[:, 1],
        s=4.5,
        facecolor="white",
        edgecolors=INK_2,
        linewidths=0.3,
        label="on time",
    )
    if len(late):
        ax.scatter(
            late[:, 0],
            late[:, 1],
            s=11,
            color=ORANGE,
            edgecolors="white",
            linewidths=0.4,
            label="late",
        )
    ax.plot([], [], color=INK, linewidth=0.9, label="jam outline")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"(a) The square at step {snap['time']}", fontsize=7.5, loc="left", pad=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False, handletextpad=0.3,
              columnspacing=0.9, borderaxespad=0.0)  # fmt: skip

    # shared regime strip over (b) and (c)
    regimes = [s["regime"] for s in snaps]
    gray = np.array([[REGIME_GRAY[r] for r in regimes]])
    horizon = len(snaps)
    for col in (1, 2):
        strip = fig.add_subplot(grid[0, col])
        strip.imshow(
            gray,
            aspect="auto",
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=(0, horizon, 0, 1),
            interpolation="nearest",
        )
        strip.set_xticks([])
        strip.set_yticks([])
        for side in strip.spines.values():
            side.set_visible(False)

    # (b) surplus
    ax = fig.add_subplot(grid[1, 1])
    surplus = result["surplus_path"]
    ax.plot(range(1, len(surplus) + 1), surplus, color=INK, linewidth=1.1)
    ax.axhline(0, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.axvline(ruin_time, color=CRITICAL, linewidth=0.9)
    ax.set_xlim(0, horizon)
    ax.set_xlabel("time (steps)")
    ax.set_ylabel("surplus")
    ax.set_title("(b) The buffer", fontsize=7.5, loc="left", pad=20)
    ax.annotate(f"ruin at step {ruin_time}:\nthe SLA is breached\nwhile surplus is\nstill positive",
                xy=(ruin_time, surplus[-1]), xytext=(150, 18), textcoords="data", ha="left", va="bottom",
                fontsize=6.5, color=CRITICAL,
                arrowprops={"arrowstyle": "-", "color": CRITICAL, "linewidth": 0.6,
                            "shrinkA": 2, "shrinkB": 1})  # fmt: skip
    _tidy(ax)

    # (c) remaining destiny, three views
    ax = fig.add_subplot(grid[1, 2])
    rows = destiny["curve"]
    rows = [r for r in rows if r["time"] <= ruin_time]
    t = [r["time"] for r in rows] + [destiny["end"]["time"]]
    end = destiny["end"]["destiny"]
    for key, color, style, marker, label in (
        ("destiny", BLUE, "-", "o", "knows the regime"),
        ("destiny_observer", ORANGE, (0, (1, 1.4)), "s", "watches pressures"),
        ("destiny_blind", AQUA, (0, (4, 2)), "^", "knows nothing"),
    ):
        y = [r[key] for r in rows] + [end]
        ax.plot(
            t,
            y,
            color=color,
            linestyle=style,
            marker=marker,
            markersize=2.6,
            linewidth=1.0,
            label=label,
        )
    ax.axvline(ruin_time, color=CRITICAL, linewidth=0.9)
    ax.set_xlim(0, horizon)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_xlabel("time (steps)")
    ax.set_ylabel("chance of finishing safely")
    ax.set_title("(c) Remaining destiny", fontsize=7.5, loc="left", pad=20)
    ax.legend(loc="lower left", frameon=False, handlelength=2.2, borderaxespad=0.1)
    _tidy(ax)

    fig.savefig(OUT / "anatomy.pdf", bbox_inches="tight")
    fig.savefig(OUT / "anatomy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- figure 2


def figure_structure():
    """Same average disruption, very different ruin."""
    data = _load("structure")["variants"]
    labels = [
        ("thin", "Memoryless"),
        ("heavy_tails", "Memoryless,\nheavy-tailed sizes"),
        ("clustered", "Bursts (hidden regime)"),
        ("full", "Bursts, heavy tails,\nlingering jams"),
    ]
    p = np.array([data[k]["ruin_probability"] for k, _ in labels]) * 100
    low = np.array([data[k]["ruin_probability_ci"][0] for k, _ in labels]) * 100
    high = np.array([data[k]["ruin_probability_ci"][1] for k, _ in labels]) * 100

    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    y = np.arange(len(labels))[::-1]
    colors = [MUTED, MUTED, BLUE, BLUE]
    ax.barh(y, p, height=0.58, color=colors, linewidth=0)
    ax.errorbar(
        p,
        y,
        xerr=[p - low, high - p],
        fmt="none",
        ecolor=INK,
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.8,
    )
    for yi, pi, hi in zip(y, p, high, strict=True):
        ax.text(hi + 0.5, yi, f"{pi:.1f}%", va="center", ha="left", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([name for _, name in labels], fontsize=7)
    ax.set_xlim(0, 24)
    ax.set_xlabel("shifts that end in ruin, % (95% interval)")
    _tidy(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / "structure.pdf", bbox_inches="tight")
    fig.savefig(OUT / "structure.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- figure 3


def figure_cause_effect():
    """Where the delay came from, against when the loss accrued."""
    att = _load("organization")["reference"]["attribution"]
    names = ["STABLE", "STRESSED", "DISRUPTED", "CHAOTIC"]
    delay = [att["delay_share_by_origin"][n] * 100 for n in names]
    loss = [att["loss_share_by_regime_when_accrued"][n] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(3.35, 1.85))
    y = np.arange(len(names))[::-1]
    ax.barh(
        y + 0.19,
        delay,
        height=0.36,
        color=BLUE,
        linewidth=0,
        label="delay: where disruption came from",
    )
    ax.barh(y - 0.19, loss, height=0.36, color="white", edgecolor=ORANGE, hatch="////", linewidth=0.8,
            label="loss: when it was paid")  # fmt: skip
    for yi, d, lo in zip(y, delay, loss, strict=True):
        ax.text(d + 0.8, yi + 0.19, f"{d:.0f}%", va="center", fontsize=6.5, color=INK)
        ax.text(lo + 0.8, yi - 0.19, f"{lo:.0f}%", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([n.title() for n in names], fontsize=7)
    ax.set_xlim(0, 78)
    ax.set_xlabel("share of the total, % (80 shifts)")
    ax.legend(loc="lower right", frameon=False, handlelength=1.6, borderaxespad=0.1)
    _tidy(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(OUT / "cause_effect.pdf", bbox_inches="tight")
    fig.savefig(OUT / "cause_effect.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- figure 4


def figure_organization(cfg):
    """The organisation of the same shift: when each jam lived, and where the loss fell."""
    recorder = StructureRecorder()
    result = run_trajectory(
        cfg, seed=SEED, store_field_snapshots=True, stop_at_ruin=True, recorder=recorder
    )
    snaps = result["snapshots"]
    summary = recorder.summary()
    jams = sorted(recorder.jams.values(), key=lambda j: j.born)
    horizon = len(snaps)
    top_id = summary["ruin_zone"]["top_jams"][0]["id"]
    biggest = max(j.peak_area for j in jams)

    fig = plt.figure(figsize=(7.0, 2.35))
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.9, 1.0], height_ratios=[0.07, 1.0], wspace=0.22, hspace=0.05
    )

    strip = fig.add_subplot(grid[0, 0])
    gray = np.array([[REGIME_GRAY[s["regime"]] for s in snaps]])
    strip.imshow(
        gray,
        aspect="auto",
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=(0, horizon, 0, 1),
        interpolation="nearest",
    )
    strip.set_xticks([])
    strip.set_yticks([])
    for side in strip.spines.values():
        side.set_visible(False)

    ax = fig.add_subplot(grid[1, 0])
    for row, jam in enumerate(jams):
        shade = 0.15 + 0.85 * np.log1p(jam.peak_area) / np.log1p(biggest)
        took = jam.id == top_id
        ax.barh(row, jam.lifetime, left=jam.born, height=1.0, color=BLUE, alpha=float(shade),
                edgecolor=CRITICAL if took else "none", linewidth=1.0 if took else 0)  # fmt: skip
        if took:
            ax.annotate(f"the jam that took the shift\n({jam.peak_area} cells at its largest)",
                        xy=(jam.born, row), xytext=(8, row + 6), fontsize=6.5, color=CRITICAL,
                        ha="left", va="center", arrowprops={"arrowstyle": "-", "color": CRITICAL, "linewidth": 0.6})  # fmt: skip
    ax.set_xlim(0, horizon)
    ax.set_ylim(len(jams) + 1, -1)
    ax.set_xlabel("time (steps)")
    ax.set_ylabel("jams, oldest first")
    ax.set_title(
        f"(a) {len(jams)} jams, each from birth to disappearance",
        fontsize=7.5,
        loc="left",
        pad=20,
    )
    ax.axvline(result["ruin_time"], color=CRITICAL, linewidth=0.8)
    _tidy(ax, grid_axis=None)

    heat = fig.add_subplot(grid[1, 1])
    loss = np.array(snaps[-1]["loss_field"])
    norm = PowerNorm(gamma=0.4, vmin=0, vmax=float(loss.max()))
    heat.imshow(loss, cmap=RAMP, norm=norm, origin="lower", interpolation="nearest")
    top = recorder.jams[top_id]
    if top.peak_cells is not None:
        heat.contour(top.peak_cells.astype(float), levels=[0.5], colors=CRITICAL, linewidths=1.0)
    heat.set_xticks([])
    heat.set_yticks([])
    heat.set_title("(b) Where the loss fell (square-root scale)", fontsize=7.5, loc="left", pad=20)

    fig.savefig(OUT / "organization.pdf", bbox_inches="tight")
    fig.savefig(OUT / "organization.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="recompute the cached analyses")
    args = parser.parse_args()
    if args.refresh:
        refresh_cache()
    OUT.mkdir(exist_ok=True)
    cfg = load_ruin_config("examples/urban_ruin_shift.yaml")
    figure_anatomy(cfg)
    figure_structure()
    figure_cause_effect()
    figure_organization(cfg)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()

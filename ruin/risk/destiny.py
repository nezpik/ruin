"""Remaining destiny: how much future a shift still has, from where it stands.

At any moment of a shift the present is known: where every Q dot is, how
much disruption each cell holds, how much surplus is left. The future is
not. `run_destiny` follows one shift and, at regular checkpoints, branches
many independent futures from the exact present state. The share of those
futures that reach the end of the shift without ruin is the shift's
*remaining destiny* at that moment:

    destiny(t) = P(no ruin before the horizon | everything known at t)

It starts near `1 - ruin probability`, moves as surplus is spent and
disruption builds or clears, and reaches 0 if the shift ruins or 1 at the
horizon if it does not. A shift "runs out of destiny" when this curve falls
to zero.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ruin.config import ConfigLike, as_config
from ruin.core.network import NetworkSurplus
from ruin.risk.ruin_probability import wilson_interval
from ruin.simulation.agent_based import advance_shift, make_network
from ruin.state_space.probability_square import ProbabilitySquare

_Branch = tuple[bool, str | None, float, float]


def _run_branches(
    args: tuple[ProbabilitySquare, NetworkSurplus, int, list[list[int]]],
) -> list[_Branch]:
    """Worker: play out several independent futures from one present state."""
    square, network, horizon, seeds = args
    out: list[_Branch] = []
    for entropy in seeds:
        future = square.fork(np.random.SeedSequence(entropy))
        surplus = network.fork()
        start = len(surplus.surplus_path)
        loss_before = surplus.cumulative_loss
        advance_shift(future, surplus, horizon, stop_at_ruin=False)
        out.append(
            (
                surplus.ruined,
                surplus.ruin_reason,
                min(surplus.surplus_path[start:], default=surplus.surplus),
                surplus.cumulative_loss - loss_before,
            )
        )
    return out


def run_destiny(
    config: ConfigLike,
    seed: int | None = None,
    every: int = 24,
    branches: int = 64,
    confidence: float | None = None,
    max_steps: int | None = None,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Follow one shift and estimate its remaining destiny every `every` steps.

    Each checkpoint reports:

    - `destiny`: share of `branches` futures that do not ruin, with a Wilson
      interval at `confidence`.
    - `required_buffer`: the surplus the shift would need *now* for a
      `confidence` chance of never touching the surplus barrier (the
      `confidence` quantile of the futures' worst drawdown). Compare it with
      `surplus`: the gap is the shift's margin, in the units of the buffer.
    - `expected_future_loss`: mean delay penalty still to come.
    """
    cfg = as_config(config)
    base_seed = int(cfg.simulation.seed if seed is None else seed)
    level = float(cfg.risk.confidence_level if confidence is None else confidence)
    horizon = int(max_steps if max_steps is not None else cfg.simulation.shift_duration)
    barrier = float(cfg.ruin.barrier)

    square = ProbabilitySquare(cfg, base_seed, store_qdot_snapshots=False)
    network = make_network(cfg)

    # Follow the observed shift, keeping a copy of the present at each checkpoint.
    checkpoints: list[tuple[int, float, str, float, ProbabilitySquare, NetworkSurplus]] = []
    for t in range(0, horizon, max(1, every)):
        advance_shift(square, network, t)
        if network.ruined:
            break
        checkpoints.append(
            (
                t,
                network.surplus,
                square.d_state.value,
                square.global_chaos_pressure,
                square.fork(0),
                network.fork(),
            )
        )
    advance_shift(square, network, horizon)

    jobs = [
        (state, surplus, horizon, [[base_seed, t, b] for b in range(branches)])
        for t, _, _, _, state, surplus in checkpoints
    ]
    workers = max(1, min(n_jobs or os.cpu_count() or 4, len(jobs) or 1))
    if workers == 1:
        outcomes = [_run_branches(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            outcomes = list(executor.map(_run_branches, jobs))

    curve = []
    for (t, surplus_now, d_state, chaos, _, _), futures in zip(checkpoints, outcomes, strict=True):
        survived = sum(1 for ruined, _, _, _ in futures if not ruined)
        low, high = wilson_interval(survived, branches, level)
        drawdowns = [surplus_now - lowest for _, _, lowest, _ in futures]
        curve.append(
            {
                "time": t,
                "surplus": round(surplus_now, 4),
                "d_state": d_state,
                "chaos_pressure": round(chaos, 4),
                "destiny": round(survived / branches, 4),
                "destiny_ci": [round(low, 4), round(high, 4)],
                "required_buffer": round(float(np.quantile(drawdowns, level)) + barrier, 4),
                "expected_future_loss": round(float(np.mean([f[3] for f in futures])), 4),
                "futures_ruined_by_surplus": sum(1 for f in futures if f[1] == "surplus"),
                "futures_ruined_by_sla": sum(1 for f in futures if f[1] == "sla"),
            }
        )

    end_time = network.ruin_time if network.ruined else horizon
    destinies = [float(row["destiny"]) for row in curve]  # type: ignore[arg-type]
    lowest = curve[destinies.index(min(destinies))] if curve else None
    return {
        "seed": base_seed,
        "horizon": horizon,
        "branches": branches,
        "confidence_level": level,
        "ruined": network.ruined,
        "ruin_time": network.ruin_time,
        "ruin_reason": network.ruin_reason,
        "final_surplus": round(network.surplus, 4),
        "initial_destiny": curve[0]["destiny"] if curve else None,
        "lowest_destiny": lowest["destiny"] if lowest else None,
        "lowest_destiny_time": lowest["time"] if lowest else None,
        "curve": curve,
        # The curve's known end point: no destiny left at ruin, all of it at the horizon.
        "end": {"time": end_time, "destiny": 0.0 if network.ruined else 1.0},
        "surplus_path": [round(v, 4) for v in network.surplus_path],
    }


# Dark chart surface and ink; series colours are slots 1 and 2 of a palette
# validated for colour-vision deficiency on this surface.
_SURFACE, _INK, _INK_2, _MUTED, _GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a"
_BLUE, _ORANGE, _CRITICAL = "#3987e5", "#d95926", "#d03b3b"


def plot_destiny(result: dict[str, Any], output: str | Path) -> Path:
    """Draw the destiny curve above the surplus it is explained by, as a PNG.

    Two panels share the time axis rather than two scales sharing one panel:
    remaining destiny with its confidence band, then surplus against the
    buffer the shift would need at each checkpoint.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output)
    curve = result["curve"]
    times = [row["time"] for row in curve] + [result["end"]["time"]]
    destiny = [row["destiny"] for row in curve] + [result["end"]["destiny"]]
    low = [row["destiny_ci"][0] for row in curve] + [result["end"]["destiny"]]
    high = [row["destiny_ci"][1] for row in curve] + [result["end"]["destiny"]]
    level = round(100 * result["confidence_level"])

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(9, 6.4), sharex=True, facecolor=_SURFACE, height_ratios=[1, 1]
    )
    for ax in (top, bottom):
        ax.set_facecolor(_SURFACE)
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.tick_params(colors=_MUTED, length=0, labelsize=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(_GRID)
        ax.set_xlim(0, result["horizon"])

    top.fill_between(times, low, high, color=_BLUE, alpha=0.18, linewidth=0)
    top.plot(times, destiny, color=_BLUE, linewidth=2, marker="o", markersize=4)
    top.set_ylim(-0.03, 1.03)
    top.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    top.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    outcome = (
        f"ruined at t = {result['ruin_time']} ({result['ruin_reason']})"
        if result["ruined"]
        else "survived the shift"
    )
    top.set_title(
        "Remaining destiny: chance of reaching the end of the shift without ruin\n"
        f"Seed {result['seed']}, this shift {outcome}. "
        f"{result['branches']} futures per checkpoint, band = {level}% interval.",
        color=_INK,
        fontsize=10,
        loc="left",
    )

    surplus = result["surplus_path"]
    bottom.plot(range(1, len(surplus) + 1), surplus, color=_BLUE, linewidth=2, label="Surplus")
    bottom.plot(
        [row["time"] for row in curve],
        [row["required_buffer"] for row in curve],
        color=_ORANGE,
        linewidth=2,
        marker="o",
        markersize=4,
        label=f"Buffer needed for a {level}% chance of no surplus ruin",
    )
    bottom.axhline(0, color=_MUTED, linewidth=1)
    bottom.set_xlabel("time (steps)", color=_INK_2, fontsize=9)
    legend = bottom.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, fontsize=9, borderpad=0
    )
    for text in legend.get_texts():
        text.set_color(_INK_2)

    if result["ruined"]:
        for ax in (top, bottom):
            ax.axvline(result["ruin_time"], color=_CRITICAL, linewidth=1.2, linestyle="--")
        top.annotate(
            "ruin",
            (result["ruin_time"], 0.5),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            color=_INK_2,
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=140, facecolor=_SURFACE)
    plt.close(fig)
    return output_path

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
from ruin.inference.observer import RegimeObserver
from ruin.risk.ruin_probability import wilson_interval
from ruin.simulation.agent_based import advance_shift, make_network
from ruin.state_space.probability_square import ProbabilitySquare

_Branch = tuple[bool, str | None, float, float]


def _run_branches(
    args: tuple[ProbabilitySquare, NetworkSurplus, int, list[list[int]], list[float] | None],
) -> list[_Branch]:
    """Worker: play out several independent futures from one present state.

    With `regime_probabilities`, each future first redraws which hidden regime
    D is in from those probabilities; without them it keeps the true regime.
    """
    square, network, horizon, seeds, regime_probabilities = args
    out: list[_Branch] = []
    for entropy in seeds:
        future = square.fork(np.random.SeedSequence(entropy))
        if regime_probabilities is not None:
            future.redraw_regime(np.asarray(regime_probabilities))
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
    blind: bool = False,
    observer: RegimeObserver | None = None,
    observer_incidents: bool = False,
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

    When the scenario has a hidden regime, the futures above start from the
    true present, regime included: what someone who could see the regime
    would conclude. With `blind=True` each checkpoint also reports
    `destiny_blind`, from futures that know the whole state of D except the
    regime, which they redraw from its long-run mix. The gap between the two
    is what knowing the regime is worth; an observer who sees the pressures
    sits somewhere in between.

    Pass a fitted `observer` (see `ruin.inference.fit_regime_observer`) to put
    that observer's view on the chart: at each checkpoint the futures redraw
    the regime from the observer's calibrated belief given everything seen so
    far, and the row gains `regime_posterior` and `destiny_observer`. With
    `observer_incidents` the observer also has an incident feed (shock counts).
    The observer filters with the regime model of the scenario it was fitted
    on, so it should be fitted on the scenario passed here.
    """
    cfg = as_config(config)
    base_seed = int(cfg.simulation.seed if seed is None else seed)
    level = float(cfg.risk.confidence_level if confidence is None else confidence)
    horizon = int(max_steps if max_steps is not None else cfg.simulation.shift_duration)
    barrier = float(cfg.ruin.barrier)

    square = ProbabilitySquare(cfg, base_seed, store_qdot_snapshots=False)
    network = make_network(cfg)

    # Follow the observed shift, keeping a copy of the present at each checkpoint.
    watcher = observer.filter(observer_incidents) if observer is not None else None
    seen = 0
    checkpoints: list[
        tuple[
            int,
            float,
            str,
            float,
            ProbabilitySquare,
            NetworkSurplus,
            str | None,
            list[float] | None,
        ]
    ] = []
    for t in range(0, horizon, max(1, every)):
        advance_shift(square, network, t)
        if watcher is not None:
            for snap in square.snapshots[seen:]:
                watcher.update(snap["chaos_pressure"], snap["order_pressure"], snap["shock_count"])
            seen = len(square.snapshots)
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
                square.regime.value if square.regime is not None else None,
                [float(p) for p in watcher.posterior] if watcher is not None else None,
            )
        )
    advance_shift(square, network, horizon)

    def jobs(mode: str) -> list[Any]:
        stationary = [float(p) for p in cfg.probability_square.regime.stationary()]
        out = []
        for t, _, _, _, state, surplus, _, posterior in checkpoints:
            probabilities = {"informed": None, "blind": stationary, "observer": posterior}[mode]
            out.append(
                (
                    state,
                    surplus,
                    horizon,
                    [[base_seed, t, b] for b in range(branches)],
                    probabilities,
                )
            )
        return out

    workers = max(1, min(n_jobs or os.cpu_count() or 4, len(checkpoints) or 1))

    def run(job_list: list[Any]) -> list[list[_Branch]]:
        if workers == 1:
            return [_run_branches(job) for job in job_list]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_run_branches, job_list))

    outcomes = run(jobs("informed"))
    has_regime = square.regime_process is not None
    blind_outcomes = run(jobs("blind")) if blind and has_regime else None
    observer_outcomes = run(jobs("observer")) if watcher is not None else None

    curve = []
    for (t, surplus_now, d_state, chaos, _, _, regime, _), futures in zip(
        checkpoints, outcomes, strict=True
    ):
        survived = sum(1 for ruined, _, _, _ in futures if not ruined)
        low, high = wilson_interval(survived, branches, level)
        drawdowns = [surplus_now - lowest for _, _, lowest, _ in futures]
        curve.append(
            {
                "time": t,
                "surplus": round(surplus_now, 4),
                "d_state": d_state,
                "regime": regime,
                "chaos_pressure": round(chaos, 4),
                "destiny": round(survived / branches, 4),
                "destiny_ci": [round(low, 4), round(high, 4)],
                "required_buffer": round(float(np.quantile(drawdowns, level)) + barrier, 4),
                "expected_future_loss": round(float(np.mean([f[3] for f in futures])), 4),
                "futures_ruined_by_surplus": sum(1 for f in futures if f[1] == "surplus"),
                "futures_ruined_by_sla": sum(1 for f in futures if f[1] == "sla"),
            }
        )

    for key, extra in (("destiny_blind", blind_outcomes), ("destiny_observer", observer_outcomes)):
        if extra is None:
            continue
        for row, futures in zip(curve, extra, strict=True):
            survived = sum(1 for ruined, _, _, _ in futures if not ruined)
            low, high = wilson_interval(survived, branches, level)
            row[key] = round(survived / branches, 4)
            row[f"{key}_ci"] = [round(low, 4), round(high, 4)]
    if observer_outcomes is not None:
        for row, (*_, posterior) in zip(curve, checkpoints, strict=True):
            row["regime_posterior"] = [round(p, 4) for p in posterior or []]

    end_time = network.ruin_time if network.ruined else horizon
    destinies = [float(row["destiny"]) for row in curve]  # type: ignore[arg-type]
    lowest = curve[destinies.index(min(destinies))] if curve else None
    return {
        "seed": base_seed,
        "observer": None if observer is None else {"incidents": observer_incidents},
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
        "regime_path": [snapshot["regime"] for snapshot in square.snapshots],
    }


# Dark chart surface and ink; series colours are slots 1 to 3 of a palette
# validated for colour-vision deficiency on this surface.
_SURFACE, _INK, _INK_2, _MUTED, _GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a"
# Categorical slots 1-4 in fixed order: knowing the regime, observing it, not knowing it, and
# (bottom panel) the buffer needed. Surplus wears primary ink.
_BLUE, _ORANGE, _AQUA, _YELLOW, _CRITICAL = "#3987e5", "#d95926", "#199e70", "#c98500", "#d03b3b"
_REGIME_NAMES = ("stable", "stressed", "disrupted", "chaotic")
# Neutral gray ramp, brighter is worse: regime is context, not a series.
_REGIME_GRAY = (0.18, 0.34, 0.56, 0.80)


def plot_destiny(result: dict[str, Any], output: str | Path) -> Path:
    """Draw the destiny curve, the hidden regime, and the surplus, as a PNG.

    Panels share the time axis rather than two scales sharing one panel:
    remaining destiny with its confidence band (and, when computed, the
    destiny of a blind observer who does not know the regime), a strip with
    the hidden regime of the followed shift (brighter is worse), then surplus
    against the buffer the shift would need at each checkpoint.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_path = Path(output)
    curve = result["curve"]
    times = [row["time"] for row in curve] + [result["end"]["time"]]
    destiny = [row["destiny"] for row in curve] + [result["end"]["destiny"]]
    low = [row["destiny_ci"][0] for row in curve] + [result["end"]["destiny"]]
    high = [row["destiny_ci"][1] for row in curve] + [result["end"]["destiny"]]
    level = round(100 * result["confidence_level"])
    regimes = [r for r in result.get("regime_path", []) if r is not None]
    has_blind = bool(curve) and "destiny_blind" in curve[0]
    has_observer = bool(curve) and "destiny_observer" in curve[0]
    with_incidents = bool((result.get("observer") or {}).get("incidents"))
    observer_label = (
        "Observing pressures and incidents" if with_incidents else "Observing the pressures"
    )

    rows = 3 if regimes else 2
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(9, 7.0 if regimes else 6.4),
        sharex=True,
        facecolor=_SURFACE,
        height_ratios=[3, 0.45, 2.6] if regimes else [1, 1],
    )
    top, bottom = axes[0], axes[-1]
    for ax in axes:
        ax.set_facecolor(_SURFACE)
        ax.tick_params(colors=_MUTED, length=0, labelsize=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(_GRID)
        ax.set_xlim(0, result["horizon"])
    for ax in (top, bottom):
        ax.grid(axis="y", color=_GRID, linewidth=0.8)

    top.fill_between(times, low, high, color=_BLUE, alpha=0.18, linewidth=0)
    top.plot(
        times, destiny, color=_BLUE, linewidth=2, marker="o", markersize=4,
        label="Knowing D's regime",
    )  # fmt: skip
    end = result["end"]["destiny"]
    if has_observer:
        seen = [row["destiny_observer"] for row in curve] + [end]
        top.plot(
            times, seen, color=_ORANGE, linewidth=2, linestyle=(0, (1, 1.5)), marker="s",
            markersize=4, label=observer_label,
        )  # fmt: skip
    if has_blind:
        blind = [row["destiny_blind"] for row in curve] + [end]
        top.plot(
            times, blind, color=_AQUA, linewidth=2, linestyle=(0, (4, 2)), marker="^",
            markersize=4, label="Not knowing the regime",
        )  # fmt: skip
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
        pad=22 if (has_blind or has_observer) else 6,
    )
    if has_blind or has_observer:
        legend = top.legend(
            loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=9,
            borderpad=0,
        )  # fmt: skip
        for text in legend.get_texts():
            text.set_color(_INK_2)

    if regimes:
        strip = axes[1]
        gray = [_REGIME_GRAY[_REGIME_NAMES.index(r.lower())] for r in regimes]
        strip.imshow(
            [gray], aspect="auto", cmap="gray", vmin=0, vmax=1,
            extent=(0, len(regimes), 0, 1), interpolation="nearest",
        )  # fmt: skip
        strip.set_yticks([])
        strip.set_ylabel("regime", color=_MUTED, fontsize=8, rotation=0, ha="right", va="center")
        strip.legend(
            handles=[
                Patch(facecolor=str(g), edgecolor=_MUTED, label=n)
                for g, n in zip(_REGIME_GRAY, _REGIME_NAMES, strict=True)
            ],
            loc="upper left", bbox_to_anchor=(0.0, -0.1), ncol=4, frameon=False, fontsize=8,
            labelcolor=_INK_2, handlelength=1.2, borderpad=0,
        )  # fmt: skip

    surplus = result["surplus_path"]
    bottom.plot(range(1, len(surplus) + 1), surplus, color=_INK, linewidth=2, label="Surplus")
    bottom.plot(
        [row["time"] for row in curve],
        [row["required_buffer"] for row in curve],
        color=_YELLOW,
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
        for ax in axes:
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

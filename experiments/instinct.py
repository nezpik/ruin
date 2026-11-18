"""Exploration: does a felt deviation from the routine day ("Inst") warn earlier than the board?

Run from the repository root:

    python experiments/instinct.py [--days 1200]

It simulates `days` shifts of the reference scenario with different seeds (the "days" of one
operation), treats the first half as the days the logistician has lived through (they define the
routine) and the second half as new days. For each new day it builds, at every step, signals that
a person could watch, and asks how early each one flags a shift that goes on to ruin, when each
is allowed to raise a false alarm on 10% of shifts that do not ruin.

This is an exploration of a concept (see RUIN.md, "Destiny and the Instinct"), not a fitted
forecasting model. The scenario is the repository's illustrative one, not real data.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ruin.config import load_ruin_config
from ruin.config_models import RuinConfig
from ruin.simulation.agent_based import run_trajectory

CONFIG = Path(__file__).resolve().parent.parent / "examples" / "urban_ruin_shift.yaml"
FIRST_SEED = 5000
WARMUP = 10  # steps before a signal may raise an alarm (the shift is still filling up)
HORIZON = 40  # "about to ruin" means within this many steps
FALSE_ALARM = 0.10


def _day(args: tuple[RuinConfig, int]) -> dict[str, np.ndarray | int]:
    config, seed = args
    r = run_trajectory(config, seed=seed, store_qdot_snapshots=False, stop_at_ruin=False)
    return {
        "late": np.array(r["late_ratio_path"]),
        "sla": np.array(r["sla_miss_rate_path"]),
        "surplus": np.array(r["surplus_path"]),
        "chaos": np.array(r["chaos_pressure_path"]),
        "field": np.array([s["field_mean"] for s in r["snapshots"]]),
        "ruin_time": -1 if r["ruin_time"] is None else int(r["ruin_time"]),
    }


def simulate(days: int) -> dict[str, np.ndarray]:
    config = load_ruin_config(str(CONFIG))
    jobs = [(config, FIRST_SEED + i) for i in range(days)]
    with ProcessPoolExecutor() as pool:
        out = list(pool.map(_day, jobs, chunksize=8))
    n = min(len(o["late"]) for o in out)  # type: ignore[arg-type]
    data = {
        k: np.stack([o[k][:n] for o in out])  # type: ignore[index]
        for k in ("late", "sla", "surplus", "chaos", "field")
    }
    data["ruin_time"] = np.array([o["ruin_time"] for o in out])
    return data


def smooth(x: np.ndarray, width: int = 9) -> np.ndarray:
    pad = width // 2
    return np.convolve(np.pad(x, pad, mode="edge"), np.ones(width) / width, mode="valid")


def ewma(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(x)
    acc = np.zeros(x.shape[0])
    for t in range(x.shape[1]):
        acc = (1 - alpha) * acc + alpha * x[:, t]
        out[:, t] = acc
    return out


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    ranks = np.concatenate([positive, negative]).argsort().argsort() + 1
    n_pos = len(positive)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * len(negative)))


def signals(data: dict[str, np.ndarray], lived: int) -> dict[str, np.ndarray]:
    """Everything a person could watch, for every day and step."""

    def against_routine(x: np.ndarray) -> np.ndarray:
        """Distance from the lived days' typical value at the same point of the shift."""
        mean = smooth(x[:lived].mean(0))
        spread = smooth(x[:lived].std(0)) + 1e-9
        return (x - mean) / spread

    def against_overall(x: np.ndarray) -> np.ndarray:
        return (x - x[:lived].mean()) / x[:lived].std()

    channels = ("late", "field", "chaos")
    routine = sum(against_routine(data[k]) for k in channels) / 3
    overall = sum(against_overall(data[k]) for k in channels) / 3
    board = np.maximum(data["sla"] / 0.08, 1 - data["surplus"] / data["surplus"][:, :1])
    return {
        "The board (SLA misses, surplus drawdown)": board,
        "Late ratio, raw": data["late"],
        "Chaos pressure, raw": data["chaos"],
        "Inst: late vs routine day": against_routine(data["late"]),
        "Inst: late, field, chaos vs routine day": routine,
        "Inst: the same, building up (EWMA)": ewma(routine, 0.15),
        "Same, but no routine (one overall scale)": overall,
        "Same, no routine, building up (EWMA)": ewma(overall, 0.15),
    }


def evaluate(
    data: dict[str, np.ndarray], lived: int
) -> list[tuple[str, float, int, int, float, float]]:
    ruin = data["ruin_time"]
    n_steps = data["late"].shape[1]
    new_days = range(lived, len(ruin))
    old_days = range(lived)

    def end(i: int) -> int:
        return int(ruin[i]) if ruin[i] >= 0 else n_steps

    rows = []
    for name, x in signals(data, lived).items():
        threshold = np.quantile(
            [x[i, WARMUP : end(i)].max() for i in old_days if ruin[i] < 0 and end(i) > WARMUP],
            1 - FALSE_ALARM,
        )
        near, far, leads = [], [], []
        alarmed = ruined = quiet = false_alarms = 0
        for i in new_days:
            for t in range(WARMUP, end(i), 4):
                if ruin[i] >= 0 and ruin[i] - t <= HORIZON:
                    near.append(x[i, t])
                elif ruin[i] < 0 or ruin[i] - t > 2 * HORIZON:
                    far.append(x[i, t])
            hits = np.where(x[i, WARMUP : end(i)] > threshold)[0]
            if ruin[i] >= 0:
                ruined += 1
                if len(hits):
                    alarmed += 1
                    leads.append(end(i) - (hits[0] + WARMUP))
            else:
                quiet += 1
                false_alarms += len(hits) > 0
        median_lead = float(np.median(leads)) if leads else float("nan")
        rows.append(
            (
                name,
                auc(np.array(near), np.array(far)),
                alarmed,
                ruined,
                median_lead,
                false_alarms / quiet,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--days", type=int, default=1200)
    days = parser.parse_args().days
    data = simulate(days)
    lived = days // 2
    ruin = data["ruin_time"]
    print(
        f"{lived} lived days (ruined: {(ruin[:lived] >= 0).sum()}), "
        f"{days - lived} new days (ruined: {(ruin[lived:] >= 0).sum()})\n"
    )
    print(f"{'signal':44s} {'AUC':>6s} {'flagged':>9s} {'median lead':>12s} {'false alarms':>13s}")
    for name, area, alarmed, ruined, lead, false_rate in evaluate(data, lived):
        print(f"{name:44s} {area:6.3f} {alarmed:>5d}/{ruined:<3d} {lead:12.0f} {false_rate:13.2f}")


if __name__ == "__main__":
    main()

"""Stylized facts of logistics disruption, and whether a scenario reproduces them.

Without data, a model cannot be calibrated, but it can be checked against
qualitative regularities that logistics operations are widely reported to
show. Each fact here is a measurable property of simulated shifts, compared
where useful with the memoryless light-tailed version of the same scenario
at the same realised mean disruption (`ruin.risk.structure`). `ruin facts`
runs the suite.

The facts, and what each one guards against:

1. Disruption clusters: counts of disruption events per window are more
   dispersed than Poisson (Fano factor above 1).
2. Bad stretches persist: adjacent windows are positively correlated.
3. Shift outcomes are heavy-tailed: a few shifts cost far more than the
   typical one.
4. Trip times are heavy-tailed: a few trips take many times longer than usual.
5. Good days and bad days: the share of delayed trips varies from shift to
   shift far more than independent disruption would make it.
6. The flaw of averages: at equal mean disruption, clustered and heavy-tailed
   disruption costs more on average than the memoryless kind.
7. The state of D is only partly observable: the D-state label follows the
   hidden regime without revealing it exactly.
8. Co-evolution: D's own pressures push the regime. When chaos pressure
   points to a worse level than the regime's, the regime escalates more often.

These are properties a model should have, not proof that it is right.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from ruin.config import ConfigLike, as_config
from ruin.config_models import RuinConfig
from ruin.risk.structure import realised_mean_shock_mass, structure_variants
from ruin.simulation.agent_based import advance_shift, make_network
from ruin.state_space.probability_square import ProbabilitySquare
from ruin.state_space.regime import RegimeProcess

_OBSERVED = {"STABLE": 0, "STRESSED": 1, "DISRUPTED": 2, "CHAOTIC": 3}
_REGIME = {"STABLE": 0, "STRESSED": 1, "DISRUPTED": 2, "CHAOTIC": 3}
WINDOW = 24  # steps per window for clustering statistics
ACF_WINDOW = 12  # steps per window for persistence


def _shift(args: tuple[RuinConfig, int]) -> dict[str, Any]:
    """One whole shift, summarised for the statistics below."""
    cfg, seed = args
    square = ProbabilitySquare(cfg, seed, store_qdot_snapshots=False)
    network = make_network(cfg)
    advance_shift(square, network, int(cfg.simulation.shift_duration), stop_at_ruin=False)
    slowdown = []
    for row in square.ledger():
        if row["type"] != "STANDARD" or row["status"] not in ("DELIVERED", "DELIVERED_LATE"):
            continue
        (ox, oy), (dx, dy) = row["origin"], row["destination"]
        distance = abs(ox - dx) + abs(oy - dy)
        if distance >= 5 and row["end_time"] is not None:
            slowdown.append((row["end_time"] - row["spawn_time"]) / distance)
    snaps = square.snapshots
    return {
        "shocks": [s["shock_count"] for s in snaps],
        "loss": network.cumulative_loss,
        "regime": [s["regime"] for s in snaps],
        "d_state": [s["d_state"] for s in snaps],
        "chaos": [s["chaos_pressure"] for s in snaps],
        "slowdown": slowdown,
    }


def _run(cfg: RuinConfig, paths: int, seed: int, n_jobs: int | None) -> list[dict[str, Any]]:
    jobs = [(cfg, seed + i) for i in range(paths)]
    workers = max(1, min(n_jobs or os.cpu_count() or 4, paths))
    if workers == 1:
        return [_shift(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_shift, jobs, chunksize=max(1, paths // (workers * 4))))


def _windows(counts: list[int], size: int) -> np.ndarray:
    n = len(counts) // size
    return np.asarray(counts[: n * size], dtype=float).reshape(n, size).sum(axis=1)


def fano_factor(shifts: list[dict[str, Any]], window: int = WINDOW) -> float:
    """Variance over mean of event counts per window, pooled over shifts."""
    pooled = np.concatenate([_windows(s["shocks"], window) for s in shifts])
    return float(pooled.var() / pooled.mean()) if pooled.mean() > 0 else float("nan")


def window_autocorrelation(shifts: list[dict[str, Any]], window: int = ACF_WINDOW) -> float:
    """Correlation between the event count of a window and the next one."""
    first: list[float] = []
    second: list[float] = []
    for s in shifts:
        w = _windows(s["shocks"], window)
        first.extend(w[:-1])
        second.extend(w[1:])
    return float(np.corrcoef(first, second)[0, 1])


def tail_ratio(values: list[float] | np.ndarray, high: float, low: float = 50) -> float:
    """The `high` percentile over the `low` percentile (default: median)."""
    v = np.asarray(values, dtype=float)
    mid = float(np.percentile(v, low))
    return float(np.percentile(v, high) / mid) if mid > 0 else float("nan")


def delayed_share_cv(shifts: list[dict[str, Any]], factor: float = 1.5) -> float:
    """Variation of the share of delayed trips from shift to shift.

    A trip is delayed if it took more than `factor` times the median steps
    per cell over all trips. Returns the coefficient of variation across
    shifts: near 0 when every shift is alike, large when there are good days
    and bad days.
    """
    median = float(np.median(np.concatenate([s["slowdown"] for s in shifts])))
    shares = np.array([np.mean(np.asarray(s["slowdown"]) > factor * median) for s in shifts])
    return float(shares.std() / shares.mean()) if shares.mean() > 0 else float("nan")


def escalation_ratio(config: RuinConfig, shifts: list[dict[str, Any]]) -> float:
    """How much likelier the regime is to escalate when pressure points higher.

    Over all steps, the chance that the regime moves up on the next step when
    this step's chaos pressure points to a worse level than the regime's,
    divided by that chance otherwise. 1 means the regime ignores pressure.
    """
    process = RegimeProcess(
        config.probability_square.regime, config.probability_square.d_state_thresholds
    )
    pointing: list[bool] = []
    escalated: list[bool] = []
    for s in shifts:
        levels = [_REGIME[r] for r in s["regime"]]
        for t in range(len(levels) - 1):
            if levels[t] < 3:
                pointing.append(process.level_from_pressure(s["chaos"][t]) > levels[t])
                escalated.append(levels[t + 1] > levels[t])
    up, hit = np.array(pointing), np.array(escalated)
    if not up.any() or not hit[~up].any():
        return float("nan")
    return float(hit[up].mean() / hit[~up].mean())


def run_stylized_facts(
    config: ConfigLike,
    paths: int = 80,
    seed: int | None = None,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Measure each stylized fact on `paths` simulated shifts and report pass/fail."""
    cfg = as_config(config)
    seed = int(cfg.simulation.seed if seed is None else seed)
    regime_cfg = cfg.probability_square.regime
    scenario = _run(cfg, paths, seed, n_jobs)

    mean_mass = realised_mean_shock_mass(cfg, shifts=max(20, paths // 2), n_jobs=n_jobs)
    thin_cfg = structure_variants(
        cfg if regime_cfg.enabled else _with_default_regime(cfg), mean_mass
    )["thin"]
    thin = _run(thin_cfg, paths, seed, n_jobs)

    facts: list[dict[str, Any]] = []

    def add(
        name: str, claim: str, measured: dict[str, Any], criterion: str, passed: bool | None
    ) -> None:
        facts.append(
            {
                "fact": name,
                "claim": claim,
                "measured": measured,
                "criterion": criterion,
                "passed": passed,
            }
        )

    f_s, f_t = fano_factor(scenario), fano_factor(thin)
    add(
        "clustering",
        "Disruption events cluster in time.",
        {"scenario": round(f_s, 2), "thin": round(f_t, 2)},
        "Fano factor of events per 24 steps >= 1.5 (Poisson = 1)",
        f_s >= 1.5,
    )
    a_s, a_t = window_autocorrelation(scenario), window_autocorrelation(thin)
    add(
        "persistence",
        "A bad stretch is followed by another bad stretch.",
        {"scenario": round(a_s, 2), "thin": round(a_t, 2)},
        "correlation between adjacent 12-step windows >= 0.15",
        a_s >= 0.15,
    )
    loss_s = [s["loss"] for s in scenario]
    loss_t = [s["loss"] for s in thin]
    r_s, r_t = tail_ratio(loss_s, 95), tail_ratio(loss_t, 95)
    add(
        "heavy_tailed_shifts",
        "A few shifts cost far more than the typical one.",
        {"scenario": round(r_s, 2), "thin": round(r_t, 2)},
        "95th over 50th percentile of whole-shift loss >= 3 and >= 1.5x thin",
        r_s >= 3.0 and r_s >= 1.5 * r_t,
    )
    d_s = tail_ratio([x for s in scenario for x in s["slowdown"]], 99)
    d_t = tail_ratio([x for s in thin for x in s["slowdown"]], 99)
    add(
        "heavy_tailed_trips",
        "A few trips take many times longer than usual.",
        {"scenario": round(d_s, 2), "thin": round(d_t, 2)},
        "99th over 50th percentile of steps per cell >= 2",
        d_s >= 2.0,
    )
    c_s, c_t = delayed_share_cv(scenario), delayed_share_cv(thin)
    add(
        "good_days_and_bad_days",
        "The share of delayed trips varies from shift to shift.",
        {"scenario": round(c_s, 2), "thin": round(c_t, 2)},
        "variation across shifts of the delayed-trip share >= 2x thin",
        c_s >= 2.0 * c_t,
    )
    m_s, m_t = float(np.mean(loss_s)), float(np.mean(loss_t))
    add(
        "flaw_of_averages",
        "At equal mean disruption, clustered heavy-tailed disruption costs more on average.",
        {
            "scenario_mean_loss": round(m_s, 1),
            "thin_mean_loss": round(m_t, 1),
            "ratio": round(m_s / m_t, 2),
        },
        "mean whole-shift loss >= 1.1x thin",
        m_s >= 1.1 * m_t,
    )

    if regime_cfg.enabled:
        pairs = [
            (_REGIME[r], _OBSERVED[d])
            for s in scenario
            for r, d in zip(s["regime"], s["d_state"], strict=True)
            if d in _OBSERVED
        ]
        latent, observed = np.array(pairs).T
        exact = float((latent == observed).mean())
        within_one = float((abs(latent - observed) <= 1).mean())
        corr = float(np.corrcoef(latent, observed)[0, 1])
        add(
            "partial_observability",
            "The D-state follows the hidden regime without revealing it exactly.",
            {
                "exact": round(exact, 2),
                "within_one_level": round(within_one, 2),
                "correlation": round(corr, 2),
            },
            "correlation >= 0.5, within one level >= 0.9, exact match < 0.9",
            corr >= 0.5 and within_one >= 0.9 and exact < 0.9,
        )
        if regime_cfg.pressure_pull > 1:
            free = cfg.model_copy(deep=True)
            free.probability_square.regime.pressure_pull = 1.0
            free.probability_square.regime.order_support = 0.0
            with_pull = escalation_ratio(cfg, scenario)
            without = escalation_ratio(free, _run(free, paths, seed, n_jobs))
            add(
                "co_evolution",
                "D's own pressures push the regime toward worse levels.",
                {"with_feedback": round(with_pull, 2), "without_feedback": round(without, 2)},
                "escalation is >= 1.3x likelier when pressure points higher (1 = ignores pressure)",
                with_pull >= 1.3,
            )
        else:
            add(
                "co_evolution",
                "D's own pressures push the regime toward worse levels.",
                {},
                "not applicable (pressure_pull = 1)",
                None,
            )
    else:
        for name in ("partial_observability", "co_evolution"):
            add(name, "Needs a hidden regime.", {}, "not applicable (regime disabled)", None)

    applicable = [f for f in facts if f["passed"] is not None]
    return {
        "paths": paths,
        "scenario": cfg.simulation.name,
        "facts": facts,
        "passed": all(f["passed"] for f in applicable),
        "n_passed": sum(1 for f in applicable if f["passed"]),
        "n_applicable": len(applicable),
    }


def _with_default_regime(cfg: RuinConfig) -> RuinConfig:
    """A scenario without a regime, with default regime settings switched on, so a
    mean-matched thin comparison can still be built."""
    out = cfg.model_copy(deep=True)
    out.probability_square.regime.enabled = True
    return out

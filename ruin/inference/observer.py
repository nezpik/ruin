"""What an observer can know about the hidden regime of D.

D's regime generates disruption, but nobody sees it: an operator sees
pressures, and perhaps an incident log. The D-state label is a crude reading
of those pressures. This module does it properly: a Bayes filter that turns
the history of what was observed into a probability for each regime,

    belief(t) = P(regime at t | everything observed up to t).

The filter is exact where it can be. The regime chain is known, and its
transition probabilities depend only on D's chaos and order pressure, which
the observer sees, so the prediction step is exact. What is not known in
closed form is how likely a pressure reading is under each regime, because
pressure carries the memory of the disruption field. That likelihood is
modelled as a Markov-switching VAR(3): within each regime, the next
observation (log chaos pressure, order pressure) is a linear function of the
previous three plus Gaussian noise, fitted by least squares on simulated
shifts where the true regime is known. An optional incident feed adds the
number of shocks each step, whose likelihood is exactly Poisson under each
regime.

Successive pressure readings are far from independent given the regime (the
field remembers), so multiplying their likelihoods overstates the evidence:
an untempered one-lag filter was measurably *worse* than ignoring the
pressures (log score -0.80 against -0.77 for the stationary prior, and "96%
disrupted or worse" came true 71% of the time). The pressure likelihood is
therefore raised to a power, the temperature, chosen on held-out shifts to
maximise the log score. Tempering fixes the overall confidence but not a
systematic over-weighting of the severe regimes (middle-range forecasts of
"disrupted or worse" came true about 35% less often than forecast), so a
final per-regime bias on the log-odds is fitted on the same held-out shifts.
The belief the filter keeps for its own recursion stays raw; `posterior` is
the calibrated one. The result is approximately calibrated rather than
exactly Bayes, and `evaluate_observer` checks it against ground truth.

Two idealisations to keep in mind. The emission model is fitted with the true
regime labels, which a real operator never has (they would use EM on
unlabelled history), so a real filter should do somewhat worse. And the
observer is assumed to know the scenario's parameters.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ruin.config import ConfigLike, as_config
from ruin.config_models import RuinConfig
from ruin.simulation.agent_based import run_trajectory
from ruin.state_space.regime import REGIMES, RegimeProcess

# Added to chaos pressure before taking logs, so a calm reading near 0 stays finite.
LOG_FLOOR = 0.01
# Fewest training rows per regime that the fit will accept.
MIN_ROWS = 200
# Memory of the emission model, in steps.
LAGS = 3
# Candidate temperatures for the pressure likelihood.
TEMPERATURES = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
# Seeds: training, validation, and evaluation shifts must not overlap each
# other or the Monte Carlo seeds (simulation.seed + i).
TRAIN_OFFSET = 50_000
VALIDATION_OFFSET = 70_000
EVAL_OFFSET = 90_000

_OBSERVED_LEVEL = {"STABLE": 0, "STRESSED": 1, "DISRUPTED": 2, "CHAOTIC": 3}


def _features(chaos: np.ndarray, order: np.ndarray) -> np.ndarray:
    """(n, 2) observation vectors: log chaos pressure and order pressure."""
    return np.column_stack([np.log(np.asarray(chaos) + LOG_FLOOR), np.clip(order, -1.0, 1.0)])


def _design(history: np.ndarray) -> np.ndarray:
    """Regressor row [1, z(t-1), z(t-2), ...] from `history` (lags, 2), most recent first."""
    return np.concatenate([[1.0], history.ravel()])


@dataclass(frozen=True)
class EmissionModel:
    """Per-regime Gaussian VAR: z_t = beta[k].T @ [1, z_(t-1), ..., z_(t-lags)] + noise(cov[k])."""

    beta: np.ndarray  # (4, 1 + 2 * lags, 2)
    cov: np.ndarray  # (4, 2, 2)
    rows: tuple[int, ...]  # training rows per regime
    lags: int = LAGS
    temperature: float = 1.0  # power applied to the pressure likelihood
    _precision: np.ndarray = field(init=False, repr=False)
    _log_norm: float = field(init=False, repr=False)
    _log_dets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_precision", np.linalg.inv(self.cov))
        object.__setattr__(self, "_log_dets", np.linalg.slogdet(self.cov)[1])
        object.__setattr__(self, "_log_norm", -math.log(2.0 * math.pi))

    def with_temperature(self, temperature: float) -> EmissionModel:
        return EmissionModel(self.beta, self.cov, self.rows, self.lags, temperature)

    def log_likelihood(self, history: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Untempered log density of `current` given `history`, under each regime."""
        resid = current - np.einsum("j,kji->ki", _design(history), self.beta)  # (4, 2)
        quad = np.einsum("ki,kij,kj->k", resid, self._precision, resid)
        return np.asarray(self._log_norm - 0.5 * self._log_dets - 0.5 * quad)


def fit_emission_model(shifts: list[dict[str, np.ndarray]], lags: int = LAGS) -> EmissionModel:
    """Least-squares fit of the per-regime VAR from labelled shifts."""
    designs, targets, labels = [], [], []
    for s in shifts:
        z = _features(s["chaos"], s["order"])
        for t in range(lags, len(z)):
            designs.append(_design(z[t - lags : t][::-1]))
            targets.append(z[t])
            labels.append(s["regime"][t])
    x, y, label = np.array(designs), np.array(targets), np.array(labels)
    beta = np.zeros((4, x.shape[1], 2))
    cov = np.zeros((4, 2, 2))
    counts = []
    for k in range(4):
        mask = label == k
        counts.append(int(mask.sum()))
        if mask.sum() < MIN_ROWS:
            raise ValueError(
                f"Only {int(mask.sum())} training steps in regime {REGIMES[k].value}; "
                f"need {MIN_ROWS}. Use more shifts or a scenario that visits every regime."
            )
        beta[k], *_ = np.linalg.lstsq(x[mask], y[mask], rcond=None)
        resid = y[mask] - x[mask] @ beta[k]
        cov[k] = np.cov(resid.T) + 1e-6 * np.eye(2)
    return EmissionModel(beta, cov, tuple(counts), lags)


def _transition_matrix(process: RegimeProcess, chaos: float, order: float) -> np.ndarray:
    """P(regime now | regime last step) given last step's observed pressures."""
    matrix = np.zeros((4, 4))
    for level in range(4):
        up, down = process.hazards(level, chaos, order)
        if level < 3:
            matrix[level, level + 1] = up
        if level > 0:
            matrix[level, level - 1] = down
        matrix[level, level] = 1.0 - up - down
    return matrix


def _prior(config: RuinConfig) -> np.ndarray:
    regime_cfg = config.probability_square.regime
    if regime_cfg.start == "stationary":
        return np.asarray(regime_cfg.stationary(), dtype=float)
    index = ("stable", "stressed", "disrupted", "chaotic").index(regime_cfg.start)
    return np.asarray(np.eye(4)[index], dtype=float)


def _fit_bias(beliefs: np.ndarray, truth: np.ndarray, iterations: int = 25) -> np.ndarray:
    """Per-regime log-odds bias that minimises the log loss of the beliefs.

    Convex in the bias, so Newton's method converges in a few steps; the first
    regime is the reference (bias 0)."""
    bias = np.zeros(4)
    onehot = np.eye(4)[truth]
    for _ in range(iterations):
        q = _apply_bias(beliefs, bias)
        gradient = (q - onehot).mean(axis=0)
        hessian = np.diag(q.mean(axis=0)) - (q[:, :, None] * q[:, None, :]).mean(axis=0)
        step = np.zeros(4)
        step[1:] = np.linalg.solve(hessian[1:, 1:] + 1e-9 * np.eye(3), gradient[1:])
        bias = np.clip(bias - step, -6.0, 6.0)
    return bias


def _apply_bias(beliefs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Shift the log-odds of each regime by `bias` and renormalise."""
    z = np.log(np.clip(beliefs, 1e-9, 1.0)) + bias
    z = z - z.max(axis=-1, keepdims=True)
    q = np.exp(z)
    out: np.ndarray = q / q.sum(axis=-1, keepdims=True)
    return out


class RegimeFilter:
    """Bayes filter over the four regimes for one observed shift.

    Call `update` once per step with what was observed after the step. The
    belief it returns is the probability of each regime *during* that step.
    """

    def __init__(
        self,
        config: RuinConfig,
        emission: EmissionModel,
        use_incidents: bool = False,
        bias: np.ndarray | None = None,
    ) -> None:
        regime_cfg = config.probability_square.regime
        if not regime_cfg.enabled:
            raise ValueError(
                "A regime filter needs a scenario with probability_square.regime.enabled"
            )
        self.process = RegimeProcess(regime_cfg, config.probability_square.d_state_thresholds)
        self.emission = emission
        self.use_incidents = use_incidents
        self.bias = np.zeros(4) if bias is None else np.asarray(bias, dtype=float)
        dt = float(config.simulation.dt)
        rates = config.disruption_field.shock_arrival_rate * np.asarray(regime_cfg.shock_rate) * dt
        self._rates = np.maximum(rates, 1e-9)
        self.belief = _prior(config)
        # Before the first step the engine's pressures are zero.
        self._prev_chaos, self._prev_order = 0.0, 0.0
        self._history: list[np.ndarray] = []  # most recent observation first

    @property
    def posterior(self) -> np.ndarray:
        """The calibrated belief: what to report and to decide on.

        The correction was fitted on beliefs that carry pressure evidence, so it
        is applied only once enough readings have arrived for that evidence to
        enter. Before then the belief is the prior moved by the chain, which is
        already exact."""
        if len(self._history) < self.emission.lags:
            return self.belief
        return _apply_bias(self.belief, self.bias)

    def update(self, chaos: float, order: float, shocks: int | None = None) -> np.ndarray:
        """Advance one step with what was observed; returns the new raw belief
        (see `posterior` for the calibrated one)."""
        predicted = self.belief @ _transition_matrix(
            self.process, self._prev_chaos, self._prev_order
        )
        z = _features(np.array([chaos]), np.array([order]))[0]
        loglik = np.zeros(4)
        if len(self._history) == self.emission.lags:
            loglik += self.emission.temperature * self.emission.log_likelihood(
                np.array(self._history), z
            )
        if self.use_incidents and shocks is not None:
            loglik += shocks * np.log(self._rates) - self._rates - math.lgamma(shocks + 1)
        weights = predicted * np.exp(loglik - float(np.max(loglik)))
        total = weights.sum()
        self.belief = weights / total if total > 0 else predicted
        self._prev_chaos, self._prev_order = chaos, order
        self._history = [z, *self._history][: self.emission.lags]
        return self.belief


@dataclass
class RegimeObserver:
    """A fitted observation model for one scenario."""

    config: RuinConfig
    emission: EmissionModel
    # Fitted log-odds bias per regime, for the pressure-only and incident filters.
    bias_pressure: np.ndarray = field(default_factory=lambda: np.zeros(4))
    bias_incident: np.ndarray = field(default_factory=lambda: np.zeros(4))

    def filter(self, use_incidents: bool = False) -> RegimeFilter:
        bias = self.bias_incident if use_incidents else self.bias_pressure
        return RegimeFilter(self.config, self.emission, use_incidents=use_incidents, bias=bias)


def _shift(args: tuple[RuinConfig, int]) -> dict[str, np.ndarray]:
    cfg, seed = args
    r = run_trajectory(cfg, seed=seed, store_qdot_snapshots=False, stop_at_ruin=False)
    levels = {regime.value: i for i, regime in enumerate(REGIMES)}
    return {
        "regime": np.array([levels[x] for x in r["regime_path"]]),
        "chaos": np.array(r["chaos_pressure_path"]),
        "order": np.array(r["order_pressure_path"]),
        "shocks": np.array(r["shock_count_path"]),
        "d_state": np.array(r["d_state_path"]),
    }


def _simulate(
    cfg: RuinConfig, shifts: int, first_seed: int, n_jobs: int | None
) -> list[dict[str, np.ndarray]]:
    jobs = [(cfg, first_seed + i) for i in range(shifts)]
    workers = max(1, min(n_jobs or os.cpu_count() or 4, shifts))
    if workers == 1:
        return [_shift(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_shift, jobs, chunksize=max(1, shifts // (workers * 4))))


def _precompute(
    process: RegimeProcess, emission: EmissionModel, shift: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-step transition matrices and untempered pressure log-likelihoods.

    Neither depends on the belief, only on what was observed, so they can be
    computed once and reused to try many temperatures cheaply."""
    z = _features(shift["chaos"], shift["order"])
    n = len(z)
    transitions = np.zeros((n, 4, 4))
    loglik = np.zeros((n, 4))
    prev_c, prev_o = 0.0, 0.0
    for t in range(n):
        transitions[t] = _transition_matrix(process, prev_c, prev_o)
        if t >= emission.lags:
            loglik[t] = emission.log_likelihood(z[t - emission.lags : t][::-1], z[t])
        prev_c, prev_o = float(shift["chaos"][t]), float(shift["order"][t])
    return transitions, loglik


def _log_score(
    prior: np.ndarray,
    prepared: list[tuple[np.ndarray, np.ndarray]],
    truth: list[np.ndarray],
    temperature: float,
) -> float:
    total, count = 0.0, 0
    for (transitions, loglik), levels in zip(prepared, truth, strict=True):
        belief = prior
        for t in range(len(levels)):
            belief = belief @ transitions[t]
            weights = belief * np.exp(temperature * (loglik[t] - float(np.max(loglik[t]))))
            belief = weights / weights.sum()
            if t > 0:
                total += math.log(max(belief[levels[t]], 1e-12))
                count += 1
    return total / count


def fit_regime_observer(
    config: ConfigLike,
    shifts: int = 100,
    seed: int | None = None,
    n_jobs: int | None = None,
    validation_shifts: int = 40,
) -> RegimeObserver:
    """Fit the observation model on simulated shifts of the scenario.

    The emission model is fitted on `shifts` shifts; the temperature is then
    chosen from `TEMPERATURES` to maximise the log score of the pressure filter
    on `validation_shifts` further shifts that the fit never saw."""
    cfg = as_config(config)
    regime_cfg = cfg.probability_square.regime
    if not regime_cfg.enabled:
        raise ValueError(
            "A regime observer needs a scenario with probability_square.regime.enabled"
        )
    base = int(cfg.simulation.seed if seed is None else seed)
    emission = fit_emission_model(_simulate(cfg, shifts, base + TRAIN_OFFSET, n_jobs))
    validation = _simulate(cfg, validation_shifts, base + VALIDATION_OFFSET, n_jobs)
    process = RegimeProcess(regime_cfg, cfg.probability_square.d_state_thresholds)
    prepared = [_precompute(process, emission, v) for v in validation]
    truth = [v["regime"] for v in validation]
    prior = _prior(cfg)
    best = max(TEMPERATURES, key=lambda temp: _log_score(prior, prepared, truth, temp))
    observer = RegimeObserver(cfg, emission.with_temperature(best))
    levels = np.concatenate([t[observer.emission.lags :] for t in truth])
    for incidents, attribute in ((False, "bias_pressure"), (True, "bias_incident")):
        lags = observer.emission.lags
        raw = np.concatenate([_run_filter_raw(observer, v, incidents)[lags:] for v in validation])
        setattr(observer, attribute, _fit_bias(raw, levels))
    return observer


def _run_filter_raw(
    observer: RegimeObserver, shift: dict[str, np.ndarray], use_incidents: bool
) -> np.ndarray:
    f = observer.filter(use_incidents)
    return np.array(
        [
            f.update(float(c), float(o), int(n))
            for c, o, n in zip(shift["chaos"], shift["order"], shift["shocks"], strict=True)
        ]
    )


def run_filter(
    observer: RegimeObserver, shift: dict[str, np.ndarray], use_incidents: bool = False
) -> np.ndarray:
    """Calibrated belief over the regimes at every step of one simulated shift: (steps, 4)."""
    bias = observer.bias_incident if use_incidents else observer.bias_pressure
    raw = _run_filter_raw(observer, shift, use_incidents)
    out = raw.copy()
    out[observer.emission.lags :] = _apply_bias(raw[observer.emission.lags :], bias)
    return out


def _calibration(
    p: np.ndarray, happened: np.ndarray, bins: int = 10
) -> tuple[list[dict[str, float]], float]:
    """Reliability table for an event forecast, and its expected calibration error."""
    edges = np.linspace(0, 1, bins + 1)
    index = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = index == b
        if m.sum() == 0:
            continue
        rows.append(
            {
                "forecast": round(float(p[m].mean()), 3),
                "observed": round(float(happened[m].mean()), 3),
                "steps": int(m.sum()),
            }
        )
        ece += m.mean() * abs(float(p[m].mean()) - float(happened[m].mean()))
    return rows, float(ece)


def _spells(truth: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """(start, end) of each spell at DISRUPTED or worse lasting at least `min_length` steps."""
    high = truth >= 2
    spells, t = [], 1
    while t < len(high):
        if high[t] and not high[t - 1]:
            end = t
            while end < len(high) and high[end]:
                end += 1
            if end - t >= min_length:
                spells.append((t, end))
            t = end
        t += 1
    return spells


def _flag_quality(
    truths: list[np.ndarray], flags: list[np.ndarray], min_length: int = 12
) -> dict[str, Any]:
    """How well a per-step flag ("DISRUPTED or worse") tracks the truth.

    Reports precision and recall over steps, and over spells how many were
    missed and the median number of steps from the start of a spell to the
    flag first being raised."""
    t_all = np.concatenate([t[1:] >= 2 for t in truths])
    f_all = np.concatenate([f[1:] for f in flags])
    tp = float((t_all & f_all).sum())
    delays, missed = [], 0
    for truth, flag in zip(truths, flags, strict=True):
        for start, end in _spells(truth, min_length):
            hit = np.flatnonzero(flag[start:end])
            if len(hit):
                delays.append(int(hit[0]))
            else:
                missed += 1
    return {
        "precision": round(tp / max(float(f_all.sum()), 1.0), 3),
        "recall": round(tp / max(float(t_all.sum()), 1.0), 3),
        "spells": len(delays) + missed,
        "missed_spells": missed,
        "median_steps_to_detect": round(float(np.median(delays)), 1) if delays else None,
    }


def _precision_at_recall(
    truths: list[np.ndarray], high: list[np.ndarray], target_recall: float
) -> dict[str, float | None]:
    """Precision of the flag "P(disrupted or worse) >= threshold" at the strictest
    threshold whose recall still reaches `target_recall`."""
    t_all = np.concatenate([t[1:] >= 2 for t in truths])
    p_all = np.concatenate([h[1:] for h in high])
    best: dict[str, float | None] = {"threshold": None, "precision": None, "recall": None}
    for threshold in np.linspace(0.02, 0.98, 49):
        flag = p_all >= threshold
        recall = float((t_all & flag).sum()) / max(float(t_all.sum()), 1.0)
        if recall >= target_recall:
            best = {
                "threshold": round(float(threshold), 2),
                "precision": round(float((t_all & flag).sum()) / max(float(flag.sum()), 1.0), 3),
                "recall": round(recall, 3),
            }
    return best


def evaluate_observer(
    config: ConfigLike,
    observer: RegimeObserver,
    shifts: int = 60,
    seed: int | None = None,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """Score the observer's beliefs against the true regime on held-out shifts.

    Compares four ways of knowing the regime: the stationary prior (blind), the
    D-state label read as a regime, the pressure filter, and the filter with an
    incident feed. Probabilistic scores (log score, Brier, calibration) are
    pooled over all steps after the first; the flag "DISRUPTED or worse" is
    scored for precision, recall, and how early spells are noticed. Exact
    match is reported but misleading: the stable regime holds about three
    quarters of the time, so always answering "stable" scores high.
    """
    cfg = as_config(config)
    base = int(cfg.simulation.seed if seed is None else seed)
    data = _simulate(cfg, shifts, base + EVAL_OFFSET, n_jobs)
    prior = _prior(cfg)
    truths = [s["regime"] for s in data]
    truth = np.concatenate([t[1:] for t in truths])

    def score(p: np.ndarray) -> dict[str, Any]:
        n = len(truth)
        picked = p.argmax(axis=1)
        high = p[:, 2] + p[:, 3]
        table, ece = _calibration(high, (truth >= 2).astype(float))
        # Unweighted mean gap over well-populated bins: unlike the weighted error, it is
        # not hidden by the many steps where the forecast is near zero.
        gaps = [abs(r["forecast"] - r["observed"]) for r in table if r["steps"] >= 100]
        gap = float(np.mean(gaps)) if gaps else float("nan")
        return {
            "exact": round(float((picked == truth).mean()), 3),
            "within_one": round(float((abs(picked - truth) <= 1).mean()), 3),
            "log_score": round(float(np.log(np.clip(p[np.arange(n), truth], 1e-12, 1)).mean()), 3),
            "brier": round(float(((p - np.eye(4)[truth]) ** 2).sum(axis=1).mean()), 4),
            "calibration_error_high_regime": round(ece, 4),
            "calibration_gap_high_regime": round(gap, 4),
            "calibration_table_high_regime": table,
        }

    result: dict[str, Any] = {
        "shifts": shifts,
        "steps": len(truth),
        "temperature": observer.emission.temperature,
        "prior_only": score(np.tile(prior, (len(truth), 1))),
    }
    probabilities: dict[str, list[np.ndarray]] = {}
    for name, incidents in (("pressure_filter", False), ("incident_filter", True)):
        beliefs = [run_filter(observer, s, incidents) for s in data]
        result[name] = score(np.concatenate([b[1:] for b in beliefs]))
        result[name]["flag_disrupted_or_worse"] = _flag_quality(
            truths, [b[:, 2] + b[:, 3] >= 0.5 for b in beliefs]
        )
        probabilities[name] = [b[:, 2] + b[:, 3] for b in beliefs]

    labels = np.concatenate([s["d_state"][1:] for s in data])
    keep = np.isin(labels, list(_OBSERVED_LEVEL))
    read = np.array([_OBSERVED_LEVEL[x] for x in labels[keep]])
    result["d_state_label"] = {
        "exact": round(float((read == truth[keep]).mean()), 3),
        "within_one": round(float((abs(read - truth[keep]) <= 1).mean()), 3),
        "share_of_steps_labelled": round(float(keep.mean()), 3),
        "flag_disrupted_or_worse": _flag_quality(
            truths, [np.isin(s["d_state"], ("DISRUPTED", "CHAOTIC")) for s in data]
        ),
    }
    label_recall = result["d_state_label"]["flag_disrupted_or_worse"]["recall"]
    for name, high in probabilities.items():
        result[name]["precision_at_label_recall"] = _precision_at_recall(truths, high, label_recall)
    return result

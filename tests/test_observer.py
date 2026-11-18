"""Tests for the regime observer: the emission fit, the filter, and its calibration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ruin import cli
from ruin.config_models import RuinConfig
from ruin.inference import evaluate_observer, fit_regime_observer
from ruin.inference import observer as O
from ruin.risk.destiny import plot_destiny, run_destiny
from ruin.state_space.probability_square import ProbabilitySquare
from ruin.state_space.regime import RegimeProcess


def _with_regime(cfg: RuinConfig, **kw) -> RuinConfig:
    out = cfg.model_copy(deep=True)
    out.probability_square.regime.enabled = True
    for key, value in kw.items():
        setattr(out.probability_square.regime, key, value)
    return out


# A known switching VAR(1) to generate data from: calm, stressed, disrupted, chaotic.
MEANS = np.array([[-2.0, 0.8], [-1.4, 0.65], [-0.9, 0.45], [-0.5, 0.25]])
COEF = np.array([[0.5, 0.0], [0.0, 0.5]])  # same persistence in every regime
SIGMA = np.array([0.20, 0.06])


def _synthetic_emission() -> O.EmissionModel:
    beta = np.zeros((4, 3, 2))
    for k in range(4):
        beta[k, 0] = MEANS[k] * (1 - 0.5)  # intercept so the regime mean is MEANS[k]
        beta[k, 1:] = COEF.T
    return O.EmissionModel(beta, np.array([np.diag(SIGMA**2)] * 4), (1, 1, 1, 1), lags=1)


def _generate(cfg: RuinConfig, shifts: int, steps: int, seed: int) -> list[dict]:
    """Shifts from the true chain (driven by its own pressures) and the known emission."""
    rng = np.random.default_rng(seed)
    regime_cfg = cfg.probability_square.regime
    process = RegimeProcess(regime_cfg, cfg.probability_square.d_state_thresholds)
    out = []
    for _ in range(shifts):
        level = int(rng.choice(4, p=regime_cfg.stationary()))
        chaos, order = 0.0, 0.0
        z = MEANS[level].copy()
        levels, cs, os_, ns = [], [], [], []
        for _ in range(steps):
            level = process.step(level, chaos, order, rng)
            z = MEANS[level] * 0.5 + COEF @ z + rng.normal(0, SIGMA)
            chaos, order = max(math.exp(z[0]) - O.LOG_FLOOR, 0.0), float(np.clip(z[1], -1, 1))
            levels.append(level)
            cs.append(chaos)
            os_.append(order)
            rate = cfg.disruption_field.shock_arrival_rate * regime_cfg.shock_rate[level]
            ns.append(int(rng.poisson(rate)))
        out.append(
            {
                "regime": np.array(levels),
                "chaos": np.array(cs),
                "order": np.array(os_),
                "shocks": np.array(ns),
                "d_state": np.array(["STABLE"] * steps),
            }
        )
    return out


@pytest.fixture(scope="module")
def regime_config(golden_ruin_config) -> RuinConfig:
    return golden_ruin_config


@pytest.fixture(scope="module")
def fitted(regime_config):
    return fit_regime_observer(regime_config, shifts=60, validation_shifts=30, n_jobs=2)


# ---------------------------------------------------------------------------
# The emission fit
# ---------------------------------------------------------------------------


def test_emission_fit_recovers_known_dynamics(regime_config):
    data = _generate(regime_config, shifts=60, steps=500, seed=1)
    fit = O.fit_emission_model(data, lags=1)
    truth = _synthetic_emission()
    np.testing.assert_allclose(fit.beta[:, 0, :], truth.beta[:, 0, :], atol=0.12)
    np.testing.assert_allclose(fit.beta[:, 1:, :], truth.beta[:, 1:, :], atol=0.1)
    np.testing.assert_allclose(
        [np.diag(c) for c in fit.cov], [np.diag(c) for c in truth.cov], rtol=0.35, atol=1e-3
    )
    assert fit.rows == pytest.approx(
        [int((np.concatenate([s["regime"][1:] for s in data]) == k).sum()) for k in range(4)]
    )


def test_emission_likelihood_prefers_the_generating_regime():
    emission = _synthetic_emission()
    rng = np.random.default_rng(2)
    wins = 0
    for k in range(4):
        prev = MEANS[k]
        for _ in range(200):
            z = MEANS[k] * 0.5 + COEF @ prev + rng.normal(0, SIGMA)
            wins += int(emission.log_likelihood(prev[None, :], z).argmax() == k)
    # One reading cannot separate adjacent regimes perfectly; chance would be 0.25.
    assert wins / 800 > 0.6


def test_fit_needs_every_regime_visited(regime_config):
    levels = np.zeros(300, dtype=int)  # only the stable regime ever occurs
    data = [{"regime": levels, "chaos": np.full(300, 0.1), "order": np.full(300, 0.7)}]
    with pytest.raises(ValueError, match="training steps in regime STRESSED"):
        O.fit_emission_model(data, lags=1)


def test_observer_needs_a_regime_scenario(thin_config):
    with pytest.raises(ValueError, match="regime"):
        fit_regime_observer(thin_config, shifts=4)
    with pytest.raises(ValueError, match="regime"):
        O.RegimeFilter(thin_config, _synthetic_emission())


# ---------------------------------------------------------------------------
# The filter
# ---------------------------------------------------------------------------


def test_filter_without_evidence_is_the_forward_chain(regime_config):
    """Temperature 0 ignores the pressure likelihood: only the chain's own pressure-driven
    transitions move the belief, so it must equal a hand-rolled forward pass."""
    emission = _synthetic_emission().with_temperature(0.0)
    f = O.RegimeFilter(regime_config, emission)
    process = f.process
    belief = np.asarray(regime_config.probability_square.regime.stationary())
    prev_c = prev_o = 0.0
    rng = np.random.default_rng(3)
    for _ in range(40):
        c, o = float(rng.uniform(0, 0.8)), float(rng.uniform(0, 1))
        matrix = np.array(
            [
                [
                    {
                        "up": process.hazards(i, prev_c, prev_o)[0],
                        "down": process.hazards(i, prev_c, prev_o)[1],
                    }["up" if j == i + 1 else "down"]
                    if j in (i - 1, i + 1)
                    else 0.0
                    for j in range(4)
                ]
                for i in range(4)
            ]
        )
        for i in range(4):
            matrix[i, i] = 1 - matrix[i].sum()
        belief = belief @ matrix
        got = f.update(c, o)
        np.testing.assert_allclose(got, belief, atol=1e-12)
        prev_c, prev_o = c, o


def test_belief_is_a_probability_distribution(fitted):
    f = fitted.filter()
    rng = np.random.default_rng(4)
    for _ in range(300):
        b = f.update(float(rng.uniform(0, 1.5)), float(rng.uniform(-1, 1)))
        assert b.min() >= 0 and b.sum() == pytest.approx(1.0)
        assert f.posterior.sum() == pytest.approx(1.0)


def test_exact_filter_is_calibrated_on_data_it_generates(regime_config):
    """With the true emission model and no tempering, the Bayes filter is exactly right,
    so its beliefs must be calibrated and beat the prior."""
    emission = _synthetic_emission()
    data = _generate(regime_config, shifts=80, steps=400, seed=5)
    observer = O.RegimeObserver(regime_config, emission)
    beliefs = np.concatenate([O.run_filter(observer, s)[2:] for s in data])
    truth = np.concatenate([s["regime"][2:] for s in data])
    prior = np.asarray(regime_config.probability_square.regime.stationary())
    log_score = np.log(beliefs[np.arange(len(truth)), truth]).mean()
    prior_score = np.log(prior[truth]).mean()
    assert log_score > prior_score + 0.3
    _, ece = O._calibration(beliefs[:, 2] + beliefs[:, 3], (truth >= 2).astype(float))
    assert ece < 0.03
    assert (beliefs.argmax(axis=1) == truth).mean() > (prior.argmax() == truth).mean()


def test_incident_counts_pull_belief_toward_the_regime_that_explains_them(regime_config):
    emission = _synthetic_emission().with_temperature(0.0)
    plain = O.RegimeFilter(regime_config, emission, use_incidents=True)
    quiet = plain.update(0.1, 0.7, shocks=0)
    busy = O.RegimeFilter(regime_config, emission, use_incidents=True).update(0.1, 0.7, shocks=8)
    assert busy[3] > quiet[3] and busy[0] < quiet[0]
    # Without the feed the counts are ignored.
    blind = O.RegimeFilter(regime_config, emission, use_incidents=False).update(0.1, 0.7, shocks=8)
    np.testing.assert_allclose(
        blind, quiet if False else O.RegimeFilter(regime_config, emission).update(0.1, 0.7)
    )


def test_bias_calibration_fixes_a_known_overforecast():
    rng = np.random.default_rng(6)
    n = 40_000
    beliefs = rng.dirichlet([1.0, 1.0, 1.0, 1.0], size=n)
    true_bias = np.array([0.0, 0.2, -0.7, -1.2])
    probs = O._apply_bias(beliefs, true_bias)
    truth = np.array([rng.choice(4, p=p) for p in probs])
    found = O._fit_bias(beliefs, truth)
    np.testing.assert_allclose(found, true_bias, atol=0.15)


def test_posterior_is_the_biased_belief(regime_config):
    f = O.RegimeFilter(regime_config, _synthetic_emission(), bias=np.array([0.0, 0.0, -2.0, -2.0]))
    f.update(0.3, 0.5)
    assert f.posterior[2] < f.belief[2] and f.posterior[3] < f.belief[3]
    assert f.posterior.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Against ground truth
# ---------------------------------------------------------------------------


def test_fitting_chooses_a_temperature_and_biases(fitted):
    assert fitted.emission.temperature in O.TEMPERATURES
    assert fitted.emission.temperature < 1.0  # untempered evidence is overconfident
    assert (
        fitted.bias_pressure[2] < 0 and fitted.bias_pressure[3] < 0
    )  # severe regimes over-weighted
    assert fitted.bias_incident[3] < 0


@pytest.fixture(scope="module")
def report(regime_config, fitted):
    return evaluate_observer(regime_config, fitted, shifts=30, n_jobs=2)


def test_filter_beats_ignoring_the_pressures(report):
    prior, pressure, incident = (
        report["prior_only"],
        report["pressure_filter"],
        report["incident_filter"],
    )
    assert pressure["log_score"] > prior["log_score"] + 0.1
    assert pressure["brier"] < prior["brier"] - 0.04
    assert incident["log_score"] > pressure["log_score"]
    assert incident["brier"] < pressure["brier"]


def test_filter_is_roughly_calibrated(report):
    for name in ("pressure_filter", "incident_filter"):
        assert report[name]["calibration_gap_high_regime"] < 0.1
        assert report[name]["calibration_error_high_regime"] < 0.04


def test_filter_flags_crises_more_precisely_than_the_d_state_label(report):
    label = report["d_state_label"]["flag_disrupted_or_worse"]
    for name in ("pressure_filter", "incident_filter"):
        matched = report[name]["precision_at_label_recall"]
        assert matched["recall"] >= label["recall"] - 1e-9
        assert matched["precision"] > label["precision"]


def test_evaluation_is_reproducible(regime_config, fitted):
    a = evaluate_observer(regime_config, fitted, shifts=4, n_jobs=1)
    b = evaluate_observer(regime_config, fitted, shifts=4, n_jobs=1)
    assert a == b


# ---------------------------------------------------------------------------
# In destiny and on the command line
# ---------------------------------------------------------------------------


def test_redraw_regime(regime_config):
    square = ProbabilitySquare(regime_config, seed=1, store_qdot_snapshots=False)
    square.redraw_regime(np.array([0.0, 0.0, 1.0, 0.0]))
    assert square.regime_level == 2
    square.redraw_regime(np.array([0.0, 0.0, 0.0, 5.0]))  # unnormalised is fine
    assert square.regime_level == 3


def test_redraw_regime_is_a_no_op_without_regimes(small_ruin_config):
    square = ProbabilitySquare(small_ruin_config, seed=1, store_qdot_snapshots=False)
    square.redraw_regime(np.array([0.0, 0.0, 1.0, 0.0]))
    assert square.regime is None


def test_destiny_with_an_observer(regime_config, fitted):
    cfg = regime_config.model_copy(deep=True)
    cfg.simulation.shift_duration = 160
    result = run_destiny(cfg, seed=3, every=80, branches=6, n_jobs=1, blind=True, observer=fitted)
    for row in result["curve"]:
        assert sum(row["regime_posterior"]) == pytest.approx(1.0, abs=1e-3)
        low, high = row["destiny_observer_ci"]
        assert 0 <= low <= row["destiny_observer"] <= high <= 1
        assert "destiny_blind" in row and "destiny" in row
    # Before anything is observed the belief is the prior.
    assert result["curve"][0]["regime_posterior"] == pytest.approx(
        cfg.probability_square.regime.stationary(), abs=1e-3
    )


def test_observer_destiny_follows_a_crisis(regime_config, fitted):
    """Once a crisis has persisted long enough to show in the pressures, the observer's belief
    moves to the severe regimes and its destiny to the informed one, away from the blind one.

    The crisis is made sticky after the observer was fitted, so the observer's own chain is
    slightly wrong here: this checks the behaviour is robust, not that it is exact."""
    cfg = regime_config.model_copy(deep=True)
    cfg.simulation.shift_duration = 160
    cfg.ruin.initial_buffer = 300.0
    cfg.probability_square.regime.start = "chaotic"
    cfg.probability_square.regime.relax = [0.05, 0.04, 0.004]
    result = run_destiny(cfg, seed=11, every=20, branches=16, n_jobs=2, blind=True, observer=fitted)
    curve = result["curve"]
    severe = [row["regime_posterior"][2] + row["regime_posterior"][3] for row in curve]
    assert severe[0] == pytest.approx(0.14, abs=0.01)  # nothing seen yet: the prior
    assert max(severe[2:]) > 0.7
    late = [row for row in curve if row["time"] >= 40]
    assert sum(abs(r["destiny_observer"] - r["destiny"]) for r in late) < sum(
        abs(r["destiny_blind"] - r["destiny"]) for r in late
    )


def test_plot_destiny_with_all_three_views(regime_config, fitted, tmp_path):
    cfg = regime_config.model_copy(deep=True)
    cfg.simulation.shift_duration = 120
    result = run_destiny(cfg, seed=2, every=40, branches=4, n_jobs=1, blind=True, observer=fitted)
    out = plot_destiny(result, tmp_path / "all.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_cli_observer(golden_config_path, caplog):
    caplog.set_level("INFO", logger="ruin.cli")
    rc = cli.main(
        ["observer", "--config", str(golden_config_path), "--fit-shifts", "40",
         "--eval-shifts", "6", "--jobs", "2"]
    )  # fmt: skip
    assert rc == 0
    assert '"pressure_filter"' in caplog.text and '"d_state_label"' in caplog.text


def test_cli_observer_rejects_memoryless_scenarios(thin_config_path, caplog):
    assert cli.main(["observer", "--config", str(thin_config_path)]) == 2
    assert "regime" in caplog.text


def test_cli_destiny_with_observer(golden_config_path, caplog):
    caplog.set_level("INFO", logger="ruin.cli")
    rc = cli.main(
        ["destiny", "--config", str(golden_config_path), "--max-steps", "60", "--every", "30",
         "--branches", "4", "--jobs", "2", "--observer", "--incidents"]
    )  # fmt: skip
    assert rc == 0
    assert '"destiny_observer"' in caplog.text

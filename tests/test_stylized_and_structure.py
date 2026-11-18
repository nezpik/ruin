"""Tests for the stylized-facts suite, the structure ensemble, and blind destiny."""

from __future__ import annotations

import numpy as np
import pytest

from ruin import cli
from ruin.risk.destiny import plot_destiny, run_destiny
from ruin.risk.structure import (
    realised_mean_shock_mass,
    run_structure_ensemble,
    structure_variants,
)
from ruin.risk.stylized import (
    delayed_share_cv,
    escalation_ratio,
    fano_factor,
    run_stylized_facts,
    tail_ratio,
    window_autocorrelation,
)


def _short(config, steps=160):
    cfg = config.model_copy(deep=True)
    cfg.simulation.shift_duration = steps
    return cfg


# ---------------------------------------------------------------------------
# Statistics on known data
# ---------------------------------------------------------------------------


def test_fano_factor_of_poisson_and_clustered_counts():
    rng = np.random.default_rng(0)
    poisson = [{"shocks": rng.poisson(1.0, 480).tolist()} for _ in range(40)]
    assert fano_factor(poisson) == pytest.approx(1.0, abs=0.15)
    # Alternate calm and stormy blocks of 24 steps: strongly over-dispersed.
    blocks = [{"shocks": (rng.poisson(0.2, 24).tolist() + rng.poisson(3.0, 24).tolist()) * 10}]
    assert fano_factor(blocks) > 3


def test_window_autocorrelation():
    rng = np.random.default_rng(1)
    white = [{"shocks": rng.poisson(1.0, 480).tolist()} for _ in range(40)]
    assert abs(window_autocorrelation(white)) < 0.15
    slow = [{"shocks": np.repeat(rng.poisson(2.0, 20), 24).tolist()} for _ in range(10)]
    assert window_autocorrelation(slow) > 0.5


def test_tail_ratio_and_delayed_share():
    values = [1.0] * 98 + [10.0, 20.0]
    assert tail_ratio(values, 99) == pytest.approx(np.percentile(values, 99))
    assert tail_ratio([0.0, 0.0], 95) != tail_ratio([0.0, 0.0], 95)  # nan when the median is 0
    even = [{"slowdown": [1.0] * 90 + [3.0] * 10} for _ in range(20)]
    uneven = [{"slowdown": [1.0] * 100} for _ in range(10)] + [
        {"slowdown": [1.0] * 40 + [3.0] * 60} for _ in range(10)
    ]
    assert delayed_share_cv(even) == pytest.approx(0.0, abs=1e-9)
    assert delayed_share_cv(uneven) > 0.5


def test_escalation_ratio_detects_a_regime_that_follows_pressure(golden_ruin_config):
    rng = np.random.default_rng(2)
    cfg = golden_ruin_config
    floor = cfg.probability_square.d_state_thresholds.chaotic_chaos_pressure

    def shifts(follows):
        out = []
        for _ in range(60):
            chaos = rng.uniform(0, 1.2 * floor, 300)
            regime, level = [], 0
            for t in range(300):
                pointed = sum(chaos[t - 1] >= f for f in (0.25, 0.37, floor)) if t else 0
                p = 0.3 if (follows and pointed > level) else 0.02
                level = (
                    min(3, level + 1)
                    if rng.random() < p
                    else (max(0, level - 1) if rng.random() < 0.1 else level)
                )
                regime.append(("STABLE", "STRESSED", "DISRUPTED", "CHAOTIC")[level])
            out.append({"regime": regime, "chaos": chaos.tolist()})
        return out

    assert escalation_ratio(cfg, shifts(True)) > 3
    assert escalation_ratio(cfg, shifts(False)) == pytest.approx(1.0, abs=0.5)


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_reference_scenario_reproduces_the_stylized_facts(golden_ruin_config):
    report = run_stylized_facts(golden_ruin_config, paths=40, n_jobs=2)
    failed = [f for f in report["facts"] if f["passed"] is False]
    assert not failed, failed
    assert report["passed"] and report["n_passed"] == report["n_applicable"] == 8


@pytest.mark.slow
def test_memoryless_scenario_fails_the_facts_it_should(thin_config):
    """The suite can fail: without a regime there is no clustering or persistence."""
    report = run_stylized_facts(thin_config, paths=40, n_jobs=2)
    by_name = {f["fact"]: f for f in report["facts"]}
    for name in ("clustering", "persistence", "heavy_tailed_shifts", "good_days_and_bad_days"):
        assert by_name[name]["passed"] is False, by_name[name]
    assert by_name["flaw_of_averages"]["measured"]["ratio"] == pytest.approx(1.0, abs=0.35)
    for name in ("partial_observability", "co_evolution"):
        assert by_name[name]["passed"] is None  # needs a hidden regime
    assert report["passed"] is False


def test_cli_facts_exit_code(golden_config_path, caplog):
    caplog.set_level("INFO", logger="ruin.cli")
    rc = cli.main(["facts", "--config", str(golden_config_path), "--paths", "6", "--jobs", "1"])
    assert rc in (0, 1)
    assert '"facts"' in caplog.text


# ---------------------------------------------------------------------------
# Structure ensemble
# ---------------------------------------------------------------------------


def test_structure_variants_are_mean_matched_and_isolate_one_change(golden_ruin_config):
    mass = 123.0
    variants = structure_variants(golden_ruin_config, mass)
    assert list(variants) == ["thin", "heavy_tails", "clustered", "full"]
    thin, heavy, clustered, full = variants.values()
    for v in (thin, heavy):
        assert not v.probability_square.regime.enabled
        assert v.disruption_field.congestion_persistence == 0
        assert (
            v.disruption_field.shock_arrival_rate * v.disruption_field.mean_initial_intensity
            == (pytest.approx(mass))
        )
    assert thin.disruption_field.shock_size_sigma == 0 < heavy.disruption_field.shock_size_sigma
    assert clustered.probability_square.regime.enabled
    assert clustered.disruption_field.shock_size_sigma == 0
    assert full == golden_ruin_config
    assert golden_ruin_config.probability_square.regime.enabled  # the original is untouched


def test_realised_mean_mass_is_matched_by_the_thin_variant(golden_ruin_config):
    cfg = _short(golden_ruin_config, 240)
    mass = realised_mean_shock_mass(cfg, shifts=24, n_jobs=2)
    thin = structure_variants(cfg, mass)["thin"]
    assert realised_mean_shock_mass(thin, shifts=24, n_jobs=2) == pytest.approx(mass, rel=0.2)


def test_structure_ensemble_runs_and_reports_a_range(golden_ruin_config):
    cfg = _short(golden_ruin_config, 120)
    result = run_structure_ensemble(cfg, paths=6, n_jobs=1, shifts=6)
    assert set(result["variants"]) == {"thin", "heavy_tails", "clustered", "full"}
    low, high = result["ruin_probability_range"]
    assert low == min(v["ruin_probability"] for v in result["variants"].values())
    assert high == max(v["ruin_probability"] for v in result["variants"].values())
    assert result["matched_mean_shock_mass_per_step"] > 0


def test_structure_ensemble_needs_a_regime(thin_config):
    with pytest.raises(ValueError, match="regime"):
        run_structure_ensemble(thin_config, paths=2, n_jobs=1)


def test_cli_structure(golden_config_path, thin_config_path, caplog, tmp_path):
    caplog.set_level("INFO", logger="ruin.cli")
    assert cli.main(["structure", "--config", str(thin_config_path), "--paths", "2"]) == 2
    assert "regime" in caplog.text


# ---------------------------------------------------------------------------
# Blind destiny
# ---------------------------------------------------------------------------


def test_destiny_reports_the_regime_and_blind_futures(golden_ruin_config):
    result = run_destiny(
        _short(golden_ruin_config),
        seed=3,
        every=80,
        branches=6,
        n_jobs=1,
        blind=True,
    )
    assert len(result["regime_path"]) <= 160
    for row in result["curve"]:
        assert row["regime"] in ("STABLE", "STRESSED", "DISRUPTED", "CHAOTIC")
        low, high = row["destiny_blind_ci"]
        assert 0 <= low <= row["destiny_blind"] <= high <= 1


def test_blind_destiny_is_left_out_without_regimes(thin_config):
    result = run_destiny(
        _short(thin_config, 100), seed=1, every=50, branches=4, n_jobs=1, blind=True
    )
    assert all("destiny_blind" not in row and row["regime"] is None for row in result["curve"])


def test_blind_and_informed_futures_differ_only_in_the_regime(golden_ruin_config):
    """With the regime already average-like, forgetting it changes nothing systematic;
    from a crisis, informed futures are worse than blind ones."""
    cfg = _short(golden_ruin_config, 200)
    cfg.probability_square.regime.start = "chaotic"
    result = run_destiny(cfg, seed=7, every=200, branches=40, n_jobs=2, blind=True)
    first = result["curve"][0]
    assert first["regime"] == "CHAOTIC"
    assert first["destiny"] <= first["destiny_blind"] + 0.05


def test_plot_destiny_with_regime_strip_and_blind_curve(golden_ruin_config, tmp_path):
    result = run_destiny(
        _short(golden_ruin_config, 120), seed=2, every=40, branches=4, n_jobs=1, blind=True
    )
    out = plot_destiny(result, tmp_path / "destiny.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_cli_destiny_blind(tmp_path, golden_config_path, caplog):
    caplog.set_level("INFO", logger="ruin.cli")
    rc = cli.main(
        [
            "destiny", "--config", str(golden_config_path), "--max-steps", "60",
            "--every", "30", "--branches", "4", "--jobs", "1", "--blind",
        ]
    )  # fmt: skip
    assert rc == 0
    assert '"destiny_blind"' in caplog.text

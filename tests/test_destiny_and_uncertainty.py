"""Tests for remaining destiny and for parameter sensitivity / uncertainty."""

from __future__ import annotations

import pytest

from ruin import cli
from ruin.config_models import RuinConfig
from ruin.risk.destiny import plot_destiny, run_destiny
from ruin.risk.uncertainty import run_parameter_uncertainty, run_sensitivity, scale_parameters
from ruin.simulation.agent_based import run_trajectory


def _harsh(config, scale=2.5, buffer=20.0):
    cfg = config.model_copy(deep=True)
    cfg.disruption_field.shock_arrival_rate *= scale
    cfg.ruin.initial_buffer = buffer
    return cfg


def _calm(config):
    cfg = config.model_copy(deep=True)
    cfg.disruption_field.shock_arrival_rate = 0.0
    return cfg


def test_destiny_follows_the_same_shift_as_run_trajectory(golden_ruin_config):
    cfg = _harsh(golden_ruin_config)
    trajectory = run_trajectory(cfg, seed=5, max_steps=160, store_qdot_snapshots=False)
    destiny = run_destiny(cfg, seed=5, every=40, branches=6, max_steps=160, n_jobs=1)
    assert destiny["ruined"] == trajectory["ruined"]
    assert destiny["ruin_time"] == trajectory["ruin_time"]
    assert destiny["surplus_path"] == pytest.approx(trajectory["surplus_path"], abs=1e-3)


def test_destiny_curve_shape_and_bounds(golden_ruin_config):
    result = run_destiny(golden_ruin_config, seed=42, every=60, branches=8, max_steps=180, n_jobs=1)
    assert [row["time"] for row in result["curve"]] == [0, 60, 120]
    for row in result["curve"]:
        low, high = row["destiny_ci"]
        assert 0.0 <= low <= row["destiny"] <= high <= 1.0
        assert row["expected_future_loss"] >= 0
        assert row["futures_ruined_by_surplus"] + row["futures_ruined_by_sla"] == round(
            (1 - row["destiny"]) * 8
        )
    assert result["end"] == {"time": 180, "destiny": 1.0}
    assert result["initial_destiny"] == result["curve"][0]["destiny"]


def test_a_calm_world_keeps_all_its_destiny(golden_ruin_config):
    result = run_destiny(
        _calm(golden_ruin_config), seed=1, every=50, branches=6, max_steps=150, n_jobs=1
    )
    assert all(row["destiny"] == 1.0 for row in result["curve"])
    # Nothing is ever lost, so no buffer is needed.
    assert all(row["required_buffer"] <= 0 for row in result["curve"])


def test_a_ruined_shift_runs_out_of_destiny(golden_ruin_config):
    cfg = _harsh(golden_ruin_config, scale=4.0, buffer=5.0)
    result = run_destiny(cfg, seed=2, every=20, branches=8, max_steps=300, n_jobs=1)
    assert result["ruined"]
    assert result["end"] == {"time": result["ruin_time"], "destiny": 0.0}
    assert all(row["time"] < result["ruin_time"] for row in result["curve"])
    assert result["curve"][-1]["destiny"] < 0.5
    # The shift needed far more buffer than it had.
    assert result["curve"][0]["required_buffer"] > cfg.ruin.initial_buffer


def test_destiny_is_reproducible_and_independent_of_workers(golden_ruin_config):
    kwargs = {"seed": 9, "every": 60, "branches": 6, "max_steps": 120}
    serial = run_destiny(golden_ruin_config, n_jobs=1, **kwargs)
    parallel = run_destiny(golden_ruin_config, n_jobs=2, **kwargs)
    assert serial == parallel


@pytest.mark.slow
def test_initial_destiny_agrees_with_monte_carlo_ruin_probability(golden_ruin_config):
    """Destiny at t=0 and 1 - ruin probability estimate the same thing."""
    from ruin.risk.ruin_probability import run_monte_carlo, wilson_interval

    cfg = _harsh(golden_ruin_config, scale=1.3, buffer=120.0)
    risk = run_monte_carlo(cfg, paths=120, max_steps=240, ci_boot=10)
    destiny = run_destiny(cfg, seed=42, every=240, branches=120, max_steps=240)
    survival = 1 - risk["ruin_probability"]
    low, high = wilson_interval(round(survival * 120), 120, 0.999)
    assert low - 0.05 <= destiny["initial_destiny"] <= high + 0.05


def test_plot_destiny_writes_a_png(golden_ruin_config, tmp_path):
    cfg = _harsh(golden_ruin_config, scale=4.0, buffer=5.0)
    result = run_destiny(cfg, seed=2, every=40, branches=4, max_steps=200, n_jobs=1)
    out = plot_destiny(result, tmp_path / "destiny.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_scale_parameters_scales_and_revalidates(golden_ruin_config):
    scaled = scale_parameters(
        golden_ruin_config,
        {"disruption_field.shock_arrival_rate": 2.0, "qdots.*.time_window": 0.5},
    )
    base = golden_ruin_config
    assert scaled.disruption_field.shock_arrival_rate == pytest.approx(
        2 * base.disruption_field.shock_arrival_rate
    )
    for name in ("standard", "express", "bulky"):
        assert getattr(scaled.qdots, name).time_window == pytest.approx(
            0.5 * getattr(base.qdots, name).time_window
        )
    assert base.disruption_field.shock_arrival_rate == 0.693  # original untouched
    with pytest.raises(Exception, match=r"never decay|memory_weight|less than or equal"):
        scale_parameters(base, {"disruption_field.memory_weight": 5.0})


def test_sensitivity_finds_the_direction_of_the_big_levers(golden_ruin_config):
    cfg = golden_ruin_config.model_copy(deep=True)
    cfg.simulation.shift_duration = 200
    result = run_sensitivity(
        cfg,
        parameters=["qdots.*.time_window", "disruption_field.mean_initial_intensity"],
        rel_step=0.25,
        paths=24,
        n_jobs=1,
    )
    by_name = {row["parameter"]: row for row in result["parameters"]}
    windows = by_name["qdots.*.time_window"]
    shocks = by_name["disruption_field.mean_initial_intensity"]
    # Tighter windows and bigger shocks both mean more ruin.
    assert windows["ruin_probability_low"] > windows["ruin_probability_high"]
    assert windows["elasticity"] < 0
    assert shocks["ruin_probability_low"] < shocks["ruin_probability_high"]
    assert shocks["elasticity"] > 0
    ranked = [abs(row["elasticity"]) for row in result["parameters"]]
    assert ranked == sorted(ranked, reverse=True)


def test_parameter_uncertainty_reports_a_distribution(golden_ruin_config):
    cfg = golden_ruin_config.model_copy(deep=True)
    cfg.simulation.shift_duration = 150
    result = run_parameter_uncertainty(
        cfg, spreads={"disruption_field.shock_arrival_rate": 0.5}, draws=5, paths=10, n_jobs=1
    )
    q = result["ruin_probability_quantiles"]
    assert 0 <= q["5%"] <= q["25%"] <= q["50%"] <= q["75%"] <= q["95%"] <= 1
    assert q["5%"] <= result["predictive_ruin_probability"] <= q["95%"]
    again = run_parameter_uncertainty(
        cfg, spreads={"disruption_field.shock_arrival_rate": 0.5}, draws=5, paths=10, n_jobs=1
    )
    assert again == result


def test_parameter_uncertainty_needs_parameters(golden_ruin_config):
    cfg = golden_ruin_config.model_copy(deep=True)
    cfg.risk.parameter_uncertainty = {}
    with pytest.raises(ValueError, match="No uncertain parameters"):
        run_parameter_uncertainty(cfg, draws=1, paths=2, n_jobs=1)


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"disruption_field.no_such_thing": 0.1}, "unknown parameter"),
        ({"ruin.rule": 0.1}, "not a continuous parameter"),
        ({"qdots.n_standard": 0.1}, "not a continuous parameter"),
        ({"processes.travel_drift": 0.0}, "relative spread"),
        ({"processes.travel_drift": 3.0}, "relative spread"),
    ],
)
def test_config_rejects_bad_uncertain_parameters(small_config, entry, message):
    small_config["risk"]["parameter_uncertainty"] = entry
    with pytest.raises(Exception, match=message):
        RuinConfig.model_validate(small_config)


def test_cli_destiny_writes_plot(tmp_path, golden_config_path, caplog):
    caplog.set_level("INFO", logger="ruin.cli")
    plot = tmp_path / "destiny.png"
    rc = cli.main(
        [
            "destiny", "--config", str(golden_config_path), "--max-steps", "60",
            "--every", "30", "--branches", "4", "--jobs", "1", "--plot", str(plot),
        ]
    )  # fmt: skip
    assert rc == 0
    assert plot.exists()
    assert '"initial_destiny"' in caplog.text

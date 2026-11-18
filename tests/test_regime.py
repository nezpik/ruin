"""Tests for the hidden regime of D: the chain itself, and what it does to the engine."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ruin.config import load_ruin_config
from ruin.config_models import DStateThresholds, RegimeConfig, RuinConfig
from ruin.core.processes import make_rng, sample_shock_events
from ruin.simulation.agent_based import run_trajectory
from ruin.state_space.disruption_field import DisruptionField
from ruin.state_space.probability_square import ProbabilitySquare
from ruin.state_space.regime import REGIMES, Regime, RegimeProcess

THRESHOLDS = DStateThresholds(
    stressed_chaos_pressure=0.25,
    disrupted_chaos_pressure=0.4,
    chaotic_chaos_pressure=0.6,
    recovering_decay_ratio=0.1,
)


def _process(**kw):
    return RegimeProcess(RegimeConfig(enabled=True, **kw), THRESHOLDS)


def _with_regime(cfg: RuinConfig, **kw) -> RuinConfig:
    out = cfg.model_copy(deep=True)
    out.probability_square.regime.enabled = True
    for key, value in kw.items():
        setattr(out.probability_square.regime, key, value)
    return out


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


def test_stationary_distribution_obeys_detailed_balance():
    cfg = RegimeConfig(escalate=[0.02, 0.01, 0.03], relax=[0.05, 0.04, 0.06])
    pi = cfg.stationary()
    assert sum(pi) == pytest.approx(1.0)
    for k in range(3):
        assert pi[k] * cfg.escalate[k] == pytest.approx(pi[k + 1] * cfg.relax[k])


def test_exogenous_mean_mass():
    cfg = RegimeConfig(shock_rate=[0.0, 1.0, 2.0, 4.0], shock_size=[1.0, 1.0, 2.0, 2.0])
    pi = cfg.stationary()
    assert cfg.exogenous_mean_mass() == pytest.approx(pi[1] + 4 * pi[2] + 8 * pi[3])


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"escalate": [0.1, 0.1]}, "needs 3 values"),
        ({"shock_rate": [1.0, 1.0, 1.0]}, "needs 4 values"),
        ({"shock_size": [1.0, -1.0, 1.0, 1.0]}, "must not be negative"),
        ({"escalate": [0.1, 1.0, 0.1]}, "probabilities in"),
        ({"relax": [0.1, 0.0, 0.1]}, "probabilities in"),
        ({"pressure_pull": 0.5}, "greater than or equal to 1"),
        ({"order_support": 1.0}, "less than 1"),
        ({"start": "calm"}, "Input should be"),
    ],
)
def test_regime_config_rejects_bad_values(kw, message):
    with pytest.raises(Exception, match=message):
        RegimeConfig(**kw)


def test_hazards_are_neutral_inside_the_regimes_own_band():
    proc = _process(pressure_pull=3.0)
    # Pressure 0.3 points to level 1 (stressed), so level 1 is left alone.
    up, down = proc.hazards(1, 0.3, 0.0)
    assert up == pytest.approx(proc.config.escalate[1])
    assert down == pytest.approx(proc.config.relax[0])


def test_hazards_pull_toward_the_level_pressure_points_to():
    proc = _process(pressure_pull=2.0)
    cfg = proc.config
    # Pressure 0.7 points to level 3. From level 1: toward = up, away = down.
    up, down = proc.hazards(1, 0.7, 0.0)
    assert up == pytest.approx(cfg.escalate[1] * 2.0)
    assert down == pytest.approx(cfg.relax[0] / 2.0)
    # Pressure 0.0 points to level 0. From level 2 the pull reverses.
    up, down = proc.hazards(2, 0.0, 0.0)
    assert up == pytest.approx(cfg.escalate[2] / 2.0)
    assert down == pytest.approx(cfg.relax[1] * 2.0)


def test_hazards_respect_the_ends_of_the_chain():
    proc = _process(pressure_pull=2.0)
    assert proc.hazards(0, 0.0, 0.0)[1] == 0.0
    assert proc.hazards(3, 1.0, 0.0)[0] == 0.0


def test_order_pressure_resists_escalation_and_speeds_recovery():
    proc = _process(order_support=0.5)
    cfg = proc.config
    up, down = proc.hazards(1, 0.3, 1.0)
    assert up == pytest.approx(cfg.escalate[1] * 0.5)
    assert down == pytest.approx(cfg.relax[0] * 1.5)
    # Order pressure is clipped to [0, 1].
    assert proc.hazards(1, 0.3, 5.0) == proc.hazards(1, 0.3, 1.0)
    assert proc.hazards(1, 0.3, -2.0) == proc.hazards(1, 0.3, 0.0)


def test_hazards_are_always_valid_probabilities():
    proc = _process(
        escalate=[0.9, 0.9, 0.9], relax=[1.0, 1.0, 1.0], pressure_pull=5.0, order_support=0.9
    )
    for level in range(4):
        for chaos in (0.0, 0.3, 0.5, 1.5):
            for order in (-1.0, 0.0, 1.0):
                up, down = proc.hazards(level, chaos, order)
                assert 0 <= up <= 1 and 0 <= down <= 1 and up + down <= 1 + 1e-12


def test_pressure_levels():
    proc = _process()
    assert [proc.level_from_pressure(p) for p in (0.0, 0.24, 0.25, 0.39, 0.4, 0.59, 0.6, 9.0)] == [
        0, 0, 1, 1, 2, 2, 3, 3,
    ]  # fmt: skip


def test_exogenous_chain_converges_to_its_stationary_distribution():
    proc = _process()  # pull 1, order_support 0: pressure is ignored
    rng = make_rng(1)
    level, counts = 0, np.zeros(4)
    for _ in range(300_000):
        level = proc.step(level, 0.0, 0.0, rng)
        counts[level] += 1
    assert counts / counts.sum() == pytest.approx(proc.config.stationary(), abs=0.01)


def test_sojourn_times_match_theory():
    proc = _process()
    cfg = proc.config
    rng = make_rng(2)
    stays, level, run = [], 1, 0
    for _ in range(400_000):
        nxt = proc.step(level, 0.0, 0.0, rng)
        run += 1
        if nxt != level:
            if level == 1:
                stays.append(run)
            level, run = nxt, 0
    expected = 1.0 / (cfg.escalate[1] + cfg.relax[0])
    assert np.mean(stays) == pytest.approx(expected, rel=0.08)


def test_steady_pressure_drives_the_regime():
    rng = make_rng(3)

    def mean_level(chaos, pull):
        proc = _process(pressure_pull=pull)
        level, levels = 0, []
        for _ in range(60_000):
            level = proc.step(level, chaos, 0.0, rng)
            levels.append(level)
        return float(np.mean(levels[1000:]))

    assert mean_level(1.0, pull=4.0) > 2.5  # sustained chaos pressure: chaotic
    assert mean_level(0.0, pull=4.0) < 0.2  # no pressure: stable
    # Without the pull, pressure is ignored.
    assert mean_level(1.0, pull=1.0) == pytest.approx(mean_level(0.0, pull=1.0), abs=0.15)


def test_initial_level():
    rng = make_rng(4)
    assert _process(start="chaotic").initial_level(rng) == 3
    assert _process(start="stable").initial_level(rng) == 0
    proc = _process(start="stationary")
    draws = np.bincount([proc.initial_level(rng) for _ in range(20_000)], minlength=4) / 20_000
    assert draws == pytest.approx(proc.config.stationary(), abs=0.02)


# ---------------------------------------------------------------------------
# Heavy-tailed shock sizes
# ---------------------------------------------------------------------------


def test_lognormal_shocks_keep_the_mean_and_fatten_the_tail():
    rng = make_rng(5)
    mean, sigma = 100.0, 1.2
    sizes = np.concatenate(
        [sample_shock_events(50.0, mean, 10, 10, rng=rng, size_sigma=sigma)[2] for _ in range(400)]
    )
    expo = np.concatenate([sample_shock_events(50.0, mean, 10, 10, rng=rng)[2] for _ in range(400)])
    assert sizes.mean() == pytest.approx(mean, rel=0.08)
    assert sizes.std() / sizes.mean() == pytest.approx(math.sqrt(math.exp(sigma**2) - 1), rel=0.25)
    assert np.percentile(sizes, 99.9) / sizes.mean() > np.percentile(expo, 99.9) / expo.mean()
    assert sizes.std() / sizes.mean() > expo.std() / expo.mean()


def test_zero_sigma_is_exactly_the_old_exponential_sampler():
    a = sample_shock_events(3.0, 10.0, 8, 8, rng=make_rng(6))
    b = sample_shock_events(3.0, 10.0, 8, 8, rng=make_rng(6), size_sigma=0.0)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


# ---------------------------------------------------------------------------
# Load-dependent persistence: jams linger, but the field cannot amplify itself
# ---------------------------------------------------------------------------


def test_gain_map_sets_per_cell_retention():
    field = DisruptionField(6, 6, 0.9, 1, 0.5, 0.0, 0.0, propagation_share=0.0)
    field.inject(1, 1, 10.0)
    field.inject(4, 4, 10.0)
    gain = np.full((6, 6), 0.5)
    gain[4, 4] = 0.95
    field.step(gain)
    assert field.grid[1, 1] == pytest.approx(5.0)
    assert field.grid[4, 4] == pytest.approx(9.5)


def test_retention_is_mass_weighted():
    field = DisruptionField(4, 4, 0.9, 1, 0.5, 0.0, 0.0)
    assert field.retention() == field.gain
    field.inject(0, 0, 3.0)
    field.inject(3, 3, 1.0)
    gain = np.full((4, 4), 0.5)
    gain[0, 0] = 0.9
    assert field.retention(gain) == pytest.approx((3 * 0.9 + 1 * 0.5) / 4)
    assert DisruptionField(4, 4, 0.9, 1, 0.5, 0.0, 0.0).retention(gain) == pytest.approx(0.9)


def _quiet_square(cfg: RuinConfig, **field_kw) -> ProbabilitySquare:
    cfg = cfg.model_copy(deep=True)
    cfg.disruption_field.shock_arrival_rate = 0.0
    cfg.probability_square.qdot_feedback_weight = 0.0
    for key, value in field_kw.items():
        setattr(cfg.disruption_field, key, value)
    return ProbabilitySquare(cfg, seed=3, store_qdot_snapshots=False)


def test_congestion_makes_disruption_linger(golden_ruin_config):
    def mass_after(congestion):
        square = _quiet_square(golden_ruin_config, congestion_persistence=1.0, max_gain=0.98)
        square.field.grid[10:14, 10:14] = 40.0
        for _ in range(15):
            square.congestion[:] = congestion
            square.step()
        return square.field.total()

    free, jammed = mass_after(0.0), mass_after(1.0)
    assert jammed > 1.5 * free


def test_persistence_off_matches_a_uniform_field(golden_ruin_config):
    square = _quiet_square(golden_ruin_config, congestion_persistence=0.0)
    square.congestion[:] = 1.0
    snap = square.step()
    assert snap["field_retention"] == pytest.approx(golden_ruin_config.disruption_field.gain)


@pytest.mark.parametrize("persistence", [0.0, 0.5, 1.0])
def test_field_can_never_amplify_itself(golden_ruin_config, persistence):
    """Total intensity at most max_gain times last step's, plus what was injected."""
    cfg = golden_ruin_config.model_copy(deep=True)
    cfg.probability_square.qdot_feedback_weight = 0.0
    cfg.disruption_field.congestion_persistence = persistence
    ceiling = max(cfg.disruption_field.gain, cfg.disruption_field.max_gain * (persistence > 0))
    square = ProbabilitySquare(cfg, seed=8, store_qdot_snapshots=False)
    previous = 0.0
    for _ in range(250):
        snap = square.step()
        total = square.field.total()
        assert total <= ceiling * previous + snap["shock_mass"] + 1e-6
        assert snap["field_retention"] <= ceiling + 1e-9
        previous = total


def test_config_rejects_max_gain_below_base_gain(small_config):
    small_config["disruption_field"].update({"congestion_persistence": 0.5, "max_gain": 0.3})
    with pytest.raises(Exception, match="max_gain"):
        RuinConfig.model_validate(small_config)


# ---------------------------------------------------------------------------
# The regime in the engine
# ---------------------------------------------------------------------------

THIN_GOLDEN = json.loads((Path(__file__).parent / "data" / "thin_golden.json").read_text())


@pytest.mark.parametrize("key", sorted(THIN_GOLDEN))
def test_thin_scenario_is_unchanged_by_the_new_mechanisms(key):
    """With regimes, heavy tails, and persistence off, results match the model as
    it was before they existed: the new mechanisms are strictly opt-in."""
    seed, steps, harsh = key.split(",")
    cfg = load_ruin_config(Path(__file__).parents[1] / "examples" / "urban_ruin_shift_thin.yaml")
    cfg.disruption_field.shock_arrival_rate *= float(harsh)
    result = run_trajectory(cfg, seed=int(seed), max_steps=int(steps), store_qdot_snapshots=False)
    expected = THIN_GOLDEN[key]
    for field, value in expected.items():
        if field == "chaos_sum":
            assert sum(result["chaos_pressure_path"]) == pytest.approx(value, abs=1e-5)
        elif isinstance(value, float):
            assert result[field] == pytest.approx(value, abs=1e-5), field
        else:
            assert result[field] == value, field
    assert result["regime_occupancy"] is None
    assert set(result["regime_path"]) == {None}


def test_regime_is_recorded_each_step(golden_ruin_config):
    result = run_trajectory(golden_ruin_config, seed=3, max_steps=200, store_qdot_snapshots=False)
    path = result["regime_path"]
    assert len(path) == len(result["d_state_path"]) == len(result["shock_count_path"])
    assert set(path) <= {r.value for r in REGIMES}
    steps = [
        REGIMES.index(Regime(a)) - REGIMES.index(Regime(b))
        for a, b in zip(path[1:], path, strict=False)
    ]
    assert max(abs(x) for x in steps) <= 1  # the chain moves one level at a time
    occupancy = result["regime_occupancy"]
    assert sum(occupancy.values()) == pytest.approx(1.0, abs=1e-3)
    assert occupancy[path[0]] > 0


def test_regime_runs_are_reproducible(golden_ruin_config):
    a = run_trajectory(golden_ruin_config, seed=4, max_steps=120, store_qdot_snapshots=False)
    b = run_trajectory(golden_ruin_config, seed=4, max_steps=120, store_qdot_snapshots=False)
    assert a == b


def test_shock_rate_follows_the_regime(small_ruin_config):
    """With feedback off, shocks per step in each regime match rate * multiplier."""
    cfg = _with_regime(
        small_ruin_config,
        shock_rate=[0.5, 1.0, 2.0, 4.0],
        escalate=[0.05, 0.05, 0.05],
        relax=[0.1, 0.1, 0.1],
    )
    cfg.disruption_field.shock_arrival_rate = 1.0
    cfg.processes.demand_arrival_rate = 0.0
    square = ProbabilitySquare(cfg, seed=2, store_qdot_snapshots=False)
    by_level = {k: [] for k in range(4)}
    for _ in range(6000):
        snap = square.step()
        by_level[REGIMES.index(Regime(snap["regime"]))].append(snap["shock_count"])
    for level, rate in enumerate((0.5, 1.0, 2.0, 4.0)):
        assert len(by_level[level]) > 100
        assert np.mean(by_level[level]) == pytest.approx(rate, rel=0.2), level


def test_shock_sizes_follow_the_regime(small_ruin_config):
    cfg = _with_regime(
        small_ruin_config,
        shock_size=[1.0, 1.0, 1.0, 5.0],
        escalate=[0.2, 0.2, 0.2],
        relax=[0.05, 0.05, 0.05],
    )
    cfg.disruption_field.shock_arrival_rate = 3.0
    cfg.processes.demand_arrival_rate = 0.0
    square = ProbabilitySquare(cfg, seed=5, store_qdot_snapshots=False)
    sizes = {False: [], True: []}
    for _ in range(3000):
        snap = square.step()
        if snap["shock_count"]:
            sizes[snap["regime"] == "CHAOTIC"].append(snap["shock_mass"] / snap["shock_count"])
    assert np.mean(sizes[True]) == pytest.approx(5 * np.mean(sizes[False]), rel=0.25)


def test_regime_clusters_disruption(golden_ruin_config):
    """Events per window are over-dispersed under the regime and Poisson without it."""
    from ruin.risk.stylized import fano_factor

    def shocks(cfg, seed):
        sq = ProbabilitySquare(cfg, seed=seed, store_qdot_snapshots=False)
        return {"shocks": [sq.step()["shock_count"] for _ in range(480)]}

    thin = golden_ruin_config.model_copy(deep=True)
    thin.probability_square.regime.enabled = False
    clustered = [shocks(golden_ruin_config, s) for s in range(12)]
    plain = [shocks(thin, s) for s in range(12)]
    assert fano_factor(clustered) > 2.0
    assert fano_factor(plain) == pytest.approx(1.0, abs=0.25)


def test_a_worse_regime_means_more_chaos_pressure(golden_ruin_config):
    chaos, level = [], []
    for seed in range(8):
        result = run_trajectory(
            golden_ruin_config,
            seed=seed,
            max_steps=480,
            store_qdot_snapshots=False,
            stop_at_ruin=False,
        )
        chaos += result["chaos_pressure_path"]
        level += [REGIMES.index(Regime(r)) for r in result["regime_path"]]
    assert np.corrcoef(level, chaos)[0, 1] > 0.4


def test_fork_carries_the_regime_and_forgetting_redraws_it(golden_ruin_config):
    square = ProbabilitySquare(golden_ruin_config, seed=2, store_qdot_snapshots=False)
    for _ in range(60):
        square.step()
    twin = square.fork(1)
    assert twin.regime == square.regime and twin.regime_level == square.regime_level
    # Forgetting draws from the long-run mix, on the fork's own random stream.
    levels = []
    for seed in range(400):
        other = square.fork(seed)
        other.forget_regime()
        levels.append(other.regime_level)
        assert square.regime_level == twin.regime_level  # the original is untouched
    mix = np.bincount(levels, minlength=4) / len(levels)
    assert mix == pytest.approx(golden_ruin_config.probability_square.regime.stationary(), abs=0.08)


def test_forgetting_is_a_no_op_without_regimes(small_ruin_config):
    square = ProbabilitySquare(small_ruin_config, seed=1, store_qdot_snapshots=False)
    square.forget_regime()
    assert square.regime is None


def test_monte_carlo_reports_regime_occupancy(golden_ruin_config):
    from ruin.risk.ruin_probability import run_monte_carlo

    result = run_monte_carlo(golden_ruin_config, paths=6, max_steps=120, n_jobs=1, ci_boot=10)
    assert sum(result["mean_regime_occupancy"].values()) == pytest.approx(1.0, abs=1e-3)
    thin = golden_ruin_config.model_copy(deep=True)
    thin.probability_square.regime.enabled = False
    assert (
        run_monte_carlo(thin, paths=3, max_steps=60, n_jobs=1, ci_boot=10)["mean_regime_occupancy"]
        is None
    )

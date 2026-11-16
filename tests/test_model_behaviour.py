"""Behavioural tests: the properties that make ruin a real question.

These check what the model means, not just that it runs.
"""

from __future__ import annotations

import pytest

from ruin.core.network import NetworkSurplus
from ruin.simulation.agent_based import run_trajectory
from ruin.state_space.probability_square import ProbabilitySquare


def _calm(config):
    cfg = config.model_copy(deep=True)
    cfg.disruption_field.shock_arrival_rate = 0.0
    return cfg


def test_same_seed_same_destiny(golden_ruin_config):
    a = run_trajectory(golden_ruin_config, seed=7, max_steps=120)
    b = run_trajectory(golden_ruin_config, seed=7, max_steps=120)
    assert a == b


def test_different_seeds_differ(golden_ruin_config):
    a = run_trajectory(golden_ruin_config, seed=7, max_steps=120, store_qdot_snapshots=False)
    b = run_trajectory(golden_ruin_config, seed=8, max_steps=120, store_qdot_snapshots=False)
    assert a["chaos_pressure_path"] != b["chaos_pressure_path"]


def test_dict_and_typed_config_agree(small_config, small_ruin_config):
    assert run_trajectory(small_config, seed=3) == run_trajectory(small_ruin_config, seed=3)


def test_no_disruption_means_no_ruin(golden_ruin_config):
    result = run_trajectory(_calm(golden_ruin_config), store_qdot_snapshots=False)
    assert result["ruined"] is False
    assert result["cumulative_loss"] == 0.0
    assert result["failed"] == 0
    assert result["on_time_rate"] == 1.0
    assert result["final_surplus"] > golden_ruin_config.ruin.initial_buffer
    assert set(result["d_state_path"]) == {"STABLE"}


def test_qdots_complete_and_population_is_conserved(golden_ruin_config):
    square = ProbabilitySquare(golden_ruin_config, seed=11, store_qdot_snapshots=False)
    for _ in range(200):
        square.step()
    assert square.delivered > 0
    terminated = square.delivered + square.delivered_late + square.failed
    assert square.spawned == terminated + square.active_count


def test_qdots_never_leave_d_and_move_on_both_axes(golden_ruin_config):
    square = ProbabilitySquare(golden_ruin_config, seed=2)
    start = {q.id: q.position for q in square.qdots}
    moved_x = moved_y = False
    for _ in range(15):
        snapshot = square.step()
        for q in snapshot["qdots"]:
            x, y = q["position"]
            assert 0 <= x < square.width and 0 <= y < square.height
            if q["id"] in start:
                moved_x |= x != start[q["id"]][0]
                moved_y |= y != start[q["id"]][1]
    assert moved_x and moved_y


def test_more_disruption_means_more_loss(golden_ruin_config):
    def mean_loss(scale):
        cfg = golden_ruin_config.model_copy(deep=True)
        cfg.disruption_field.shock_arrival_rate *= scale
        cfg.ruin.rule = "surplus"
        cfg.ruin.initial_buffer = 1e9  # never ruin: compare full-horizon losses
        runs = [
            run_trajectory(cfg, seed=s, max_steps=240, store_qdot_snapshots=False) for s in range(6)
        ]
        return sum(r["cumulative_loss"] for r in runs) / len(runs)

    assert mean_loss(0.0) == 0.0
    assert mean_loss(1.0) < mean_loss(3.0)


def test_larger_buffer_never_ruins_sooner(golden_ruin_config):
    def ruin_times(buffer):
        cfg = golden_ruin_config.model_copy(deep=True)
        cfg.ruin.rule = "surplus"
        cfg.ruin.initial_buffer = buffer
        cfg.disruption_field.shock_arrival_rate *= 2
        out = []
        for seed in range(8):
            r = run_trajectory(cfg, seed=seed, store_qdot_snapshots=False)
            out.append(r["ruin_time"] if r["ruined"] else float("inf"))
        return out

    small, large = ruin_times(20.0), ruin_times(200.0)
    assert any(t != float("inf") for t in small)
    assert all(a <= b for a, b in zip(small, large))


@pytest.mark.parametrize(
    "rule, surplus_breach, sla_breach, expected",
    [
        ("surplus_or_sla", True, False, "surplus"),
        ("surplus_or_sla", False, True, "sla"),
        ("surplus", False, True, None),
        ("surplus", True, True, "surplus"),
        ("sla", True, False, None),
        ("sla", True, True, "sla"),
    ],
)
def test_ruin_rule_is_honoured(rule, surplus_breach, sla_breach, expected):
    network = NetworkSurplus(initial_buffer=10.0, barrier=0.0, sla_threshold=0.5, rule=rule)
    network.update(
        time=1,
        penalty=20.0 if surplus_breach else 1.0,
        late_count=9 if sla_breach else 0,
        active_count=10,
    )
    assert network.ruin_reason == expected
    assert network.ruined is (expected is not None)


def test_surplus_accounts_for_income_and_recovery():
    network = NetworkSurplus(initial_buffer=10.0, barrier=0.0, sla_threshold=1.0, recovery_rate=0.5)
    network.update(time=1, penalty=3.0, late_count=0, active_count=5, income=2.0)
    assert network.surplus == pytest.approx(10.0 + 0.5 + 2.0 - 3.0)
    network.update(time=2, penalty=0.0, late_count=0, active_count=5, income=1.0)
    assert network.surplus_path == pytest.approx([9.5, 11.0])


def test_ruined_trajectory_ends_in_ruined_state(golden_ruin_config):
    cfg = golden_ruin_config.model_copy(deep=True)
    cfg.ruin.initial_buffer = 1.0
    cfg.disruption_field.shock_arrival_rate *= 3
    result = run_trajectory(cfg, seed=1, store_qdot_snapshots=False)
    assert result["ruined"] is True
    assert result["final_d_state"] == "RUINED"
    assert result["d_state_path"][-1] == "RUINED"
    assert len(result["surplus_path"]) == result["ruin_time"]

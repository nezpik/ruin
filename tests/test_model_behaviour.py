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
    assert all(a <= b for a, b in zip(small, large, strict=True))


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
    network = NetworkSurplus(
        initial_buffer=10.0, barrier=0.0, sla_threshold=0.5, rule=rule, sla_window=1
    )
    network.update(
        time=1,
        penalty=20.0 if surplus_breach else 1.0,
        late_count=0,
        active_count=10,
        missed=9 if sla_breach else 0,
        on_time=1 if sla_breach else 10,
    )
    assert network.ruin_reason == expected
    assert network.ruined is (expected is not None)


def test_sla_is_a_rolling_deadline_miss_rate():
    network = NetworkSurplus(
        initial_buffer=1e9, barrier=0.0, sla_threshold=0.2, rule="sla", sla_window=3
    )

    def step(t, missed, on_time):
        network.update(t, 0.0, late_count=0, active_count=10, missed=missed, on_time=on_time)

    # A terrible first step is not judged: a full window has not been seen yet.
    step(1, 5, 5)
    step(2, 0, 20)
    assert network.sla_miss_rate_path == pytest.approx([0.5, 5 / 30])
    assert not network.ruined
    # Window is now full: 5 missed of 50 judged = 10%, under the 20% threshold.
    step(3, 0, 20)
    assert network.sla_miss_rate_path[-1] == pytest.approx(0.1)
    assert not network.ruined
    # The bad first step rolls out of the window.
    step(4, 0, 20)
    assert network.sla_miss_rate_path[-1] == 0.0
    # 12 missed against 48 on time over the last three steps = 20%: breach.
    step(5, 12, 8)
    assert network.sla_miss_rate_path[-1] == pytest.approx(12 / 60)
    assert network.ruined and network.ruin_reason == "sla" and network.ruin_time == 5


def test_in_flight_lateness_alone_does_not_breach_the_sla():
    network = NetworkSurplus(
        initial_buffer=1e9, barrier=0.0, sla_threshold=0.05, rule="sla", sla_window=1
    )
    network.update(1, 0.0, late_count=9, active_count=10, missed=0, on_time=10)
    assert network.late_ratio_path == [0.9]
    assert not network.ruined


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


def test_running_past_ruin_keeps_the_ruin_time_and_the_full_loss(golden_ruin_config):
    cfg = golden_ruin_config.model_copy(deep=True)
    cfg.ruin.initial_buffer = 1.0
    cfg.disruption_field.shock_arrival_rate *= 3
    stopped = run_trajectory(cfg, seed=1, max_steps=300, store_qdot_snapshots=False)
    full = run_trajectory(
        cfg, seed=1, max_steps=300, store_qdot_snapshots=False, stop_at_ruin=False
    )
    assert stopped["ruined"] and full["ruined"]
    assert full["ruin_time"] == stopped["ruin_time"] < 300
    assert len(full["surplus_path"]) == 300
    assert full["cumulative_loss"] >= stopped["cumulative_loss"]
    assert full["surplus_path"][: stopped["ruin_time"]] == stopped["surplus_path"]
    assert set(full["d_state_path"][stopped["ruin_time"] - 1 :]) == {"RUINED"}
    assert full["min_surplus"] == pytest.approx(min(full["surplus_path"]))


def test_loss_by_type_adds_up(golden_ruin_config):
    cfg = golden_ruin_config.model_copy(deep=True)
    cfg.disruption_field.shock_arrival_rate *= 3
    result = run_trajectory(cfg, seed=4, max_steps=200, store_qdot_snapshots=False)
    assert set(result["loss_by_type"]) == {"STANDARD", "EXPRESS", "BULKY"}
    assert result["cumulative_loss"] > 0
    assert sum(result["loss_by_type"].values()) == pytest.approx(
        result["cumulative_loss"], abs=1e-3
    )


def test_missed_deadlines_are_counted_once(golden_ruin_config):
    cfg = golden_ruin_config.model_copy(deep=True)
    cfg.disruption_field.shock_arrival_rate *= 2
    square = ProbabilitySquare(cfg, seed=6, store_qdot_snapshots=False)
    missed = sum(square.step()["missed"] for _ in range(300))
    assert missed > 0
    in_flight_late = sum(1 for q in square.qdots if q.late)
    assert missed == square.delivered_late + square.failed + in_flight_late


def test_loss_ledger_accounts_for_every_qdot_and_all_loss(golden_ruin_config):
    cfg = golden_ruin_config.model_copy(deep=True)
    cfg.disruption_field.shock_arrival_rate *= 2
    result = run_trajectory(cfg, seed=4, max_steps=250, store_qdot_snapshots=False)
    square = ProbabilitySquare(cfg, seed=4, store_qdot_snapshots=False)
    penalties = sum(square.step()["penalty"] for _ in range(len(result["surplus_path"])))
    ledger = square.ledger()

    assert len(ledger) == square.spawned == len({row["id"] for row in ledger})
    assert sum(row["loss"] for row in ledger) == pytest.approx(penalties, rel=1e-6)
    assert square.loss_field.sum() == pytest.approx(penalties, rel=1e-6)
    # Only Q dots that missed their deadline carry loss.
    assert all(row["loss"] == 0 for row in ledger if row["status"] in ("ACTIVE", "DELIVERED"))
    assert all(row["end_time"] is None for row in ledger if row["status"] in ("ACTIVE", "LATE"))

    top = result["top_contributors"]
    assert top and top[0]["loss"] == max(row["loss"] for row in ledger)
    assert [r["loss"] for r in top] == sorted((r["loss"] for r in top), reverse=True)
    assert 0 < sum(r["share_of_loss"] for r in top) <= 1.0 + 1e-6
    concentration = result["loss_concentration"]
    assert concentration["qdots"] == square.spawned
    assert 0 < concentration["qdots_with_loss"] < concentration["qdots"]
    assert 0 < concentration["top_1pct_share"] <= 1


def test_fork_shares_the_past_and_not_the_future(golden_ruin_config):
    square = ProbabilitySquare(golden_ruin_config, seed=3, store_qdot_snapshots=False)
    for _ in range(40):
        square.step()
    field_before = square.field.grid.copy()
    positions_before = [q.position for q in square.qdots]

    twin_a, twin_b, other = square.fork(7), square.fork(7), square.fork(8)
    assert twin_a.time == square.time and twin_a.spawned == square.spawned
    for _ in range(30):
        twin_a.step()
        twin_b.step()
        other.step()

    # Stepping the forks left the original untouched.
    assert square.time == 40
    assert (square.field.grid == field_before).all()
    assert [q.position for q in square.qdots] == positions_before
    # Same seed, same future; different seed, different future.
    assert twin_a.snapshots == twin_b.snapshots
    assert [s["chaos_pressure"] for s in twin_a.snapshots] != [
        s["chaos_pressure"] for s in other.snapshots
    ]

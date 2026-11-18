"""Tests for chaotic organization: jams, pile-up, diversion, corridors, zones, attribution."""

from __future__ import annotations

import numpy as np
import pytest

from ruin import cli
from ruin.analysis import (
    StepFields,
    StructureRecorder,
    aggregate,
    label_components,
    plot_organization,
    run_organization,
)
from ruin.analysis.organization import (
    CORRIDOR_EXCESS,
    JAM_THRESHOLD,
    MIN_BASELINE_STEPS,
    dilate,
)
from ruin.core.processes import batch_qdot_travel_step, make_rng
from ruin.simulation.agent_based import run_trajectory
from ruin.state_space.probability_square import ProbabilitySquare

SHAPE = (10, 10)
HOT = JAM_THRESHOLD * 2  # comfortably above the jam threshold


def fields(
    t, intensity=None, density=None, flow=None, chaos=None, accrued=None, loss=None, regime=None
):
    zero = np.zeros(SHAPE)
    return StepFields(
        time=t,
        regime=regime,
        intensity=zero if intensity is None else intensity,
        density=np.ones(SHAPE) if density is None else density,
        flow=np.ones(SHAPE) if flow is None else flow,
        chaos=zero if chaos is None else chaos,
        accrued=zero if accrued is None else accrued,
        loss=zero if loss is None else loss,
    )


def blob(rows, cols, value=HOT):
    grid = np.zeros(SHAPE)
    grid[rows, cols] = value
    return grid


# ---------------------------------------------------------------------------
# Components and dilation
# ---------------------------------------------------------------------------


def test_label_components_is_four_connected():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[1:3, 1:3] = True  # a 2x2 block
    mask[3, 3] = True  # touches the block only diagonally
    mask[7, 5:8] = True
    labels, count = label_components(mask)
    assert count == 3
    assert len({labels[1, 1], labels[3, 3], labels[7, 6]}) == 3
    assert (labels > 0).sum() == mask.sum()
    assert label_components(np.zeros(SHAPE, dtype=bool))[1] == 0
    assert label_components(np.ones(SHAPE, dtype=bool))[1] == 1


def test_dilate_grows_in_manhattan_distance_and_clips_at_the_edge():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[5, 5] = True
    assert dilate(mask, 0).sum() == 1
    assert dilate(mask, 1).sum() == 5
    assert dilate(mask, 2).sum() == 13  # a diamond
    corner = np.zeros(SHAPE, dtype=bool)
    corner[0, 0] = True
    assert dilate(corner, 2).sum() == 6
    assert (dilate(mask, 2) >= mask).all()


# ---------------------------------------------------------------------------
# Tracking jams
# ---------------------------------------------------------------------------


def test_a_growing_jam_keeps_its_identity_and_records_its_peak():
    rec = StructureRecorder()
    for t, width in enumerate((2, 3, 4, 5, 4, 3), start=1):
        rec.observe(fields(t, intensity=blob(slice(4, 6), slice(2, 2 + width))))
    assert len(rec.jams) == 1
    jam = rec.jams[1]
    assert jam.born == 1 and jam.last_seen == 6 and jam.lifetime == 6
    assert jam.peak_area == 10 and jam.peak_time == 4
    assert jam.peak_cells is not None and jam.peak_cells.sum() == 10
    assert jam.alive


def test_a_drifting_jam_keeps_its_identity_but_a_jump_makes_a_new_one():
    rec = StructureRecorder()
    for t in range(1, 5):  # slides one column per step: consecutive masks overlap
        rec.observe(fields(t, intensity=blob(slice(2, 4), slice(t, t + 3))))
    assert len(rec.jams) == 1
    rec.observe(fields(5, intensity=blob(slice(7, 9), slice(0, 3))))  # nothing overlaps
    assert len(rec.jams) == 2
    assert not rec.jams[1].alive and rec.jams[2].alive and rec.jams[2].parent is None


def test_specks_are_not_jams():
    rec = StructureRecorder()
    rec.observe(fields(1, intensity=blob(slice(2, 3), slice(2, 4))))  # two cells
    assert rec.jams == {}


def test_when_two_jams_merge_the_larger_keeps_the_identity():
    rec = StructureRecorder()
    big, small = blob(slice(1, 4), slice(0, 4)), blob(slice(1, 4), slice(6, 8))
    rec.observe(fields(1, intensity=big + small))
    assert len(rec.jams) == 2
    merged = blob(slice(1, 4), slice(0, 8))
    rec.observe(fields(2, intensity=merged))
    alive = [j for j in rec.jams.values() if j.alive]
    assert [j.id for j in alive] == [1]  # the larger one (born first, 12 cells) continues
    assert not rec.jams[2].alive


def test_when_a_jam_splits_the_parts_inherit_and_are_born():
    rec = StructureRecorder()
    rec.observe(fields(1, intensity=blob(slice(1, 4), slice(0, 8))))
    split = blob(slice(1, 4), slice(0, 3)) + blob(slice(1, 4), slice(5, 8))
    rec.observe(fields(2, intensity=split))
    assert len(rec.jams) == 2
    child = rec.jams[2]
    assert child.parent == 1 and child.born == 2
    assert rec.summary()["jams"]["splits"] == 1


def test_jams_are_born_in_a_regime_and_record_loss_and_traffic():
    rec = StructureRecorder()
    jam = blob(slice(3, 6), slice(3, 6))
    for t in range(1, 4):
        rec.observe(
            fields(
                t,
                intensity=jam,
                density=np.full(SHAPE, 2.0),
                accrued=blob(slice(3, 6), slice(3, 6), 0.5),
                regime="CHAOTIC",
            )
        )
    j = rec.jams[1]
    assert j.born_regime == "CHAOTIC"
    assert j.dot_steps == pytest.approx(3 * 9 * 2.0)
    assert j.loss == pytest.approx(3 * 9 * 0.5)
    assert rec.summary()["jams"]["born_by_regime"] == {"CHAOTIC": 1}


# ---------------------------------------------------------------------------
# Pile-up, diversion, corridors
# ---------------------------------------------------------------------------


def _jam_on(t):
    """A jam that holds for 20 steps, then leaves 40 quiet ones: long enough for baselines."""
    return t % 60 < 20


def _alternating(rec, steps, jam_cells, density_in_jam=1.0, ring_flow=1.0, ring_cells=None, seed=0):
    """A jam over `jam_cells` comes and goes; density and flow follow the arguments."""
    rng = np.random.default_rng(seed)
    jam_mask = np.zeros(SHAPE, dtype=bool)
    jam_mask[jam_cells] = True
    ring_mask = dilate(jam_mask, 2) & ~jam_mask if ring_cells is None else ring_cells
    for t in range(1, steps + 1):
        on = _jam_on(t)
        density = rng.poisson(1.0, SHAPE).astype(float)
        flow = rng.poisson(1.0, SHAPE).astype(float)
        if on:
            density[jam_mask] = rng.poisson(density_in_jam, jam_mask.sum())
            flow[ring_mask] = rng.poisson(ring_flow, ring_mask.sum())
        rec.observe(
            fields(t, intensity=np.where(jam_mask & on, HOT, 0.0), density=density, flow=flow)
        )


def test_pile_up_is_the_density_inside_jams_over_the_cells_own_baseline():
    trapped, passing = StructureRecorder(), StructureRecorder()
    _alternating(trapped, 600, (slice(3, 6), slice(3, 6)), density_in_jam=3.0)
    _alternating(passing, 600, (slice(3, 6), slice(3, 6)), density_in_jam=1.0)
    assert trapped.summary()["pile_up"] == pytest.approx(3.0, rel=0.12)
    assert passing.summary()["pile_up"] == pytest.approx(1.0, rel=0.12)


def test_diversion_is_flow_beside_jams_not_density():
    funnelled, plain = StructureRecorder(), StructureRecorder()
    jam = (slice(3, 6), slice(3, 6))
    _alternating(funnelled, 600, jam, ring_flow=2.0)
    _alternating(plain, 600, jam, ring_flow=1.0)
    assert funnelled.summary()["diversion"] == pytest.approx(2.0, rel=0.12)
    assert plain.summary()["diversion"] == pytest.approx(1.0, rel=0.12)


def test_queueing_beside_a_jam_is_not_diversion():
    """Density beside a jam can rise because Q dots queue; flow does not, so no diversion."""
    rec = StructureRecorder()
    jam_mask = np.zeros(SHAPE, dtype=bool)
    jam_mask[3:6, 3:6] = True
    ring = dilate(jam_mask, 2) & ~jam_mask
    rng = np.random.default_rng(1)
    for t in range(1, 601):
        on = _jam_on(t)
        density = rng.poisson(1.0, SHAPE).astype(float)
        if on:
            density[ring] = rng.poisson(3.0, ring.sum())  # a queue
        rec.observe(
            fields(t, intensity=np.where(jam_mask & on, HOT, 0.0), density=density,
                   flow=rng.poisson(1.0, SHAPE).astype(float))
        )  # fmt: skip
    assert rec.summary()["diversion"] == pytest.approx(1.0, rel=0.12)


def test_a_route_beside_a_jam_is_a_corridor_and_noise_is_not():
    jam = (slice(3, 6), slice(3, 6))
    strip = np.zeros(SHAPE, dtype=bool)
    strip[2, 1:8] = True  # a seven-cell route along the jam's upper side
    routed, noise = StructureRecorder(), StructureRecorder()
    _alternating(routed, 800, jam, ring_flow=3.0, ring_cells=strip)
    _alternating(noise, 800, jam, ring_flow=1.0)
    found = routed.summary()["corridors"]
    assert found["corridors"] >= 1 and found["largest_corridor_cells"] >= 3
    assert found["mean_excess"] >= CORRIDOR_EXCESS
    assert noise.summary()["corridors"]["corridors"] == 0


def test_cells_without_enough_history_get_no_baseline():
    rec = StructureRecorder()
    for t in range(1, MIN_BASELINE_STEPS // 2):  # a jam always present: no far-from-jam steps
        rec.observe(fields(t, intensity=blob(slice(3, 6), slice(3, 6))))
    assert rec.summary()["pile_up"] is None


# ---------------------------------------------------------------------------
# Pressure, recovery, loss, ruin
# ---------------------------------------------------------------------------


def test_pressure_zones_concentrate_and_persist_only_if_they_stay_put():
    rng = np.random.default_rng(2)
    fixed, moving = StructureRecorder(), StructureRecorder()
    fixed.set_horizon(200)
    moving.set_horizon(200)
    for t in range(1, 201):
        hot = np.zeros(SHAPE)
        hot[2:4, 2:4] = 1.0
        fixed.observe(fields(t, chaos=0.1 + hot + rng.uniform(0, 0.05, SHAPE)))
        # A hotspot placed uniformly at random on a torus: every cell is equally likely.
        wander = np.roll(
            hot, shift=(int(rng.integers(0, 10)), int(rng.integers(0, 10))), axis=(0, 1)
        )
        moving.observe(fields(t, chaos=0.1 + wander + rng.uniform(0, 0.05, SHAPE)))
    a, b = fixed.summary()["pressure"], moving.summary()["pressure"]
    assert a["map_correlation_between_halves"] > 0.9
    assert abs(b["map_correlation_between_halves"]) < 0.5
    assert a["top_decile_share_of_pressure"] > 0.3
    # The top decile is ten cells: the four hot ones plus a few noise cells beside them.
    assert a["zones"][0]["centroid"] == pytest.approx((2.5, 2.5), abs=1.0)


def test_recovery_measures_clearing_time_and_flow_back_in():
    rec = StructureRecorder()
    jam = blob(slice(3, 6), slice(3, 6))
    mass = np.zeros(SHAPE)
    for cycle in range(40):
        t0 = cycle * 30
        for k in range(1, 6):  # jam builds and holds for five steps
            rec.observe(fields(t0 + k, intensity=jam * (1 + 0.1 * k)))
        for k in range(6, 31):  # then clears; Q dots flood back into the cleared cells for a while
            density = np.ones(SHAPE)
            if k <= 12:
                density[3:6, 3:6] = 2.0
            rec.observe(fields(t0 + k, intensity=mass, density=density))
    r = rec.summary()["recovery"]
    assert r["jams_cleared"] >= 39
    assert r["median_steps_peak_to_clear"] == 0.0  # peak mass is the last step before it clears
    # Eight recovering steps, seven of them at double density: (7 * 2 + 1) / 8.
    assert r["flow_into_recovering_cells"] == pytest.approx(15 / 8, rel=0.05)


def test_loss_concentration_and_enrichment_inside_jams():
    rec = StructureRecorder()
    jam = blob(slice(3, 6), slice(3, 6))
    total = np.zeros(SHAPE)
    for t in range(1, 101):
        on = t <= 25
        accrued = blob(slice(3, 6), slice(3, 6), 1.0) if on else np.zeros(SHAPE)
        total = total + accrued
        rec.observe(
            fields(t, intensity=jam if on else np.zeros(SHAPE), accrued=accrued, loss=total)
        )
    s = rec.summary()["loss"]
    assert s["loss_share_inside_jams"] == pytest.approx(1.0)
    assert s["cells_carrying_half_the_loss"] == 5  # nine equal cells: five make over half
    assert s["jam_share_of_cell_steps"] == pytest.approx(9 * 25 / (100 * 100), rel=0.01)
    assert s["loss_enrichment_inside_jams"] == pytest.approx(
        1.0 / s["jam_share_of_cell_steps"], rel=0.01
    )


def test_ruin_zone_names_the_jam_that_carried_the_loss():
    rec = StructureRecorder()
    big, small = blob(slice(1, 5), slice(1, 5)), blob(slice(7, 9), slice(7, 10))
    accrued_big, accrued_small = (
        blob(slice(1, 5), slice(1, 5), 3.0),
        blob(slice(7, 9), slice(7, 10), 1.0),
    )
    stray = np.zeros(SHAPE)
    stray[9, 0] = 4.0
    for t in range(1, 41):
        rec.observe(
            fields(
                t,
                intensity=big + small,
                accrued=accrued_big + accrued_small + stray,
                regime="DISRUPTED",
            )
        )
    rec.mark_ruin(40)
    for t in range(41, 60):  # the shift runs on; the recorded ruin window must not change
        rec.observe(fields(t, intensity=np.zeros(SHAPE), accrued=np.zeros(SHAPE)))
    zone = rec.summary()["ruin_zone"]
    assert zone["ruin_time"] == 40 and zone["window_steps"] == 24
    top = zone["top_jams"][0]
    assert top["peak_area"] == 16 and top["born_regime"] == "DISRUPTED"
    total = 24 * (16 * 3.0 + 6 * 1.0 + 4.0)
    assert top["share"] == pytest.approx(24 * 16 * 3.0 / total, abs=0.001)
    assert zone["share_outside_any_jam"] == pytest.approx(24 * 4.0 / total, abs=0.001)
    assert StructureRecorder().mark_ruin(1) is None  # harmless before any step


def test_summary_needs_at_least_one_step():
    with pytest.raises(ValueError, match="not observed"):
        StructureRecorder().summary()


# ---------------------------------------------------------------------------
# In the engine
# ---------------------------------------------------------------------------


def test_recording_does_not_change_the_simulation(golden_ruin_config):
    plain = run_trajectory(golden_ruin_config, seed=9, max_steps=150, store_qdot_snapshots=False)
    watched = run_trajectory(
        golden_ruin_config, seed=9, max_steps=150, store_qdot_snapshots=False,
        recorder=StructureRecorder(), attribute_sources=True,
    )  # fmt: skip
    for key in (
        "cumulative_loss",
        "final_surplus",
        "delivered",
        "failed",
        "ruin_time",
        "chaos_pressure_path",
    ):
        assert plain[key] == watched[key], key
    assert plain["structure"] is None and watched["structure"] is not None


def test_every_unit_of_loss_is_accounted_for_inside_outside_or_beside_jams(golden_ruin_config):
    rec = StructureRecorder()
    result = run_trajectory(
        golden_ruin_config,
        seed=81,
        max_steps=480,
        store_qdot_snapshots=False,
        stop_at_ruin=False,
        recorder=rec,
    )
    assert sum(rec._loss_in.values()) == pytest.approx(result["cumulative_loss"], rel=1e-6)
    assert rec._final.loss.sum() == pytest.approx(result["cumulative_loss"], rel=1e-6)
    summary = result["structure"]
    assert summary["jams"]["count"] > 0 and summary["jams"]["median_lifetime"] >= 1
    assert summary["loss"]["loss_enrichment_inside_jams"] > 1.5  # loss piles up in jams
    assert summary["pile_up"] > 1.2  # and so do Q dots
    assert summary["ruin_zone"] is not None and summary["ruin_zone"]["top_jams"]


def test_ruin_is_marked_when_it_happens_even_if_the_shift_runs_on(golden_ruin_config):
    rec = StructureRecorder()
    result = run_trajectory(
        golden_ruin_config,
        seed=81,
        max_steps=480,
        store_qdot_snapshots=False,
        stop_at_ruin=False,
        recorder=rec,
    )
    assert result["ruined"] and rec.ruin_time == result["ruin_time"] < 480


def test_avoidance_steers_dots_toward_the_clear_axis():
    n = 6000
    x = y = np.full(n, 5)
    dest = np.full(n, 15)
    blocked_x = np.full(n, 10.0)  # next cell along x is fully blocked
    clear_y = np.zeros(n)

    def share_going_x(avoidance):
        nx = batch_qdot_travel_step(
            x, y, dest, dest, np.zeros(n), np.zeros(n), np.ones(n), np.ones(n),
            1.0, 0.0, 0.0, 0.0, make_rng(3), avoidance=avoidance, ahead_x=blocked_x, ahead_y=clear_y,
        )[0]  # fmt: skip
        return float((nx > 5).mean())

    assert share_going_x(0.0) == pytest.approx(0.5, abs=0.03)
    assert share_going_x(1.0) < 0.3
    assert share_going_x(8.0) < 0.01


def test_avoidance_cannot_help_when_every_way_is_blocked_or_only_one_remains():
    n = 4000
    x, y = np.full(n, 5), np.full(n, 5)

    def moved_x(dest_x, dest_y, ahead_x, ahead_y):
        nx, *_ = batch_qdot_travel_step(
            x, y, np.full(n, dest_x), np.full(n, dest_y), np.zeros(n), np.zeros(n), np.ones(n),
            np.ones(n), 1.0, 0.0, 0.0, 0.0, make_rng(4), avoidance=6.0,
            ahead_x=np.full(n, ahead_x), ahead_y=np.full(n, ahead_y),
        )  # fmt: skip
        return float((nx > 5).mean())

    assert moved_x(15, 15, 10.0, 10.0) == pytest.approx(
        0.5, abs=0.04
    )  # both blocked: no preference
    assert moved_x(15, 5, 10.0, 0.0) == 1.0  # only x remains, blocked or not


def test_avoidance_zero_is_the_old_model(thin_config):
    """avoidance 0 must be bit-for-bit what the engine did before it existed."""
    explicit = thin_config.model_copy(deep=True)
    explicit.processes.avoidance = 0.0
    a = run_trajectory(thin_config, seed=3, max_steps=100, store_qdot_snapshots=False)
    b = run_trajectory(explicit, seed=3, max_steps=100, store_qdot_snapshots=False)
    assert a["chaos_pressure_path"] == b["chaos_pressure_path"]


@pytest.mark.slow
def test_steering_reduces_pile_up_in_jams(golden_ruin_config):
    def pile_up(avoidance):
        cfg = golden_ruin_config.model_copy(deep=True)
        cfg.processes.avoidance = avoidance
        values = []
        for seed in range(10):
            rec = StructureRecorder()
            run_trajectory(
                cfg,
                seed=seed,
                max_steps=300,
                store_qdot_snapshots=False,
                stop_at_ruin=False,
                recorder=rec,
            )
            values.append(rec.summary()["pile_up"])
        return float(np.mean(values))

    assert pile_up(8.0) < pile_up(0.0) - 0.2


def test_loss_by_regime_adds_up_and_is_none_without_regimes(golden_ruin_config, thin_config):
    result = run_trajectory(
        golden_ruin_config, seed=81, max_steps=480, store_qdot_snapshots=False, stop_at_ruin=False
    )
    assert sum(result["loss_by_regime"].values()) == pytest.approx(
        result["cumulative_loss"], rel=1e-5
    )
    assert set(result["loss_by_regime"]) == {"STABLE", "STRESSED", "DISRUPTED", "CHAOTIC"}
    assert (
        run_trajectory(thin_config, seed=1, max_steps=30, store_qdot_snapshots=False)[
            "loss_by_regime"
        ]
        is None
    )


# ---------------------------------------------------------------------------
# Attribution of delay by origin
# ---------------------------------------------------------------------------


def test_origins_always_add_up_to_the_field(golden_ruin_config):
    square = ProbabilitySquare(
        golden_ruin_config, seed=5, store_qdot_snapshots=False, attribute_sources=True
    )
    for _ in range(200):
        square.step()
        total = sum(source.grid for source in square.sources)
        np.testing.assert_allclose(total, square.field.grid, atol=1e-8)
    assert square.source_names == ["STABLE", "STRESSED", "DISRUPTED", "CHAOTIC", "FEEDBACK"]


def test_without_regimes_there_is_one_exogenous_origin(thin_config):
    square = ProbabilitySquare(
        thin_config, seed=1, store_qdot_snapshots=False, attribute_sources=True
    )
    assert square.source_names == ["EXOGENOUS", "FEEDBACK"]
    for _ in range(50):
        square.step()
    np.testing.assert_allclose(sum(s.grid for s in square.sources), square.field.grid, atol=1e-8)


def test_delay_is_split_completely_among_origins(golden_ruin_config):
    result = run_trajectory(
        golden_ruin_config,
        seed=81,
        max_steps=300,
        store_qdot_snapshots=False,
        stop_at_ruin=False,
        attribute_sources=True,
    )
    delay = result["delay_by_source"]
    assert delay["total_delay_steps"] > 0
    assert sum(delay["share_by_source"].values()) == pytest.approx(1.0, abs=1e-3)
    assert delay["unattributed_share"] == pytest.approx(0.0, abs=1e-6)
    assert (
        run_trajectory(golden_ruin_config, seed=1, max_steps=20, store_qdot_snapshots=False)[
            "delay_by_source"
        ]
        is None
    )


def test_all_delay_goes_to_the_only_regime_that_shocks(golden_ruin_config):
    cfg = golden_ruin_config.model_copy(deep=True)
    regime = cfg.probability_square.regime
    regime.start = "chaotic"
    regime.shock_rate = [0.0, 0.0, 0.0, 1.0]
    regime.relax = [0.05, 0.04, 1e-4]  # a crisis that does not end
    regime.escalate = [0.015, 0.02, 0.02]
    cfg.probability_square.qdot_feedback_weight = 0.0
    result = run_trajectory(
        cfg, seed=2, max_steps=120, store_qdot_snapshots=False, attribute_sources=True
    )
    shares = result["delay_by_source"]["share_by_source"]
    assert shares["CHAOTIC"] > 0.95
    assert shares["STABLE"] == shares["STRESSED"] == shares["FEEDBACK"] == 0.0


def test_feedback_from_late_dots_is_its_own_origin(golden_ruin_config):
    cfg = golden_ruin_config.model_copy(deep=True)
    cfg.disruption_field.shock_arrival_rate = 0.0  # nothing exogenous ...
    cfg.probability_square.qdot_feedback_weight = 400.0  # ... except what late dots stir up
    cfg.qdots.standard.time_window = 3.0
    cfg.qdots.express.time_window = 3.0
    cfg.qdots.bulky.time_window = 3.0  # everyone is late almost at once
    result = run_trajectory(
        cfg,
        seed=4,
        max_steps=100,
        store_qdot_snapshots=False,
        stop_at_ruin=False,
        attribute_sources=True,
    )
    shares = result["delay_by_source"]["share_by_source"]
    assert shares["FEEDBACK"] > 0.9
    assert result["delay_by_source"]["total_delay_steps"] > 0


def test_fork_carries_the_origin_fields(golden_ruin_config):
    square = ProbabilitySquare(
        golden_ruin_config, seed=6, store_qdot_snapshots=False, attribute_sources=True
    )
    for _ in range(60):
        square.step()
    twin = square.fork(1)
    for _ in range(30):
        twin.step()
    np.testing.assert_allclose(sum(s.grid for s in twin.sources), twin.field.grid, atol=1e-8)
    assert square.time == 60 and twin.recorder is None
    np.testing.assert_allclose(sum(s.grid for s in square.sources), square.field.grid, atol=1e-8)


# ---------------------------------------------------------------------------
# Reports and the command line
# ---------------------------------------------------------------------------


def _short(cfg, steps=120):
    out = cfg.model_copy(deep=True)
    out.simulation.shift_duration = steps
    return out


def test_run_organization_reports_every_section(golden_ruin_config):
    report = run_organization(_short(golden_ruin_config, 200), shifts=6, n_jobs=2)
    assert report["shifts"] == 6
    for section in (
        "jams",
        "pile_up",
        "diversion",
        "corridors",
        "pressure",
        "recovery",
        "loss",
        "attribution",
    ):
        assert section in report
    assert report["pile_up"]["n"] > 0 and report["pile_up"]["mean"] > 1.0
    shares = report["attribution"]["delay_share_by_origin"]
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-3)
    assert sum(
        report["attribution"]["loss_share_by_regime_when_accrued"].values()
    ) == pytest.approx(1.0, abs=1e-3)


def test_organization_report_is_reproducible(golden_ruin_config):
    cfg = _short(golden_ruin_config, 100)
    assert run_organization(cfg, shifts=3, n_jobs=1) == run_organization(cfg, shifts=3, n_jobs=2)


def test_ruin_zone_is_reported_for_ruined_shifts(golden_ruin_config):
    cfg = golden_ruin_config.model_copy(deep=True)
    cfg.ruin.initial_buffer = 5.0
    cfg.disruption_field.shock_arrival_rate *= 3
    report = run_organization(_short(cfg, 200), shifts=4, n_jobs=2)
    zone = report["ruin_zone"]
    assert zone["ruined_shifts"] >= 1
    assert 0 <= zone["loss_share_outside_any_jam"]["mean"] <= 1


def test_aggregate_handles_shifts_that_never_ruin(thin_config):
    cfg = _short(thin_config, 60)
    cfg.disruption_field.shock_arrival_rate = 0.0
    report = run_organization(cfg, shifts=2, n_jobs=1)
    assert "ruin_zone" not in report and report["ruin_probability"] == 0.0
    assert aggregate is not None


def test_plot_organization(golden_ruin_config, tmp_path):
    rec = StructureRecorder()
    result = run_trajectory(
        _short(golden_ruin_config, 150),
        seed=81,
        store_qdot_snapshots=False,
        stop_at_ruin=False,
        recorder=rec,
        attribute_sources=True,
    )
    out = plot_organization(result, rec, tmp_path / "org.png")
    assert out.exists() and out.stat().st_size > 20_000


def test_cli_organize_aggregates_and_overrides_avoidance(golden_config_path, caplog):
    caplog.set_level("INFO", logger="ruin.cli")
    rc = cli.main(
        [
            "organize",
            "--config",
            str(golden_config_path),
            "--shifts",
            "3",
            "--jobs",
            "1",
            "--avoidance",
            "4",
        ]
    )
    assert rc == 0
    assert '"avoidance": 4.0' in caplog.text and '"pile_up"' in caplog.text


def test_cli_organize_draws_one_shift(golden_config_path, tmp_path, caplog):
    caplog.set_level("INFO", logger="ruin.cli")
    plot = tmp_path / "org.png"
    rc = cli.main(
        ["organize", "--config", str(golden_config_path), "--seed", "81", "--plot", str(plot)]
    )
    assert rc == 0 and plot.exists()
    assert "Organization chart written" in caplog.text


def test_jam_outlines_do_not_break_the_animation(golden_ruin_config, tmp_path):
    from ruin.viz.square import save_field_gif

    cfg = _short(golden_ruin_config, 40)
    result = run_trajectory(cfg, seed=2, store_field_snapshots=True, store_qdot_snapshots=True)
    out = save_field_gif(result, cfg, output=tmp_path / "jams.gif", fps=4, max_frames=8)
    assert out.exists() and out.stat().st_size > 10_000

"""Validation of the risk layer against classical Cramér-Lundberg ruin theory."""

from __future__ import annotations

import math

import pytest

from ruin import cli
from ruin.risk.classical import (
    cramer_lundberg_ruin_probability,
    simulate_classical_ruin,
    simulate_classical_ruin_stepped,
    validate_against_cramer_lundberg,
)
from ruin.risk.ruin_probability import wilson_interval

# claim rate 1, mean claim 1, premium 1.5: a 50% safety loading.
C, LAM, MU = 1.5, 1.0, 1.0


def test_closed_form_known_values():
    assert cramer_lundberg_ruin_probability(0.0, C, LAM, MU) == pytest.approx(2 / 3)
    assert cramer_lundberg_ruin_probability(3.0, C, LAM, MU) == pytest.approx(
        (2 / 3) * math.exp(-1.0)
    )
    # No safety loading: ruin is certain whatever the buffer.
    assert cramer_lundberg_ruin_probability(100.0, 1.0, LAM, MU) == 1.0
    assert cramer_lundberg_ruin_probability(100.0, 0.5, LAM, MU) == 1.0
    # No claims: ruin is impossible.
    assert cramer_lundberg_ruin_probability(1.0, C, 0.0, MU) == 0.0


def test_closed_form_decreases_with_buffer_and_premium():
    by_buffer = [cramer_lundberg_ruin_probability(u, C, LAM, MU) for u in (0, 1, 2, 5, 10)]
    assert by_buffer == sorted(by_buffer, reverse=True)
    by_premium = [cramer_lundberg_ruin_probability(3.0, c, LAM, MU) for c in (1.1, 1.5, 2.0, 4.0)]
    assert by_premium == sorted(by_premium, reverse=True)


def test_exact_simulation_matches_closed_form():
    report = validate_against_cramer_lundberg(paths=20_000, seed=42)
    assert report["passed"], report["rows"]
    for row in report["rows"]:
        assert row["estimate"] == pytest.approx(row["exact"], abs=0.012)


def test_network_surplus_reproduces_closed_form():
    """The engine's own surplus object, stepped in time, lands on the same answer."""
    paths, u = 1500, 2.0
    exact = cramer_lundberg_ruin_probability(u, C, LAM, MU)
    ruined = simulate_classical_ruin_stepped(
        u, C, LAM, MU, horizon=100, paths=paths, dt=0.1, seed=1
    )
    low, high = wilson_interval(ruined, paths, 0.999)
    # Checking only at step ends misses a little ruin, so allow a small downward bias.
    assert low - 0.02 <= exact <= high + 0.02


def test_coarser_steps_miss_more_ruin():
    kwargs = {"horizon": 60, "paths": 1500, "seed": 3}
    fine = simulate_classical_ruin_stepped(2.0, C, LAM, MU, dt=0.1, **kwargs)
    coarse = simulate_classical_ruin_stepped(2.0, C, LAM, MU, dt=2.0, **kwargs)
    assert coarse < fine


def test_longer_horizon_never_lowers_ruin_count():
    short = simulate_classical_ruin(2.0, C, LAM, MU, horizon=2, paths=4000, seed=5)
    long = simulate_classical_ruin(2.0, C, LAM, MU, horizon=200, paths=4000, seed=5)
    assert short < long


def test_cli_validate(caplog):
    caplog.set_level("INFO", logger="ruin.cli")
    assert cli.main(["validate", "--paths", "4000"]) == 0
    assert '"passed": true' in caplog.text

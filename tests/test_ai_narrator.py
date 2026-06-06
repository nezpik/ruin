"""Tests for the optional Codex-powered AI research-report narrator."""

from __future__ import annotations

import importlib.util
import json

import pytest

from ruin import cli
from ruin.ai import narrator
from ruin.ai.narrator import (
    CodexNarration,
    _build_config_summary,
    _build_risk_facts,
    _build_risk_prompt,
    _build_trajectory_facts,
    _build_trajectory_prompt,
    _summarize_d_state_transitions,
    _summarize_path,
    detect_result_kind,
    generate_report,
)
from ruin.config_models import RuinConfig
from ruin.risk.ruin_probability import run_monte_carlo
from ruin.simulation.agent_based import run_trajectory


# ---------------------------------------------------------------------------
# Pure-function tests — real simulation/risk results, no Codex involved.
# ---------------------------------------------------------------------------


@pytest.fixture
def trajectory_result(small_config):
    return run_trajectory(small_config, seed=42, max_steps=20)


@pytest.fixture
def risk_result(small_config):
    return run_monte_carlo(small_config, paths=4, max_steps=15, n_jobs=1)


def test_detect_result_kind(trajectory_result, risk_result):
    assert detect_result_kind(trajectory_result) == "trajectory"
    assert detect_result_kind(risk_result) == "risk"


def test_summarize_path_compresses_series():
    summary = _summarize_path([1.0, 2.0, 3.0, 4.0], "surplus")
    assert summary.startswith("surplus: ")
    assert "start=1" in summary
    assert "end=4" in summary
    assert "min=1" in summary
    assert "max=4" in summary
    assert "mean=2.5" in summary
    assert "len=4" in summary


def test_summarize_path_handles_empty():
    assert _summarize_path([], "surplus") == "surplus: (no data)"


def test_summarize_d_state_transitions_run_length_encodes():
    transitions = _summarize_d_state_transitions(
        ["STABLE", "STABLE", "STRESSED", "DISRUPTED", "DISRUPTED", "DISRUPTED"]
    )
    assert transitions == "STABLE(2) → STRESSED(1) → DISRUPTED(3)"


def test_summarize_d_state_transitions_handles_empty():
    assert _summarize_d_state_transitions([]) == "(no D-state history recorded)"


def test_build_config_summary_dict(small_config):
    summary = _build_config_summary(small_config)
    assert "Scenario: tiny_test (seed=123)" in summary
    assert "Grid D: 8x8" in summary
    assert "5 standard, 2 express, 1 bulky" in summary
    assert "rule=surplus_or_sla" in summary


def test_build_config_summary_typed(small_ruin_config):
    summary = _build_config_summary(small_ruin_config)
    assert isinstance(small_ruin_config, RuinConfig)
    assert "Scenario: tiny_test (seed=123)" in summary
    assert "Grid D: 8x8" in summary
    assert "rule=surplus_or_sla" in summary


def test_build_trajectory_prompt_uses_vocabulary_and_compresses_data(small_config, trajectory_result):
    prompt = _build_trajectory_prompt(trajectory_result, small_config)

    # Grounded in RUIN's vocabulary.
    assert "D-state" in prompt
    assert "Q dot" in prompt
    assert "Surplus process" in prompt
    assert "Scenario: tiny_test" in prompt

    # Path data must be compressed, never dumped raw.
    assert "start=" in prompt
    assert "surplus_path" not in prompt
    assert "chaos_pressure_path" not in prompt
    assert "snapshots" not in prompt
    # Per-step snapshot dicts (with keys like "qdots": [...] and "field_exposure")
    # must never be dumped raw into the prompt — only the compressed summaries above.
    assert '"qdots"' not in prompt
    assert "field_exposure" not in prompt
    assert "d_state_exposure" not in prompt


def test_build_risk_prompt_uses_vocabulary_and_compresses_data(small_config, risk_result):
    prompt = _build_risk_prompt(risk_result, small_config)

    assert "tail risk" in prompt.lower()
    assert "Surplus process" in prompt
    assert "Scenario: tiny_test" in prompt

    # The compact MC stats should be present...
    assert "ruin_probability=" in prompt
    assert "value_at_risk=" in prompt
    assert "expected_shortfall=" in prompt

    # ...but the raw per-path time_to_ruin list must not be dumped verbatim.
    assert "time_to_ruin: " in prompt
    assert json.dumps(risk_result.get("time_to_ruin")) not in prompt


# ---------------------------------------------------------------------------
# Mocked-Codex tests — exercise generate_report end to end via the single
# SDK indirection point (_call_codex), without needing live Codex auth.
# ---------------------------------------------------------------------------


FAKE_NARRATION = CodexNarration(
    text="This is the narrated analysis body.",
    thread_id="thread_fake0001",
    turn_id="turn_fake0002",
    duration_ms=4242,
    total_tokens=1337,
)


@pytest.fixture
def fake_codex(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_call_codex(prompt: str, model: str | None = None, effort: str | None = None) -> CodexNarration:
        calls.append({"prompt": prompt, "model": model, "effort": effort})
        return FAKE_NARRATION

    monkeypatch.setattr(narrator, "_call_codex", fake_call_codex)
    return calls


def test_generate_report_trajectory(tmp_path, small_config, trajectory_result, fake_codex):
    out_path = tmp_path / "trajectory_report.md"

    written = generate_report(trajectory_result, small_config, out_path)

    assert written == out_path
    assert written.exists()
    content = written.read_text(encoding="utf-8")

    assert "# RUIN AI Research Report (single trajectory)" in content
    assert "Result kind: `trajectory`" in content
    assert "This is the narrated analysis body." in content
    assert "AI-generated by OpenAI Codex" in content
    assert "RUIN AI Narrator" in content

    # Provenance of the one non-deterministic step is recorded in the header...
    assert "thread_fake0001" in content
    assert "turn_fake0002" in content
    assert "4242 ms" in content
    assert "1337 tokens" in content
    assert "model=`default`" in content
    assert "effort=`default`" in content

    # ...and the deterministic data Codex saw is echoed verbatim in an appendix.
    assert "## Appendix: source data (verbatim, as given to Codex)" in content
    assert "## Scenario configuration" in content.split("Appendix")[1]
    assert "## D-state regime sequence" in content.split("Appendix")[1]

    assert len(fake_codex) == 1
    assert fake_codex[0]["model"] is None
    assert fake_codex[0]["effort"] is None


def test_generate_report_risk(tmp_path, small_config, risk_result, fake_codex):
    out_path = tmp_path / "risk_report.md"

    written = generate_report(risk_result, small_config, out_path)

    assert written == out_path
    content = written.read_text(encoding="utf-8")
    assert "# RUIN AI Research Report (Monte Carlo risk analysis)" in content
    assert "Result kind: `risk`" in content
    assert "This is the narrated analysis body." in content
    assert "## Appendix: source data (verbatim, as given to Codex)" in content
    assert "## Monte Carlo risk result" in content.split("Appendix")[1]


def test_generate_report_appendix_matches_facts_byte_for_byte(
    tmp_path, small_config, trajectory_result, risk_result, fake_codex
):
    """The appendix must echo exactly what `_build_*_facts` computed — the one
    part of the report that is reproducible byte-for-byte across re-runs."""
    trajectory_path = generate_report(trajectory_result, small_config, tmp_path / "t.md")
    trajectory_appendix = trajectory_path.read_text(encoding="utf-8").split("Appendix")[1]
    assert _build_trajectory_facts(trajectory_result, small_config) in trajectory_appendix

    risk_path = generate_report(risk_result, small_config, tmp_path / "r.md")
    risk_appendix = risk_path.read_text(encoding="utf-8").split("Appendix")[1]
    assert _build_risk_facts(risk_result, small_config) in risk_appendix


def test_generate_report_records_requested_model_and_effort_in_header(
    tmp_path, small_config, trajectory_result, fake_codex
):
    out_path = tmp_path / "report.md"

    written = generate_report(trajectory_result, small_config, out_path, model="gpt-5-codex", effort="xhigh")
    content = written.read_text(encoding="utf-8")

    assert "model=`gpt-5-codex`" in content
    assert "effort=`xhigh`" in content


def test_generate_report_forwards_model_kwarg(tmp_path, small_config, trajectory_result, fake_codex):
    out_path = tmp_path / "report.md"

    generate_report(trajectory_result, small_config, out_path, model="gpt-5-codex")

    assert len(fake_codex) == 1
    assert fake_codex[0]["model"] == "gpt-5-codex"


def test_generate_report_forwards_effort_kwarg(tmp_path, small_config, trajectory_result, fake_codex):
    out_path = tmp_path / "report.md"

    generate_report(trajectory_result, small_config, out_path, effort="low")

    assert len(fake_codex) == 1
    assert fake_codex[0]["effort"] == "low"


def test_generate_report_creates_parent_directories(tmp_path, small_config, trajectory_result, fake_codex):
    out_path = tmp_path / "nested" / "dir" / "report.md"

    written = generate_report(trajectory_result, small_config, out_path)

    assert written.exists()
    assert written.parent.is_dir()


def test_wrap_report_handles_empty_narrative(small_config):
    empty = CodexNarration(text="   \n  ")
    content = narrator._wrap_report(empty, small_config, "trajectory", model=None, effort=None)
    assert "_Codex returned no narrative content for this run._" in content


def test_wrap_report_records_provenance_and_handles_missing_fields(small_config):
    """Provenance fields are optional on `CodexNarration` — absent ones must
    render as readable placeholders rather than the literal string `None`."""
    bare = CodexNarration(text="narrated body")
    content = narrator._wrap_report(bare, small_config, "trajectory", model=None, effort="low")

    assert "model=`default`" in content
    assert "effort=`low`" in content
    assert "n/a" in content
    assert "None" not in content


# ---------------------------------------------------------------------------
# CLI wiring — `--explain PATH` on both `simulate` and `risk`.
# ---------------------------------------------------------------------------


def test_cli_simulate_explain_writes_report(tmp_path, golden_config_path, fake_codex, caplog):
    out_path = tmp_path / "explain.md"
    caplog.set_level("INFO", logger="ruin.cli")

    rc = cli.main(
        [
            "simulate",
            "--config",
            str(golden_config_path),
            "--max-steps",
            "5",
            "--output",
            str(tmp_path / "frames.txt"),
            "--explain",
            str(out_path),
        ]
    )

    assert rc == 0
    assert out_path.exists()
    assert "AI report written to" in caplog.text


def test_cli_simulate_explain_effort_forwards_to_codex(tmp_path, golden_config_path, fake_codex, caplog):
    out_path = tmp_path / "explain.md"
    caplog.set_level("INFO", logger="ruin.cli")

    rc = cli.main(
        [
            "simulate",
            "--config",
            str(golden_config_path),
            "--max-steps",
            "5",
            "--output",
            str(tmp_path / "frames.txt"),
            "--explain",
            str(out_path),
            "--explain-effort",
            "low",
        ]
    )

    assert rc == 0
    assert len(fake_codex) == 1
    assert fake_codex[0]["effort"] == "low"


def test_cli_explain_effort_rejects_invalid_choice(golden_config_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "simulate",
                "--config",
                str(golden_config_path),
                "--explain-effort",
                "ludicrous",
            ]
        )

    assert "invalid choice" in capsys.readouterr().err


def test_cli_risk_explain_writes_report(tmp_path, golden_config_path, fake_codex, caplog):
    out_path = tmp_path / "explain.md"
    caplog.set_level("INFO", logger="ruin.cli")

    rc = cli.main(
        [
            "risk",
            "--config",
            str(golden_config_path),
            "--paths",
            "4",
            "--max-steps",
            "10",
            "--jobs",
            "1",
            "--explain",
            str(out_path),
        ]
    )

    assert rc == 0
    assert out_path.exists()
    assert "AI report written to" in caplog.text


def test_cli_simulate_without_explain_unchanged(tmp_path, golden_config_path, caplog):
    rc = cli.main(
        [
            "simulate",
            "--config",
            str(golden_config_path),
            "--max-steps",
            "5",
            "--output",
            str(tmp_path / "frames.txt"),
        ]
    )

    assert rc == 0
    assert "AI report written to" not in caplog.text


def test_cli_explain_missing_dependency_does_not_crash(tmp_path, golden_config_path, monkeypatch, caplog):
    """A missing 'ai' extra must not turn a successful run into a crash (see _call_codex's ImportError)."""

    def missing_dependency(prompt: str, model: str | None = None, effort: str | None = None) -> str:
        raise ImportError("Codex narration requires the optional 'ai' extra: pip install 'ruin[ai]'")

    monkeypatch.setattr(narrator, "_call_codex", missing_dependency)
    caplog.set_level("INFO", logger="ruin.cli")
    out_path = tmp_path / "explain.md"

    rc = cli.main(
        [
            "simulate",
            "--config",
            str(golden_config_path),
            "--max-steps",
            "5",
            "--output",
            str(tmp_path / "frames.txt"),
            "--explain",
            str(out_path),
        ]
    )

    assert rc == 0
    assert not out_path.exists()
    assert "Skipping --explain" in caplog.text
    assert "pip install 'ruin[ai]'" in caplog.text


# ---------------------------------------------------------------------------
# Optional-dependency boundary.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("openai_codex") is not None,
    reason="openai_codex is installed in this environment; this documents the missing-dependency path",
)
def test_call_codex_raises_helpful_error_without_optional_dependency():
    with pytest.raises(ImportError, match=r"ruin\[ai\]"):
        narrator._call_codex("prompt")

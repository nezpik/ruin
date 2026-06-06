"""Codex-powered narration of RUIN simulation and risk-analysis results.

Codex is used purely as a text-generation engine here: this module reads and
summarizes simulation output itself, builds a complete prompt grounded in
RUIN's own vocabulary (see ``RUIN.md``), and asks a Codex thread to write a
markdown research report from that summary. Codex is given a read-only
sandbox — it never touches the filesystem; this module writes the report.

The optional ``openai_codex`` dependency is imported lazily (only inside
``_call_codex``) so the rest of ``ruin`` stays importable without the ``ai``
extra installed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Union

from ruin.config import to_dict
from ruin.config_models import RuinConfig

ConfigLike = Union[dict[str, Any], RuinConfig]


RUIN_VOCABULARY_GLOSSARY = """\
## RUIN vocabulary (use these terms precisely)

- **D** (the destiny frame / Probability Square): a bounded grid through which
  Q dots move. Q dots can drift, jump, stall, cluster, fail, or recover, but
  they can never leave D.
- **Q dot**: a quantized delivery commitment — STANDARD (balanced),
  EXPRESS (tighter time window, higher SLA pressure), or BULKY (slower,
  higher drag and penalty if delayed) — carrying time-window slack, penalty
  exposure, and local disruption-field exposure.
- **Disruption Propagation Field**: spatial memory of shocks that intensifies,
  spreads to nearby cells, decays over time, and recovers.
- **D-state**: the grid's regime — one of STABLE, STRESSED, DISRUPTED,
  CHAOTIC, RECOVERING, or RUINED — driven by the balance of chaos pressure
  and order pressure.
- **chaos_pressure**: a global measure of how strongly disruption, congestion
  memory, and local Q-dot failure dominate the grid.
- **order_pressure**: a global measure of how well active and moving Q dots
  preserve flow against failure drag.
- **Surplus process**: `U(t) = U(0) + C(t) - L(t)`, where `U(0)` is the
  initial operational buffer, `C(t)` is recovered/planned capacity, and
  `L(t)` is cumulative disruption loss, delay penalty, and SLA cost.
- **Ruin**: the shift fails when `U(t) <= barrier` (surplus ruin) and/or
  `late_qdots / active_qdots >= sla_threshold` (SLA ruin). The configured
  `rule` (`surplus_or_sla`, `surplus`, or `sla`) decides which trigger(s)
  actually apply for this scenario.
"""


_TRAJECTORY_REPORT_INSTRUCTIONS = """\
## Your task

Write a clear, well-structured markdown research report explaining this
single Urban Ruin Shift trajectory for a logistics operations audience. Use
RUIN's vocabulary precisely (D, Q dots, D-state, chaos/order pressure,
surplus, ruin). Structure the report with these sections:

1. **Executive Summary** — what happened, in one paragraph.
2. **D-State Narrative** — how the grid's regime evolved over the shift and
   why, tying the transitions to the chaos/order pressure dynamics.
3. **Surplus Dynamics** — how the operational surplus moved over time and
   what drained or protected it.
4. **Ruin Analysis** — explain precisely why the shift ruined, or survived,
   citing the configured ruin rule and the relevant trigger(s).
5. **Policy Suggestions** — one or two concrete ideas for what a hedging,
   buffer-sizing, or recovery policy might target next.

Ground every claim strictly in the numbers given below. Do not invent
statistics, events, or path values that are not present in this data.
"""

_RISK_REPORT_INSTRUCTIONS = """\
## Your task

Write a clear, well-structured markdown research report explaining this Monte
Carlo ruin-risk analysis for a logistics operations audience. Use RUIN's
vocabulary precisely (D, Q dots, surplus, ruin, tail risk). Structure the
report with these sections:

1. **Executive Summary** — what the headline ruin probability means in plain
   operational terms.
2. **Tail Risk** — interpret Value at Risk and Expected Shortfall as
   logistics-surplus / cumulative-loss exposure, and explain what the width
   of the ruin-probability confidence interval implies about how precise
   this estimate is.
3. **Time to Ruin** — what the time-to-ruin figures (if any ruin events
   occurred) suggest about how quickly this shift tends to fail.
4. **Policy Suggestions** — one or two concrete ideas for buffer sizing,
   hedging, or recovery capacity that these numbers point toward.

Ground every claim strictly in the numbers given below. Do not invent
statistics that are not present in this data.
"""


def detect_result_kind(result: dict[str, Any]) -> Literal["trajectory", "risk"]:
    """Distinguish a `run_monte_carlo` result from a `run_trajectory` result."""
    return "risk" if "ruin_probability" in result else "trajectory"


def _summarize_path(path: list[float], name: str) -> str:
    if not path:
        return f"{name}: (no data)"
    values = [float(v) for v in path]
    return (
        f"{name}: start={values[0]:.4g} end={values[-1]:.4g} "
        f"min={min(values):.4g} max={max(values):.4g} "
        f"mean={(sum(values) / len(values)):.4g} len={len(values)}"
    )


def _summarize_d_state_transitions(d_state_path: list[str]) -> str:
    if not d_state_path:
        return "(no D-state history recorded)"
    runs: list[list[Any]] = []
    for state in d_state_path:
        if runs and runs[-1][0] == state:
            runs[-1][1] += 1
        else:
            runs.append([state, 1])
    return " → ".join(f"{state}({count})" for state, count in runs)


def _build_config_summary(config: ConfigLike) -> str:
    if isinstance(config, RuinConfig):
        sim = config.simulation
        qdots = config.qdots
        ruin_cfg = config.ruin
        name, seed = sim.name, sim.seed
        grid = f"{sim.grid_width}x{sim.grid_height}"
        duration = sim.shift_duration
        n_standard, n_express, n_bulky = qdots.n_standard, qdots.n_express, qdots.n_bulky
        initial_buffer, barrier = ruin_cfg.initial_buffer, ruin_cfg.barrier
        sla_threshold, rule = ruin_cfg.sla_threshold, ruin_cfg.rule
    else:
        cfg = to_dict(config)
        sim = cfg.get("simulation", {})
        qdots = cfg.get("qdots", {})
        ruin_cfg = cfg.get("ruin", {})
        name = sim.get("name", "unknown")
        seed = sim.get("seed", 42)
        grid = f"{sim.get('grid_width', 30)}x{sim.get('grid_height', 30)}"
        duration = sim.get("shift_duration", 480)
        n_standard = qdots.get("n_standard", 0)
        n_express = qdots.get("n_express", 0)
        n_bulky = qdots.get("n_bulky", 0)
        initial_buffer = ruin_cfg.get("initial_buffer", 0)
        barrier = ruin_cfg.get("barrier", 0)
        sla_threshold = ruin_cfg.get("sla_threshold", 0.12)
        rule = ruin_cfg.get("rule", "surplus_or_sla")

    return (
        f"Scenario: {name} (seed={seed})\n"
        f"Grid D: {grid}, shift_duration={duration} steps\n"
        f"Q dots: {n_standard} standard, {n_express} express, {n_bulky} bulky\n"
        f"Ruin setup: initial_buffer={initial_buffer}, barrier={barrier}, "
        f"sla_threshold={sla_threshold}, rule={rule}"
    )


def _build_trajectory_prompt(result: dict[str, Any], config: ConfigLike) -> str:
    stats = (
        f"ruined={result.get('ruined')} ruin_time={result.get('ruin_time')} "
        f"ruin_reason={result.get('ruin_reason')}\n"
        f"cumulative_loss={result.get('cumulative_loss')} "
        f"final_surplus={result.get('final_surplus')} "
        f"final_d_state={result.get('final_d_state')}"
    )
    paths = "\n".join(
        _summarize_path(result.get(key, []), label)
        for key, label in (
            ("surplus_path", "surplus"),
            ("late_ratio_path", "late_ratio"),
            ("chaos_pressure_path", "chaos_pressure"),
            ("order_pressure_path", "order_pressure"),
        )
    )
    transitions = _summarize_d_state_transitions(result.get("d_state_path", []))

    return (
        f"{RUIN_VOCABULARY_GLOSSARY}\n"
        f"## Scenario configuration\n{_build_config_summary(config)}\n\n"
        f"## Trajectory result\n{stats}\n\n"
        f"## Path summaries (compressed — the only numeric series available)\n{paths}\n\n"
        f"## D-state regime sequence (run-length encoded)\n{transitions}\n\n"
        f"{_TRAJECTORY_REPORT_INSTRUCTIONS}"
    )


def _build_risk_prompt(result: dict[str, Any], config: ConfigLike) -> str:
    ruin_times = result.get("time_to_ruin") or []
    if ruin_times:
        ttr_summary = (
            f"ruin observed in {len(ruin_times)} of {result.get('paths')} paths "
            f"(min={min(ruin_times)}, max={max(ruin_times)}, "
            f"mean_time_to_ruin={result.get('mean_time_to_ruin')})"
        )
    else:
        ttr_summary = "no ruin events observed in any sampled path"

    stats = (
        f"paths={result.get('paths')} confidence_level={result.get('confidence_level')}\n"
        f"ruin_probability={result.get('ruin_probability')} "
        f"(bootstrap CI {result.get('ruin_probability_ci')})\n"
        f"mean_loss={result.get('mean_loss')} loss_std={result.get('loss_std')}\n"
        f"value_at_risk={result.get('value_at_risk')} "
        f"expected_shortfall={result.get('expected_shortfall')}\n"
        f"time_to_ruin: {ttr_summary}"
    )

    return (
        f"{RUIN_VOCABULARY_GLOSSARY}\n"
        f"## Scenario configuration\n{_build_config_summary(config)}\n\n"
        f"## Monte Carlo risk result\n{stats}\n\n"
        f"{_RISK_REPORT_INSTRUCTIONS}"
    )


def _call_codex(prompt: str, model: str | None = None) -> str:
    """The single point where the optional Codex SDK is touched (and mocked in tests)."""
    try:
        from openai_codex import Codex, Sandbox
    except ImportError as exc:
        raise ImportError(
            "Codex narration requires the optional 'ai' extra: pip install 'ruin[ai]'"
        ) from exc

    with Codex() as codex:
        thread = codex.thread_start(sandbox=Sandbox.read_only, model=model)
        turn = thread.run(prompt)
        return turn.final_response or ""


def _wrap_report(raw_text: str, config: ConfigLike, kind: str) -> str:
    scenario_line = _build_config_summary(config).splitlines()[0]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    kind_label = "single trajectory" if kind == "trajectory" else "Monte Carlo risk analysis"

    header = (
        f"# RUIN AI Research Report ({kind_label})\n\n"
        f"> AI-generated by OpenAI Codex from RUIN simulation data — every figure\n"
        f"> below is sourced from the simulation output; verify against the raw\n"
        f"> JSON results before citing.\n\n"
        f"- {scenario_line}\n"
        f"- Result kind: `{kind}`\n"
        f"- Generated: {generated_at}\n\n"
        f"---\n\n"
    )
    footer = (
        "\n\n---\n"
        "*Report generated by the RUIN AI Narrator (`ruin.ai.narrator`) via the "
        "OpenAI Codex Python SDK. Treat the narrative above as an interpretive "
        "aid, not a substitute for the underlying data.*\n"
    )
    body = raw_text.strip() or "_Codex returned no narrative content for this run._"
    return header + body + footer


def generate_report(
    result: dict[str, Any],
    config: ConfigLike,
    output_path: str | Path,
    model: str | None = None,
) -> Path:
    """Narrate a `run_trajectory`/`run_monte_carlo` result via Codex and write it to disk.

    Codex is used solely as a text-generation engine: this function builds the
    full prompt itself (RUIN vocabulary + a faithful, compressed summary of
    `result`/`config`) and writes Codex's response to `output_path` wrapped in
    a disclosure header/footer. Returns the written `Path`.
    """
    kind = detect_result_kind(result)
    prompt = (
        _build_trajectory_prompt(result, config)
        if kind == "trajectory"
        else _build_risk_prompt(result, config)
    )
    narrative = _call_codex(prompt, model=model)
    content = _wrap_report(narrative, config, kind)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path

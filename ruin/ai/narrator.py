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

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Union

from ruin.config import to_dict
from ruin.config_models import RuinConfig

ConfigLike = Union[dict[str, Any], RuinConfig]


@dataclass(frozen=True)
class CodexNarration:
    """Narrative text plus the traceable provenance of the Codex turn that produced it.

    Codex exposes no seed/temperature knob, so its prose can never be made
    reproducible the way the rest of RUIN is. What *can* be pinned down is
    exactly which call produced it — the same "simple, traceable loop"
    discipline RUIN.md asks of the simulation core, applied to the narration
    layer instead.
    """

    text: str
    thread_id: str | None = None
    turn_id: str | None = None
    duration_ms: int | None = None
    total_tokens: int | None = None


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


def _build_trajectory_facts(result: dict[str, Any], config: ConfigLike) -> str:
    """The deterministic per-run data block — a pure function of `result`/`config`.

    Reused both inside the prompt (so Codex sees it) and verbatim in the
    report's appendix (so a reader can check the narrative against the exact
    figures Codex was given, without leaving the document).
    """
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
        f"## Scenario configuration\n{_build_config_summary(config)}\n\n"
        f"## Trajectory result\n{stats}\n\n"
        f"## Path summaries (compressed — the only numeric series available)\n{paths}\n\n"
        f"## D-state regime sequence (run-length encoded)\n{transitions}"
    )


def _build_trajectory_prompt(result: dict[str, Any], config: ConfigLike) -> str:
    return (
        f"{RUIN_VOCABULARY_GLOSSARY}\n"
        f"{_build_trajectory_facts(result, config)}\n\n"
        f"{_TRAJECTORY_REPORT_INSTRUCTIONS}"
    )


def _build_risk_facts(result: dict[str, Any], config: ConfigLike) -> str:
    """The deterministic per-run data block — a pure function of `result`/`config`.

    Reused both inside the prompt (so Codex sees it) and verbatim in the
    report's appendix (so a reader can check the narrative against the exact
    figures Codex was given, without leaving the document).
    """
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

    return f"## Scenario configuration\n{_build_config_summary(config)}\n\n## Monte Carlo risk result\n{stats}"


def _build_risk_prompt(result: dict[str, Any], config: ConfigLike) -> str:
    return (
        f"{RUIN_VOCABULARY_GLOSSARY}\n"
        f"{_build_risk_facts(result, config)}\n\n"
        f"{_RISK_REPORT_INSTRUCTIONS}"
    )


def _call_codex(prompt: str, model: str | None = None, effort: str | None = None) -> CodexNarration:
    """The single point where the optional Codex SDK is touched (and mocked in tests).

    `effort` is passed through as a plain string (e.g. "low", "high") rather
    than the SDK's `ReasoningEffort` enum — Codex's own Pydantic validation
    coerces valid level names and rejects invalid ones, so we don't need to
    import the enum just to forward a value un-altered.

    Returns a `CodexNarration`, not a bare string: the turn's id/thread id,
    duration, and token usage are recorded alongside the text so the report
    can carry full provenance of the one genuinely non-deterministic step.
    """
    try:
        from openai_codex import Codex, Sandbox
    except ImportError as exc:
        raise ImportError(
            "Codex narration requires the optional 'ai' extra: pip install 'ruin[ai]'"
        ) from exc

    with Codex() as codex:
        thread = codex.thread_start(sandbox=Sandbox.read_only, model=model)
        # `effort` is `str | None` (see docstring): Codex's own Pydantic layer
        # coerces valid level names to `ReasoningEffort` and rejects invalid
        # ones, but its stub types `run(effort=...)` as `ReasoningEffort | None`.
        turn = thread.run(prompt, effort=effort)  # type: ignore[arg-type]
        total_tokens = turn.usage.total.total_tokens if turn.usage is not None else None
        return CodexNarration(
            text=turn.final_response or "",
            thread_id=thread.id,
            turn_id=turn.id,
            duration_ms=turn.duration_ms,
            total_tokens=total_tokens,
        )


def _build_appendix(facts: str) -> str:
    """A verbatim, byte-reproducible echo of the data block fed to Codex.

    Unlike the narrative above it, this section is generated by RUIN itself
    from `result`/`config` — re-running the same scenario and seed reproduces
    it exactly, so a reader can check every claim in the prose against it
    without leaving the report.
    """
    return (
        "\n\n---\n\n"
        "## Appendix: source data (verbatim, as given to Codex)\n\n"
        "> Generated directly by RUIN, not Codex, from the same simulation\n"
        "> `result`/`config` that produced the prompt above — deterministic,\n"
        "> and reproducible byte-for-byte by re-running this scenario and seed.\n"
        "> Cross-check every figure in the narrative against this block.\n\n"
        f"{facts}\n"
    )


def _wrap_report(
    narration: CodexNarration,
    config: ConfigLike,
    kind: str,
    *,
    model: str | None,
    effort: str | None,
) -> str:
    scenario_line = _build_config_summary(config).splitlines()[0]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    kind_label = "single trajectory" if kind == "trajectory" else "Monte Carlo risk analysis"

    duration = f"{narration.duration_ms} ms" if narration.duration_ms is not None else "n/a"
    tokens = f"{narration.total_tokens} tokens" if narration.total_tokens is not None else "n/a"
    thread_id = narration.thread_id or "n/a"
    turn_id = narration.turn_id or "n/a"

    header = (
        f"# RUIN AI Research Report ({kind_label})\n\n"
        f"> AI-generated by OpenAI Codex from RUIN simulation data — every figure\n"
        f"> below is sourced from the simulation output; verify against the raw\n"
        f"> JSON results (and the appendix at the end of this report) before citing.\n\n"
        f"- {scenario_line}\n"
        f"- Result kind: `{kind}`\n"
        f"- Generated: {generated_at}\n"
        f"- Codex turn: model=`{model or 'default'}` effort=`{effort or 'default'}` "
        f"· {duration} · {tokens}\n"
        f"- Provenance: thread=`{thread_id}` turn=`{turn_id}`\n\n"
        f"---\n\n"
    )
    footer = (
        "\n\n---\n"
        "*Report generated by the RUIN AI Narrator (`ruin.ai.narrator`) via the "
        "OpenAI Codex Python SDK. Treat the narrative above as an interpretive "
        "aid, not a substitute for the underlying data — Codex's prose cannot be "
        "reproduced, but the call that produced it (above) and the data it saw "
        "(appendix below) can be.*\n"
    )
    body = narration.text.strip() or "_Codex returned no narrative content for this run._"
    return header + body + footer


def generate_report(
    result: dict[str, Any],
    config: ConfigLike,
    output_path: str | Path,
    model: str | None = None,
    effort: str | None = None,
) -> Path:
    """Narrate a `run_trajectory`/`run_monte_carlo` result via Codex and write it to disk.

    Codex is used solely as a text-generation engine: this function builds the
    full prompt itself (RUIN vocabulary + a faithful, compressed summary of
    `result`/`config`) and writes Codex's response to `output_path` wrapped in
    a disclosure header/footer. Returns the written `Path`.

    `effort` (e.g. "minimal", "low", "medium", "high", "xhigh") trades report
    depth for speed/cost — forwarded straight through to the Codex turn.

    Codex's prose is the one place this report is genuinely non-deterministic
    (the SDK exposes no seed/temperature control). Rather than pretend
    otherwise, the report brackets that prose with two things RUIN *can* keep
    deterministic and traceable: a header recording exactly which Codex turn
    produced it (`CodexNarration` — thread/turn id, duration, token usage),
    and an appendix containing the same compressed `result`/`config` summary
    Codex was given, byte-reproducible by re-running the scenario and seed.
    """
    kind = detect_result_kind(result)
    if kind == "trajectory":
        facts = _build_trajectory_facts(result, config)
        prompt = _build_trajectory_prompt(result, config)
    else:
        facts = _build_risk_facts(result, config)
        prompt = _build_risk_prompt(result, config)

    narration = _call_codex(prompt, model=model, effort=effort)
    content = (
        _wrap_report(narration, config, kind, model=model, effort=effort)
        + _build_appendix(facts)
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ruin.analysis import StructureRecorder, plot_organization, run_organization
from ruin.config import load_ruin_config
from ruin.config_models import RuinConfig
from ruin.inference.observer import evaluate_observer, fit_regime_observer
from ruin.risk.classical import validate_against_cramer_lundberg
from ruin.risk.destiny import plot_destiny, run_destiny
from ruin.risk.ruin_probability import run_monte_carlo
from ruin.risk.structure import run_structure_ensemble
from ruin.risk.stylized import run_stylized_facts
from ruin.risk.uncertainty import run_parameter_uncertainty, run_sensitivity
from ruin.simulation.agent_based import run_trajectory
from ruin.viz.square import save_field_gif, save_text_frames

logger = logging.getLogger("ruin.cli")

# Codex `ReasoningEffort` levels, low → high — accepted as plain strings and
# forwarded to the SDK, which validates and coerces them itself.
EXPLAIN_EFFORT_CHOICES = ["none", "minimal", "low", "medium", "high", "xhigh"]
EXPLAIN_HELP = (
    "Write an AI-narrated markdown research report to PATH via Codex "
    "(requires the 'ai' extra: pip install 'ruin[ai]')"
)
EXPLAIN_EFFORT_HELP = (
    "Codex reasoning effort for --explain — trade narrative depth for speed/cost "
    "(default: Codex's own default)"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ruin", description="RUIN: destiny-framed ruin risk for logistics flow"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Run one visible Urban Ruin Shift trajectory")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--frames", type=int, default=20)
    simulate.add_argument("--max-steps", type=int, default=None)
    simulate.add_argument("--seed", type=int, default=None, help="Override the config's seed")
    simulate.add_argument("--output", default="ruin_frames.txt")
    simulate.add_argument("--json", action="store_true")
    simulate.add_argument(
        "--visualize",
        action="store_true",
        help="Generate animated GIF of the real disruption field + Q-dots",
    )
    simulate.add_argument(
        "--explain",
        metavar="PATH",
        default=None,
        help=EXPLAIN_HELP,
    )
    simulate.add_argument(
        "--explain-effort",
        choices=EXPLAIN_EFFORT_CHOICES,
        default=None,
        help=EXPLAIN_EFFORT_HELP,
    )

    risk = subparsers.add_parser("risk", help="Run Monte Carlo ruin-risk estimation")
    risk.add_argument("--config", required=True)
    risk.add_argument("--paths", type=int, default=None)
    risk.add_argument("--confidence", type=float, default=None)
    risk.add_argument("--max-steps", type=int, default=None)
    risk.add_argument(
        "--jobs", type=int, default=None, help="Number of parallel workers (default: auto)"
    )
    risk.add_argument(
        "--explain",
        metavar="PATH",
        default=None,
        help=EXPLAIN_HELP,
    )
    risk.add_argument(
        "--explain-effort",
        choices=EXPLAIN_EFFORT_CHOICES,
        default=None,
        help=EXPLAIN_EFFORT_HELP,
    )

    destiny = subparsers.add_parser(
        "destiny",
        help="Follow one shift and estimate its remaining destiny (chance of no ruin) over time",
    )
    destiny.add_argument("--config", required=True)
    destiny.add_argument("--seed", type=int, default=None, help="Override the config's seed")
    destiny.add_argument("--every", type=int, default=24, help="Steps between checkpoints")
    destiny.add_argument("--branches", type=int, default=64, help="Futures per checkpoint")
    destiny.add_argument("--max-steps", type=int, default=None)
    destiny.add_argument("--jobs", type=int, default=None)
    destiny.add_argument("--plot", metavar="PATH", default=None, help="Write the curve as a PNG")
    destiny.add_argument(
        "--observer",
        action="store_true",
        help="Also estimate destiny for an observer who sees the pressures (fits a filter first)",
    )
    destiny.add_argument(
        "--incidents",
        action="store_true",
        help="With --observer: give the observer an incident feed (shock counts)",
    )
    destiny.add_argument(
        "--blind",
        action="store_true",
        help="Also estimate destiny for a viewer who does not know D's hidden regime",
    )

    sensitivity = subparsers.add_parser(
        "sensitivity", help="Rank parameters by how strongly they move the ruin probability"
    )
    sensitivity.add_argument("--config", required=True)
    sensitivity.add_argument("--paths", type=int, default=200)
    sensitivity.add_argument("--step", type=float, default=0.1, help="Relative change applied")
    sensitivity.add_argument("--jobs", type=int, default=None)
    sensitivity.add_argument(
        "--parameter",
        action="append",
        default=None,
        help="Dotted parameter path (repeatable; default: a standard set)",
    )

    uncertainty = subparsers.add_parser(
        "uncertainty",
        help="Ruin probability when the parameters in risk.parameter_uncertainty are uncertain",
    )
    uncertainty.add_argument("--config", required=True)
    uncertainty.add_argument("--draws", type=int, default=40)
    uncertainty.add_argument("--paths", type=int, default=100, help="Paths per draw")
    uncertainty.add_argument("--jobs", type=int, default=None)

    facts = subparsers.add_parser(
        "facts",
        help="Check a scenario against stylized facts of logistics disruption",
    )
    facts.add_argument("--config", required=True)
    facts.add_argument("--paths", type=int, default=80, help="Simulated shifts per measurement")
    facts.add_argument("--jobs", type=int, default=None)

    structure = subparsers.add_parser(
        "structure",
        help="Ruin risk under memoryless, heavy-tailed, and clustered disruption of equal mean",
    )
    structure.add_argument("--config", required=True)
    structure.add_argument("--paths", type=int, default=300)
    structure.add_argument("--jobs", type=int, default=None)

    organize = subparsers.add_parser(
        "organize",
        help="Measure the structure that forms in D: jams, pile-up, corridors, zones, recovery",
    )
    organize.add_argument("--config", required=True)
    organize.add_argument("--shifts", type=int, default=60)
    organize.add_argument(
        "--avoidance",
        type=float,
        default=None,
        help="Override processes.avoidance (how strongly Q dots steer around blocked cells)",
    )
    organize.add_argument("--seed", type=int, default=None, help="With --plot: the shift to draw")
    organize.add_argument(
        "--plot",
        metavar="PATH",
        default=None,
        help="Draw one shift as a PNG instead of aggregating",
    )
    organize.add_argument("--jobs", type=int, default=None)

    watch = subparsers.add_parser(
        "observer",
        help="Fit a regime observer and score it against the true regime on held-out shifts",
    )
    watch.add_argument("--config", required=True)
    watch.add_argument("--fit-shifts", type=int, default=150, help="Shifts used to fit the model")
    watch.add_argument("--eval-shifts", type=int, default=80, help="Held-out shifts to score on")
    watch.add_argument("--jobs", type=int, default=None)

    validate = subparsers.add_parser(
        "validate",
        help="Check the risk layer against the closed-form Cramér-Lundberg ruin probability",
    )
    validate.add_argument("--paths", type=int, default=20000)
    validate.add_argument("--seed", type=int, default=42)

    return parser


def _maybe_explain(
    result: dict[str, Any],
    config: RuinConfig,
    explain_path: str | None,
    effort: str | None = None,
) -> None:
    if explain_path is None:
        return
    from ruin.ai.narrator import generate_report

    try:
        out = generate_report(result, config, output_path=explain_path, effort=effort)
    except ImportError as exc:
        logger.error("Skipping --explain: %s", exc)
        return
    except Exception as exc:  # noqa: BLE001 - narration is optional and must never fail a run
        logger.warning(
            "--explain failed and will be skipped (simulation output is unaffected): %s", exc
        )
        return
    logger.info("AI report written to %s", out)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if args.command == "validate":
        report = validate_against_cramer_lundberg(paths=args.paths, seed=args.seed)
        logger.info(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1

    try:
        config = load_ruin_config(args.config)
    except (OSError, TypeError, ValueError, ValidationError) as exc:
        logger.error("Invalid config %s:\n%s", args.config, exc)
        return 2

    if args.command == "simulate":
        store_field = bool(args.visualize)

        result = run_trajectory(
            config, seed=args.seed, max_steps=args.max_steps, store_field_snapshots=store_field
        )

        save_text_frames(result, Path(args.output), limit=args.frames)

        if args.visualize:
            gif_path = Path(args.output).with_suffix(".gif")
            save_field_gif(result, config, output=gif_path, fps=8)
            logger.info("Animation written to %s", gif_path)

        summary = {
            "ruined": result["ruined"],
            "ruin_time": result["ruin_time"],
            "ruin_reason": result["ruin_reason"],
            "final_surplus": result["final_surplus"],
            "min_surplus": result["min_surplus"],
            "loss_by_type": result["loss_by_type"],
            "loss_concentration": result["loss_concentration"],
            "top_contributors": result["top_contributors"][:5],
            "final_d_state": result["final_d_state"],
            "delivered": result["delivered"],
            "delivered_late": result["delivered_late"],
            "failed": result["failed"],
            "on_time_rate": result["on_time_rate"],
            "frames_written": args.output,
        }
        logger.info(json.dumps(summary if not args.json else result, indent=2))
        _maybe_explain(result, config, args.explain, effort=args.explain_effort)
        return 0

    if args.command == "destiny":
        observer = None
        if args.observer:
            try:
                observer = fit_regime_observer(config, n_jobs=args.jobs)
            except ValueError as exc:
                logger.error("%s", exc)
                return 2
        result = run_destiny(
            config,
            seed=args.seed,
            every=args.every,
            branches=args.branches,
            max_steps=args.max_steps,
            n_jobs=args.jobs,
            blind=args.blind,
            observer=observer,
            observer_incidents=args.incidents,
        )
        if args.plot:
            logger.info("Destiny curve written to %s", plot_destiny(result, args.plot))
        result.pop("surplus_path")
        result.pop("regime_path")
        logger.info(json.dumps(result, indent=2))
        return 0

    if args.command == "organize":
        if args.avoidance is not None:
            config = config.model_copy(deep=True)
            config.processes.avoidance = args.avoidance
        if args.plot:
            recorder = StructureRecorder()
            result = run_trajectory(
                config,
                seed=args.seed,
                stop_at_ruin=False,
                store_qdot_snapshots=False,
                recorder=recorder,
                attribute_sources=True,
            )
            logger.info(
                "Organization chart written to %s", plot_organization(result, recorder, args.plot)
            )
            logger.info(json.dumps(result["structure"], indent=2, default=str))
            return 0
        logger.info(
            json.dumps(run_organization(config, shifts=args.shifts, n_jobs=args.jobs), indent=2)
        )
        return 0

    if args.command == "observer":
        try:
            fitted = fit_regime_observer(config, shifts=args.fit_shifts, n_jobs=args.jobs)
        except ValueError as exc:
            logger.error("%s", exc)
            return 2
        report = evaluate_observer(config, fitted, shifts=args.eval_shifts, n_jobs=args.jobs)
        logger.info(json.dumps(report, indent=2))
        return 0

    if args.command == "facts":
        report = run_stylized_facts(config, paths=args.paths, n_jobs=args.jobs)
        logger.info(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1

    if args.command == "structure":
        try:
            result = run_structure_ensemble(config, paths=args.paths, n_jobs=args.jobs)
        except ValueError as exc:
            logger.error("%s", exc)
            return 2
        logger.info(json.dumps(result, indent=2))
        return 0

    if args.command == "sensitivity":
        result = run_sensitivity(
            config,
            parameters=args.parameter,
            rel_step=args.step,
            paths=args.paths,
            n_jobs=args.jobs,
        )
        logger.info(json.dumps(result, indent=2))
        return 0

    if args.command == "uncertainty":
        try:
            result = run_parameter_uncertainty(
                config, draws=args.draws, paths=args.paths, n_jobs=args.jobs
            )
        except ValueError as exc:
            logger.error("%s", exc)
            return 2
        logger.info(json.dumps(result, indent=2))
        return 0

    if args.command == "risk":
        result = run_monte_carlo(
            config,
            paths=args.paths,
            confidence=args.confidence,
            max_steps=args.max_steps,
            n_jobs=args.jobs,
        )
        logger.info(json.dumps(result, indent=2))
        _maybe_explain(result, config, args.explain, effort=args.explain_effort)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ruin.config import load_ruin_config
from ruin.config_models import RuinConfig
from ruin.risk.classical import validate_against_cramer_lundberg
from ruin.risk.ruin_probability import run_monte_carlo
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

from __future__ import annotations

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Any

from ruin.config import load_config, load_ruin_config
from ruin.config_models import RuinConfig
from ruin.risk.ruin_probability import run_monte_carlo
from ruin.simulation.agent_based import run_trajectory
from ruin.viz.square import save_text_frames, save_field_gif


logger = logging.getLogger("ruin.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ruin", description="RUIN: destiny-framed ruin risk for logistics flow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Run one visible Urban Ruin Shift trajectory")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--frames", type=int, default=20)
    simulate.add_argument("--max-steps", type=int, default=None)
    simulate.add_argument("--output", default="ruin_frames.txt")
    simulate.add_argument("--json", action="store_true")
    simulate.add_argument("--visualize", action="store_true", help="Generate animated GIF of the real disruption field + Q-dots")
    simulate.add_argument(
        "--explain",
        metavar="PATH",
        default=None,
        help="Write an AI-narrated markdown research report to PATH via Codex (requires the 'ai' extra: pip install 'ruin[ai]')",
    )

    risk = subparsers.add_parser("risk", help="Run Monte Carlo ruin-risk estimation")
    risk.add_argument("--config", required=True)
    risk.add_argument("--paths", type=int, default=None)
    risk.add_argument("--confidence", type=float, default=None)
    risk.add_argument("--max-steps", type=int, default=None)
    risk.add_argument("--jobs", type=int, default=None, help="Number of parallel workers (default: auto)")
    risk.add_argument(
        "--explain",
        metavar="PATH",
        default=None,
        help="Write an AI-narrated markdown research report to PATH via Codex (requires the 'ai' extra: pip install 'ruin[ai]')",
    )

    return parser


def _maybe_explain(result: dict[str, Any], config: RuinConfig | dict[str, Any], explain_path: str | None) -> None:
    if explain_path is None:
        return
    from ruin.ai.narrator import generate_report

    out = generate_report(result, config, output_path=explain_path)
    logger.info("AI report written to %s", out)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    # Prefer validated Pydantic config (v0.2) — pass the model directly now that functions accept it
    try:
        config: RuinConfig | dict[str, Any] = load_ruin_config(args.config)
    except Exception:
        config = load_config(args.config)

    if args.command == "simulate":
        store_field = bool(args.visualize)

        result = run_trajectory(config, max_steps=args.max_steps, store_field_snapshots=store_field)

        save_text_frames(result, Path(args.output), limit=args.frames)

        if args.visualize:
            gif_path = Path(args.output).with_suffix(".gif")
            save_field_gif(result, config, output=gif_path, fps=8, max_frames=120)
            logger.info("Animation written to %s", gif_path)

        summary = {
            "ruined": result["ruined"],
            "ruin_time": result["ruin_time"],
            "ruin_reason": result["ruin_reason"],
            "final_surplus": result["final_surplus"],
            "final_d_state": result["final_d_state"],
            "frames_written": args.output,
        }
        logger.info(json.dumps(summary if not args.json else result, indent=2))
        _maybe_explain(result, config, args.explain)
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
        _maybe_explain(result, config, args.explain)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

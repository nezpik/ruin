from __future__ import annotations

import argparse
import json
from pathlib import Path

from ruin.config import load_config
from ruin.risk.ruin_probability import run_monte_carlo
from ruin.simulation.agent_based import run_trajectory
from ruin.viz.square import save_text_frames, save_field_gif


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ruin", description="RUIN: destiny-framed ruin risk for logistics flow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Run one visible Urban Ruin Shift trajectory")
    simulate.add_argument("--config", required=True)
    simulate.add_argument("--frames", type=int, default=20)
    simulate.add_argument("--max-steps", type=int, default=None)
    simulate.add_argument("--output", default="ruin_frames.txt")
    simulate.add_argument("--json", action="store_true")
    simulate.add_argument("--visualize", action="store_true", help="Also generate a GIF animation of the field + Q-dots")

    risk = subparsers.add_parser("risk", help="Run Monte Carlo ruin-risk estimation")
    risk.add_argument("--config", required=True)
    risk.add_argument("--paths", type=int, default=None)
    risk.add_argument("--confidence", type=float, default=None)
    risk.add_argument("--max-steps", type=int, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "simulate":
        result = run_trajectory(config, max_steps=args.max_steps)
        save_text_frames(result, Path(args.output), limit=args.frames)

        if args.visualize:
            gif_path = Path(args.output).with_suffix(".gif")
            save_field_gif(result, config, output=gif_path, fps=8, max_frames=120)
            print(f"Animation written to {gif_path}")

        summary = {
            "ruined": result["ruined"],
            "ruin_time": result["ruin_time"],
            "ruin_reason": result["ruin_reason"],
            "final_surplus": result["final_surplus"],
            "final_d_state": result["final_d_state"],
            "frames_written": args.output,
        }
        print(json.dumps(summary if not args.json else result, indent=2))
        return 0

    if args.command == "risk":
        result = run_monte_carlo(config, paths=args.paths, confidence=args.confidence, max_steps=args.max_steps)
        print(json.dumps(result, indent=2))
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

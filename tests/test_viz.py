"""Smoke tests for v0.2 visualization (text + GIF generation)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ruin.config import load_config
from ruin.simulation.agent_based import run_trajectory
from ruin.viz.square import save_text_frames, save_field_gif


def test_save_text_frames(small_config):
    result = run_trajectory(small_config, seed=123, max_steps=15)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "frames.txt"
        save_text_frames(result, out, limit=5)
        assert out.exists()
        content = out.read_text()
        assert "t=" in content
        assert "D=" in content


@pytest.mark.slow
def test_save_field_gif_smoke(small_ruin_config):
    """Generate a short GIF — must not crash and must produce a file > 10KB."""
    result = run_trajectory(small_ruin_config.model_dump(), seed=42, max_steps=25)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test_field.gif"
        gif_path = save_field_gif(result, small_ruin_config.model_dump(), output=out, fps=4, max_frames=20)
        assert gif_path.exists()
        assert gif_path.stat().st_size > 10_000, "GIF too small — likely failed to render"

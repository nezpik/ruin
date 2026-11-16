"""Smoke tests for v0.2 visualization with real field storage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ruin.simulation.agent_based import run_trajectory
from ruin.viz.square import save_field_gif, save_text_frames


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
def test_save_field_gif_with_real_field(small_ruin_config):
    """Generate GIF with real field data stored in snapshots (typed RuinConfig path)."""
    # v0.2: pass RuinConfig object directly — no .model_dump()
    result = run_trajectory(
        small_ruin_config,
        seed=42,
        max_steps=30,
        store_field_snapshots=True,
    )

    # Verify real field data is present
    assert len(result["snapshots"]) > 0
    assert "field" in result["snapshots"][0]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test_real_field.gif"
        gif_path = save_field_gif(
            result,
            small_ruin_config,  # pass model directly
            output=out,
            fps=6,
            max_frames=25,
        )
        assert gif_path.exists()
        assert gif_path.stat().st_size > 50_000  # larger because real data


@pytest.mark.slow
@pytest.mark.parametrize("heatmap", ["chaos_pressure", "order_pressure"])
def test_gif_follows_visualization_config(small_ruin_config, heatmap, tmp_path):
    from PIL import Image

    cfg = small_ruin_config.model_copy(deep=True)
    cfg.visualization.heatmap_field = heatmap
    cfg.visualization.frames = 6
    cfg.visualization.show_qdot_types = False
    cfg.visualization.show_d_state = False
    result = run_trajectory(cfg, seed=42, max_steps=30, store_field_snapshots=True)
    out = save_field_gif(result, cfg, output=tmp_path / "viz.gif", fps=6)
    with Image.open(out) as gif:
        assert gif.n_frames == 6  # spread over the trajectory, not just the first frames


def test_frame_selection_spans_the_trajectory():
    from ruin.viz.square import _select_frames

    snapshots = [{"time": t, "qdots": []} for t in range(1, 101)]
    picked = [s["time"] for s in _select_frames(snapshots, 5)]
    assert picked[0] == 1 and picked[-1] == 100 and len(picked) == 5
    assert _select_frames(snapshots[:3], 5) == snapshots[:3]

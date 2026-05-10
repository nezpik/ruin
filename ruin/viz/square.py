from __future__ import annotations

from pathlib import Path
from typing import Any


def save_text_frames(result: dict[str, Any], output: str | Path, limit: int = 10) -> None:
    path = Path(output)
    lines: list[str] = []
    snapshots = result.get("snapshots", [])
    if len(snapshots) <= limit:
        selected = snapshots
    else:
        step = max(1, len(snapshots) // max(limit, 1))
        selected = snapshots[::step][:limit]
        if snapshots[-1] not in selected:
            selected[-1] = snapshots[-1]

    for snapshot in selected:
        lines.append(
            f"t={snapshot['time']} D={snapshot['d_state']} chaos={snapshot['chaos_pressure']} order={snapshot['order_pressure']} penalty={snapshot['penalty']}"
        )
        ruined = [q for q in snapshot.get("qdots", []) if q.get("ruined")]
        lines.append(f"  active={snapshot['active_count']} late={snapshot['late_count']} ruined_sample={len(ruined)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

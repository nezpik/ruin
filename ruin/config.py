from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"\'')


def load_config(path: str | Path) -> dict[str, Any]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        text = raw.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if text.startswith("- "):
            item = parse_scalar(text[2:])
            if not isinstance(parent, list):
                raise ValueError(f"List item without list parent: {raw}")
            parent.append(item)
            continue

        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value == "":
            next_container: Any = {}
            if isinstance(parent, dict):
                parent[key] = next_container
            stack.append((indent, next_container))
        else:
            if isinstance(parent, dict):
                parent[key] = parse_scalar(value)

        if value == "" and _next_non_empty_is_list(lines, raw):
            parent[key] = []
            stack[-1] = (indent, parent[key])

    return root


def _next_non_empty_is_list(lines: list[str], current: str) -> bool:
    index = lines.index(current)
    for line in lines[index + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        return line.strip().startswith("- ")
    return False

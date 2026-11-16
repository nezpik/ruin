"""Configuration loading for RUIN.

There is one configuration path: YAML (or a plain dict) is validated into a
`RuinConfig`. Everything downstream works on the validated model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml

from .config_models import RuinConfig

ConfigLike = Union[dict[str, Any], RuinConfig]


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file into a plain dict (not yet validated)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level")
    return raw


def load_ruin_config(path: str | Path) -> RuinConfig:
    """Read and strictly validate a YAML config file."""
    return RuinConfig.model_validate(load_config(path))


def as_config(config: ConfigLike) -> RuinConfig:
    """Return a validated `RuinConfig`, validating a plain dict if needed."""
    if isinstance(config, RuinConfig):
        return config
    return RuinConfig.model_validate(config)

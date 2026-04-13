"""Tiny typed config layer over YAML.

We deliberately avoid heavyweight config frameworks (Hydra, OmegaConf) so the
repo stays trivially runnable on a fresh Colab kernel. A ``Config`` is a plain
dataclass-like dotted-access dict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class Config(dict):
    """A dict that also supports attribute access and nested dotted access."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Read ``a.b.c`` from a nested config, returning ``default`` if absent."""
        node: Any = self
        for key in dotted.split("."):

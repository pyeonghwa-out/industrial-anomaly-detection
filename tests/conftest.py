"""Shared fixtures: a tiny synthetic MVTec-style dataset on disk.

These let the whole pipeline be unit-tested in seconds with no dataset download
and no pretrained-weight download (models are built with ``pretrained=False``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Make src/ importable without an editable install.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@pytest.fixture(scope="session")

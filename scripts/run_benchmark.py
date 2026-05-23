"""Run a method across MVTec AD categories and write a results table.

Examples::

    # All 15 categories with PatchCore
    python scripts/run_benchmark.py --config configs/patchcore.yaml

    # A subset, on CPU, with PaDiM
    python scripts/run_benchmark.py --config configs/padim.yaml \
        --categories bottle cable --device cpu

    # Override the dataset location
    python scripts/run_benchmark.py --config configs/patchcore.yaml \
        --root /kaggle/input/mvtec-ad
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
import pandas as pd

from iad.datasets.mvtec import MVTEC_CATEGORIES
from iad.runner import run_experiment
from iad.utils.config import load_config


def _resolve_categories(config, cli_categories) -> list[str]:
    if cli_categories:
        return cli_categories

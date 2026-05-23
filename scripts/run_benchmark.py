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
    cfg_cats = config.get("categories", "all")
    return MVTEC_CATEGORIES if cfg_cats == "all" else list(cfg_cats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an anomaly method on MVTec AD.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--categories", nargs="*", help="Subset of categories (default: all).")
    parser.add_argument("--root", help="Override the dataset root in the config.")
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto).")
    parser.add_argument("--output", default="results/runs", help="Where to write CSV/MD.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.root:
        config.setdefault("data", {})["root"] = args.root

    categories = _resolve_categories(config, args.categories)
    method = config["method"]
    print(f"Running '{method}' on {len(categories)} categories: {categories}")

    rows = []
    for cat in categories:
        print(f"\n=== {cat} ===")
        metrics = run_experiment(config, cat, device=args.device)
        rows.append(metrics)
        print(
            f"  image AUROC={metrics['image_auroc']:.3f}  "
            f"pixel AUROC={metrics['pixel_auroc']:.3f}  "
            f"PRO={metrics.get('pro', float('nan')):.3f}  "
            f"{metrics['inference_ms_per_image']:.1f} ms/img"
        )

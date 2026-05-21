"""Download MVTec AD and lay it out in the standard folder structure.

MVTec rotated its official download link, so this script pulls the dataset from a
public, ungated Hugging Face mirror (``TheoM55/mvtec_all_objects_split``) that
hosts all 15 categories as parquet with ``image_path``, ``mask_path``, named
``defect`` types and a binary ``label``. It then materializes the canonical MVTec
layout on disk::

    <root>/<category>/
        train/good/*.png
        test/good/*.png
        test/<defect_type>/*.png
        ground_truth/<defect_type>/*_mask.png

so the file-based loader in ``iad/datasets/mvtec.py`` works unchanged.

Requirements: ``pip install datasets pillow`` (both are in requirements.txt).
No Hugging Face account or token is required — the mirror is public.

Examples::

    # All 15 categories (~5.3 GB)
    python scripts/download_data.py --root ./data/mvtec_ad

    # Just a couple of categories (much smaller, good for a quick run)
    python scripts/download_data.py --root ./data/mvtec_ad --categories capsule bottle

On Kaggle you can instead add the "MVTec AD" dataset and skip this entirely by
passing ``--root /kaggle/input/mvtec-ad`` to the benchmark scripts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

HF_REPO = "TheoM55/mvtec_all_objects_split"

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
    "carpet", "grid", "leather", "tile", "wood",
]

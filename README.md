# Unsupervised Industrial Anomaly Detection & Localization

Detect and localize **manufacturing defects from defect-free images only** — no
labelled defects, no fine-tuning, no GPU training required. This repository
implements and benchmarks the two methods that define the state of the art in
training-free industrial inspection — **PatchCore** and **PaDiM** — plus a small
trained autoencoder baseline for contrast, across the **15 categories of MVTec AD**.
Everything runs on free compute (Google Colab T4 / Kaggle GPU).

<p align="center">
  <img src="docs/pipeline.svg" alt="Training-free anomaly detection pipeline" width="92%"/>
</p>

---

## Highlights

- **Two training-free detectors + a baseline.** PatchCore (coreset memory bank) and
  PaDiM (per-patch Gaussian) on a frozen ImageNet backbone — no gradient training —
  plus a conv autoencoder to show what the frozen-backbone methods buy you.
- **Detection *and* localization.** Image-level score (defective yes/no) and a
  pixel-level heatmap of *where* the defect is.
- **Honest, reproducible evaluation.** Image AUROC, pixel AUROC, and the PRO score,
  with fixed seeds and one-command runs.
- **Verified on a free Colab T4.** PatchCore reproduces a **0.986 mean image AUROC /
  0.977 pixel AUROC across all 15 categories** end-to-end ([details below](#results)).
- **Clean, typed, tested code.** `src`-layout package, YAML-configured experiments,
  unit tests that need neither the dataset nor a GPU.

---

## Getting started

After cloning, you can be running a benchmark in three commands:

```bash
git clone https://github.com/SHARANcanCODE/industrial-anomaly-detection.git
cd industrial-anomaly-detection
pip install -e .                                   # 1. install the `iad` package

python scripts/download_data.py --root ./data/mvtec_ad   # 2. get the dataset (see "Dataset")

python scripts/run_benchmark.py \                  # 3. run PatchCore on all 15 categories
    --config configs/patchcore.yaml --root ./data/mvtec_ad
```

No GPU or dataset? Verify the install on CPU in seconds:

```bash
pip install -e ".[dev]" && pytest                  # unit tests use synthetic data only
```

Prefer zero local setup? Open

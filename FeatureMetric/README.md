# FeatureMetric — Realism Metric from Self-Supervised Encoders

Subgroup work for *Metric for Realism of Atmospheric Fields* (ETH PMLR 2026).

This folder implements a **latent-space realism metric** for atmospheric fields. Two
self-supervised encoders — **MAE** and **I-JEPA** — are trained on ERA5 reanalysis and
evaluated against a suite of physically-motivated corruptions and against ML forecasts
(Pangu, GraphCast). It is the encoder/embedding counterpart to the `Discriminator/`
subgroup's adversarial approach.

> Environment setup, ERA5 download, and the shared cluster storage layout are documented
> in the repository's **root `README.md`**. This README covers only the encoder work.

## Layout

```
FeatureMetric/
├── utils/        models, dataset, corruptions, masking, feature extraction, temporal compose
├── train/        train_mae.py, train_ijepa.py, train_twins.py
├── eval/         eval_probe.py, eval_distances.py, eval_real_vs_forecast.py, eval_embedding_probe.py, ...
├── scripts/      poster/plot utilities + forecast download
├── tests/        dataset exploration
├── shelved/      embedding-adapter discriminator experiment (reference only, not wired in)
├── data/         (git-ignored) local NetCDF datasets
└── checkpoints/  (git-ignored) model weights + data_mean.npy / data_std.npy
```

## Quick start (local)

Use the explicit interpreter — never `conda activate`. All commands run **from this
folder** (`FeatureMetric/`), which is where `data/` and `checkpoints/` live.

```bash
P=/opt/miniconda3/envs/pmlr/bin/python

# Training (local quick runs)
$P train/train_mae.py   --local --epochs 2 --batch-size 16
$P train/train_ijepa.py --local --smoke-test --smoke-samples 64

# Temporal variants: --temporal-mode {none,diff,concat,phase}
$P train/train_mae.py   --local --epochs 2 --batch-size 16 --temporal-mode concat
```

Pass `--large-local` for the 5-year local dataset, or omit local flags on the cluster.
Checkpoints and stats (`best_mae_model*.pth`, `best_ijepa_model*.pth`, `data_mean.npy`,
`data_std.npy`) are written to `checkpoints/`.

## Evaluation

Both eval protocols support `--model {mae,ijepa}` and `--local` / `--large-local` / cluster.

```bash
# Protocol 1 — linear probe: regress corruption severity from frozen latents (reports R²)
$P eval/eval_probe.py     --model mae --local

# Protocol 2 — Fréchet & MMD distance vs. severity ladder
$P eval/eval_distances.py --model mae --local

# Real ERA5 vs. ML forecast separation
$P eval/eval_real_vs_forecast.py --model mae --local
```

## Corruptions

`utils/corruptions.py` defines six corruption types (severity ∈ [0, 1]):

| Corruption | Description | Severity → parameter |
|---|---|---|
| Gaussian Blur | Spatial smoothing | σ ∈ [0, 1.125] |
| High-Freq Noise | Per-pixel iid Gaussian noise | std ∈ [0, 0.25] |
| GRF Noise | Spatially correlated Gaussian Random Field noise | std ∈ [0, 0.375] |
| Random Pixel Replace | Replaces random pixels with Gaussian samples | prob ∈ [0, 0.3] |
| Spatial Shuffle (wind only) | Shuffles wind patches; T2M/MSL unchanged | shuffled fraction ∈ [0, 1] |
| Channel Rotation (wind only) | Rotates wind vectors; T2M/MSL unchanged | angle ∈ [0°, 90°] |

```bash
$P eval/visualize_corruptions.py
```

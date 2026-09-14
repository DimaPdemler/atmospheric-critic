# Atmospheric Field Discriminator

This directory contains the discriminator pipeline for assigning realism scores
to atmospheric fields. A binary classifier is trained on ERA5 (real) versus
forecasts and synthetic corruptions (fake). The raw logit serves as the realism
score.

## Directory Structure

- `conf/` — Hydra configuration files.
  - `baseline_pipeline.yaml` — pipeline orchestration (stages, matching, W&B,
    temporal resampling).
  - `config.yaml` — default data paths, variables, train/test ranges, model
    choice, augmentation, and logging.
  - `baseline_config.yaml` — baseline metrics, corruptions, and plotting.
  - `target_discriminator_baselines.yaml` — target-specific discriminator
    settings (SqueezeNet, interpretability, severity).
- `scripts/` — pipeline, training, evaluation, and plotting code.
  - `run_baseline_pipeline.py` — pipeline orchestrator, runs stages in
    dependency order.
  - `train_discriminator.py` — trains one ResNet18 or SqueezeNet discriminator.
  - `train_target_discriminator_baselines.py` — target discriminator training,
    IG interpretability, logit histograms.
  - `evaluate_standard_metric_baselines.py` — standard metric computation
    (SCWD, MMD, Wasserstein, etc.).
  - `evaluate_discriminator_metric_baselines.py` — discriminator metric
    evaluation across severity/lead-time sweeps.
  - `corruptions.py` — tensor corruptions for training and evaluation.
  - `plot_standard_metric_baselines.py` — metric plotting.
  - `plot_bundles.py` — figure bundle infrastructure (PNG+PDF+NPZ).
  - `temporal_resampling_pipeline.py` — temporal resampling wrapper.
  - `evaluate_discriminator.py` — test accuracy/loss and spatial logit maps.
  - `analysis_utils.py` — shared inference helpers.
- `tests/` — unit and integration tests.

## Training

Train the default discriminator:

```bash
python scripts/train_discriminator.py
```

Use a custom data directory:

```bash
DATA_DIR=/path/to/netcdf python scripts/train_discriminator.py
```

Train a single-field discriminator:

```bash
python scripts/train_discriminator.py variables=null selected_variable=2m_temperature
```

Choose a backbone:

```bash
python scripts/train_discriminator.py model_name=resnet18
python scripts/train_discriminator.py model_name=squeezenet
```

## Baseline Pipeline

`scripts/run_baseline_pipeline.py` is the canonical entry point. It executes
selected stages within one pipeline-run directory:

1. Optional histogram or moment matching fitting.
2. Target-specific discriminator training and held-out testing.
3. Standard metric evaluation.
4. Discriminator metric evaluation.
5. Plotting.

Run all stages:

```bash
python scripts/run_baseline_pipeline.py \
  "pipeline.stages=[train_discriminators,evaluate_standard_metrics,evaluate_discriminator_metrics,plot]"
```

Reuse checkpoints for evaluation only:

```bash
python scripts/run_baseline_pipeline.py \
  "pipeline.stages=[evaluate_discriminator_metrics,plot]" \
  pipeline.input_checkpoint_dir=/path/to/models/target_discriminators
```

Disable temporal resampling for single-split debugging:

```bash
python scripts/run_baseline_pipeline.py temporal_resampling.enabled=false
```

### Output Structure

Each pipeline run writes to an immutable directory:

```
<pipeline-id>/
├── resolved_config.yaml
├── manifest.json
└── <variable-tag>/
    ├── data/           (CSV and NPZ metric draws, resample status)
    ├── models/         (discriminator checkpoints per fold)
    └── plots/          (PNG/NPZ bundles, optional PDF)
```

### Standard Metrics

The baseline pipeline compares unpaired forecast or corrupted-field
distributions with ERA5. Available metrics include mean bias, std ratio error,
field energy distance, zonal-spectrum L2, sliced Wasserstein/Cramer-Wold, RBF
MMD, global-mean Wasserstein, and SCWD.

Temporal resampling is enabled by default. Learned metrics train five
independent critics with seven-day test windows. Fixed metrics reuse those folds
and add 45 deterministic schedules. Standard curves show the mean and 5th-95th
percentiles over temporal resamples.

### Corruptions

The default corruption suite includes Gaussian blur, high-frequency noise, GRF,
pixel replacement, wind patch shuffling and rotation, checkerboard, scanlines,
hemisphere splice, and field splice. Corruption critics draw uniformly from
the discrete nonzero severity grid used at test time.

### Matching

Histogram matching and moment matching are both disabled by default and mutually
exclusive. When enabled, they fit dataset-level scalar transforms on training
data only, applied before standardization.

## Logging

Training does not require W&B by default:

- `logger=csv` — local CSV logs (default).
- `logger=wandb` — Weights & Biases logging.
- `logger=none` — disable logging.

## Important Options

- `data_dir` — base directory for NetCDF files (defaults to `DATA_DIR` env var).
- `variables` — list of input channels.
- `model_name` — `squeezenet` or `resnet18`.
- `batch_size`, `epochs`, `learning_rate` — training controls.
- `output_dir` — target directory for weights, plots, and logs.

## Analyses

### Learned-Metric Blind-Spot Search

`scripts/train_adversarial_corruption.py` searches for coherent temperature
perturbations within the null variability of SCWD and zonal log-spectrum
distance. It trains a coarse residual U-Net cooperatively with a SqueezeNet
discriminator at fixed normalized RMS values.

### Finite-SCWD Null-Space Case Study

`scripts/plot_scwd_null_space_case_study.py` projects a deterministic
checkerboard pattern into the null space of the finite Wendland response matrix,
demonstrating a sampling-based blind spot of SCWD.

### Temporal Logit Map Case Study

`scripts/plot_temporal_logit_map_case_study.py` ranks held-out forecasts by
logit, selects representative examples, and plots forecast, ERA5, and spatial
logit maps.

## Notes

- The default config expects data under `./data` unless `DATA_DIR` is set.
- Train/test time splits are validated before training.
- Logits are not calibrated probabilities — use them for relative comparisons
  under the same discriminator and normalization.

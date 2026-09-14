# Metric for Realism of Atmospheric Fields

ETH Zurich, 2026.

## Overview

This project develops a quantitative realism metric for atmospheric fields: a score that
captures how physically plausible a surface weather field is. The metric is trained on
[ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) reanalysis, treated
as the reference "real" distribution, and validated against machine-learning weather
forecasts (Pangu-Weather, GraphCast, FuXi, and others) as well as a suite of
physically-motivated synthetic corruptions.

The canonical experiments, including the baseline pipeline, use the same four 1.5-degree
surface fields by default: `2m_temperature`, `10m_u_component_of_wind`,
`10m_v_component_of_wind`, and `mean_sea_level_pressure`.

## Approach

A binary classifier (the *discriminator*) is trained to separate real ERA5 fields
(label 1) from fakes (label 0), where fakes consist of ML weather forecasts and
synthetically corrupted fields. The raw classifier logit serves as the realism score.

- Backbones: ResNet18 and SqueezeNet (torchvision, ImageNet initialization, adapted for
  the weather-channel input).
- Hydra-based configuration, with CSV or Weights and Biases logging.
- Analyses: logit versus forecast lead time, logit versus corruption severity,
  leave-one-model-out k-fold for numerical-model comparisons, and paper figures.

Refer to [`Discriminator/README.md`](Discriminator/README.md) for the full pipeline
and configuration options.

## Repository layout

```
.
├── Discriminator/     discriminator pipeline (training, evaluation, plotting)
├── download/          data-download utilities (ERA5 and forecasts from WeatherBench2)
├── ucast/             UCast forecast postprocessing and evaluation
├── FeatureMetric/     SWIFT data conversion utilities
├── environment.yml
├── .gitignore
└── README.md
```

Generated and large artifacts (`data/`, `checkpoints/`, `results/`, `plots/`, `wandb/`, and
`*.out`) are git-ignored and not committed.

## Data

All fields are obtained from the [WeatherBench2](https://weatherbench2.readthedocs.io/) Google
Cloud buckets at 1.5-degree resolution and 6-hourly cadence. Download utilities are in
[`download/`](download/), e.g.
[`download/download_era5_netcdf.py`](download/download_era5_netcdf.py) downloads an ERA5 or
forecast variable and time slice from a WeatherBench2 zarr store to NetCDF.

The ERA5 dataset used in this work covers 1.5-degree resolution, 2004 to 2023, for the four
surface fields listed above.

## Setup

Create the conda environment from the provided file:

```bash
conda env create -f environment.yml
conda activate atmospheric-fields
```

See [`Discriminator/README.md`](Discriminator/README.md) for pipeline-specific usage.

## Getting started

| Task | Location |
|------|----------|
| Train or evaluate the discriminator | [`Discriminator/README.md`](Discriminator/README.md) |
| Download ERA5 or forecast data | [`download/`](download/) |

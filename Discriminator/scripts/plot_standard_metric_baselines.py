"""Distributional spatial-realism baselines for forecast/corruption experiments.

The discriminator plots score whether samples look real, not whether individual
forecasts match paired ERA5 fields. This script therefore compares candidate
sample distributions against an ERA5 reference distribution. It deliberately
does not use per-sample forecast/ERA5 pairs. The configured variables are
standardized with ERA5 moments and evaluated jointly; configure a single
variable when field-wise metrics are desired.

Metrics:
- `mean_bias`: unweighted candidate grid-cell mean minus ERA5 mean.
- `std_ratio_error`: unweighted candidate grid-cell standard deviation divided by
  ERA5 standard deviation, minus one.
- `crps_like_field_energy`: half-energy distance between complete multi-field
  states, using cosine-latitude-weighted mean absolute differences.
- `zonal_energy_spectrum_l2`: relative L2 distance between mean zonal spectra.
- `sliced_wasserstein`: sliced Wasserstein distance between unweighted flattened
  spatial field distributions.
- `sliced_wasserstein_lon_corrected`: same metric after applying the spherical
  surface-Jacobian correction for lat-lon cell areas.
- `global_mean_wasserstein`: Vissio et al. joint quadratic Wasserstein distance
  between distributions of cosine-area-weighted global-mean vectors.
- `mmd_rbf`: maximum mean discrepancy between flattened spatial-field
  distributions, using a Gaussian RBF kernel with a median-distance bandwidth.
- `scwd`: spherical convolutional Wasserstein distance approximation. Scalar
  fields use the compact-Wendland quantile approximation from Garrett et al.
  (2024); multi-field states use exact empirical joint W2 at every anchor.
- `scwd_area_weighted`: same SCWD responses, with cosine-area weighting when
  combining regular latitude-longitude anchor costs.
"""

import csv
import gzip
import json
import math
from fractions import Fraction
import tempfile
import textwrap
import zlib
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import hydra
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.spatial.distance
import torch
import xarray as xr
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


# Make independently labelled curves distinguishable without relying on colour.
SERIES_MARKERS = ("o", "s", "^", "v", "D", "P", "X", "<", ">", "h")


def series_marker(index):
    return SERIES_MARKERS[int(index) % len(SERIES_MARKERS)]

try:
    from .plot_bundles import (
        CANDIDATE_COLOR, CRITIC_COLOR, PLOT_PALETTE, REFERENCE_COLOR, categorical_colors,
        configure_plot_bundle_saving_from_cfg, displayed_model_name, lead_time_colors, model_colors, save_figure_bundle,
    )
    from .fake_matching_apply import match_standardized
    from .fake_matching_checkpoint import validate_binding
    from .temporal_resampling import active_schedule, settings, file_sha256, write_csv_gz, write_split_membership
    from .corruptions import U10_CHANNEL, V10_CHANNEL
    from .train_discriminator import (
        WeatherDiscriminator,
        apply_configured_corruption,
        normalize_prediction_timedelta,
        safe_open_dataset,
        select_time_ranges,
    )
    from .monthly_split import (
        concatenate_forecasts,
        coverage_metadata,
        evenly_spaced_pairs,
        forecast_pairs,
        select_era5_split,
    )
except ImportError:
    from plot_bundles import (
        CANDIDATE_COLOR, CRITIC_COLOR, PLOT_PALETTE, REFERENCE_COLOR, categorical_colors,
        configure_plot_bundle_saving_from_cfg, displayed_model_name, lead_time_colors, model_colors, save_figure_bundle,
    )
    from fake_matching_apply import match_standardized
    from fake_matching_checkpoint import validate_binding
    from temporal_resampling import active_schedule, settings, file_sha256, write_csv_gz, write_split_membership
    from corruptions import U10_CHANNEL, V10_CHANNEL
    from train_discriminator import (
        WeatherDiscriminator,
        apply_configured_corruption,
        normalize_prediction_timedelta,
        safe_open_dataset,
        select_time_ranges,
    )
    from monthly_split import (
        concatenate_forecasts,
        coverage_metadata,
        evenly_spaced_pairs,
        forecast_pairs,
        select_era5_split,
    )


DEFAULT_METRICS = [
    "mean_bias",
    "std_ratio_error",
    "crps_like_field_energy",
    "zonal_energy_spectrum_l2",
    "zonal_energy_spectrum_log_l2",
    "sliced_wasserstein",
    "sliced_wasserstein_lon_corrected",
    "global_mean_wasserstein",
    "mmd_rbf",
    "scwd",
]

ERA5_NULL_LABEL = "ERA5 test-vs-train null"
PLOTTING_DISABLED_METRICS = {
    "crps_like_field_energy",
    "sliced_wasserstein",
    "sliced_wasserstein_lon_corrected",
    "zonal_energy_spectrum_l2",
    # Keep the unweighted SCWD as the reported SCWD variant. The
    # area-weighted alternative may still be evaluated for analysis, but it
    # must not appear in the standard plotting/W&B gallery.
    "scwd_area_weighted",
}


# These diagnostics remain evaluated and get their own figures, but combining
# their lower-moment scales with distributional distances obscures the latter.
MULTIPLOT_EXCLUDED_METRICS = {
    "mean_bias",
    "std_ratio_error",
}

def plotted_metric_names(metric_names):
    """Return metrics suitable for figures containing several metrics."""
    return [
        name for name in metric_names
        if name not in PLOTTING_DISABLED_METRICS | MULTIPLOT_EXCLUDED_METRICS
    ]


def standalone_metric_names(metric_names):
    """Return metrics that retain their dedicated, single-metric figures."""
    return [name for name in metric_names if name not in PLOTTING_DISABLED_METRICS]


AVAILABLE_METRICS = {
    "mean_bias",
    "mean_abs_diff",
    "std_ratio_error",
    "std_abs_diff",
    "crps_like_field_energy",
    "zonal_energy_spectrum_l2",
    "zonal_energy_spectrum_log_l2",
    "sliced_wasserstein",
    "sliced_wasserstein_lon_corrected",
    "global_mean_wasserstein",
    "sliced_cramer_wold",
    "sliced_cramer_wold_lon_corrected",
    "mmd_rbf",
    "scwd_area_weighted",
    "scwd",
}

_SCWD_WEIGHT_CACHE = {}
_SCWD_TORCH_WEIGHT_CACHE = {}
_SCWD_NUMPY_WEIGHT_CACHE = {}

STRUCTURED_NEAR_NULL_CORRUPTIONS = {
    "equatorial_checker_texture",
    "meridional_scanlines",
    "checkerboard_2px",
    "zonal_scanlines",
}
DATA_DEPENDENT_CORRUPTIONS = {"hemisphere_splice", "field_splice"}


def cfg_get(cfg, key, default):
    """Return a config value, treating explicit YAML null as missing."""
    value = cfg.get(key)
    return default if value is None else value


def baseline_get(cfg, key, default=None):
    """Return a baseline setting, treating explicit YAML null as missing."""
    baseline = cfg.get("baseline")
    if baseline is None:
        raise ValueError("Missing required baseline configuration section.")
    value = baseline.get(key)
    return default if value is None else value


def torch_metric_device(cfg):
    """Return the configured Torch metric device, or None for NumPy metrics."""
    if cfg is None:
        return None
    backend = str(baseline_get(cfg, "backend", "auto")).lower()
    if backend in {"numpy", "np", "cpu"}:
        return None
    if backend not in {"auto", "torch", "cuda", "gpu"}:
        raise ValueError("baseline.backend must be one of: auto, torch, cuda, gpu, numpy")
    if backend == "auto" and not torch.cuda.is_available():
        return None
    configured_device = baseline_get(cfg, "device", None)
    if configured_device is None:
        configured_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(configured_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return None
    return device


def torch_metric_dtype(cfg):
    """Return the floating dtype for Torch metric kernels."""
    dtype_name = str(baseline_get(cfg, "dtype", "float32")).lower()
    if dtype_name in {"float64", "double"}:
        return torch.float64
    if dtype_name in {"float32", "single"}:
        return torch.float32
    raise ValueError("baseline.dtype must be float32 or float64")


def torch_tensor(values, device, dtype):
    """Move a NumPy-like array to the metric device."""
    return torch.as_tensor(np.asarray(values), dtype=dtype, device=device)


def variables_from_config(cfg):
    """Return fields configured for the distribution baseline."""
    variables = baseline_get(cfg, "variables", None)
    if not variables:
        raise ValueError("baseline.variables must contain at least one field.")
    return list(variables)


def metric_names_from_config(cfg):
    """Return requested distributional baseline metrics."""
    metric_names = list(baseline_get(cfg, "metrics", DEFAULT_METRICS))
    unknown = sorted(set(metric_names) - AVAILABLE_METRICS)
    if unknown:
        raise ValueError(f"Unknown baseline metric(s): {unknown}. Available metrics: {sorted(AVAILABLE_METRICS)}")
    return metric_names


def select_level(ds, level):
    """Select a pressure level when the dataset has one."""
    if level is None:
        return ds
    if "level" in ds.dims:
        return ds.sel(level=level)
    if "pressure_level" in ds.dims:
        return ds.sel(pressure_level=level)
    return ds


def lead_hours(ds):
    """Return forecast lead times as integer hours."""
    ds = normalize_prediction_timedelta(ds)
    if "prediction_timedelta" not in ds.coords:
        return np.array([0], dtype=int)
    return np.asarray(ds.prediction_timedelta.values).astype(int)


def sample_time_indices(ds, max_samples):
    """Return evenly spaced time indices for an xarray dataset."""
    n_time = ds.sizes.get("time", 0)
    if n_time == 0:
        return []
    if max_samples <= 0 or n_time <= max_samples:
        return list(range(n_time))
    return np.linspace(0, n_time - 1, max_samples, dtype=int).tolist()


def pairwise_sample_positions(n_samples, max_samples):
    """Return deterministic positions retained by quadratic metrics."""
    if n_samples <= 0:
        return []
    if max_samples <= 0 or n_samples <= max_samples:
        return list(range(n_samples))
    return np.linspace(0, n_samples - 1, max_samples, dtype=int).tolist()


def corruption_sample_seed(base_seed, corruption_type, sample_key):
    """Return a stable per-corruption, per-sample seed shared across severities."""
    corruption_hash = zlib.crc32(str(corruption_type).encode("utf-8"))
    return int((int(base_seed) + corruption_hash + int(sample_key)) % (2**31 - 1))


def latitude_weights(latitudes):
    """Return cosine-latitude weights normalized by their mean."""
    weights = np.cos(np.deg2rad(np.asarray(latitudes, dtype=np.float64)))
    return weights / np.nanmean(weights)


def latitude_weighted_moment_totals(values, latitudes):
    """Return surface-area-weighted sum, squared sum, and finite weight mass."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2] != len(latitudes):
        raise ValueError("Moment values must end in latitude-longitude dimensions.")
    weights = np.maximum(np.cos(np.deg2rad(np.asarray(latitudes, dtype=np.float64))), 0.0)
    shape = (1,) * (values.ndim - 2) + (len(weights), 1)
    weights = weights.reshape(shape)
    finite = np.isfinite(values)
    safe_values = np.where(finite, values, 0.0)
    weighted = weights * finite
    return (
        float(np.sum(safe_values * weighted, dtype=np.float64)),
        float(np.sum(safe_values * safe_values * weighted, dtype=np.float64)),
        float(np.sum(weighted, dtype=np.float64)),
    )



def pointwise_moment_totals(values):
    """Return unweighted sum, squared sum, and count of finite grid cells."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    safe_values = np.where(finite, values, 0.0)
    return (
        float(np.sum(safe_values, dtype=np.float64)),
        float(np.sum(safe_values * safe_values, dtype=np.float64)),
        float(np.sum(finite, dtype=np.float64)),
    )


def area_weighted_global_means(fields, latitudes):
    """Return one cosine-area-weighted global mean per sample and field."""
    values = np.asarray(fields, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("Global-mean fields must have shape sample x channel x latitude x longitude.")
    weights = np.maximum(np.cos(np.deg2rad(np.asarray(latitudes, dtype=np.float64))), 0.0)
    denominator = float(np.sum(weights) * values.shape[-1])
    if denominator <= 0.0:
        raise ValueError("Latitude weights have zero total area.")
    return np.sum(values * weights[None, None, :, None], axis=(-2, -1)) / denominator


def structured_near_null_pattern(corruption_type, latitudes, n_longitudes):
    """Return a mean-centred, unit-area-RMS spatial carrier for baseline probes."""
    n_latitudes = len(latitudes)
    rows, columns = np.indices((n_latitudes, int(n_longitudes)))
    if corruption_type == "equatorial_checker_texture":
        pattern = np.where((rows + columns) % 2, -1.0, 1.0)
        pattern *= 0.15 + 0.85 * np.exp(
            -(np.asarray(latitudes, dtype=np.float64)[:, None] / 25.0) ** 8
        )
    elif corruption_type == "meridional_scanlines":
        pattern = np.where(columns % 2, -1.0, 1.0)
    elif corruption_type == "checkerboard_2px":
        pattern = np.where(((rows // 2) + (columns // 2)) % 2, -1.0, 1.0)
    elif corruption_type == "zonal_scanlines":
        pattern = np.where(rows % 2, -1.0, 1.0)
    else:
        raise ValueError(f"Unknown structured near-null corruption: {corruption_type}.")
    weights = np.maximum(np.cos(np.deg2rad(np.asarray(latitudes))), 0.0)[:, None]
    denominator = float(np.sum(weights) * int(n_longitudes))
    if corruption_type == "zonal_scanlines":
        # Zonal scanlines are intended to be symmetric in the ordinary
        # grid-cell mean used by mean_bias, rather than in spherical area.
        pattern -= np.mean(pattern)
    else:
        pattern -= np.sum(pattern * weights) / denominator
    rms = float(np.sqrt(np.sum(pattern**2 * weights) / denominator))
    if rms <= 1e-12:
        raise ValueError(f"Structured corruption {corruption_type} has zero energy.")
    return (pattern / rms).astype(np.float32)


def deranged_sample_positions(size, seed):
    """Return a reproducible cyclic donor permutation without self-pairs."""
    size = int(size)
    if size < 2:
        raise ValueError("Data-dependent splice requires at least two samples.")
    rng = np.random.default_rng(int(seed))
    return np.roll(np.arange(size, dtype=int), int(rng.integers(1, size)))


def fieldwise_deranged_sample_positions(size, n_fields, seed):
    """Return one reproducible no-self-pair donor permutation per field."""
    return np.stack([
        deranged_sample_positions(size, int(np.random.SeedSequence([int(seed), field]).generate_state(1)[0]))
        for field in range(int(n_fields))
    ])


def apply_special_baseline_corruption(
    standardized,
    corruption_type,
    severity,
    latitudes,
    cfg,
    donor=None,
    maximum_severity=None,
    random_seed=None,
):
    """Apply structured or data-dependent corruptions in standardized space."""
    standardized = np.asarray(standardized, dtype=np.float32)
    severity = float(severity)
    if severity <= 0.0:
        return standardized
    if corruption_type in STRUCTURED_NEAR_NULL_CORRUPTIONS:
        pattern = structured_near_null_pattern(
            corruption_type, latitudes, standardized.shape[-1]
        )
        return standardized + severity * pattern[None, :, :]
    if corruption_type in {"hemisphere_splice", "field_splice"}:
        if donor is None:
            raise ValueError(f"{corruption_type} requires a donor ERA5 sample.")
        maximum = float(
            corruption_max_severity(corruption_type, cfg)
            if maximum_severity is None else maximum_severity
        )
        replace_probability = np.clip(severity / max(maximum, 1e-12), 0.0, 1.0)
        result = standardized.copy()
        rng = np.random.default_rng(random_seed) if random_seed is not None else np.random
        if corruption_type == "hemisphere_splice":
            boundary = float(baseline_get(cfg, "hemisphere_splice_latitude", 0.0))
            south = np.asarray(latitudes, dtype=np.float64) < boundary
            # One Bernoulli draw replaces the whole southern hemisphere.
            if rng.random() < replace_probability:
                result[:, south, :] = np.asarray(donor)[:, south, :]
        else:
            # Each field has an independently permuted donor and one Bernoulli
            # draw: no within-field pixel blending occurs.
            replace_fields = rng.random(standardized.shape[0]) < replace_probability
            result[replace_fields] = np.asarray(donor)[replace_fields]
        return result.astype(np.float32)
    raise ValueError(f"Unknown special baseline corruption: {corruption_type}.")


def weighted_quadratic_wasserstein_1d(support_a, mass_a, support_b, mass_b):
    """Compute exact one-dimensional W2 between two weighted discrete measures."""
    support_a = np.asarray(support_a, dtype=np.float64)
    support_b = np.asarray(support_b, dtype=np.float64)
    mass_a = np.asarray(mass_a, dtype=np.float64)
    mass_b = np.asarray(mass_b, dtype=np.float64)
    keep_a = mass_a > 0.0
    keep_b = mass_b > 0.0
    if not np.any(keep_a) or not np.any(keep_b):
        return np.nan
    support_a, mass_a = support_a[keep_a], mass_a[keep_a]
    support_b, mass_b = support_b[keep_b], mass_b[keep_b]
    mass_a = mass_a / np.sum(mass_a)
    mass_b = mass_b / np.sum(mass_b)
    cdf_a = np.cumsum(mass_a)
    cdf_b = np.cumsum(mass_b)
    boundaries = np.unique(np.concatenate(([0.0], cdf_a, cdf_b, [1.0])))
    widths = np.diff(boundaries)
    midpoints = boundaries[:-1] + 0.5 * widths
    idx_a = np.minimum(np.searchsorted(cdf_a, midpoints, side="right"), support_a.size - 1)
    idx_b = np.minimum(np.searchsorted(cdf_b, midpoints, side="right"), support_b.size - 1)
    return float(np.sqrt(np.sum(widths * (support_a[idx_a] - support_b[idx_b]) ** 2)))


def _finite_joint_samples(values, n_fields=None):
    """Return finite rows from a scalar or joint sample array."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        return np.empty((0, 0), dtype=np.float64)
    if n_fields is not None:
        values = values[:, :int(n_fields)]
    return values[np.all(np.isfinite(values), axis=1)]


def fit_vissio_ulam_grid(distributions, n_bins=20):
    """Fit one experiment-wide, per-field Ulam grid to pooled distributions."""
    arrays = [_finite_joint_samples(values) for values in distributions]
    arrays = [values for values in arrays if values.size]
    if not arrays:
        raise ValueError("Cannot fit a Vissio Ulam grid without finite samples.")
    n_fields = arrays[0].shape[1]
    if any(values.shape[1] != n_fields for values in arrays):
        raise ValueError("All Ulam-grid distributions must have the same field count.")
    if int(n_bins) < 2:
        raise ValueError("The Vissio Ulam grid requires at least two bins per field.")
    pooled = np.concatenate(arrays, axis=0)
    return {
        "lower": np.min(pooled, axis=0),
        "upper": np.max(pooled, axis=0),
        "n_bins": int(n_bins),
    }


def vissio_ulam_measure(values, grid):
    """Discretize joint samples into occupied normalized Ulam cells."""
    lower = np.asarray(grid["lower"], dtype=np.float64)
    upper = np.asarray(grid["upper"], dtype=np.float64)
    n_bins = int(grid["n_bins"])
    values = _finite_joint_samples(values, lower.size)
    if not values.size:
        return np.empty((0, lower.size)), np.empty(0), np.empty((0, lower.size), dtype=np.int64)
    scale = upper - lower
    varying = scale > 1e-12
    normalized = np.zeros_like(values)
    normalized[:, varying] = (values[:, varying] - lower[varying]) / scale[varying]
    normalized = np.clip(normalized, 0.0, np.nextafter(1.0, 0.0))
    cells = np.floor(normalized * n_bins).astype(np.int64)
    cells[:, ~varying] = 0
    occupied, counts = np.unique(cells, axis=0, return_counts=True)
    support = (occupied.astype(np.float64) + 0.5) / n_bins
    support[:, ~varying] = 0.5
    return support, counts.astype(np.float64) / counts.sum(), occupied


def weighted_quadratic_wasserstein_nd(support_a, mass_a, support_b, mass_b):
    """Compute exact discrete multivariate W2 with Euclidean ground geometry."""
    support_a = np.asarray(support_a, dtype=np.float64)
    support_b = np.asarray(support_b, dtype=np.float64)
    mass_a = np.asarray(mass_a, dtype=np.float64)
    mass_b = np.asarray(mass_b, dtype=np.float64)
    keep_a, keep_b = mass_a > 0.0, mass_b > 0.0
    support_a, mass_a = support_a[keep_a], mass_a[keep_a]
    support_b, mass_b = support_b[keep_b], mass_b[keep_b]
    if not support_a.size or not support_b.size:
        return np.nan
    mass_a, mass_b = mass_a / mass_a.sum(), mass_b / mass_b.sum()
    cost = scipy.spatial.distance.cdist(support_a, support_b, metric="sqeuclidean")

    def integer_masses(masses):
        fractions = [Fraction(float(value)).limit_denominator(100000) for value in masses]
        denominator = math.lcm(*(value.denominator for value in fractions))
        counts = np.asarray([value.numerator * (denominator // value.denominator) for value in fractions])
        divisor = np.gcd.reduce(counts)
        return counts // max(int(divisor), 1)

    counts_a, counts_b = integer_masses(mass_a), integer_masses(mass_b)
    common_total = math.lcm(int(counts_a.sum()), int(counts_b.sum()))
    expanded_count = common_total
    if expanded_count <= 4096:
        counts_a = counts_a * (common_total // int(counts_a.sum()))
        counts_b = counts_b * (common_total // int(counts_b.sum()))
        expanded_a = np.repeat(support_a, counts_a, axis=0)
        expanded_b = np.repeat(support_b, counts_b, axis=0)
        expanded_cost = scipy.spatial.distance.cdist(expanded_a, expanded_b, metric="sqeuclidean")
        rows, cols = scipy.optimize.linear_sum_assignment(expanded_cost)
        return float(np.sqrt(np.mean(expanded_cost[rows, cols])))

    n_a, n_b = len(mass_a), len(mass_b)
    row_indices = np.repeat(np.arange(n_a), n_b)
    col_indices = np.tile(np.arange(n_b), n_a)
    variable_indices = np.arange(n_a * n_b)
    constraints = scipy.sparse.coo_matrix(
        (np.ones(2 * n_a * n_b),
         (np.concatenate([row_indices, n_a + col_indices]),
          np.concatenate([variable_indices, variable_indices]))),
        shape=(n_a + n_b, n_a * n_b),
    ).tocsr()
    result = scipy.optimize.linprog(
        cost.ravel(), A_eq=constraints, b_eq=np.concatenate([mass_a, mass_b]),
        bounds=(0.0, None), method="highs",
    )
    if not result.success:
        raise RuntimeError(f"Joint Ulam transport failed: {result.message}")
    return float(np.sqrt(max(float(result.fun), 0.0)))


def vissio_global_mean_wasserstein(candidate_means, reference_means, n_bins=20, grid=None):
    """Return Vissio-style joint Ulam-binned W2 of global-mean vectors."""
    candidate = _finite_joint_samples(candidate_means)
    reference = _finite_joint_samples(reference_means)
    if not candidate.size or not reference.size or candidate.shape[1] != reference.shape[1]:
        return np.nan
    if grid is None:
        grid = fit_vissio_ulam_grid([candidate, reference], n_bins=n_bins)
    support_a, mass_a, _ = vissio_ulam_measure(candidate, grid)
    support_b, mass_b, _ = vissio_ulam_measure(reference, grid)
    if candidate.shape[1] == 1:
        return weighted_quadratic_wasserstein_1d(
            support_a[:, 0], mass_a, support_b[:, 0], mass_b
        )
    return weighted_quadratic_wasserstein_nd(support_a, mass_a, support_b, mass_b)


def zonal_energy_spectrum(values, latitudes):
    """Latitude-weighted zonal power spectrum for one 2D field."""
    values = np.nan_to_num(np.asarray(values, dtype=np.float64))
    n_lat = len(latitudes)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D latitude-longitude field, got shape={values.shape}")
    if values.shape[0] != n_lat and values.shape[1] == n_lat:
        values = values.T
    if values.shape[0] != n_lat:
        raise ValueError(
            "Latitude dimension does not match field shape for zonal spectrum: "
            f"field={values.shape}, n_lat={n_lat}"
        )
    fft_values = np.fft.rfft(values, axis=-1)
    power = np.abs(fft_values) ** 2
    return np.average(power, axis=0, weights=latitude_weights(latitudes))


def zonal_energy_spectrum_log_l2(candidate_spectrum, reference_spectrum, cfg):
    """Root mean squared difference between log-power spectra."""
    common_spectrum_len = min(candidate_spectrum.size, reference_spectrum.size)
    if not common_spectrum_len:
        return np.nan
    candidate_values = np.maximum(np.asarray(candidate_spectrum[:common_spectrum_len], dtype=np.float64), 0.0)
    reference_values = np.maximum(np.asarray(reference_spectrum[:common_spectrum_len], dtype=np.float64), 0.0)
    finite_reference = reference_values[np.isfinite(reference_values)]
    positive_reference = finite_reference[finite_reference > 0.0]
    reference_scale = float(np.max(positive_reference)) if positive_reference.size else 1.0
    eps_factor = float(baseline_get(cfg, "spectrum_log_eps_factor", 1e-12))
    eps = max(reference_scale * eps_factor, np.finfo(np.float64).tiny)
    diff = np.log(candidate_values + eps) - np.log(reference_values + eps)
    return float(np.sqrt(np.nanmean(diff * diff))) if diff.size else np.nan


def canonical_latlon(values, latitudes):
    """Return a 2D field in latitude-longitude order."""
    values = np.nan_to_num(np.asarray(values, dtype=np.float64))
    n_lat = len(latitudes)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D latitude-longitude field, got shape={values.shape}")
    if values.shape[0] != n_lat and values.shape[1] == n_lat:
        return values.T
    if values.shape[0] != n_lat:
        raise ValueError(f"Could not identify latitude axis: field={values.shape}, n_lat={n_lat}")
    return values


def invalid_field_reason(values, *, zero_atol=1e-12, min_std=1e-12):
    """Return why a field should be skipped, or None if it looks usable."""
    raw_values = np.asarray(values, dtype=np.float64)
    finite = raw_values[np.isfinite(raw_values)]
    if finite.size == 0:
        return "all_nonfinite"
    if np.all(np.abs(finite) <= zero_atol):
        return "all_zero"
    if float(np.nanstd(finite)) <= min_std:
        return "near_constant"
    return None


def should_skip_field(cfg, values):
    """Return true when invalid-field filtering should drop this field."""
    if not bool(baseline_get(cfg, "filter_invalid_fields", True)):
        return False
    zero_atol = float(baseline_get(cfg, "invalid_zero_atol", 1e-12))
    min_std = float(baseline_get(cfg, "invalid_min_std", 1e-12))
    return invalid_field_reason(values, zero_atol=zero_atol, min_std=min_std) is not None


def field_vector(values, latitudes, max_pixels):
    """Flatten one field with cosine-latitude area weighting."""
    values = canonical_latlon(values, latitudes)
    weighted = values * np.sqrt(latitude_weights(latitudes))[:, None]
    flat = weighted.ravel()
    if max_pixels <= 0 or flat.size <= max_pixels:
        return flat
    indices = np.linspace(0, flat.size - 1, max_pixels, dtype=int)
    return flat[indices]


def unweighted_field_vector(values, latitudes, max_pixels):
    """Flatten one field without surface-area correction."""
    values = canonical_latlon(values, latitudes)
    flat = values.ravel()
    if max_pixels <= 0 or flat.size <= max_pixels:
        return flat
    indices = np.linspace(0, flat.size - 1, max_pixels, dtype=int)
    return flat[indices]


def sliced_wasserstein_distance(candidate_vectors, reference_vectors, n_projections, seed, cfg=None):
    """Approximate Wasserstein distance with random 1D projections."""
    device = torch_metric_device(cfg)
    if device is not None:
        return sliced_wasserstein_distance_torch(candidate_vectors, reference_vectors, n_projections, seed, cfg, device)

    candidate = np.asarray(candidate_vectors, dtype=np.float64)
    reference = np.asarray(reference_vectors, dtype=np.float64)
    if candidate.size == 0 or reference.size == 0:
        return np.nan
    if candidate.ndim != 2 or reference.ndim != 2:
        return np.nan
    if candidate.shape[1] != reference.shape[1]:
        n_dim = min(candidate.shape[1], reference.shape[1])
        candidate = candidate[:, :n_dim]
        reference = reference[:, :n_dim]

    rng = np.random.default_rng(seed)
    distances = []
    n_quantiles = min(candidate.shape[0], reference.shape[0])
    for _ in range(n_projections):
        direction = rng.normal(size=candidate.shape[1])
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            continue
        direction /= norm
        candidate_proj = np.sort(candidate @ direction)
        reference_proj = np.sort(reference @ direction)
        candidate_q = np.linspace(0, candidate_proj.size - 1, n_quantiles)
        reference_q = np.linspace(0, reference_proj.size - 1, n_quantiles)
        candidate_values = np.interp(candidate_q, np.arange(candidate_proj.size), candidate_proj)
        reference_values = np.interp(reference_q, np.arange(reference_proj.size), reference_proj)
        distances.append(float(np.mean(np.abs(candidate_values - reference_values))))
    return float(np.mean(distances)) if distances else np.nan


def sliced_wasserstein_distance_torch(candidate_vectors, reference_vectors, n_projections, seed, cfg, device):
    """Approximate sliced Wasserstein with batched Torch projections."""
    candidate = np.asarray(candidate_vectors, dtype=np.float64)
    reference = np.asarray(reference_vectors, dtype=np.float64)
    if candidate.size == 0 or reference.size == 0:
        return np.nan
    if candidate.ndim != 2 or reference.ndim != 2:
        return np.nan
    n_dim = min(candidate.shape[1], reference.shape[1])
    if n_dim == 0:
        return np.nan
    candidate = candidate[:, :n_dim]
    reference = reference[:, :n_dim]

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(int(n_projections), n_dim))
    norms = np.linalg.norm(directions, axis=1)
    keep = norms > 1e-12
    if not np.any(keep):
        return np.nan
    directions = directions[keep] / norms[keep, None]

    dtype = torch_metric_dtype(cfg)
    candidate_t = torch_tensor(candidate, device, dtype)
    reference_t = torch_tensor(reference, device, dtype)
    directions_t = torch_tensor(directions.T, device, dtype)
    quantiles = torch.linspace(
        0.0,
        1.0,
        min(candidate_t.shape[0], reference_t.shape[0]),
        dtype=dtype,
        device=device,
    )

    candidate_proj = candidate_t @ directions_t
    reference_proj = reference_t @ directions_t
    candidate_q = torch.quantile(candidate_proj, quantiles, dim=0)
    reference_q = torch.quantile(reference_proj, quantiles, dim=0)
    return float(torch.mean(torch.abs(candidate_q - reference_q)).detach().cpu())


def sliced_wasserstein_from_projections(candidate_projections, reference_projections):
    """Compute sliced Wasserstein from compact, precomputed field projections."""
    candidate = np.asarray(candidate_projections, dtype=np.float64)
    reference = np.asarray(reference_projections, dtype=np.float64)
    if candidate.size == 0 or reference.size == 0:
        return np.nan
    if candidate.ndim != 2 or reference.ndim != 2:
        return np.nan
    n_projections = min(candidate.shape[1], reference.shape[1])
    n_quantiles = min(candidate.shape[0], reference.shape[0])
    if n_projections == 0 or n_quantiles == 0:
        return np.nan
    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    candidate_q = np.quantile(candidate[:, :n_projections], quantiles, axis=0)
    reference_q = np.quantile(reference[:, :n_projections], quantiles, axis=0)
    return float(np.mean(np.abs(candidate_q - reference_q)))


def aligned_vector_sets(candidate_vectors, reference_vectors):
    """Return candidate/reference vector matrices with matching dimensions."""
    candidate = np.asarray(candidate_vectors, dtype=np.float64)
    reference = np.asarray(reference_vectors, dtype=np.float64)
    if candidate.size == 0 or reference.size == 0:
        return None, None
    if candidate.ndim != 2 or reference.ndim != 2:
        return None, None
    n_dim = min(candidate.shape[1], reference.shape[1])
    if n_dim == 0:
        return None, None
    return candidate[:, :n_dim], reference[:, :n_dim]


def standardize_for_mmd(candidate_vectors, reference_vectors, cfg):
    """Standardize MMD vectors using reference-distribution moments."""
    candidate, reference = aligned_vector_sets(candidate_vectors, reference_vectors)
    if candidate is None:
        return None, None
    if not bool(baseline_get(cfg, "mmd_standardize", True)):
        return candidate, reference

    mean = np.nanmean(reference, axis=0)
    std = np.nanstd(reference, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (candidate - mean) / std, (reference - mean) / std


def squared_euclidean_distances(left, right):
    """Compute pairwise squared Euclidean distances without materializing diffs."""
    left_norm = np.sum(left * left, axis=1)[:, None]
    right_norm = np.sum(right * right, axis=1)[None, :]
    distances = left_norm + right_norm - 2.0 * (left @ right.T)
    return np.maximum(distances, 0.0)


def mmd_rbf_bandwidth(reference, configured_bandwidth):
    """Fit an RBF bandwidth on the reference distribution alone."""
    if configured_bandwidth is not None and float(configured_bandwidth) > 0:
        return float(configured_bandwidth)
    distances = squared_euclidean_distances(reference, reference)
    positive = distances[distances > 1e-12]
    if positive.size == 0:
        return 1.0
    return float(np.sqrt(np.median(positive)))


def normalized_mmd_field_slices(n_dimensions, field_slices):
    """Validate contiguous field blocks covering one concatenated MMD vector."""
    if field_slices is None:
        return [(0, int(n_dimensions))]
    slices = [(int(start), int(stop)) for start, stop in field_slices]
    expected_start = 0
    for start, stop in slices:
        if start != expected_start or stop <= start or stop > n_dimensions:
            raise ValueError(
                "MMD field slices must be positive, contiguous blocks covering the vector; "
                f"got {slices} for dimension {n_dimensions}."
            )
        expected_start = stop
    if not slices or expected_start != n_dimensions:
        raise ValueError(
            f"MMD field slices cover {expected_start} of {n_dimensions} vector coordinates."
        )
    return slices


def mmd_rbf_bandwidths(reference, field_slices, configured_bandwidth=None):
    """Fit one ERA5-reference median-heuristic bandwidth per field block."""
    slices = normalized_mmd_field_slices(reference.shape[1], field_slices)
    if configured_bandwidth is None:
        configured = [None] * len(slices)
    elif np.isscalar(configured_bandwidth):
        configured = [float(configured_bandwidth)] * len(slices)
    else:
        configured = list(configured_bandwidth)
        if len(configured) != len(slices):
            raise ValueError(
                f"Expected {len(slices)} configured MMD bandwidths, got {len(configured)}."
            )
    return np.asarray(
        [
            mmd_rbf_bandwidth(reference[:, start:stop], bandwidth)
            for (start, stop), bandwidth in zip(slices, configured)
        ],
        dtype=np.float64,
    )


def rbf_kernel_mean(left, right, bandwidth):
    """Mean Gaussian RBF kernel value between two vector samples."""
    distances = squared_euclidean_distances(left, right)
    scale = 2.0 * max(float(bandwidth), 1e-12) ** 2
    return float(np.mean(np.exp(-distances / scale)))


def rbf_kernel_mean_per_field(left, right, bandwidths, field_slices):
    """Mean joint product-RBF value with a separate scale for each field."""
    slices = normalized_mmd_field_slices(left.shape[1], field_slices)
    if len(bandwidths) != len(slices):
        raise ValueError(f"Expected {len(slices)} MMD bandwidths, got {len(bandwidths)}.")
    scaled_distance = np.zeros((left.shape[0], right.shape[0]), dtype=np.float64)
    for (start, stop), bandwidth in zip(slices, bandwidths):
        scale = max(float(bandwidth), 1e-12) ** 2
        scaled_distance += squared_euclidean_distances(
            left[:, start:stop], right[:, start:stop]
        ) / scale
    scaled_distance /= len(slices)
    return float(np.mean(np.exp(-0.5 * scaled_distance)))


def mmd_rbf_distance(candidate_vectors, reference_vectors, cfg, field_slices=None):
    """Biased joint RBF-MMD, optionally using one bandwidth per field block."""
    candidate, reference = standardize_for_mmd(candidate_vectors, reference_vectors, cfg)
    if candidate is None:
        return np.nan
    mode = str(baseline_get(cfg, "mmd_bandwidth_mode", "per_field")).lower()
    if mode not in {"per_field", "joint_median"}:
        raise ValueError(
            f"Unknown mmd_bandwidth_mode={mode!r}; expected per_field or joint_median."
        )
    slices = normalized_mmd_field_slices(reference.shape[1], field_slices)
    device = torch_metric_device(cfg)
    if device is not None:
        return mmd_rbf_distance_torch(candidate, reference, cfg, device, slices)
    configured = baseline_get(cfg, "mmd_bandwidth", None)
    if mode == "joint_median":
        bandwidth = mmd_rbf_bandwidth(reference, configured)
        kernel = lambda left, right: rbf_kernel_mean(left, right, bandwidth)
    else:
        bandwidths = mmd_rbf_bandwidths(reference, slices, configured)
        kernel = lambda left, right: rbf_kernel_mean_per_field(
            left, right, bandwidths, slices
        )
    k_xx = kernel(candidate, candidate)
    k_yy = kernel(reference, reference)
    k_xy = kernel(candidate, reference)
    return float(np.sqrt(max(k_xx + k_yy - 2.0 * k_xy, 0.0)))


def squared_euclidean_distances_torch(left, right):
    """Compute pairwise squared Euclidean distances in Torch."""
    left_norm = torch.sum(left * left, dim=1, keepdim=True)
    right_norm = torch.sum(right * right, dim=1, keepdim=True).T
    return torch.clamp(left_norm + right_norm - 2.0 * (left @ right.T), min=0.0)


def mmd_rbf_bandwidth_torch(reference, configured_bandwidth):
    """Fit an RBF bandwidth on the reference distribution on the metric device."""
    if configured_bandwidth is not None and float(configured_bandwidth) > 0:
        return torch.as_tensor(float(configured_bandwidth), dtype=reference.dtype, device=reference.device)
    distances = squared_euclidean_distances_torch(reference, reference)
    positive = distances[distances > 1e-12]
    if positive.numel() == 0:
        return torch.as_tensor(1.0, dtype=reference.dtype, device=reference.device)
    return torch.sqrt(torch.quantile(positive, 0.5))


def mmd_rbf_bandwidths_torch(reference, field_slices, configured_bandwidth=None):
    """Torch equivalent of the ERA5-reference per-field bandwidth fit."""
    if configured_bandwidth is None:
        configured = [None] * len(field_slices)
    elif np.isscalar(configured_bandwidth):
        configured = [float(configured_bandwidth)] * len(field_slices)
    else:
        configured = list(configured_bandwidth)
        if len(configured) != len(field_slices):
            raise ValueError(
                f"Expected {len(field_slices)} configured MMD bandwidths, got {len(configured)}."
            )
    return torch.stack(
        [
            mmd_rbf_bandwidth_torch(reference[:, start:stop], bandwidth)
            for (start, stop), bandwidth in zip(field_slices, configured)
        ]
    )


def rbf_kernel_mean_torch(left, right, bandwidth):
    """Mean Gaussian RBF kernel value between two vector samples in Torch."""
    distances = squared_euclidean_distances_torch(left, right)
    scale = 2.0 * torch.clamp(bandwidth, min=1e-12) ** 2
    return torch.mean(torch.exp(-distances / scale))


def rbf_kernel_mean_per_field_torch(left, right, bandwidths, field_slices):
    """Torch joint product-RBF with one ERA5-fitted scale per field."""
    scaled_distance = torch.zeros(
        (left.shape[0], right.shape[0]), dtype=left.dtype, device=left.device
    )
    for field_index, (start, stop) in enumerate(field_slices):
        scale = torch.clamp(bandwidths[field_index], min=1e-12) ** 2
        scaled_distance += squared_euclidean_distances_torch(
            left[:, start:stop], right[:, start:stop]
        ) / scale
    scaled_distance /= len(field_slices)
    return torch.mean(torch.exp(-0.5 * scaled_distance))


def mmd_rbf_distance_torch(candidate, reference, cfg, device, field_slices=None):
    """Biased joint RBF-MMD with Torch pairwise distances."""
    dtype = torch_metric_dtype(cfg)
    candidate_t = torch_tensor(candidate, device, dtype)
    reference_t = torch_tensor(reference, device, dtype)
    slices = normalized_mmd_field_slices(reference_t.shape[1], field_slices)
    configured = baseline_get(cfg, "mmd_bandwidth", None)
    mode = str(baseline_get(cfg, "mmd_bandwidth_mode", "per_field")).lower()
    if mode == "joint_median":
        bandwidth = mmd_rbf_bandwidth_torch(reference_t, configured)
        kernel = lambda left, right: rbf_kernel_mean_torch(left, right, bandwidth)
    else:
        bandwidths = mmd_rbf_bandwidths_torch(reference_t, slices, configured)
        kernel = lambda left, right: rbf_kernel_mean_per_field_torch(
            left, right, bandwidths, slices
        )
    k_xx = kernel(candidate_t, candidate_t)
    k_yy = kernel(reference_t, reference_t)
    k_xy = kernel(candidate_t, reference_t)
    value = torch.sqrt(torch.clamp(k_xx + k_yy - 2.0 * k_xy, min=0.0))
    return float(value.detach().cpu())


def features_by_key(features):
    """Map sample keys to row indices for a finalized feature dictionary."""
    return {key: idx for idx, key in enumerate(features.get("sample_keys", []))}


def standardized_field(field, reference_features):
    """Standardize one field with the reference distribution for its variable."""
    std = reference_features["std"] if reference_features["std"] > 1e-12 else 1.0
    return (np.asarray(field, dtype=np.float64) - reference_features["mean"]) / std


def joint_features_from_variables(features_by_variable, reference_by_variable, variables, cfg):
    """Combine configured variables into one standardized multi-field feature set."""
    if not variables:
        raise ValueError("No variables configured for joint metric evaluation.")

    key_sets = []
    index_maps = {}
    for variable in variables:
        features = features_by_variable[variable]
        index_maps[variable] = features_by_key(features)
        key_sets.append(set(index_maps[variable]))

    common_keys = sorted(set.intersection(*key_sets)) if key_sets else []
    if not common_keys:
        return {
            "values": np.array([], dtype=np.float64),
            "sample_values": [],
            "spectrum": np.array([], dtype=np.float64),
            "spectra": np.empty((0, 0), dtype=np.float64),
            "fields": np.empty((0, 0, 0), dtype=np.float64),
            "channel_fields": np.empty((0, 0, 0, 0), dtype=np.float64),
            "vectors": np.empty((0, 0), dtype=np.float64),
            "unweighted_vectors": np.empty((0, 0), dtype=np.float64),
            "mmd_field_slices": [],
            "sample_keys": [],
            "n_valid_samples": 0,
            "latitudes": None,
            "longitudes": None,
            "mean_field": np.array([], dtype=np.float64),
            "channel_mean_fields": np.array([], dtype=np.float64),
            "mean": np.nan,
            "std": np.nan,
        }

    max_values = int(baseline_get(cfg, "value_samples", 20000))
    values_per_sample_variable = max(1, max_values // max(len(common_keys) * len(variables), 1))
    per_variable_pixels = max(1, int(baseline_get(cfg, "swd_pixels", 4096)) // len(variables))

    values_parts = []
    sample_values = []
    spectra = []
    fields = []
    channel_fields = []
    vectors = []
    unweighted_vectors = []
    latitudes = None
    longitudes = None

    for key in common_keys:
        sample_value_parts = []
        sample_spectra = []
        sample_fields = []
        sample_vectors = []
        sample_unweighted_vectors = []
        for variable in variables:
            source = features_by_variable[variable]
            reference = reference_by_variable[variable]
            field = source["fields"][index_maps[variable][key]]
            standardized = standardized_field(field, reference)
            sample_value_parts.append(subsample_flat_values(standardized, values_per_sample_variable))
            sample_spectra.append(zonal_energy_spectrum(standardized, source["latitudes"]))
            sample_fields.append(standardized)
            sample_vectors.append(field_vector(standardized, source["latitudes"], per_variable_pixels))
            sample_unweighted_vectors.append(unweighted_field_vector(standardized, source["latitudes"], per_variable_pixels))
            if latitudes is None:
                latitudes = source["latitudes"]
            if longitudes is None:
                longitudes = source["longitudes"]

        values = np.concatenate(sample_value_parts)
        values_parts.append(values)
        sample_values.append(values)
        spectra.append(np.concatenate(sample_spectra))
        channel_stack = np.stack(sample_fields)
        channel_fields.append(channel_stack)
        fields.append(np.concatenate(sample_fields, axis=1))
        vectors.append(np.concatenate(sample_vectors))
        unweighted_vectors.append(np.concatenate(sample_unweighted_vectors))

    all_values = np.concatenate(values_parts) if values_parts else np.array([], dtype=np.float64)
    spectra = np.stack(spectra) if spectra else np.empty((0, 0), dtype=np.float64)
    fields = np.stack(fields) if fields else np.empty((0, 0, 0), dtype=np.float64)
    channel_fields = np.stack(channel_fields) if channel_fields else np.empty((0, 0, 0, 0), dtype=np.float64)
    vectors = np.stack(vectors) if vectors else np.empty((0, 0), dtype=np.float64)
    unweighted_vectors = np.stack(unweighted_vectors) if unweighted_vectors else np.empty((0, 0), dtype=np.float64)
    vector_block_size = vectors.shape[1] // len(variables) if vectors.size else 0
    mmd_field_slices = [
        (field_index * vector_block_size, (field_index + 1) * vector_block_size)
        for field_index in range(len(variables))
    ] if vector_block_size else []

    return {
        "values": all_values,
        "sample_values": sample_values,
        "spectrum": np.nanmean(spectra, axis=0) if spectra.size else np.array([], dtype=np.float64),
        "spectra": spectra,
        "fields": fields,
        "channel_fields": channel_fields,
        "vectors": vectors,
        "unweighted_vectors": unweighted_vectors,
        "mmd_field_slices": mmd_field_slices,
        "sample_keys": common_keys,
        "n_valid_samples": int(fields.shape[0]) if fields.ndim == 3 else 0,
        "latitudes": latitudes,
        "longitudes": longitudes,
        "mean_field": np.nanmean(fields, axis=0) if fields.size else np.array([], dtype=np.float64),
        "channel_mean_fields": np.nanmean(channel_fields, axis=0) if channel_fields.size else np.array([], dtype=np.float64),
        "mean": float(np.nanmean(all_values)) if all_values.size else np.nan,
        "std": float(np.nanstd(all_values)) if all_values.size else np.nan,
    }


def regular_sphere_centers(n_lat, n_lon):
    """Return regular latitude-longitude cell centers for SCWD slicing."""
    latitudes = np.linspace(-90.0 + 90.0 / n_lat, 90.0 - 90.0 / n_lat, n_lat)
    longitudes = (np.arange(n_lon, dtype=np.float64) + 0.5) * 360.0 / n_lon
    return latitudes, longitudes


def nearest_indices(values, targets):
    """Return nearest indices in a possibly unsorted 1D coordinate array."""
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    order = np.argsort(values)
    sorted_values = values[order]
    right = np.searchsorted(sorted_values, targets, side="left")
    right = np.clip(right, 0, len(sorted_values) - 1)
    left = np.clip(right - 1, 0, len(sorted_values) - 1)
    choose_right = np.abs(sorted_values[right] - targets) < np.abs(sorted_values[left] - targets)
    return order[np.where(choose_right, right, left)]


def nearest_longitude_indices(longitudes, targets):
    """Return nearest periodic longitude indices."""
    longitudes = np.mod(np.asarray(longitudes, dtype=np.float64), 360.0)
    targets = np.mod(np.asarray(targets, dtype=np.float64), 360.0)
    order = np.argsort(longitudes)
    sorted_lons = longitudes[order]
    right = np.searchsorted(sorted_lons, targets, side="left") % len(sorted_lons)
    left = (right - 1) % len(sorted_lons)
    right_distance = np.abs(((sorted_lons[right] - targets + 180.0) % 360.0) - 180.0)
    left_distance = np.abs(((sorted_lons[left] - targets + 180.0) % 360.0) - 180.0)
    return order[np.where(right_distance < left_distance, right, left)]


def wendland_kernel(chordal_distance, radius_chordal):
    """Compact Wendland kernel from the SCWD paper, scaled to the range radius."""
    scaled = chordal_distance / max(float(radius_chordal), 1e-12)
    values = np.zeros_like(scaled, dtype=np.float64)
    mask = scaled <= 1.0
    d = scaled[mask]
    values[mask] = ((1.0 - d) ** 6) * (35.0 * d**2 + 18.0 * d + 3.0) / 3.0
    return values


def scwd_weight_vectors(reference, cfg):
    """Build and cache sparse Wendland convolution weights for one grid."""
    n_lat = int(baseline_get(cfg, "scwd_anchor_lat_points", 60))
    n_lon = int(baseline_get(cfg, "scwd_anchor_lon_points", 120))
    domain_n_lat = int(baseline_get(cfg, "scwd_domain_lat_points", 361))
    domain_n_lon = int(baseline_get(cfg, "scwd_domain_lon_points", 720))
    radius_km = float(baseline_get(cfg, "scwd_radius_km", 1000.0))
    latitudes = np.asarray(reference["latitudes"], dtype=np.float64)
    longitudes = np.asarray(reference["longitudes"], dtype=np.float64)
    cache_key = (
        tuple(np.round(latitudes, 8)),
        tuple(np.round(np.mod(longitudes, 360.0), 8)),
        n_lat,
        n_lon,
        domain_n_lat,
        domain_n_lon,
        radius_km,
    )
    if cache_key in _SCWD_WEIGHT_CACHE:
        return _SCWD_WEIGHT_CACHE[cache_key]

    earth_radius_km = 6371.0
    radius_radians = radius_km / earth_radius_km
    radius_degrees = np.rad2deg(radius_radians)
    radius_chordal = 2.0 * np.sin((radius_km / earth_radius_km) / 2.0)
    anchor_lats, anchor_lons = regular_sphere_centers(n_lat, n_lon)
    domain_lats = np.linspace(-90.0, 90.0, domain_n_lat)
    domain_lons = np.arange(domain_n_lon, dtype=np.float64) * 360.0 / domain_n_lon

    weights = []
    for lat0, lon0 in zip(np.deg2rad(np.repeat(anchor_lats, n_lon)), np.deg2rad(np.tile(anchor_lons, n_lat))):
        lat0_deg = np.rad2deg(lat0)
        lat_mask = (domain_lats >= lat0_deg - radius_degrees) & (domain_lats <= lat0_deg + radius_degrees)
        candidate_lats = domain_lats[lat_mask]
        if candidate_lats.size == 0:
            weights.append((np.array([], dtype=int), np.array([], dtype=np.float64)))
            continue

        lat_grid, lon_grid = np.meshgrid(candidate_lats, domain_lons, indexing="ij")
        flat_lat = np.deg2rad(lat_grid.ravel())
        flat_lon = np.deg2rad(lon_grid.ravel())
        cos_distance = (
            np.sin(lat0) * np.sin(flat_lat)
            + np.cos(lat0) * np.cos(flat_lat) * np.cos(flat_lon - lon0)
        )
        chordal_distance = np.sqrt(np.maximum(2.0 - 2.0 * np.clip(cos_distance, -1.0, 1.0), 0.0))
        kernel = wendland_kernel(chordal_distance, radius_chordal) * np.maximum(np.cos(flat_lat), 0.0)
        support = np.flatnonzero(kernel > 0)
        if support.size == 0:
            weights.append((support, kernel[support]))
            continue

        source_lat_idx = nearest_indices(latitudes, np.rad2deg(flat_lat[support]))
        source_lon_idx = nearest_longitude_indices(longitudes, np.rad2deg(flat_lon[support]))
        source_idx = source_lat_idx * len(longitudes) + source_lon_idx
        unique_source_idx, inverse = np.unique(source_idx, return_inverse=True)
        source_weights = np.bincount(inverse, weights=kernel[support], minlength=len(unique_source_idx))
        weights.append((unique_source_idx, source_weights / np.sum(source_weights)))

    _SCWD_WEIGHT_CACHE[cache_key] = weights
    return weights


def scwd_sparse_weight_matrix_torch(weights, n_pixels, cfg, device):
    """Return a cached sparse anchor-by-pixel SCWD weight matrix."""
    dtype = torch_metric_dtype(cfg)
    cache_key = (id(weights), int(n_pixels), str(device), str(dtype))
    if cache_key in _SCWD_TORCH_WEIGHT_CACHE:
        return _SCWD_TORCH_WEIGHT_CACHE[cache_key]

    rows = []
    cols = []
    values = []
    row_idx = 0
    for support, support_weights in weights:
        if support.size == 0:
            continue
        rows.append(np.full(support.size, row_idx, dtype=np.int64))
        cols.append(np.asarray(support, dtype=np.int64))
        values.append(np.asarray(support_weights, dtype=np.float64))
        row_idx += 1

    if not rows:
        matrix = None
    else:
        row_values = np.concatenate(rows)
        col_values = np.concatenate(cols)
        weight_values = np.concatenate(values)
        indices = torch.as_tensor(np.stack([row_values, col_values]), dtype=torch.long, device=device)
        data = torch.as_tensor(weight_values, dtype=dtype, device=device)
        matrix = torch.sparse_coo_tensor(indices, data, size=(row_idx, int(n_pixels)), device=device).coalesce()

    _SCWD_TORCH_WEIGHT_CACHE[cache_key] = matrix
    return matrix


def scwd_sparse_weight_matrix_numpy(weights, n_pixels):
    """Return a cached CSR anchor-by-pixel SCWD weight matrix."""
    cache_key = (id(weights), int(n_pixels))
    if cache_key in _SCWD_NUMPY_WEIGHT_CACHE:
        return _SCWD_NUMPY_WEIGHT_CACHE[cache_key]

    rows = []
    cols = []
    values = []
    for row_idx, (support, support_weights) in enumerate(weights):
        if support.size == 0:
            continue
        rows.append(np.full(support.size, row_idx, dtype=np.int64))
        cols.append(np.asarray(support, dtype=np.int64))
        values.append(np.asarray(support_weights, dtype=np.float32))
    if not rows:
        matrix = None
    else:
        matrix = scipy.sparse.csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(weights), int(n_pixels)),
            dtype=np.float32,
        )
    _SCWD_NUMPY_WEIGHT_CACHE[cache_key] = matrix
    return matrix


def scwd_response_batch(fields, numpy_weights, torch_weights, cfg, device):
    """Project a BxCxHxW field batch onto every SCWD spatial anchor."""
    fields = np.asarray(fields, dtype=np.float32)
    if device is None:
        flat = fields.reshape(fields.shape[0] * fields.shape[1], -1)
        responses = flat @ numpy_weights.T
        return np.asarray(responses, dtype=np.float32).reshape(fields.shape[0], fields.shape[1], -1)

    dtype = torch_metric_dtype(cfg)
    fields_t = torch_tensor(fields, device, dtype).reshape(fields.shape[0], fields.shape[1], -1)
    by_channel = [
        torch.sparse.mm(torch_weights, fields_t[:, channel_idx, :].T).T
        for channel_idx in range(fields.shape[1])
    ]
    return torch.stack(by_channel, dim=1).detach().cpu().numpy().astype(np.float32)


def empirical_joint_w2_cost(candidate, reference):
    """Return the squared exact empirical W2 cost for equally weighted vectors."""
    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.ndim != 2 or reference.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError(
            "Exact empirical joint W2 requires equally sized sample-by-field arrays; "
            f"got {candidate.shape} and {reference.shape}."
        )
    if not candidate.size:
        return np.nan
    cost = scipy.spatial.distance.cdist(candidate, reference, metric="sqeuclidean")
    rows, cols = scipy.optimize.linear_sum_assignment(cost)
    return float(np.mean(cost[rows, cols]))


def _selected_scwd_responses(candidate_response, reference_response, cfg):
    """Return equally sized, evenly distributed deterministic subsets for joint OT."""
    maximum = int(baseline_get(
        cfg, "scwd_ot_samples", baseline_get(cfg, "pairwise_eval_samples", 256)
    ))
    n_samples = min(candidate_response.shape[0], reference_response.shape[0])
    if maximum > 0:
        n_samples = min(n_samples, maximum)
    if n_samples == 0:
        return candidate_response[:0], reference_response[:0]
    candidate_indices = pairwise_sample_positions(candidate_response.shape[0], n_samples)
    reference_indices = pairwise_sample_positions(reference_response.shape[0], n_samples)
    return candidate_response[candidate_indices], reference_response[reference_indices]


def scwd_anchor_transport_costs(candidate, reference, cfg):
    """Return per-anchor transport costs for scalar or joint multi-field SCWD."""
    candidate_response = np.asarray(candidate.get("scwd_responses"))
    reference_response = np.asarray(reference.get("scwd_responses"))
    if candidate_response.size == 0 or reference_response.size == 0:
        return np.array([], dtype=np.float64)
    if candidate_response.ndim != 3 or reference_response.ndim != 3:
        return np.array([], dtype=np.float64)
    n_channels = min(candidate_response.shape[1], reference_response.shape[1])
    n_anchors = min(candidate_response.shape[2], reference_response.shape[2])
    if n_channels == 0 or n_anchors == 0:
        return np.array([], dtype=np.float64)

    order = float(baseline_get(cfg, "scwd_order", 2.0))
    if n_channels == 1:
        n_quantiles = int(baseline_get(cfg, "scwd_quantiles", 200))
        quantiles = np.linspace(0.0, 1.0, n_quantiles)
        anchor_chunk = max(1, int(baseline_get(cfg, "scwd_anchor_chunk_size", 128)))
        costs = np.zeros(n_anchors, dtype=np.float64)
        for start in range(0, n_anchors, anchor_chunk):
            stop = min(start + anchor_chunk, n_anchors)
            candidate_q = np.quantile(candidate_response[:, 0, start:stop], quantiles, axis=0)
            reference_q = np.quantile(reference_response[:, 0, start:stop], quantiles, axis=0)
            costs[start:stop] = np.mean(np.abs(candidate_q - reference_q) ** order, axis=0)
        return costs
    candidate_response, reference_response = _selected_scwd_responses(
        candidate_response[:, :n_channels, :n_anchors],
        reference_response[:, :n_channels, :n_anchors], cfg,
    )
    if not candidate_response.size or not reference_response.size:
        return np.array([], dtype=np.float64)

    if not np.isclose(order, 2.0):
        raise ValueError("Joint multi-field SCWD currently implements quadratic W2 only (scwd_order=2).")
    costs = np.empty(n_anchors, dtype=np.float64)
    anchors = range(n_anchors)
    if bool(baseline_get(cfg, "scwd_ot_progress", True)):
        anchors = tqdm(anchors, desc="Joint SCWD anchor OT", leave=False)
    for anchor_idx in anchors:
        costs[anchor_idx] = empirical_joint_w2_cost(
            candidate_response[:, :, anchor_idx], reference_response[:, :, anchor_idx]
        )
    return costs


def scwd_anchor_joint_distributions(candidate, reference, cfg, n_top):
    """Rank anchors by local joint W2 and retain their per-field responses."""
    costs = scwd_anchor_transport_costs(candidate, reference, cfg)
    candidate_response = np.asarray(candidate.get("scwd_responses"))
    reference_response = np.asarray(reference.get("scwd_responses"))
    if costs.size == 0 or candidate_response.ndim != 3 or reference_response.ndim != 3:
        return np.array([], dtype=np.float64), []
    n_channels = min(candidate_response.shape[1], reference_response.shape[1])
    candidate_response, reference_response = _selected_scwd_responses(
        candidate_response[:, :n_channels, :costs.size],
        reference_response[:, :n_channels, :costs.size], cfg,
    )
    local_wasserstein = np.sqrt(np.maximum(costs, 0.0))
    n_top = min(max(int(n_top), 0), costs.size)
    distributions = []
    for anchor_idx in np.argsort(local_wasserstein)[::-1][:n_top]:
        distributions.append({
            "anchor_index": int(anchor_idx),
            "wasserstein": float(local_wasserstein[anchor_idx]),
            "candidate": np.asarray(candidate_response[:, :, anchor_idx], dtype=np.float32),
            "reference": np.asarray(reference_response[:, :, anchor_idx], dtype=np.float32),
        })
    return local_wasserstein, distributions


def scwd_anchor_w1_distributions(candidate, reference, cfg, n_top):
    """Compatibility alias for joint-W2 anchor diagnostics."""
    local_wasserstein, distributions = scwd_anchor_joint_distributions(
        candidate, reference, cfg, n_top
    )
    for item in distributions:
        item["w1"] = item["wasserstein"]
    return local_wasserstein, distributions


def scwd_anchor_mean_response_difference(candidate, reference, cfg):
    """Return per-field candidate-minus-reference mean response at each anchor."""
    candidate_response = np.asarray(candidate.get("scwd_responses"))
    reference_response = np.asarray(reference.get("scwd_responses"))
    if candidate_response.ndim != 3 or reference_response.ndim != 3:
        return np.empty((0, 0), dtype=np.float64)
    n_channels = min(candidate_response.shape[1], reference_response.shape[1])
    n_anchors = min(candidate_response.shape[2], reference_response.shape[2])
    if n_channels == 0 or n_anchors == 0:
        return np.empty((0, 0), dtype=np.float64)
    candidate_mean = np.mean(candidate_response[:, :n_channels, :n_anchors], axis=0)
    reference_mean = np.mean(reference_response[:, :n_channels, :n_anchors], axis=0)
    return np.asarray(candidate_mean - reference_mean, dtype=np.float64)


def scwd_anchor_area_weights(cfg):
    """Return normalized surface-area weights for the regular SCWD anchor grid."""
    n_lat = int(baseline_get(cfg, "scwd_anchor_lat_points", 60))
    n_lon = int(baseline_get(cfg, "scwd_anchor_lon_points", 120))
    anchor_latitudes, _ = regular_sphere_centers(n_lat, n_lon)
    weights = np.repeat(np.maximum(np.cos(np.deg2rad(anchor_latitudes)), 0.0), n_lon)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("SCWD anchor grid has zero total surface-area weight.")
    return weights / total


def scwd_area_weighted_from_responses(candidate, reference, cfg):
    """Aggregate SCWD anchor transport costs using spherical surface-area weights."""
    costs = scwd_anchor_transport_costs(candidate, reference, cfg)
    if costs.size == 0:
        return np.nan
    weights = scwd_anchor_area_weights(cfg)
    if costs.size != weights.size:
        raise ValueError(f"SCWD anchor-weight mismatch: got {costs.size} costs and {weights.size} weights.")
    order = float(baseline_get(cfg, "scwd_order", 2.0))
    return float(np.sum(weights * costs) ** (1.0 / order))


def scwd_from_responses(candidate, reference, cfg):
    """Compute SCWD from disk-backed sample-by-channel-by-anchor responses."""
    costs = scwd_anchor_transport_costs(candidate, reference, cfg)
    order = float(baseline_get(cfg, "scwd_order", 2.0))
    return float(np.mean(costs) ** (1.0 / order)) if costs.size else np.nan


def scwd_filter_responses_torch(fields, weight_matrix, cfg, device):
    """Apply all SCWD spatial filters to a batch of fields."""
    dtype = torch_metric_dtype(cfg)
    fields_t = torch_tensor(fields, device, dtype).reshape(fields.shape[0], -1)
    return torch.sparse.mm(weight_matrix, fields_t.T).T


def spherical_convolutional_wasserstein_torch(candidate, reference, cfg, device):
    """Torch SCWD implementation for scalar fields."""
    candidate_fields = np.asarray(candidate["fields"], dtype=np.float64)
    reference_fields = np.asarray(reference["fields"], dtype=np.float64)
    if candidate_fields.size == 0 or reference_fields.size == 0:
        return np.nan
    if candidate_fields.ndim != 3 or reference_fields.ndim != 3:
        return np.nan
    if candidate_fields.shape[1:] != reference_fields.shape[1:]:
        return np.nan

    weights = scwd_weight_vectors(reference, cfg)
    weight_matrix = scwd_sparse_weight_matrix_torch(weights, candidate_fields.shape[1] * candidate_fields.shape[2], cfg, device)
    if weight_matrix is None or weight_matrix.shape[0] == 0:
        return np.nan

    dtype = torch_metric_dtype(cfg)
    r = float(baseline_get(cfg, "scwd_order", 2.0))
    n_quantiles = int(baseline_get(cfg, "scwd_quantiles", 200))
    quantiles = torch.linspace(0.0, 1.0, n_quantiles, dtype=dtype, device=device)

    candidate_response = scwd_filter_responses_torch(candidate_fields, weight_matrix, cfg, device)
    reference_response = scwd_filter_responses_torch(reference_fields, weight_matrix, cfg, device)
    candidate_quantiles = torch.quantile(candidate_response, quantiles, dim=0)
    reference_quantiles = torch.quantile(reference_response, quantiles, dim=0)
    total = torch.sum(torch.abs(candidate_quantiles - reference_quantiles) ** r)
    denom = float(weight_matrix.shape[0] * n_quantiles)
    return float(((total / denom) ** (1.0 / r)).detach().cpu())


def joint_spherical_convolutional_wasserstein_torch(candidate, reference, cfg, device):
    """Compute joint SCWD responses on-device, followed by exact CPU anchor OT."""
    candidate_fields = np.asarray(candidate["channel_fields"], dtype=np.float32)
    reference_fields = np.asarray(reference["channel_fields"], dtype=np.float32)
    if candidate_fields.size == 0 or reference_fields.size == 0:
        return np.nan
    if candidate_fields.ndim != 4 or reference_fields.ndim != 4:
        return np.nan
    if candidate_fields.shape[1:] != reference_fields.shape[1:]:
        return np.nan
    weights = scwd_weight_vectors(reference, cfg)
    n_pixels = candidate_fields.shape[2] * candidate_fields.shape[3]
    weight_matrix = scwd_sparse_weight_matrix_torch(weights, n_pixels, cfg, device)
    if weight_matrix is None or weight_matrix.shape[0] == 0:
        return np.nan
    candidate_response = scwd_response_batch(candidate_fields, None, weight_matrix, cfg, device)
    reference_response = scwd_response_batch(reference_fields, None, weight_matrix, cfg, device)
    return scwd_from_responses(
        {"scwd_responses": candidate_response},
        {"scwd_responses": reference_response}, cfg,
    )


def spherical_convolutional_wasserstein(candidate, reference, cfg):
    """Approximate SCWD using the quantile algorithm from Garrett et al. (2024)."""
    if "scwd_responses" in candidate and "scwd_responses" in reference:
        return scwd_from_responses(candidate, reference, cfg)
    if np.asarray(candidate.get("channel_fields", [])).ndim == 4:
        candidate_channels = np.asarray(candidate["channel_fields"], dtype=np.float64)
        reference_channels = np.asarray(reference["channel_fields"], dtype=np.float64)
        if candidate_channels.shape[1] > 1:
            return joint_spherical_convolutional_wasserstein(candidate, reference, cfg)

    candidate_fields = np.asarray(candidate["fields"], dtype=np.float64)
    reference_fields = np.asarray(reference["fields"], dtype=np.float64)
    if candidate_fields.size == 0 or reference_fields.size == 0:
        return np.nan
    if candidate_fields.ndim != 3 or reference_fields.ndim != 3:
        return np.nan
    if candidate_fields.shape[1:] != reference_fields.shape[1:]:
        return np.nan

    latitudes = reference["latitudes"]
    longitudes = reference["longitudes"]
    if latitudes is None or longitudes is None:
        return np.nan

    weights = scwd_weight_vectors(reference, cfg)
    if not weights:
        return np.nan

    device = torch_metric_device(cfg)
    if device is not None:
        return spherical_convolutional_wasserstein_torch(candidate, reference, cfg, device)

    r = float(baseline_get(cfg, "scwd_order", 2.0))
    n_quantiles = int(baseline_get(cfg, "scwd_quantiles", 200))
    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    candidate_flat = candidate_fields.reshape(candidate_fields.shape[0], -1)
    reference_flat = reference_fields.reshape(reference_fields.shape[0], -1)

    total = 0.0
    n_used = 0
    for support, support_weights in weights:
        if support.size == 0:
            continue
        candidate_slice = candidate_flat[:, support] @ support_weights
        reference_slice = reference_flat[:, support] @ support_weights
        candidate_quantiles = np.quantile(candidate_slice, quantiles)
        reference_quantiles = np.quantile(reference_slice, quantiles)
        total += float(np.sum(np.abs(candidate_quantiles - reference_quantiles) ** r))
        n_used += 1
    # Approximate the SCWD integrals with means over convolution centers and
    # quantile locations. Summing here would make the value depend on the chosen
    # numerical grids rather than on the field distributions.
    return float((total / (n_used * n_quantiles)) ** (1.0 / r)) if n_used else np.nan


def joint_spherical_convolutional_wasserstein(candidate, reference, cfg):
    """Joint empirical multi-field W2 of spherical convolution responses."""
    candidate_fields = np.asarray(candidate["channel_fields"], dtype=np.float32)
    reference_fields = np.asarray(reference["channel_fields"], dtype=np.float32)
    if candidate_fields.size == 0 or reference_fields.size == 0:
        return np.nan
    if candidate_fields.ndim != 4 or reference_fields.ndim != 4:
        return np.nan
    if candidate_fields.shape[1:] != reference_fields.shape[1:]:
        return np.nan
    weights = scwd_weight_vectors(reference, cfg)
    if not weights:
        return np.nan
    device = torch_metric_device(cfg)
    if device is not None:
        return joint_spherical_convolutional_wasserstein_torch(candidate, reference, cfg, device)
    n_pixels = candidate_fields.shape[2] * candidate_fields.shape[3]
    weight_matrix = scwd_sparse_weight_matrix_numpy(weights, n_pixels)
    if weight_matrix is None or weight_matrix.shape[0] == 0:
        return np.nan
    candidate_response = scwd_response_batch(candidate_fields, weight_matrix, None, cfg, None)
    reference_response = scwd_response_batch(reference_fields, weight_matrix, None, cfg, None)
    return scwd_from_responses(
        {"scwd_responses": candidate_response},
        {"scwd_responses": reference_response}, cfg,
    )


def gaussian_density_inner_mean_1d(left, right, bandwidth):
    """Mean Gaussian-density inner product between two 1D empirical samples."""
    diffs = left[:, None] - right[None, :]
    variance = float(bandwidth) ** 2
    normalizer = np.sqrt(4.0 * np.pi * variance)
    return float(np.mean(np.exp(-(diffs ** 2) / (4.0 * variance)) / normalizer))


def projection_bandwidth(left, right, configured_bandwidth):
    """Choose Cramer-Wold bandwidth, falling back to a median-distance heuristic."""
    if configured_bandwidth is not None and float(configured_bandwidth) > 0:
        return float(configured_bandwidth)
    values = np.sort(np.concatenate([left, right]))
    if values.size < 2:
        return 1.0
    diffs = np.abs(values[:, None] - values[None, :])
    positive = diffs[diffs > 1e-12]
    if positive.size == 0:
        scale = float(np.std(values))
        return scale if scale > 1e-12 else 1.0
    return float(np.median(positive))


def sliced_cramer_wold_distance(candidate_vectors, reference_vectors, n_projections, seed, bandwidth=None):
    """Approximate sliced Cramer-Wold distance with Gaussian-kernel projections."""
    candidate = np.asarray(candidate_vectors, dtype=np.float64)
    reference = np.asarray(reference_vectors, dtype=np.float64)
    if candidate.size == 0 or reference.size == 0:
        return np.nan
    if candidate.ndim != 2 or reference.ndim != 2:
        return np.nan
    if candidate.shape[1] != reference.shape[1]:
        n_dim = min(candidate.shape[1], reference.shape[1])
        candidate = candidate[:, :n_dim]
        reference = reference[:, :n_dim]

    rng = np.random.default_rng(seed)
    distances = []
    for _ in range(n_projections):
        direction = rng.normal(size=candidate.shape[1])
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            continue
        direction /= norm
        candidate_proj = candidate @ direction
        reference_proj = reference @ direction
        bw = projection_bandwidth(candidate_proj, reference_proj, bandwidth)
        k_xx = gaussian_density_inner_mean_1d(candidate_proj, candidate_proj, bw)
        k_yy = gaussian_density_inner_mean_1d(reference_proj, reference_proj, bw)
        k_xy = gaussian_density_inner_mean_1d(candidate_proj, reference_proj, bw)
        distances.append(float(np.sqrt(max(k_xx + k_yy - 2.0 * k_xy, 0.0))))
    return float(np.mean(distances)) if distances else np.nan


def field_l1_pairwise_mean(left_fields, right_fields, latitudes, chunk_size):
    """Mean cosine-latitude-weighted L1 distance over all field pairs."""
    left = np.asarray(left_fields, dtype=np.float32)
    right = np.asarray(right_fields, dtype=np.float32)
    if left.size == 0 or right.size == 0:
        return np.nan
    if left.ndim != 3 or right.ndim != 3 or left.shape[1:] != right.shape[1:]:
        return np.nan

    weights_1d = latitude_weights(latitudes).astype(np.float32)
    if left.shape[1] == weights_1d.size:
        weights = weights_1d
    elif left.shape[1] % weights_1d.size == 0:
        weights = np.tile(weights_1d, left.shape[1] // weights_1d.size)
    else:
        return np.nan
    weights = weights[None, :, None]
    chunk_size = max(1, int(chunk_size))
    total = 0.0
    n_pairs = 0
    for start in range(0, left.shape[0], chunk_size):
        chunk = left[start:start + chunk_size]
        distances = np.mean(np.abs(chunk[:, None, :, :] - right[None, :, :, :]) * weights, axis=(2, 3))
        total += float(np.sum(distances))
        n_pairs += int(distances.size)
    return total / n_pairs if n_pairs else np.nan


def cached_reference_field_self_distance(reference, cfg):
    """Return and cache E[d(Y,Y_prime)] on the reference feature object."""
    fields = np.asarray(reference["fields"])
    latitudes = reference["latitudes"]
    if fields.size == 0 or latitudes is None:
        return np.nan
    chunk_size = int(baseline_get(cfg, "field_energy_chunk_size", 4))
    cache = reference.setdefault("_field_self_distance_cache", {})
    cache_key = (tuple(np.round(np.asarray(latitudes, dtype=np.float64), 8)), chunk_size)
    if cache_key not in cache:
        cache[cache_key] = field_l1_pairwise_mean(fields, fields, latitudes, chunk_size)
    return cache[cache_key]


def crps_like_field_energy(candidate, reference, cfg):
    """Half-energy distance with a weighted L1 distance between complete fields."""
    fields = np.asarray(candidate["fields"])
    reference_fields = np.asarray(reference["fields"])
    latitudes = reference["latitudes"]
    if fields.size == 0 or reference_fields.size == 0 or latitudes is None:
        return np.nan

    chunk_size = int(baseline_get(cfg, "field_energy_chunk_size", 4))
    cross = field_l1_pairwise_mean(fields, reference_fields, latitudes, chunk_size)
    candidate_self = field_l1_pairwise_mean(fields, fields, latitudes, chunk_size)
    reference_self = cached_reference_field_self_distance(reference, cfg)
    return float(max(cross - 0.5 * candidate_self - 0.5 * reference_self, 0.0))


def subsample_flat_values(values, max_values):
    """Take a deterministic subset from a flattened field."""
    flat = np.ravel(np.asarray(values, dtype=np.float64))
    flat = flat[np.isfinite(flat)]
    if max_values <= 0 or flat.size <= max_values:
        return flat
    indices = np.linspace(0, flat.size - 1, max_values, dtype=int)
    return flat[indices]


def empty_features(variables):
    """Create mutable feature accumulators for each variable."""
    return {
        variable: {
            "values": [],
            "spectra": [],
            "fields": [],
            "vectors": [],
            "unweighted_vectors": [],
            "sample_keys": [],
            "latitudes": None,
            "longitudes": None,
        }
        for variable in variables
    }


def finalize_features(features, max_values):
    """Concatenate sampled values and average spectra for each variable."""
    finalized = {}
    for variable, parts in features.items():
        values = np.concatenate(parts["values"]) if parts["values"] else np.array([], dtype=np.float64)
        if max_values > 0 and values.size > max_values:
            values = values[np.linspace(0, values.size - 1, max_values, dtype=int)]
        spectra = np.stack(parts["spectra"]) if parts["spectra"] else np.empty((0, 0), dtype=np.float64)
        fields = np.stack(parts["fields"]) if parts["fields"] else np.empty((0, 0, 0), dtype=np.float64)
        vectors = np.stack(parts["vectors"]) if parts["vectors"] else np.empty((0, 0), dtype=np.float64)
        unweighted_vectors = (
            np.stack(parts["unweighted_vectors"]) if parts["unweighted_vectors"] else np.empty((0, 0), dtype=np.float64)
        )
        finalized[variable] = {
            "values": values,
            "sample_values": parts["values"],
            "spectrum": np.nanmean(spectra, axis=0) if spectra.size else np.array([], dtype=np.float64),
            "spectra": spectra,
            "fields": fields,
            "vectors": vectors,
            "unweighted_vectors": unweighted_vectors,
            "sample_keys": parts["sample_keys"],
            "n_valid_samples": int(fields.shape[0]) if fields.ndim == 3 else 0,
            "latitudes": parts["latitudes"],
            "longitudes": parts["longitudes"],
            "mean_field": np.nanmean(fields, axis=0) if fields.size else np.array([], dtype=np.float64),
            "mean": float(np.nanmean(values)) if values.size else np.nan,
            "std": float(np.nanstd(values)) if values.size else np.nan,
        }
    return finalized


def normalized_tensor(ds_slice, variables, means, stds):
    """Convert one ERA5 slice to the normalized tensor used by training corruptions."""
    channels = []
    for variable in variables:
        raw = ds_slice[variable].transpose("latitude", "longitude").values.astype(np.float32)
        channels.append(np.nan_to_num((raw - means[variable]) / stds[variable], nan=0.0))
    return torch.tensor(np.stack(channels), dtype=torch.float32)


def unnormalize_channel(channel, variable, means, stds):
    """Convert a normalized tensor channel back to physical units."""
    return channel.detach().cpu().numpy().astype(np.float32) * stds[variable] + means[variable]


def distribution_features(
    cfg,
    ds,
    variables,
    time_indices,
    max_values,
    *,
    lead_idx=None,
    corruption_type=None,
    severity=0.0,
    means=None,
    stds=None,
):
    """Extract unpaired distribution features from a dataset subset."""
    features = empty_features(variables)
    values_per_field = max(1, max_values // max(len(time_indices), 1)) if max_values > 0 else 0
    vector_pixels = int(baseline_get(cfg, "swd_pixels", 4096))

    for time_idx in time_indices:
        ds_slice = ds.isel(time=time_idx)
        if lead_idx is not None and "prediction_timedelta" in ds_slice.dims:
            ds_slice = ds_slice.isel(prediction_timedelta=lead_idx)

        corrupted_channels = None
        if corruption_type is not None:
            sample = normalized_tensor(ds_slice, variables, means, stds)
            corrupted_channels = apply_configured_corruption(sample, corruption_type, float(severity))

        for channel_idx, variable in enumerate(variables):
            if corrupted_channels is None:
                values = ds_slice[variable].transpose("latitude", "longitude").values.astype(np.float64)
            else:
                values = unnormalize_channel(corrupted_channels[channel_idx], variable, means, stds)
            latitudes = ds_slice[variable].latitude.values
            longitudes = ds_slice[variable].longitude.values
            if should_skip_field(cfg, values):
                continue
            values = canonical_latlon(values, latitudes)
            features[variable]["values"].append(subsample_flat_values(values, values_per_field))
            features[variable]["spectra"].append(zonal_energy_spectrum(values, latitudes))
            features[variable]["fields"].append(values)
            features[variable]["vectors"].append(field_vector(values, latitudes, vector_pixels))
            features[variable]["unweighted_vectors"].append(unweighted_field_vector(values, latitudes, vector_pixels))
            features[variable]["sample_keys"].append(time_idx)
            if features[variable]["latitudes"] is None:
                features[variable]["latitudes"] = np.asarray(latitudes, dtype=np.float64)
            if features[variable]["longitudes"] is None:
                features[variable]["longitudes"] = np.asarray(longitudes, dtype=np.float64)
    return finalize_features(features, max_values)


def streaming_reference_stats(cfg, ds, variables, time_indices):
    """Compute surface-area-weighted per-variable reference moments in bounded memory."""
    chunk_size = max(1, int(baseline_get(cfg, "feature_chunk_size", 32)))
    totals = {variable: [0.0, 0.0, 0.0] for variable in variables}
    for start in tqdm(range(0, len(time_indices), chunk_size), desc="ERA5 reference moments"):
        chunk = ds.isel(time=time_indices[start:start + chunk_size])
        for variable in variables:
            values = np.asarray(
                chunk[variable].transpose("time", "latitude", "longitude").values, dtype=np.float64
            )
            total, total_sq, weight_mass = latitude_weighted_moment_totals(
                values, chunk[variable].latitude.values
            )
            totals[variable][0] += total
            totals[variable][1] += total_sq
            totals[variable][2] += weight_mass
    stats = {}
    for variable, (total, total_sq, weight_mass) in totals.items():
        if weight_mass <= 0.0:
            raise ValueError(f"No finite reference values for {variable}.")
        mean = total / weight_mass
        variance = max(total_sq / weight_mass - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        stats[variable] = {"mean": float(mean), "std": std if std > 1e-8 else 1.0}
    return stats


def streaming_pointwise_reference_stats(cfg, ds, variables, time_indices):
    """Compute ordinary pooled-grid-cell moments for affine fake matching."""
    chunk_size = max(1, int(baseline_get(cfg, "feature_chunk_size", 32)))
    totals = {variable: [0.0, 0.0, 0.0] for variable in variables}
    for start in tqdm(range(0, len(time_indices), chunk_size),
                      desc="Pointwise reference moments"):
        chunk = ds.isel(time=time_indices[start:start + chunk_size])
        for variable in variables:
            values = np.asarray(
                chunk[variable].transpose("time", "latitude", "longitude").values,
                dtype=np.float64,
            )
            total, total_sq, count = pointwise_moment_totals(values)
            totals[variable][0] += total
            totals[variable][1] += total_sq
            totals[variable][2] += count
    result = {}
    for variable, (total, total_sq, count) in totals.items():
        if count <= 0.0:
            raise ValueError(f"No finite pointwise values for {variable}.")
        mean = total / count
        variance = max(total_sq / count - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        result[variable] = {"mean": float(mean), "std": std if std > 1e-8 else 1.0}
    return result


def global_moment_correction(source_stats, target_stats, variables):
    """Fit one per-variable global affine forecast-to-ERA5 correction."""
    corrections = {}
    for variable in variables:
        source_mean = float(source_stats[variable]["mean"])
        source_std = float(source_stats[variable]["std"])
        target_mean = float(target_stats[variable]["mean"])
        target_std = float(target_stats[variable]["std"])
        if not np.isfinite(source_std) or source_std <= 1e-8:
            raise ValueError(
                f"Cannot moment-match {variable}: forecast training standard deviation "
                f"is {source_std!r}."
            )
        corrections[variable] = {
            "forecast_train_mean": source_mean,
            "forecast_train_std": source_std,
            "era5_train_mean": target_mean,
            "era5_train_std": target_std,
            "scale": target_std / source_std,
            "shift": target_mean - source_mean * target_std / source_std,
        }
    return corrections


def apply_global_moment_correction(values, correction):
    """Apply a train-fitted global affine forecast calibration."""
    return (
        (np.asarray(values, dtype=np.float32) - float(correction["forecast_train_mean"]))
        * float(correction["scale"]) + float(correction["era5_train_mean"])
    ).astype(np.float32)


def mmd_global_moment_matching_settings(cfg):
    """Configuration for the isolated train-fitted MMD calibration experiment."""
    return baseline_get(cfg, "mmd_global_moment_matching", {}) or {}


def mmd_global_moment_matching_output_dir(cfg, variables):
    settings = mmd_global_moment_matching_settings(cfg)
    return baseline_output_dir(cfg, variables) / str(
        settings.get("output_subdir", "mmd_global_moment_matching")
    )


def mmd_safe_filename(value):
    return "".join(character if character.isalnum() else "_" for character in str(value))


def write_mmd_global_moment_matching(rows, output_root):
    path = output_root / "data" / "mmd_global_moment_matching.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.unlink(missing_ok=True)
        return path
    fields = list(rows[0])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_mmd_global_moment_matching(output_root):
    path = output_root / "data" / "mmd_global_moment_matching.csv"
    if not path.is_file():
        return []
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("lead_hour", "n_train_pairs", "n_test_pairs", "pairwise_n_samples"):
            row[key] = int(row[key])
        for key, value in list(row.items()):
            if key in {"label", "variables"} or key in {"lead_hour", "n_train_pairs", "n_test_pairs", "pairwise_n_samples"}:
                continue
            row[key] = float(value)
    return rows


def plot_mmd_global_moment_matching(rows, output_root):
    """Compare raw and train-moment-matched test MMD by forecast lead."""
    if not rows:
        return []
    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    labels = sorted({row["label"] for row in rows})
    colors = model_colors(labels)
    for label_index, (color, label) in enumerate(zip(colors, labels)):
        series = sorted((row for row in rows if row["label"] == label), key=lambda row: row["lead_hour"])
        leads = [row["lead_hour"] for row in series]
        axis.plot(leads, [row["raw_mmd_rbf"] for row in series], marker=series_marker(2 * label_index), color=color,
                  linestyle="--", alpha=0.7, label=f"{label} raw")
        axis.plot(leads, [row["matched_mmd_rbf"] for row in series], marker=series_marker(2 * label_index + 1), color=color,
                  linestyle="-", linewidth=2.0, label=f"{label} moment-matched")
    axis.set(
        xlabel="Lead time (hours)", ylabel="RBF MMD",
        title="Test MMD after train-fitted global mean/variance matching",
    )
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    path = plot_root / "lead_time_raw_vs_global_moment_matched_mmd.png"
    save_figure_bundle(figure, path, plot_type="mmd_global_moment_matching", payload={"rows": np.asarray([str(row) for row in rows])}, dpi=220)
    plt.close(figure)
    return [path]


def evaluate_mmd_global_moment_matching(cfg):
    """Evaluate raw and train-fitted globally moment-matched forecast MMD only."""
    variables = variables_from_config(cfg)
    settings = mmd_global_moment_matching_settings(cfg)
    if not bool(settings.get("enabled", False)):
        return []
    output_root = mmd_global_moment_matching_output_dir(cfg, variables)
    real_ds = select_level(safe_open_dataset(cfg.real_nc_file), cfg.get("level"))
    requested_leads = {int(value) for value in cfg.lead_times}
    max_samples = int(settings.get("eval_samples", baseline_get(cfg, "eval_samples", 0)))
    rows = []
    scratch_dir = baseline_get(cfg, "scratch_dir", None)
    with tempfile.TemporaryDirectory(
        prefix="mmd-global-moment-matching-",
        dir=None if scratch_dir is None else str(scratch_dir),
    ) as temporary_dir:
        for label, configured_paths in baseline_get(cfg, "forecast_files", {}).items():
            opened = [normalize_prediction_timedelta(safe_open_dataset(path)) for path in configured_paths]
            forecast_ds = select_level(concatenate_forecasts(opened), cfg.get("level"))
            missing = [variable for variable in variables if variable not in forecast_ds.data_vars]
            if missing:
                print(f"Skipping {label}: missing variables {missing}")
                forecast_ds.close()
                continue
            for lead_idx, lead_hour in enumerate(lead_hours(forecast_ds)):
                lead_hour = int(lead_hour)
                if lead_hour not in requested_leads:
                    continue
                train_pairs = evenly_spaced_pairs(
                    [pair for pair in forecast_pairs(forecast_ds, real_ds, cfg, "train", [lead_hour])
                     if pair.lead_index == lead_idx], max_samples,
                )
                test_pairs = evenly_spaced_pairs(
                    [pair for pair in forecast_pairs(forecast_ds, real_ds, cfg, "test", [lead_hour])
                     if pair.lead_index == lead_idx], max_samples,
                )
                if not train_pairs or not test_pairs:
                    print(f"Skipping {label} +{lead_hour}h: missing paired train or test samples.")
                    continue
                train_forecast = forecast_ds.isel(prediction_timedelta=lead_idx)
                train_forecast_stats = streaming_pointwise_reference_stats(
                    cfg, train_forecast, variables, [pair.forecast_index for pair in train_pairs]
                )
                train_era5 = real_ds.isel(time=[pair.era5_index for pair in train_pairs])
                train_era5_stats = streaming_pointwise_reference_stats(
                    cfg, train_era5, variables, list(range(len(train_pairs)))
                )
                correction = global_moment_correction(train_forecast_stats, train_era5_stats, variables)
                test_era5 = real_ds.isel(time=[pair.era5_index for pair in test_pairs])
                test_indices = list(range(len(test_pairs)))
                test_stats = streaming_reference_stats(cfg, test_era5, variables, test_indices)
                reference = streaming_joint_features(
                    cfg, test_era5, variables, test_stats, test_indices, ["mmd_rbf"],
                    Path(temporary_dir) / f"{mmd_safe_filename(label)}_{lead_hour}_reference.dat",
                    description=f"{label} +{lead_hour}h ERA5 test MMD reference",
                )
                forecast_indices = [pair.forecast_index for pair in test_pairs]
                raw = streaming_joint_features(
                    cfg, forecast_ds, variables, test_stats, forecast_indices, ["mmd_rbf"],
                    Path(temporary_dir) / f"{mmd_safe_filename(label)}_{lead_hour}_raw.dat",
                    lead_idx=lead_idx, description=f"{label} +{lead_hour}h raw forecast MMD",
                )
                matched = streaming_joint_features(
                    cfg, forecast_ds, variables, test_stats, forecast_indices, ["mmd_rbf"],
                    Path(temporary_dir) / f"{mmd_safe_filename(label)}_{lead_hour}_matched.dat",
                    lead_idx=lead_idx, global_moment_correction=correction,
                    description=f"{label} +{lead_hour}h moment-matched forecast MMD",
                )
                raw_mmd = mmd_rbf_distance(
                    raw["vectors"], reference["vectors"], cfg, reference["mmd_field_slices"]
                )
                matched_mmd = mmd_rbf_distance(
                    matched["vectors"], reference["vectors"], cfg, reference["mmd_field_slices"]
                )
                row = {
                    "label": label, "variables": ",".join(variables), "lead_hour": lead_hour,
                    "n_train_pairs": len(train_pairs), "n_test_pairs": len(test_pairs),
                    "pairwise_n_samples": min(len(raw["vectors"]), len(reference["vectors"])),
                    "raw_mmd_rbf": raw_mmd, "matched_mmd_rbf": matched_mmd,
                }
                for variable in variables:
                    for key, value in correction[variable].items():
                        row[f"{variable}__{key}"] = value
                rows.append(row)
                close_feature_memmaps(reference)
                close_feature_memmaps(raw)
                close_feature_memmaps(matched)
            forecast_ds.close()
    output_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=output_root / "resolved_config.yaml", resolve=True)
    csv_path = write_mmd_global_moment_matching(rows, output_root)
    real_ds.close()
    return [output_root / "resolved_config.yaml", csv_path] if csv_path.is_file() else [output_root / "resolved_config.yaml"]


def projection_directions(cfg, n_dimensions):
    """Return the deterministic directions used by both sliced-Wasserstein variants."""
    rng = np.random.default_rng(int(baseline_get(cfg, "swd_seed", 0)))
    directions = rng.normal(size=(int(baseline_get(cfg, "swd_projections", 64)), n_dimensions))
    norms = np.linalg.norm(directions, axis=1)
    directions = directions[norms > 1e-12]
    norms = norms[norms > 1e-12]
    return directions / norms[:, None]


def close_feature_memmaps(features):
    """Flush and close any disk-backed arrays owned by a compact feature summary."""
    for key in ("scwd_responses",):
        values = features.get(key)
        if isinstance(values, np.memmap):
            values.flush()
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()


def streaming_joint_features(
    cfg,
    ds,
    variables,
    reference_stats,
    time_indices,
    metric_names,
    scwd_path,
    *,
    lead_idx=None,
    corruption_type=None,
    severity=0.0,
    global_moment_correction=None,
    histogram_target=None,
    histogram_coordinate=None,
    description="distribution features",
):
    """Build compact all-time metric summaries in bounded memory."""
    if lead_idx is not None and "prediction_timedelta" in ds.dims:
        ds = ds.isel(prediction_timedelta=lead_idx)
    if not time_indices:
        raise ValueError(f"No timestamps available for {description}.")
    requested = set(metric_names)
    pairwise_cap = int(baseline_get(cfg, "pairwise_eval_samples", 256))
    pairwise_positions = set(pairwise_sample_positions(len(time_indices), pairwise_cap))
    chunk_size = max(1, int(baseline_get(cfg, "feature_chunk_size", 32)))
    latitudes = np.asarray(ds[variables[0]].latitude.values, dtype=np.float64)
    longitudes = np.asarray(ds[variables[0]].longitude.values, dtype=np.float64)
    per_variable_pixels = max(1, int(baseline_get(cfg, "swd_pixels", 4096)) // len(variables))
    need_swd = bool({"sliced_wasserstein", "sliced_wasserstein_lon_corrected"} & requested)
    need_global_mean_wd = "global_mean_wasserstein" in requested
    need_scwd = bool({"scwd", "scwd_area_weighted"} & requested)
    need_pair_fields = "crps_like_field_energy" in requested
    need_pair_vectors = "mmd_rbf" in requested
    directions = None
    swd_projections = None
    swd_lon_projections = None
    pair_fields = []
    pair_vectors = []
    global_means = []
    pair_count = 0
    spectrum_total = None
    value_total = 0.0
    value_total_sq = 0.0
    value_weight_mass = 0.0
    n_valid = 0

    scwd_responses = None
    numpy_weights = None
    torch_weights = None
    device = None
    if need_scwd:
        reference_grid = {"latitudes": latitudes, "longitudes": longitudes}
        weights = scwd_weight_vectors(reference_grid, cfg)
        n_pixels = len(latitudes) * len(longitudes)
        device = torch_metric_device(cfg)
        if device is None:
            numpy_weights = scwd_sparse_weight_matrix_numpy(weights, n_pixels)
            n_anchors = 0 if numpy_weights is None else numpy_weights.shape[0]
        else:
            torch_weights = scwd_sparse_weight_matrix_torch(weights, n_pixels, cfg, device)
            n_anchors = 0 if torch_weights is None else torch_weights.shape[0]
        if n_anchors == 0:
            raise ValueError("SCWD produced no spatial anchors for the configured grid.")
        scwd_responses = np.memmap(
            scwd_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(time_indices), len(variables), n_anchors),
        )

    base_seed = int(baseline_get(cfg, "corruption_seed", 0))
    special_corruption = (
        corruption_type in STRUCTURED_NEAR_NULL_CORRUPTIONS | DATA_DEPENDENT_CORRUPTIONS
    )
    donor_positions = None
    if corruption_type == "hemisphere_splice":
        donor_positions = deranged_sample_positions(len(time_indices), base_seed)
    elif corruption_type == "field_splice":
        donor_positions = fieldwise_deranged_sample_positions(
            len(time_indices), len(variables), base_seed
        )
    for start in tqdm(range(0, len(time_indices), chunk_size), desc=description):
        chunk_indices = time_indices[start:start + chunk_size]
        chunk_ds = ds.isel(time=chunk_indices)
        raw_by_variable = {
            variable: np.asarray(
                chunk_ds[variable].transpose("time", "latitude", "longitude").values,
                dtype=np.float32,
            )
            for variable in variables
        }
        donor_raw_by_variable = None
        if donor_positions is not None:
            donor_raw_by_variable = {}
            for channel, variable in enumerate(variables):
                positions = donor_positions if donor_positions.ndim == 1 else donor_positions[channel]
                donor_time_indices = [
                    time_indices[int(positions[position])]
                    for position in range(start, start + len(chunk_indices))
                ]
                donor_chunk = ds.isel(time=donor_time_indices)
                donor_raw_by_variable[variable] = np.asarray(
                    donor_chunk[variable].transpose("time", "latitude", "longitude").values,
                    dtype=np.float32,
                )
        valid_fields = []
        valid_positions = []
        for local_idx, time_idx in enumerate(chunk_indices):
            raw_fields = [raw_by_variable[variable][local_idx] for variable in variables]
            if global_moment_correction is not None:
                raw_fields = [
                    apply_global_moment_correction(
                        values, global_moment_correction[variable]
                    )
                    for variable, values in zip(variables, raw_fields)
                ]
            if any(should_skip_field(cfg, values) for values in raw_fields):
                continue
            donor_fields = None
            if donor_raw_by_variable is not None:
                donor_fields = [
                    donor_raw_by_variable[variable][local_idx] for variable in variables
                ]
                if any(should_skip_field(cfg, values) for values in donor_fields):
                    continue
            standardized = np.stack(
                [
                    (canonical_latlon(values, latitudes) - reference_stats[variable]["mean"])
                    / reference_stats[variable]["std"]
                    for variable, values in zip(variables, raw_fields)
                ]
            ).astype(np.float32)
            if corruption_type is not None:
                if special_corruption:
                    donor = None
                    if donor_fields is not None:
                        donor = np.stack(
                            [
                                (canonical_latlon(values, latitudes) - reference_stats[variable]["mean"])
                                / reference_stats[variable]["std"]
                                for variable, values in zip(variables, donor_fields)
                            ]
                        ).astype(np.float32)
                    standardized = apply_special_baseline_corruption(
                        standardized, corruption_type, severity, latitudes, cfg, donor,
                        random_seed=corruption_sample_seed(base_seed, corruption_type, time_idx),
                    )
                else:
                    seed = corruption_sample_seed(base_seed, corruption_type, time_idx)
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(seed)
                        standardized = apply_configured_corruption(
                            torch.from_numpy(standardized), corruption_type, float(severity)
                        ).detach().cpu().numpy().astype(np.float32)
            if histogram_target is not None:
                kind = "corruption" if corruption_type is not None else "forecast"
                standardized = match_standardized(
                    cfg, standardized, variables,
                    {v: reference_stats[v]["mean"] for v in variables},
                    {v: reference_stats[v]["std"] for v in variables},
                    "standard", kind, histogram_target, histogram_coordinate,
                )
            valid_fields.append(standardized)
            valid_positions.append(start + local_idx)

        if not valid_fields:
            continue
        field_batch = np.stack(valid_fields)
        batch_size = field_batch.shape[0]

        if need_global_mean_wd:
            global_means.append(area_weighted_global_means(field_batch, latitudes))

        # Mean-bias and standard-ratio baselines are pointwise grid-cell moments;
        # area weighting remains specific to metrics that explicitly require it.
        total, total_sq, weight_mass = pointwise_moment_totals(field_batch)
        value_total += total
        value_total_sq += total_sq
        value_weight_mass += weight_mass

        batch_spectra = []
        weighted_vectors = []
        unweighted_vectors = []
        for sample_fields in field_batch:
            sample_spectra = [zonal_energy_spectrum(field, latitudes) for field in sample_fields]
            batch_spectra.append(np.concatenate(sample_spectra))
            if need_swd or need_pair_vectors:
                weighted_vectors.append(
                    np.concatenate(
                        [field_vector(field, latitudes, per_variable_pixels) for field in sample_fields]
                    )
                )
            if need_swd:
                unweighted_vectors.append(
                    np.concatenate(
                        [unweighted_field_vector(field, latitudes, per_variable_pixels) for field in sample_fields]
                    )
                )
        batch_spectrum_total = np.sum(np.stack(batch_spectra), axis=0, dtype=np.float64)
        spectrum_total = (
            batch_spectrum_total if spectrum_total is None else spectrum_total + batch_spectrum_total
        )

        if need_swd:
            weighted_vectors_array = np.asarray(weighted_vectors, dtype=np.float32)
            unweighted_vectors_array = np.asarray(unweighted_vectors, dtype=np.float32)
            if directions is None:
                directions = projection_directions(cfg, weighted_vectors_array.shape[1])
                n_projection = directions.shape[0]
                swd_projections = np.empty((len(time_indices), n_projection), dtype=np.float32)
                swd_lon_projections = np.empty((len(time_indices), n_projection), dtype=np.float32)
            swd_projections[n_valid:n_valid + batch_size] = unweighted_vectors_array @ directions.T
            swd_lon_projections[n_valid:n_valid + batch_size] = weighted_vectors_array @ directions.T
        if need_scwd:
            scwd_responses[n_valid:n_valid + batch_size] = scwd_response_batch(
                field_batch, numpy_weights, torch_weights, cfg, device
            )

        weighted_vectors_array = (
            np.asarray(weighted_vectors, dtype=np.float32) if weighted_vectors else None
        )
        for batch_idx, original_position in enumerate(valid_positions):
            if original_position not in pairwise_positions:
                continue
            pair_count += 1
            if need_pair_fields:
                pair_fields.append(np.concatenate(field_batch[batch_idx], axis=0))
            if need_pair_vectors:
                pair_vectors.append(weighted_vectors_array[batch_idx])
        n_valid += batch_size

    if n_valid == 0:
        raise ValueError(f"No valid fields available for {description}.")
    if value_weight_mass <= 0.0:
        raise ValueError(f"No finite weighted values available for {description}.")
    mean = value_total / value_weight_mass
    variance = max(value_total_sq / value_weight_mass - mean * mean, 0.0)
    if scwd_responses is not None:
        scwd_responses.flush()

    return {
        "field_names": tuple(str(variable) for variable in variables),
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
        "spectrum": spectrum_total / n_valid,
        "swd_projections": (
            swd_projections[:n_valid] if swd_projections is not None else np.empty((0, 0), dtype=np.float32)
        ),
        "swd_lon_projections": (
            swd_lon_projections[:n_valid]
            if swd_lon_projections is not None
            else np.empty((0, 0), dtype=np.float32)
        ),
        "scwd_responses": (
            scwd_responses[:n_valid]
            if scwd_responses is not None
            else np.empty((0, 0, 0), dtype=np.float32)
        ),
        "fields": (
            np.stack(pair_fields).astype(np.float32)
            if pair_fields
            else np.empty((0, 0, 0), dtype=np.float32)
        ),
        "vectors": (
            np.stack(pair_vectors).astype(np.float32)
            if pair_vectors
            else np.empty((0, 0), dtype=np.float32)
        ),
        "unweighted_vectors": np.empty((0, 0), dtype=np.float32),
        "mmd_field_slices": [
            (field_index * min(len(latitudes) * len(longitudes), per_variable_pixels),
             (field_index + 1) * min(len(latitudes) * len(longitudes), per_variable_pixels))
            for field_index in range(len(variables))
        ],
        "global_means": (
            np.concatenate(global_means, axis=0).astype(np.float32)
            if global_means
            else np.empty((0, len(variables)), dtype=np.float32)
        ),
        "latitudes": latitudes,
        "longitudes": longitudes,
        "n_valid_samples": int(n_valid),
        "pairwise_n_samples": int(pair_count),
    }


def distribution_metric_values(candidate, reference, cfg, metric_names=None):
    """Compute scalar distributional metrics from candidate/reference features."""
    requested = set(metric_names or AVAILABLE_METRICS)
    n_projections = int(baseline_get(cfg, "swd_projections", 64))
    seed = int(baseline_get(cfg, "swd_seed", 0))
    cramer_wold_bandwidth = baseline_get(cfg, "cramer_wold_bandwidth", None)
    candidate_unweighted = candidate.get("unweighted_vectors", candidate["vectors"])
    reference_unweighted = reference.get("unweighted_vectors", reference["vectors"])

    metrics = {}
    if "mean_bias" in requested:
        metrics["mean_bias"] = candidate["mean"] - reference["mean"]
    if "mean_abs_diff" in requested:
        metrics["mean_abs_diff"] = abs(candidate["mean"] - reference["mean"])
    if "std_ratio_error" in requested:
        metrics["std_ratio_error"] = candidate["std"] / (reference["std"] + 1e-12) - 1.0
    if "std_abs_diff" in requested:
        metrics["std_abs_diff"] = abs(candidate["std"] - reference["std"])
    if "crps_like_field_energy" in requested:
        metrics["crps_like_field_energy"] = crps_like_field_energy(candidate, reference, cfg)
    if {"zonal_energy_spectrum_l2", "zonal_energy_spectrum_log_l2"} & requested:
        candidate_spectrum = candidate["spectrum"]
        reference_spectrum = reference["spectrum"]
        common_spectrum_len = min(candidate_spectrum.size, reference_spectrum.size)
        if "zonal_energy_spectrum_l2" in requested:
            metrics["zonal_energy_spectrum_l2"] = (
                float(
                    np.linalg.norm(
                        candidate_spectrum[:common_spectrum_len] - reference_spectrum[:common_spectrum_len]
                    )
                    / (np.linalg.norm(reference_spectrum[:common_spectrum_len]) + 1e-12)
                )
                if common_spectrum_len
                else np.nan
            )
        if "zonal_energy_spectrum_log_l2" in requested:
            metrics["zonal_energy_spectrum_log_l2"] = zonal_energy_spectrum_log_l2(
                candidate_spectrum, reference_spectrum, cfg
            )
    if "sliced_wasserstein" in requested:
        if "swd_projections" in candidate and "swd_projections" in reference:
            metrics["sliced_wasserstein"] = sliced_wasserstein_from_projections(
                candidate["swd_projections"], reference["swd_projections"]
            )
        else:
            metrics["sliced_wasserstein"] = sliced_wasserstein_distance(
                candidate_unweighted, reference_unweighted, n_projections, seed, cfg
            )
    if "sliced_wasserstein_lon_corrected" in requested:
        if "swd_lon_projections" in candidate and "swd_lon_projections" in reference:
            metrics["sliced_wasserstein_lon_corrected"] = sliced_wasserstein_from_projections(
                candidate["swd_lon_projections"], reference["swd_lon_projections"]
            )
        else:
            metrics["sliced_wasserstein_lon_corrected"] = sliced_wasserstein_distance(
                candidate["vectors"], reference["vectors"], n_projections, seed, cfg
            )
    if "global_mean_wasserstein" in requested:
        metrics["global_mean_wasserstein"] = vissio_global_mean_wasserstein(
            candidate.get("global_means", np.empty((0, 0))),
            reference.get("global_means", np.empty((0, 0))),
            baseline_get(cfg, "global_mean_wd_bins", 20),
        )
    if "sliced_cramer_wold" in requested:
        metrics["sliced_cramer_wold"] = sliced_cramer_wold_distance(
            candidate_unweighted, reference_unweighted, n_projections, seed, cramer_wold_bandwidth
        )
    if "sliced_cramer_wold_lon_corrected" in requested:
        metrics["sliced_cramer_wold_lon_corrected"] = sliced_cramer_wold_distance(
            candidate["vectors"], reference["vectors"], n_projections, seed, cramer_wold_bandwidth
        )
    if "mmd_rbf" in requested:
        metrics["mmd_rbf"] = mmd_rbf_distance(
            candidate["vectors"], reference["vectors"], cfg,
            reference.get("mmd_field_slices", candidate.get("mmd_field_slices")),
        )
    if "scwd_area_weighted" in requested:
        metrics["scwd_area_weighted"] = scwd_area_weighted_from_responses(candidate, reference, cfg)
    if "scwd" in requested:
        metrics["scwd"] = spherical_convolutional_wasserstein(candidate, reference, cfg)
    return metrics


def selected_distribution_metric_values(candidate, reference, metric_names, cfg):
    """Compute selected full-distribution metrics."""
    metrics = distribution_metric_values(candidate, reference, cfg, metric_names)
    metrics["n_samples"] = int(candidate.get("n_valid_samples", 0))
    metrics["pairwise_n_samples"] = int(candidate.get("pairwise_n_samples", 0))
    return metrics


def variable_tag(variables):
    """Return a collision-safe output tag for the configured fields."""
    return "__".join(str(variable).replace(" ", "_") for variable in variables)


def joint_variable_name(variables):
    """Return the output variable label for the configured joint sample."""
    return "all_fields" if len(variables) > 1 else variables[0]


def reference_features_from_config(cfg, truth_ds, variables, metric_names, temporary_dir):
    """Fit metric parameters on test ERA5 and retain train ERA5 for a shift diagnostic."""
    max_samples = int(baseline_get(cfg, "eval_samples", cfg_get(cfg, "max_samples", 100)))
    evaluation_range = baseline_get(cfg, "evaluation_time_range")
    evaluation_ds = select_level(select_time_ranges(truth_ds, evaluation_range), cfg.get("level"))
    evaluation_indices = sample_time_indices(evaluation_ds, max_samples)
    if not evaluation_indices:
        raise ValueError("No ERA5 samples in baseline.evaluation_time_range.")
    reference_stats = streaming_reference_stats(cfg, evaluation_ds, variables, evaluation_indices)
    evaluation_features = streaming_joint_features(
        cfg,
        evaluation_ds,
        variables,
        reference_stats,
        evaluation_indices,
        metric_names,
        Path(temporary_dir) / "evaluation-reference-scwd.dat",
        description="ERA5 test-reference features",
    )

    train_ranges = baseline_get(cfg, "reference_real_ranges", cfg_get(cfg, "train_real_range", None))
    train_ds = select_level(select_time_ranges(truth_ds, train_ranges), cfg.get("level"))
    train_indices = sample_time_indices(train_ds, max_samples)
    if not train_indices:
        raise ValueError("No ERA5 samples in baseline.reference_real_ranges.")
    train_features = streaming_joint_features(
        cfg,
        train_ds,
        variables,
        reference_stats,
        train_indices,
        metric_names,
        Path(temporary_dir) / "train-shift-reference-scwd.dat",
        description="ERA5 train-shift features",
    )
    return evaluation_features, train_features, reference_stats


def evaluate_era5_train_shift_metrics(evaluation_features, train_features, variables, metric_names, cfg):
    """Report the distinct ERA5 train-to-test distribution-shift diagnostic."""
    metrics = selected_distribution_metric_values(evaluation_features, train_features, metric_names, cfg)
    row = {
        "label": "ERA5 Test vs Train",
        "variable": joint_variable_name(variables),
        "lead_hour": 0,
        "n_samples": metrics["n_samples"],
        "pairwise_n_samples": metrics["pairwise_n_samples"],
    }
    row.update({metric_name: metrics[metric_name] for metric_name in metric_names})
    return row



def global_mean_wasserstein_diagnostic(
    candidate, reference, cfg, label, lead_hour, scalar_distance=np.nan, *,
    comparison_kind="forecast", severity=None, row=None,
):
    """Retain joint global-mean samples and the row receiving the final score."""
    return {
        "label": str(label),
        "comparison_kind": str(comparison_kind),
        "severity": np.nan if severity is None else float(severity),
        "lead_hour": int(lead_hour),
        "candidate": np.asarray(candidate["global_means"], dtype=np.float32),
        "reference": np.asarray(reference["global_means"], dtype=np.float32),
        "distance": float(scalar_distance),
        "n_bins": int(baseline_get(cfg, "global_mean_wd_bins", 20)),
        "row": row,
    }


def finalize_global_mean_wasserstein(diagnostics, cfg):
    """Fit one pooled experiment grid and recompute every joint GWD comparison."""
    if not diagnostics:
        return None
    n_bins = int(baseline_get(cfg, "global_mean_wd_bins", 20))
    distributions = []
    for item in diagnostics:
        distributions.extend([item["candidate"], item["reference"]])
    grid = fit_vissio_ulam_grid(distributions, n_bins=n_bins)
    for item in tqdm(diagnostics, desc="Joint global-mean Ulam W2"):
        item["distance"] = vissio_global_mean_wasserstein(
            item["candidate"], item["reference"], n_bins=n_bins, grid=grid,
        )
        item["grid"] = grid
        if item.get("row") is not None:
            item["row"]["global_mean_wasserstein"] = item["distance"]
    return grid


def write_global_mean_wasserstein_diagnostics(diagnostics, variables, output_root):
    """Persist joint GWD samples, pooled grid, sparse Ulam measures, and scores."""
    output_path = output_root / "data" / "global_mean_wasserstein_distributions.nc"
    if not diagnostics:
        output_path.unlink(missing_ok=True)
        return
    n_fields = len(variables)
    grid = diagnostics[0].get("grid")
    if grid is None:
        raise ValueError("Joint GWD diagnostics were not finalized with an experiment-wide grid.")
    candidate_size = max(item["candidate"].shape[0] for item in diagnostics)
    reference_size = max(item["reference"].shape[0] for item in diagnostics)
    candidate = np.full((len(diagnostics), n_fields, candidate_size), np.nan, dtype=np.float32)
    reference = np.full((len(diagnostics), n_fields, reference_size), np.nan, dtype=np.float32)
    candidate_count = np.zeros(len(diagnostics), dtype=np.int64)
    reference_count = np.zeros(len(diagnostics), dtype=np.int64)
    measures = []
    for item in diagnostics:
        cand_support, cand_mass, cand_cells = vissio_ulam_measure(item["candidate"], grid)
        ref_support, ref_mass, ref_cells = vissio_ulam_measure(item["reference"], grid)
        measures.append((cand_support, cand_mass, cand_cells, ref_support, ref_mass, ref_cells))
    candidate_support_size = max(len(item[0]) for item in measures)
    reference_support_size = max(len(item[3]) for item in measures)
    candidate_support = np.full((len(diagnostics), candidate_support_size, n_fields), np.nan)
    reference_support = np.full((len(diagnostics), reference_support_size, n_fields), np.nan)
    candidate_cells = np.full((len(diagnostics), candidate_support_size, n_fields), -1, dtype=np.int32)
    reference_cells = np.full((len(diagnostics), reference_support_size, n_fields), -1, dtype=np.int32)
    candidate_mass = np.zeros((len(diagnostics), candidate_support_size), dtype=np.float64)
    reference_mass = np.zeros((len(diagnostics), reference_support_size), dtype=np.float64)
    for comparison, (item, measure) in enumerate(zip(diagnostics, measures)):
        candidate_values, reference_values = item["candidate"], item["reference"]
        if candidate_values.shape[1] != n_fields or reference_values.shape[1] != n_fields:
            raise ValueError("Global-mean diagnostic field count does not match configured variables.")
        candidate_count[comparison], reference_count[comparison] = len(candidate_values), len(reference_values)
        candidate[comparison, :, :len(candidate_values)] = candidate_values.T
        reference[comparison, :, :len(reference_values)] = reference_values.T
        cand_support, cand_mass, cand_cells, ref_support, ref_mass, ref_cells = measure
        candidate_support[comparison, :len(cand_support)] = cand_support
        reference_support[comparison, :len(ref_support)] = ref_support
        candidate_cells[comparison, :len(cand_cells)] = cand_cells
        reference_cells[comparison, :len(ref_cells)] = ref_cells
        candidate_mass[comparison, :len(cand_mass)] = cand_mass
        reference_mass[comparison, :len(ref_mass)] = ref_mass
    dataset = xr.Dataset(
        data_vars={
            "candidate_global_mean": (("comparison", "field", "candidate_sample"), candidate),
            "reference_global_mean": (("comparison", "field", "reference_sample"), reference),
            "candidate_sample_count": ("comparison", candidate_count),
            "reference_sample_count": ("comparison", reference_count),
            "global_mean_wasserstein": ("comparison", np.asarray([item["distance"] for item in diagnostics])),
            "ulam_lower": ("field", np.asarray(grid["lower"], dtype=np.float64)),
            "ulam_upper": ("field", np.asarray(grid["upper"], dtype=np.float64)),
            "candidate_ulam_support": (("comparison", "candidate_support", "field"), candidate_support),
            "reference_ulam_support": (("comparison", "reference_support", "field"), reference_support),
            "candidate_ulam_cell": (("comparison", "candidate_support", "field"), candidate_cells),
            "reference_ulam_cell": (("comparison", "reference_support", "field"), reference_cells),
            "candidate_ulam_mass": (("comparison", "candidate_support"), candidate_mass),
            "reference_ulam_mass": (("comparison", "reference_support"), reference_mass),
        },
        coords={
            "comparison": np.arange(len(diagnostics), dtype=np.int64),
            "label": ("comparison", [item["label"] for item in diagnostics]),
            "comparison_kind": ("comparison", [item.get("comparison_kind", "forecast") for item in diagnostics]),
            "severity": ("comparison", [item.get("severity", np.nan) for item in diagnostics]),
            "lead_hour": ("comparison", [item["lead_hour"] for item in diagnostics]),
            "field": list(variables),
        },
        attrs={
            "description": "Joint Vissio-style global-mean W2 diagnostics.",
            "schema_version": 2,
            "estimator": "joint_ulam_w2",
            "n_bins": int(grid["n_bins"]),
            "binning": "One experiment-wide pooled min/max grid; normalized cell-center Euclidean ground cost.",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)
    dataset.close()
    print(f"Saved global-mean Wasserstein distributions to: {output_path}")


def read_global_mean_wasserstein_diagnostics(output_root):
    """Load persisted global-mean distributions for artifact-only plotting."""
    path = output_root / "data" / "global_mean_wasserstein_distributions.nc"
    if not path.exists():
        return []
    diagnostics = []
    with xr.open_dataset(path) as dataset:
        if int(dataset.attrs.get("schema_version", 0)) < 2:
            raise ValueError(f"{path} uses marginal or pairwise-grid GWD; rerun standard metric evaluation.")
        required = {"candidate_global_mean", "reference_global_mean", "candidate_sample_count", "reference_sample_count"}
        if not required.issubset(dataset.variables):
            raise ValueError(f"{path} predates artifact-only plotting; rerun baseline evaluation once.")
        fields = [str(value) for value in dataset.field.values]
        for comparison in range(dataset.sizes["comparison"]):
            n_candidate = int(dataset.candidate_sample_count.values[comparison])
            n_reference = int(dataset.reference_sample_count.values[comparison])
            diagnostics.append({
                "label": str(dataset.label.values[comparison]),
                "comparison_kind": str(dataset.comparison_kind.values[comparison]),
                "severity": float(dataset.severity.values[comparison]),
                "lead_hour": int(dataset.lead_hour.values[comparison]),
                "candidate": np.asarray(dataset.candidate_global_mean.values[comparison, :, :n_candidate]).T,
                "reference": np.asarray(dataset.reference_global_mean.values[comparison, :, :n_reference]).T,
                "distance": float(dataset.global_mean_wasserstein.values[comparison]),
                "n_bins": int(dataset.attrs.get("n_bins", 20)),
                "fields": fields,
                "grid": {
                    "lower": np.asarray(dataset.ulam_lower.values),
                    "upper": np.asarray(dataset.ulam_upper.values),
                    "n_bins": int(dataset.attrs["n_bins"]),
                },
            })
    return diagnostics


def repeated_diagnostic_labels(diagnostics, representative_only=False):
    """Return all labels, or one representative label per comparison kind."""
    labels = sorted({
        (str(item.get("comparison_kind", "comparison")), str(item["label"]))
        for item in diagnostics
    })
    if not representative_only:
        return labels
    representatives = []
    seen_kinds = set()
    for comparison_kind, label in labels:
        if comparison_kind not in seen_kinds:
            representatives.append((comparison_kind, label))
            seen_kinds.add(comparison_kind)
    return representatives


def plot_global_mean_wasserstein_distributions(
    diagnostics, output_root, representative_only=False,
):
    """Overlay forecast and matched-ERA5 global-mean distributions by lead time."""
    if not diagnostics:
        return
    root = output_root / "plots" / "global_mean_wasserstein"
    root.mkdir(parents=True, exist_ok=True)
    for comparison_kind, label in repeated_diagnostic_labels(
        diagnostics, representative_only=representative_only,
    ):
        series = sorted(
            [item for item in diagnostics
             if str(item.get("comparison_kind", "comparison")) == comparison_kind
             and str(item["label"]) == label],
            key=lambda item: item["lead_hour"],
        )
        fields = series[0]["fields"]
        n_panels = len(series) * len(fields)
        n_cols = min(2, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        figure, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.6 * n_rows), squeeze=False)
        flat_axes = axes.ravel()
        panel = 0
        for item in series:
            for field_index, field in enumerate(fields):
                axis = flat_axes[panel]
                candidate = item["candidate"][:, field_index]
                reference = item["reference"][:, field_index]
                candidate = candidate[np.isfinite(candidate)]
                reference = reference[np.isfinite(reference)]
                if candidate.size and reference.size:
                    lower = float(item["grid"]["lower"][field_index])
                    upper = float(item["grid"]["upper"][field_index])
                    if upper > lower:
                        edges = np.linspace(lower, upper, item["n_bins"] + 1)
                    else:
                        edges = np.linspace(lower - 0.5, upper + 0.5, item["n_bins"] + 1)
                    axis.hist(reference, bins=edges, density=True, histtype="step", linewidth=1.8, color="black", label="ERA5 test")
                    axis.hist(np.clip(candidate, edges[0], edges[-1]), bins=edges, density=True, histtype="step", linewidth=1.8, color=CANDIDATE_COLOR, label=label)
                axis.set_title(f"+{item['lead_hour']} h | {field}\nJoint global-mean W2={item['distance']:.4g}", fontsize=10)
                axis.set_xlabel("Cosine-area-weighted global mean (standardized)")
                axis.set_ylabel("Density")
                axis.grid(True, alpha=0.25)
                panel += 1
        for axis in flat_axes[panel:]:
            axis.set_visible(False)
        flat_axes[0].legend(fontsize=9)
        figure.suptitle(f"Global-mean Wasserstein distributions: {label}", fontsize=14)
        figure.tight_layout(rect=[0, 0, 1, 0.94])
        output_path = root / f"{label.replace(' ', '_').replace('/', '_')}.png"
        payload = {"fields": np.asarray(fields)}
        for index, item in enumerate(series):
            payload[f"candidate_global_means_{index}"] = item["candidate"]
            payload[f"reference_global_means_{index}"] = item["reference"]
            payload[f"lead_hour_{index}"] = np.asarray(item["lead_hour"])
            payload[f"wasserstein_distance_{index}"] = np.asarray(item["distance"])
        save_figure_bundle(
            figure, output_path, plot_type="global_mean_wasserstein_distributions",
            payload=payload, dpi=220, bbox_inches="tight",
        )
        plt.close(figure)
        print(f"Saved global-mean Wasserstein distributions to: {output_path}")

def scwd_anchor_diagnostic(
    candidate, reference, cfg, label, lead_hour, scalar_scwd,
    comparison_kind="forecast", severity=None,
):
    """Build one labelled joint-SCWD anchor diagnostic."""
    costs = scwd_anchor_transport_costs(candidate, reference, cfg)
    n_lat = int(baseline_get(cfg, "scwd_anchor_lat_points", 60))
    n_lon = int(baseline_get(cfg, "scwd_anchor_lon_points", 120))
    if costs.size != n_lat * n_lon:
        raise ValueError(f"SCWD anchor count mismatch: got {costs.size}, expected {n_lat} x {n_lon}.")
    order = float(baseline_get(cfg, "scwd_order", 2.0))
    anchor_lats, anchor_lons = regular_sphere_centers(n_lat, n_lon)
    reconstructed = float(np.mean(costs) ** (1.0 / order))
    if not np.isclose(reconstructed, scalar_scwd, rtol=1e-6, atol=1e-8):
        raise RuntimeError(
            f"SCWD anchor contributions do not reconstruct scalar value: {reconstructed} != {scalar_scwd}."
        )
    local_wasserstein, top_distributions = scwd_anchor_joint_distributions(
        candidate, reference, cfg, baseline_get(cfg, "scwd_top_wasserstein_anchors",
                                                baseline_get(cfg, "scwd_top_w1_anchors", 6)),
    )
    mean_difference = scwd_anchor_mean_response_difference(candidate, reference, cfg)
    if local_wasserstein.size != costs.size or mean_difference.shape[-1] != costs.size:
        raise RuntimeError("SCWD diagnostic anchor counts do not match transport contributions.")
    field_names = list(candidate.get("field_names", ()))
    if len(field_names) != mean_difference.shape[0]:
        field_names = [f"field_{index}" for index in range(mean_difference.shape[0])]
    for item in top_distributions:
        lat_idx, lon_idx = divmod(item["anchor_index"], n_lon)
        item["latitude"] = float(anchor_lats[lat_idx])
        item["longitude"] = float(anchor_lons[lon_idx])
    return {
        "label": str(label),
        "comparison_kind": str(comparison_kind),
        "severity": np.nan if severity is None else float(severity),
        "lead_hour": int(lead_hour),
        "field_names": field_names,
        "anchor_latitudes": anchor_lats,
        "anchor_longitudes": anchor_lons,
        "anchor_transport_cost": costs.reshape(n_lat, n_lon),
        "anchor_local_wasserstein": local_wasserstein.reshape(n_lat, n_lon),
        "anchor_mean_response_difference": mean_difference.reshape(len(field_names), n_lat, n_lon),
        "top_wasserstein_distributions": top_distributions,
        "scwd": float(scalar_scwd),
        "scwd_order": order,
    }


def write_scwd_anchor_diagnostics(diagnostics, output_root):
    """Write versioned joint-SCWD diagnostics needed for artifact-only plotting."""
    output_path = output_root / "data" / "scwd_anchor_contributions.nc"
    if not diagnostics:
        output_path.unlink(missing_ok=True)
        return
    first = diagnostics[0]
    field_names = list(first["field_names"])
    if any(list(item["field_names"]) != field_names for item in diagnostics):
        raise ValueError("All SCWD diagnostics must use the same ordered fields.")
    top_count = max((len(item["top_wasserstein_distributions"]) for item in diagnostics), default=0)
    response_count = max(
        (max(top["candidate"].shape[0], top["reference"].shape[0])
         for item in diagnostics for top in item["top_wasserstein_distributions"]), default=0,
    )
    top_shape = (len(diagnostics), top_count)
    response_shape = (len(diagnostics), top_count, len(field_names), response_count)
    top_anchor_index = np.full(top_shape, -1, dtype=np.int64)
    top_wasserstein = np.full(top_shape, np.nan, dtype=np.float64)
    top_latitude = np.full(top_shape, np.nan, dtype=np.float64)
    top_longitude = np.full(top_shape, np.nan, dtype=np.float64)
    candidate_count = np.zeros(top_shape, dtype=np.int64)
    reference_count = np.zeros(top_shape, dtype=np.int64)
    candidate_response = np.full(response_shape, np.nan, dtype=np.float32)
    reference_response = np.full(response_shape, np.nan, dtype=np.float32)
    for comparison, item in enumerate(diagnostics):
        for rank, top in enumerate(item["top_wasserstein_distributions"]):
            candidate = np.asarray(top["candidate"], dtype=np.float32)
            reference = np.asarray(top["reference"], dtype=np.float32)
            top_anchor_index[comparison, rank] = int(top["anchor_index"])
            top_wasserstein[comparison, rank] = float(top["wasserstein"])
            top_latitude[comparison, rank] = float(top["latitude"])
            top_longitude[comparison, rank] = float(top["longitude"])
            candidate_count[comparison, rank] = candidate.shape[0]
            reference_count[comparison, rank] = reference.shape[0]
            candidate_response[comparison, rank, :, :candidate.shape[0]] = candidate.T
            reference_response[comparison, rank, :, :reference.shape[0]] = reference.T

    dataset = xr.Dataset(
        data_vars={
            "anchor_transport_cost": (("comparison", "anchor_latitude", "anchor_longitude"),
                                      np.stack([item["anchor_transport_cost"] for item in diagnostics])),
            "anchor_local_wasserstein": (("comparison", "anchor_latitude", "anchor_longitude"),
                                         np.stack([item["anchor_local_wasserstein"] for item in diagnostics])),
            "anchor_mean_response_difference": (
                ("comparison", "field", "anchor_latitude", "anchor_longitude"),
                np.stack([item["anchor_mean_response_difference"] for item in diagnostics]),
            ),
            "scwd": ("comparison", np.asarray([item["scwd"] for item in diagnostics], dtype=np.float64)),
            "top_anchor_index": (("comparison", "top_rank"), top_anchor_index),
            "top_local_wasserstein": (("comparison", "top_rank"), top_wasserstein),
            "top_latitude": (("comparison", "top_rank"), top_latitude),
            "top_longitude": (("comparison", "top_rank"), top_longitude),
            "candidate_response_count": (("comparison", "top_rank"), candidate_count),
            "reference_response_count": (("comparison", "top_rank"), reference_count),
            "candidate_response": (("comparison", "top_rank", "field", "response_sample"), candidate_response),
            "reference_response": (("comparison", "top_rank", "field", "response_sample"), reference_response),
        },
        coords={
            "comparison": np.arange(len(diagnostics), dtype=np.int64),
            "label": ("comparison", [item["label"] for item in diagnostics]),
            "comparison_kind": ("comparison", [item.get("comparison_kind", "forecast") for item in diagnostics]),
            "severity": ("comparison", [item.get("severity", np.nan) for item in diagnostics]),
            "lead_hour": ("comparison", [item["lead_hour"] for item in diagnostics]),
            "field": field_names,
            "anchor_latitude": first["anchor_latitudes"],
            "anchor_longitude": first["anchor_longitudes"],
        },
        attrs={
            "description": "Joint-SCWD per-anchor transport diagnostics.",
            "schema_version": 2,
            "estimator": "joint_empirical_w2",
            "scwd_order": first["scwd_order"],
            "reconstruction": "scwd = mean(anchor_transport_cost) ** (1 / scwd_order)",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)
    dataset.close()
    print(f"Saved SCWD anchor diagnostics to: {output_path}")


def read_scwd_anchor_diagnostics(output_root):
    """Load versioned joint-SCWD diagnostics without reopening source data."""
    path = output_root / "data" / "scwd_anchor_contributions.nc"
    if not path.exists():
        return []
    diagnostics = []
    with xr.open_dataset(path) as dataset:
        if int(dataset.attrs.get("schema_version", 0)) < 2:
            raise ValueError(
                f"{path} uses the legacy projected-channel SCWD schema; rerun standard metric evaluation."
            )
        required = {"candidate_response", "reference_response", "top_local_wasserstein"}
        if not required.issubset(dataset.variables):
            raise ValueError(f"{path} is missing joint-SCWD diagnostics; rerun baseline evaluation.")
        field_names = [str(value) for value in dataset.field.values]
        for comparison in range(dataset.sizes["comparison"]):
            distributions = []
            for rank in range(dataset.sizes.get("top_rank", 0)):
                anchor_index = int(dataset.top_anchor_index.values[comparison, rank])
                if anchor_index < 0:
                    continue
                n_candidate = int(dataset.candidate_response_count.values[comparison, rank])
                n_reference = int(dataset.reference_response_count.values[comparison, rank])
                distributions.append({
                    "anchor_index": anchor_index,
                    "wasserstein": float(dataset.top_local_wasserstein.values[comparison, rank]),
                    "latitude": float(dataset.top_latitude.values[comparison, rank]),
                    "longitude": float(dataset.top_longitude.values[comparison, rank]),
                    "candidate": np.asarray(dataset.candidate_response.values[comparison, rank, :, :n_candidate]).T,
                    "reference": np.asarray(dataset.reference_response.values[comparison, rank, :, :n_reference]).T,
                })
            diagnostics.append({
                "label": str(dataset.label.values[comparison]),
                "comparison_kind": (str(dataset.comparison_kind.values[comparison])
                                    if "comparison_kind" in dataset.coords else "forecast"),
                "severity": (float(dataset.severity.values[comparison])
                             if "severity" in dataset.coords else np.nan),
                "lead_hour": int(dataset.lead_hour.values[comparison]),
                "field_names": field_names,
                "anchor_latitudes": np.asarray(dataset.anchor_latitude.values),
                "anchor_longitudes": np.asarray(dataset.anchor_longitude.values),
                "anchor_transport_cost": np.asarray(dataset.anchor_transport_cost.values[comparison]),
                "anchor_local_wasserstein": np.asarray(dataset.anchor_local_wasserstein.values[comparison]),
                "anchor_mean_response_difference": np.asarray(dataset.anchor_mean_response_difference.values[comparison]),
                "top_wasserstein_distributions": distributions,
                "scwd": float(dataset.scwd.values[comparison]),
                "scwd_order": float(dataset.attrs["scwd_order"]),
            })
    return diagnostics


def scwd_comparison_kind(item):
    return str(item.get("comparison_kind", "forecast"))


def scwd_comparison_detail(item):
    comparison_kind = scwd_comparison_kind(item)
    if comparison_kind == "corruption":
        return f"severity={float(item['severity']):.3g}"
    if comparison_kind == "null":
        return "ERA5 test vs buffered training complement"
    return f"+{int(item['lead_hour'])} h"


def scwd_plot_root(output_root, comparison_kind, top_w1=False):
    root = output_root / "plots" / "scwd"
    if comparison_kind == "corruption":
        root = root / "corruptions"
    elif comparison_kind == "null":
        root = root / "null"
    if top_w1:
        root = root / "top_wasserstein_distributions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def scwd_output_stem(item):
    label = str(item["label"]).replace(" ", "_").replace("/", "_")
    if scwd_comparison_kind(item) == "corruption":
        return f"{label}_severity_{float(item['severity']):.3g}"
    return label


def plot_scwd_anchor_diagnostics(diagnostics, output_root, representative_only=False):
    """Plot shared-scale SCWD local-Wasserstein maps in one figure per model."""
    if not diagnostics:
        return
    values = np.stack([item["anchor_local_wasserstein"] for item in diagnostics])
    vmax = max(float(np.nanpercentile(values, 99)), 1e-12)
    labels = repeated_diagnostic_labels(
        [{**item, "comparison_kind": scwd_comparison_kind(item)} for item in diagnostics],
        representative_only=representative_only,
    )

    for comparison_kind, label in labels:
        series = sorted(
            [item for item in diagnostics
             if scwd_comparison_kind(item) == comparison_kind and item["label"] == label],
            key=lambda item: item["lead_hour"],
        )
        n_cols = min(2, len(series))
        n_rows = int(np.ceil(len(series) / n_cols))
        figure, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.2 * n_cols, 3.6 * n_rows),
            squeeze=False,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        flat_axes = axes.ravel()
        image = None
        for axis, item in zip(flat_axes, series):
            image = axis.pcolormesh(
                item["anchor_longitudes"],
                item["anchor_latitudes"],
                item["anchor_local_wasserstein"],
                shading="auto",
                transform=ccrs.PlateCarree(),
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
                rasterized=True,
            )
            axis.set_global()
            axis.coastlines(linewidth=0.55)
            axis.add_feature(cfeature.BORDERS, linewidth=0.35, alpha=0.45)
            axis.set_title(f"{scwd_comparison_detail(item)}  |  SCWD={item['scwd']:.4g}", fontsize=10)
        for axis in flat_axes[len(series):]:
            axis.set_visible(False)
        figure.suptitle(f"SCWD anchor contributions: {label} ({comparison_kind})", fontsize=14)
        figure.subplots_adjust(left=0.03, right=0.87, bottom=0.03, top=0.88, wspace=0.08, hspace=0.24)
        figure.colorbar(
            image, ax=flat_axes[:len(series)].tolist(), fraction=0.035, pad=0.035,
            label="Local SCWD magnitude",
        )
        kind_root = scwd_plot_root(output_root, comparison_kind)
        output_path = kind_root / f"{scwd_output_stem(series[0])}.png"
        save_figure_bundle(figure, output_path, plot_type="baseline_plot", dpi=220, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved SCWD anchor map to: {output_path}")


def plot_scwd_top_wasserstein_distributions(
    diagnostics, cfg, output_root, representative_only=False,
):
    """Plot per-field marginals at anchors with the largest local joint W2."""
    if not diagnostics:
        return
    n_bins = max(2, int(baseline_get(cfg, "scwd_distribution_bins", 40)))
    selected = diagnostics
    if representative_only:
        selected = []
        seen_kinds = set()
        for diagnostic in sorted(
            diagnostics,
            key=lambda item: (
                scwd_comparison_kind(item), str(item["label"]),
                int(item.get("lead_hour", 0)), float(item.get("severity", 0.0)),
            ),
        ):
            comparison_kind = scwd_comparison_kind(diagnostic)
            if comparison_kind not in seen_kinds:
                selected.append(diagnostic)
                seen_kinds.add(comparison_kind)
    for diagnostic in selected:
        distributions = diagnostic["top_wasserstein_distributions"]
        fields = diagnostic["field_names"]
        if not distributions:
            continue
        n_rows, n_cols = len(distributions), len(fields)
        figure, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.8 * n_cols, 2.8 * n_rows), squeeze=False,
        )
        payload = {"fields": np.asarray(fields, dtype=str)}
        for rank, item in enumerate(distributions):
            for field_index, field in enumerate(fields):
                axis = axes[rank, field_index]
                candidate = item["candidate"][:, field_index]
                reference = item["reference"][:, field_index]
                edges = np.histogram_bin_edges(np.concatenate([candidate, reference]), bins=n_bins)
                axis.hist(reference, bins=edges, density=True, histtype="step", linewidth=1.5,
                          color="black", label=("ERA5 training complement" if scwd_comparison_kind(diagnostic) == "null" else "ERA5 test"))
                axis.hist(candidate, bins=edges, density=True, histtype="step", linewidth=1.5,
                          color=CANDIDATE_COLOR, label=diagnostic["label"])
                latitude = item["latitude"]
                longitude = item["longitude"]
                local_wasserstein = item["wasserstein"]
                axis.set_title(
                    f"{field}\n{latitude:.1f}°, {longitude:.1f}°; joint W2={local_wasserstein:.4g}",
                    fontsize=9,
                )
                axis.set_xlabel("SCWD filter response")
                axis.set_ylabel("Density")
                axis.grid(True, alpha=0.25)
                payload[f"candidate_responses_{rank}_{field_index}"] = candidate
                payload[f"reference_responses_{rank}_{field_index}"] = reference
                payload[f"bin_edges_{rank}_{field_index}"] = edges
            payload[f"anchor_latitude_{rank}"] = np.asarray(item["latitude"])
            payload[f"anchor_longitude_{rank}"] = np.asarray(item["longitude"])
            payload[f"local_wasserstein_{rank}"] = np.asarray(item["wasserstein"])
        axes[0, 0].legend(fontsize=8)
        detail = scwd_comparison_detail(diagnostic)
        figure.suptitle(f"Highest local joint-W2 SCWD anchors: {diagnostic.get('label')} ({detail})", fontsize=13)
        figure.tight_layout(rect=[0, 0, 1, 0.97])
        kind_root = scwd_plot_root(output_root, scwd_comparison_kind(diagnostic), top_w1=True)
        suffix = f"_{int(diagnostic.get("lead_hour")):03d}h" if scwd_comparison_kind(diagnostic) == "forecast" else ""
        output_path = kind_root / f"{scwd_output_stem(diagnostic)}{suffix}.png"
        save_figure_bundle(
            figure, output_path, plot_type="scwd_top_joint_w2_distributions",
            payload=payload, dpi=180, bbox_inches="tight",
        )
        plt.close(figure)
        print(f"Saved top joint-W2 SCWD response distributions to: {output_path}")


def plot_scwd_mean_response_differences(diagnostics, output_root, representative_only=False):
    """Plot per-field candidate-minus-reference mean SCWD responses."""
    if not diagnostics:
        return
    values = np.stack([item["anchor_mean_response_difference"] for item in diagnostics])
    vmax = max(float(np.nanpercentile(np.abs(values), 99)), 1e-12)
    labels = repeated_diagnostic_labels(
        [{**item, "comparison_kind": scwd_comparison_kind(item)} for item in diagnostics],
        representative_only=representative_only,
    )
    for comparison_kind, label in labels:
        series = sorted(
            [item for item in diagnostics
             if scwd_comparison_kind(item) == comparison_kind and item["label"] == label],
            key=lambda item: item["lead_hour"],
        )
        fields = series[0]["field_names"]
        n_panels = len(series) * len(fields)
        n_cols = min(2, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        figure, axes = plt.subplots(
            n_rows, n_cols, figsize=(5.2 * n_cols, 3.6 * n_rows), squeeze=False,
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        flat_axes = axes.ravel()
        image = None
        panel = 0
        for item in series:
            for field_index, field in enumerate(fields):
                axis = flat_axes[panel]
                image = axis.pcolormesh(
                    item["anchor_longitudes"], item["anchor_latitudes"],
                    item["anchor_mean_response_difference"][field_index],
                    shading="auto", transform=ccrs.PlateCarree(), cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, rasterized=True,
                )
                axis.set_global()
                axis.coastlines(linewidth=0.55)
                axis.add_feature(cfeature.BORDERS, linewidth=0.35, alpha=0.45)
                axis.set_title(f"{scwd_comparison_detail(item)} | {field}", fontsize=9)
                panel += 1
        for axis in flat_axes[panel:]:
            axis.set_visible(False)
        figure.suptitle(f"Mean SCWD filter-response difference: {label} ({comparison_kind})", fontsize=13)
        figure.subplots_adjust(left=0.03, right=0.87, bottom=0.03, top=0.9, wspace=0.08, hspace=0.24)
        figure.colorbar(
            image, ax=flat_axes[:panel].tolist(), fraction=0.035, pad=0.035,
            label="Mean response difference (candidate − reference)",
        )
        kind_root = scwd_plot_root(output_root, comparison_kind)
        output_path = kind_root / f"{scwd_output_stem(series[0])}_mean_response_difference.png"
        save_figure_bundle(
            figure, output_path, plot_type="scwd_mean_response_difference",
            payload={
                "fields": np.asarray(fields, dtype=str),
                "mean_response_difference": np.stack([item["anchor_mean_response_difference"] for item in series]),
                "anchor_latitudes": np.asarray(series[0]["anchor_latitudes"]),
                "anchor_longitudes": np.asarray(series[0]["anchor_longitudes"]),
            }, dpi=180, bbox_inches="tight",
        )
        plt.close(figure)
        print(f"Saved per-field mean SCWD response-difference map to: {output_path}")


def evaluate_lead_metrics(cfg, truth_ds, variables, metric_names, temporary_dir):
    """Score forecasts against ERA5 at their exact valid timestamps."""
    results = []
    scwd_diagnostics = []
    global_mean_diagnostics = []
    max_samples = int(baseline_get(cfg, "eval_samples", cfg_get(cfg, "max_samples", 100)))
    variable_name = joint_variable_name(variables)
    configured_leads = {int(value) for value in cfg.lead_times}
    forecast_files = baseline_get(cfg, "forecast_files", None)
    if not forecast_files:
        raise ValueError("baseline.forecast_files must contain at least one named forecast file.")

    for label, configured_paths in forecast_files.items():
        opened = [normalize_prediction_timedelta(safe_open_dataset(path)) for path in configured_paths]
        forecast_ds = select_level(concatenate_forecasts(opened), cfg.get("level"))
        if forecast_ds.sizes.get("time", 0) == 0:
            print(f"Skipping {label}: forecast file contains no samples.")
            forecast_ds.close()
            continue

        missing_variables = [variable for variable in variables if variable not in forecast_ds.data_vars]
        if missing_variables:
            print(f"Skipping {label}: missing variables {missing_variables}")
            forecast_ds.close()
            continue

        available_leads = lead_hours(forecast_ds)
        lead_pairs = [
            (lead_idx, int(lead_hour))
            for lead_idx, lead_hour in enumerate(available_leads)
            if int(lead_hour) in configured_leads
        ]
        missing_leads = sorted(configured_leads - {lead_hour for _, lead_hour in lead_pairs})
        if missing_leads:
            print(f"Warning: {label} is missing configured lead hours: {missing_leads}")
        if not lead_pairs:
            print(f"Skipping {label}: none of the configured lead hours are available.")
            forecast_ds.close()
            continue

        for lead_idx, lead_hour in tqdm(lead_pairs, desc=f"{label} lead distribution metrics"):
            pairs = [
                pair for pair in forecast_pairs(
                    forecast_ds, truth_ds, cfg, "test", [lead_hour]
                ) if pair.lead_index == lead_idx
            ]
            pairs = evenly_spaced_pairs(pairs, max_samples)
            if not pairs:
                print(f"Skipping {label} +{lead_hour}h: no exact monthly test pairs.")
                continue
            time_indices = [pair.forecast_index for pair in pairs]
            reference_ds = truth_ds.isel(time=[pair.era5_index for pair in pairs])
            reference_indices = list(range(len(pairs)))
            reference_stats = streaming_reference_stats(
                cfg, reference_ds, variables, reference_indices
            )
            reference_features = streaming_joint_features(
                cfg, reference_ds, variables, reference_stats, reference_indices,
                metric_names, Path(temporary_dir) / "forecast-reference-scwd.dat",
                description=f"{label} +{lead_hour}h exact ERA5 reference",
            )
            candidate_features = streaming_joint_features(
                cfg,
                forecast_ds,
                variables,
                reference_stats,
                time_indices,
                metric_names,
                Path(temporary_dir) / "forecast-candidate-scwd.dat",
                lead_idx=lead_idx,
                histogram_target=label, histogram_coordinate=lead_hour,
                description=f"{label} +{lead_hour}h features",
            )
            metrics = selected_distribution_metric_values(
                candidate_features, reference_features, metric_names, cfg
            )
            row = {
                "label": label,
                "variable": variable_name,
                "lead_hour": lead_hour,
                "n_samples": metrics["n_samples"],
                "pairwise_n_samples": metrics["pairwise_n_samples"],
                "is_null": False,
            }
            row.update(coverage_metadata(pairs))
            row.update({metric_name: metrics[metric_name] for metric_name in metric_names})
            if "scwd" in metric_names:
                scwd_diagnostics.append(
                    scwd_anchor_diagnostic(
                        candidate_features,
                        reference_features,
                        cfg,
                        label,
                        lead_hour,
                        metrics["scwd"],
                    )
                )
            if "global_mean_wasserstein" in metric_names:
                global_mean_diagnostics.append(
                    global_mean_wasserstein_diagnostic(
                        candidate_features, reference_features, cfg, label, lead_hour,
                        metrics["global_mean_wasserstein"], row=row,
                    ))
            if row["n_samples"] == 0:
                print(f"Warning: no valid joint samples for {label} lead={lead_hour} variables={variables}")
            results.append(row)
            close_feature_memmaps(candidate_features)
            close_feature_memmaps(reference_features)
        forecast_ds.close()

    return results, scwd_diagnostics, global_mean_diagnostics


def corruption_types_from_config(cfg):
    """Return every corruption configured specifically for baseline probes."""
    return [str(value) for value in baseline_get(cfg, "corruptions", [])]


def is_u_wind_variable(variable):
    """Return true when a variable name looks like a 10m U-wind component."""
    normalized = str(variable).lower()
    return (
        "10m_u_component" in normalized
        or normalized in {"u10", "10u", "u_component_of_wind"}
        or normalized.endswith("_u_component_of_wind")
    )


def is_v_wind_variable(variable):
    """Return true when a variable name looks like a 10m V-wind component."""
    normalized = str(variable).lower()
    return (
        "10m_v_component" in normalized
        or normalized in {"v10", "10v", "v_component_of_wind"}
        or normalized.endswith("_v_component_of_wind")
    )


def can_apply_wind_vector_corruption(variables):
    """Return whether variables match the channel layout used by wind corruptions."""
    return (
        len(variables) > V10_CHANNEL
        and is_u_wind_variable(variables[U10_CHANNEL])
        and is_v_wind_variable(variables[V10_CHANNEL])
    )


def compatible_corruption_types(corruption_types, variables):
    """Drop standard metric corruption probes that are incompatible with variables."""
    fixed_wind_corruptions = {"wind_patch_shuffle", "wind_shuffled", "wind_rotation", "wind_rotated"}
    compatible = []
    skipped = []
    for corruption_type in corruption_types:
        if corruption_type in fixed_wind_corruptions and not can_apply_wind_vector_corruption(variables):
            skipped.append(corruption_type)
            continue
        compatible.append(corruption_type)
    if skipped:
        print(
            "Skipping baseline corruption(s) incompatible with variables "
            f"{variables}: {', '.join(skipped)}"
        )
    return compatible


def corruption_levels(corruption_type, cfg):
    """Return severity levels for one train-time corruption."""
    n_steps = int(baseline_get(cfg, "corruption_steps", 7))
    max_severity = corruption_max_severity(corruption_type, cfg)
    if n_steps <= 1:
        return [max_severity]
    return np.linspace(0.0, max_severity, n_steps).tolist()


def corruption_max_severity(corruption_type, cfg):
    """Return the configured maximum, honoring per-corruption overrides."""
    maximum = baseline_get(cfg, "corruption_severity_max", cfg_get(cfg, "corruption_severity_max", 1.0))
    overrides = baseline_get(cfg, "corruption_severity_max_overrides", {}) or {}
    return float(overrides.get(str(corruption_type), maximum))


def representative_corruption_time_index(time_indices, configured_index=None):
    """Choose a reproducible sample from the same times used by the corruption sweep."""
    if not time_indices:
        raise ValueError("No timestamps available for a corruption visualization.")
    if configured_index is not None:
        index = int(configured_index)
        if index not in time_indices:
            raise ValueError(
                "baseline.corruption_visualization_time_index must be one of the "
                "timestamps sampled by the corruption sweep."
            )
        return index
    return int(time_indices[len(time_indices) // 2])


def normalization_stats_for_corruptions(cfg, truth_ds, variables):
    """Compute normalization stats required by training-time corruptions."""
    stats_ds = select_level(
        select_era5_split(truth_ds, cfg, "train", coverage="corruption"), cfg.get("level")
    )
    means = {}
    stds = {}
    for variable in variables:
        means[variable] = float(stats_ds[variable].mean())
        std = float(stats_ds[variable].std())
        stds[variable] = std if std > 1e-8 else 1.0
    return means, stds


def evaluate_corruption_metrics(
    cfg, truth_ds, _reference_stats, variables, metric_names, temporary_dir,
    return_scwd_diagnostics=False,
):
    """Compare corrupted test fields to matched clean test ERA5."""
    results = []
    scwd_diagnostics = []
    global_mean_diagnostics = []
    max_samples = int(
        baseline_get(
            cfg,
            "corruption_eval_samples",
            baseline_get(cfg, "eval_samples", cfg_get(cfg, "max_samples", 100)),
        )
    )
    variable_name = joint_variable_name(variables)
    test_ds = select_level(
        select_era5_split(truth_ds, cfg, "test", coverage="corruption"), cfg.get("level")
    )
    test_indices = sample_time_indices(test_ds, max_samples)
    if not test_indices:
        raise ValueError("No ERA5 test samples available for corruption probes.")
    train_ds = select_level(
        select_era5_split(truth_ds, cfg, "train", coverage="corruption"), cfg.get("level")
    )
    null_ds = select_level(
        select_era5_split(truth_ds, cfg, "null", coverage="corruption"), cfg.get("level")
    )
    train_indices = sample_time_indices(train_ds, max_samples)
    if not train_indices:
        raise ValueError("No ERA5 train samples available for corruption probes.")
    null_indices = sample_time_indices(null_ds, max_samples)
    reference_stats = streaming_reference_stats(cfg, test_ds, variables, test_indices)
    null_reference_stats = streaming_reference_stats(cfg, train_ds, variables, train_indices)
    test_features = streaming_joint_features(
        cfg,
        test_ds,
        variables,
        reference_stats,
        test_indices,
        metric_names,
        Path(temporary_dir) / "corruption-test-scwd.dat",
        description="clean ERA5 test corruption features",
    )
    train_features = streaming_joint_features(
        cfg, train_ds, variables, null_reference_stats,
        train_indices,
        metric_names,
        Path(temporary_dir) / "corruption-train-scwd.dat",
        description="ERA5 train corruption-reference features",
    )
    null_features = streaming_joint_features(
        cfg, null_ds, variables, null_reference_stats, null_indices, metric_names,
        Path(temporary_dir) / "corruption-null-scwd.dat",
        description="ERA5 test-vs-train null features",
    )

    for corruption_type in compatible_corruption_types(corruption_types_from_config(cfg), variables):
        levels = corruption_levels(corruption_type, cfg)
        full_severity = float(max(levels))
        zero_metrics = selected_distribution_metric_values(test_features, test_features, metric_names, cfg)
        zero_row = {
            "label": "ERA5 test matched reference",
            "variable": variable_name,
            "corruption": corruption_type,
            "severity": 0.0,
            "n_samples": zero_metrics["n_samples"],
            "pairwise_n_samples": zero_metrics["pairwise_n_samples"],
            "is_null": False,
        }
        zero_row.update({metric_name: zero_metrics[metric_name] for metric_name in metric_names})
        results.append(zero_row)
        if "global_mean_wasserstein" in metric_names:
            global_mean_diagnostics.append(global_mean_wasserstein_diagnostic(
                test_features, test_features, cfg, corruption_type, 0,
                zero_metrics["global_mean_wasserstein"], comparison_kind="corruption",
                severity=0.0, row=zero_row,
            ))
        null_metrics = selected_distribution_metric_values(
            null_features, train_features, metric_names, cfg
        )
        null_row = {
            "label": ERA5_NULL_LABEL,
            "variable": variable_name,
            "corruption": corruption_type,
            "severity": 0.0,
            "n_samples": null_metrics["n_samples"],
            "pairwise_n_samples": null_metrics["pairwise_n_samples"],
            "is_null": True,
        }
        null_row.update({metric_name: null_metrics[metric_name] for metric_name in metric_names})
        results.append(null_row)
        if "global_mean_wasserstein" in metric_names:
            global_mean_diagnostics.append(global_mean_wasserstein_diagnostic(
                null_features, train_features, cfg, ERA5_NULL_LABEL, 0,
                null_metrics["global_mean_wasserstein"], comparison_kind="null",
                severity=0.0, row=null_row,
            ))
        for severity in tqdm(levels, desc=f"{corruption_type} distribution metrics"):
            if float(severity) == 0.0:
                continue
            candidate_features = streaming_joint_features(
                cfg,
                test_ds,
                variables,
                reference_stats,
                test_indices,
                metric_names,
                Path(temporary_dir) / "corruption-candidate-scwd.dat",
                corruption_type=corruption_type,
                severity=severity,
                histogram_target=corruption_type, histogram_coordinate=severity,
                description=f"{corruption_type} severity={severity:.3g}",
            )
            base_row = {
                "label": "ERA5 test corrupted vs matched test",
                "variable": variable_name,
                "corruption": corruption_type,
                "severity": float(severity),
                "is_null": False,
            }
            metrics = selected_distribution_metric_values(
                candidate_features, test_features, metric_names, cfg
            )
            row = dict(base_row)
            row.update({metric_name: metrics[metric_name] for metric_name in metric_names})
            row["n_samples"] = metrics["n_samples"]
            row["pairwise_n_samples"] = metrics["pairwise_n_samples"]
            if "global_mean_wasserstein" in metric_names:
                global_mean_diagnostics.append(global_mean_wasserstein_diagnostic(
                    candidate_features, test_features, cfg, corruption_type, 0,
                    metrics["global_mean_wasserstein"], comparison_kind="corruption",
                    severity=float(severity), row=row,
                ))
            if "scwd" in metric_names and np.isclose(float(severity), full_severity):
                scwd_diagnostics.append(scwd_anchor_diagnostic(
                    candidate_features, test_features, cfg, corruption_type, 0, metrics["scwd"],
                    comparison_kind="corruption", severity=float(severity),
                ))
            if row["n_samples"] == 0:
                print(
                    "Warning: no valid joint samples for "
                    f"corruption={corruption_type} severity={severity} variables={variables}"
                )
            results.append(row)
            close_feature_memmaps(candidate_features)

    close_feature_memmaps(test_features)
    close_feature_memmaps(train_features)
    close_feature_memmaps(null_features)
    return (results, scwd_diagnostics, global_mean_diagnostics) if return_scwd_diagnostics else results


def baseline_output_dir(cfg, variables):
    """Return the namespaced output directory for one baseline variant."""
    return Path(str(baseline_get(cfg, "output_dir"))) / variable_tag(variables)


def write_metric_csv(rows, metric_names, output_path, experiment):
    """Write one experiment type with a purpose-specific schema."""
    if experiment == "lead_time":
        fieldnames = [
            "label", "variable", "lead_hour", "is_null", "n_samples",
            "pairwise_n_samples", "n_pairs", "initialization_start",
            "initialization_end", "valid_start", "valid_end",
        ] + metric_names
    elif experiment == "corruption_strength":
        fieldnames = [
            "label", "variable", "corruption", "severity", "is_null", "n_samples", "pairwise_n_samples"
        ] + metric_names
    else:
        raise ValueError(f"Unknown baseline experiment: {experiment}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {experiment} metric values to: {output_path}")


def read_metric_csv(output_path, metric_names, experiment):
    """Read persisted metric rows and restore the numeric types used by plotting."""
    if not output_path.exists():
        raise FileNotFoundError(
            f"Missing baseline evaluation artifact: {output_path}. "
            "Run scripts/evaluate_standard_metric_baselines.py first."
        )
    with open(output_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = set(metric_names) | {f"{name}_{suffix}" for name in metric_names for suffix in ("lower", "upper")} | {"n_samples", "pairwise_n_samples", "n_resamples"}
    numeric.add("lead_hour" if experiment == "lead_time" else "severity")
    for row in rows:
        for field in numeric:
            if field in row and row[field] != "":
                row[field] = float(row[field])
        for field in ("n_samples", "pairwise_n_samples", "lead_hour"):
            if field in row and row[field] != "":
                row[field] = int(float(row[field]))
    return rows


def row_is_null(row):
    return str(row.get("is_null", False)).lower() == "true"

ABSOLUTE_DISPLAY_METRICS = {"mean_bias", "std_ratio_error"}

METRIC_DISPLAY_NAMES = {
    "mean_bias": "Mean bias",
    "std_ratio_error": "Std. ratio error",
    "crps_like_field_energy": "Field energy",
    "zonal_energy_spectrum_l2": "Zonal spectrum L2",
    "zonal_energy_spectrum_log_l2": "Log zonal spectrum L2",
    "sliced_wasserstein": "Sliced WD",
    "sliced_wasserstein_lon_corrected": "Area-corrected sliced WD",
    "global_mean_wasserstein": "Global-mean WD",
    "global_mean_wasserstein_area_weighted": "Area-weighted global-mean WD",
    "mmd_rbf": "RBF MMD",
    "scwd": "SCWD",
    "scwd_area_weighted": "Area-weighted SCWD",
}

CORRUPTION_DISPLAY_NAMES = {
    "gaussian_blur": "Gaussian blur",
    "grf": "GRF noise",
    "checkerboard_2px": "2-pixel checkerboard",
    "equatorial_checker_texture": "Equatorial checkerboard",
    "zonal_scanlines": "Zonal scanlines",
    "meridional_scanlines": "Meridional scanlines",
    "hemisphere_splice": "Hemisphere splice",
    "field_splice": "Field splice",
    "hf_noise": "High-frequency noise",
    "pixel_replace": "Pixel replacement",
    "wind_patch_shuffle": "Wind-patch shuffle",
    "wind_rotation": "Wind-vector rotation",
}

ARCHITECTURE_DISPLAY_NAMES = {
    "squeezenet": "SqueezeNet",
    "squeezenet_attention": "Attention SqueezeNet",
    "squeezenet_equator_mask": "Equator-masked SqueezeNet",
    "sfno_linear": "SFNO + linear probe",
    "sfno_mlp": "SFNO + MLP probe",
}


def displayed_corruption_name(name):
    return CORRUPTION_DISPLAY_NAMES.get(str(name), str(name).replace("_", " ").title())


def displayed_architecture_name(name):
    return ARCHITECTURE_DISPLAY_NAMES.get(str(name), str(name).replace("_", " ").title())


def displayed_variable_name(name):
    return "Four surface fields" if str(name) == "all_fields" else str(name).replace("_", " ")


def display_metric_value(row, metric_name):
    """Return a presentation-only metric value for baseline figures."""
    raw = row.get(metric_name, np.nan)
    value = np.nan if raw in (None, "") else float(raw)
    return abs(value) if metric_name in ABSOLUTE_DISPLAY_METRICS else value


def display_metric_bounds(row, metric_name):
    fallback = row.get(metric_name, np.nan)
    raw_lower = row.get(f"{metric_name}_lower", fallback)
    raw_upper = row.get(f"{metric_name}_upper", fallback)
    lower = np.nan if raw_lower in (None, "") else float(raw_lower)
    upper = np.nan if raw_upper in (None, "") else float(raw_upper)
    if metric_name in ABSOLUTE_DISPLAY_METRICS:
        if lower <= 0.0 <= upper:
            return 0.0, max(abs(lower), abs(upper))
        return min(abs(lower), abs(upper)), max(abs(lower), abs(upper))
    return lower, upper


def displayed_metric_name(metric_name):
    label = METRIC_DISPLAY_NAMES.get(metric_name, metric_name.replace("_", " "))
    return f"|{label}|" if metric_name in ABSOLUTE_DISPLAY_METRICS else label


def relative_corruption_coordinates(rows, value_key):
    """Map a corruption-specific native range onto the common visual [0, 1] range."""
    values = [float(row[value_key]) for row in rows]
    maximum = max(values, default=0.0)
    if maximum <= 0.0:
        return [0.0 for _ in values]
    return [value / maximum for value in values]


def corruption_range_label(label, rows, value_key):
    maximum = max((float(row[value_key]) for row in rows), default=0.0)
    short_names = {
        "checkerboard_2px": "2-pixel checkerboard",
        "equatorial_checker_texture": "Equatorial checker",
        "gaussian_blur": "Gaussian blur",
        "grf": "GRF noise",
        "hemisphere_splice": "Hemisphere splice",
        "hf_noise": "HF noise",
        "meridional_scanlines": "Meridional lines",
        "pixel_replace": "Pixel replace",
        "zonal_scanlines": "Zonal lines",
    }
    return f"{short_names.get(str(label), displayed_corruption_name(label))} ({maximum:g})"


def metric_normalization_scales(rows, metric_names):
    """Return one robust, global-per-figure normalization factor per metric."""
    scales = {}
    for metric_name in metric_names:
        values = np.asarray([display_metric_value(row, metric_name) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        positive = values[values > 0.0]
        scale = positive.max() if positive.size else (np.abs(values).max() if values.size else 0.0)
        scales[metric_name] = float(scale) if scale > 0.0 else 1.0
    return scales


def normalized_metric_value(row, metric_name, scales):
    """Normalize one presentation metric value by its global figure scale."""
    value = display_metric_value(row, metric_name)
    return value / scales[metric_name] if np.isfinite(value) else np.nan


def normalized_metric_bounds(row, metric_name, scales):
    lower, upper = display_metric_bounds(row, metric_name)
    return lower / scales[metric_name], upper / scales[metric_name]


def null_diamond_positions(count, negative_extent):
    """Dodge coincident null markers within the reserved space left of zero."""
    count = int(count)
    if count <= 1:
        return np.asarray([0.0])
    return np.linspace(-float(negative_extent), 0.0, count)


def plot_null_diamond(axis, x, value, lower, upper, color, marker="D"):
    """Draw one dodged null estimate with its persisted uncertainty interval."""
    value, lower, upper = float(value), float(lower), float(upper)
    axis.errorbar(
        [float(x)], [value],
        yerr=[[max(value - lower, 0.0)], [max(upper - value, 0.0)]],
        marker=marker, markersize=3.6, linestyle="None", color=color,
        elinewidth=0.75, capsize=1.5, capthick=0.75, zorder=5,
    )


def metric_colors(metric_names):
    return {
        metric_name: color
        for metric_name, color in zip(
            metric_names, categorical_colors(max(len(metric_names), 1))
        )
    }


def plot_normalized_lead_metrics_by_model(
        rows, metric_names, variables, output_root, discriminator_rows=None):
    """Plot one forecast target per panel with fixed metrics and our critic."""
    metric_names = plotted_metric_names(metric_names)
    if not metric_names:
        return
    variable = joint_variable_name(variables)
    variable_rows = [row for row in rows if row["variable"] == variable]
    labels = sorted({
        row["label"] for row in variable_rows if row["label"] != ERA5_NULL_LABEL
    })
    if not labels:
        return
    critic_rows = [
        row for row in (discriminator_rows or [])
        if row.get("architecture") == "squeezenet" and row.get("kind") == "forecast"
    ]
    critic_values = np.asarray([
        row["score"] for row in critic_rows if np.isfinite(row.get("score", np.nan))
    ], dtype=float)
    critic_positive = critic_values[critic_values > 0.0]
    critic_scale = (
        float(critic_positive.max()) if critic_positive.size
        else (float(np.abs(critic_values).max()) if critic_values.size else 1.0)
    )
    if critic_scale <= 0.0:
        critic_scale = 1.0

    scales = metric_normalization_scales(variable_rows, metric_names)
    colors = metric_colors(metric_names)
    n_cols = 2
    n_rows = int(np.ceil((len(labels) + 1) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(10.0, 3.0 * n_rows), squeeze=False,
        layout="constrained",
    )
    era5_rows = [row for row in variable_rows if row["label"] == ERA5_NULL_LABEL]
    null_positions = null_diamond_positions(len(metric_names) + 1, 4.5)
    for panel_index, (axis, label) in enumerate(zip(axes.ravel(), labels)):
        series = sorted(
            [row for row in variable_rows if row["label"] == label],
            key=lambda row: row["lead_hour"],
        )
        for metric_index, metric_name in enumerate(metric_names):
            x_values = [row["lead_hour"] for row in series]
            axis.plot(
                x_values,
                [normalized_metric_value(row, metric_name, scales) for row in series],
                marker=series_marker(metric_index), linewidth=1.5,
                color=colors[metric_name], label=displayed_metric_name(metric_name),
            )
            if series and f"{metric_name}_lower" in series[0]:
                bounds = [normalized_metric_bounds(row, metric_name, scales) for row in series]
                axis.fill_between(
                    x_values, [bound[0] for bound in bounds], [bound[1] for bound in bounds],
                    color=colors[metric_name], alpha=0.14, linewidth=0,
                )
            if era5_rows:
                null_value = normalized_metric_value(era5_rows[0], metric_name, scales)
                null_lower, null_upper = normalized_metric_bounds(
                    era5_rows[0], metric_name, scales,
                )
                plot_null_diamond(
                    axis, null_positions[metric_index], null_value,
                    null_lower, null_upper, colors[metric_name],
                )

        target_rows = [row for row in critic_rows if row.get("target") == label]
        critic_series = sorted(
            [row for row in target_rows if not row["is_era5_test_null"]],
            key=lambda row: row["x"],
        )
        if critic_series:
            critic_x = [row["x"] for row in critic_series]
            axis.plot(
                critic_x, [row["score"] / critic_scale for row in critic_series],
                marker=series_marker(len(metric_names)), linestyle="--", linewidth=1.6,
                color=CRITIC_COLOR, label="Learned critic",
            )
            if "score_lower" in critic_series[0]:
                axis.fill_between(
                    critic_x,
                    [row["score_lower"] / critic_scale for row in critic_series],
                    [row["score_upper"] / critic_scale for row in critic_series],
                    color=CRITIC_COLOR, alpha=0.12, linewidth=0,
                )
            critic_null = next(
                (row for row in target_rows if row["is_era5_test_null"]), None
            )
            if critic_null is not None:
                critic_value = critic_null["score"] / critic_scale
                critic_lower = critic_null.get("score_lower", critic_null["score"]) / critic_scale
                critic_upper = critic_null.get("score_upper", critic_null["score"]) / critic_scale
                plot_null_diamond(
                    axis, null_positions[-1], critic_value, critic_lower, critic_upper,
                    CRITIC_COLOR,
                )

        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        axis.set_title(displayed_model_name(label))
        axis.set_xlim(-6.0, max([row["lead_hour"] for row in series], default=192) * 1.03)
        if panel_index // 2 == n_rows - 1:
            axis.set_xlabel("Lead time (hours)")
        if panel_index % 2 == 0:
            axis.set_ylabel("Normalized divergence")
        axis.grid(True, alpha=0.3)
    legend_axis = axes.ravel()[len(labels)]
    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    legend_axis.set_axis_off()
    legend_axis.legend(
        handles, legend_labels, loc="center", frameon=False,
        title="Metrics", fontsize=7, title_fontsize=8,
    )
    for axis in axes.ravel()[len(labels) + 1:]:
        axis.set_axis_off()
    fig.suptitle(
        f"Normalized distributional metrics by forecast model: "
        f"{displayed_variable_name(variable)}\n"
        "Diamonds: ERA5 test-vs-train null (horizontally offset)"
    )
    output_path = output_root / "plots" / "lead_time_by_model_normalized.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_bundle(fig, output_path, plot_type="baseline_plot", dpi=220)
    plt.close(fig)
    print(f"Saved normalized lead-time-by-model plot to: {output_path}")

def plot_normalized_corruption_metrics_by_type(rows, metric_names, variables, output_root):
    """Plot one corruption per panel on a common relative-severity axis."""
    metric_names = plotted_metric_names(metric_names)
    if not metric_names:
        return
    variable = joint_variable_name(variables)
    variable_rows = [row for row in rows if row["variable"] == variable]
    corruptions = sorted({row["corruption"] for row in variable_rows})
    if not corruptions:
        return
    scales = metric_normalization_scales(variable_rows, metric_names)
    colors = metric_colors(metric_names)
    n_cols = 2
    n_rows = int(np.ceil((len(corruptions) + 1) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(10.0, 3.0 * n_rows), squeeze=False,
        layout="constrained",
    )
    null_positions = null_diamond_positions(len(metric_names), 0.03)
    for panel_index, (axis, corruption) in enumerate(zip(axes.ravel(), corruptions)):
        series = sorted(
            [
                row for row in variable_rows
                if row["corruption"] == corruption and not row_is_null(row)
            ],
            key=lambda row: row["severity"],
        )
        x_values = relative_corruption_coordinates(series, "severity")
        null_row = next(
            (
                row for row in variable_rows
                if row["corruption"] == corruption and row_is_null(row)
            ),
            None,
        )
        for metric_index, metric_name in enumerate(metric_names):
            axis.plot(
                x_values,
                [normalized_metric_value(row, metric_name, scales) for row in series],
                marker=series_marker(metric_index), linewidth=1.5,
                color=colors[metric_name], label=displayed_metric_name(metric_name),
            )
            if series and f"{metric_name}_lower" in series[0]:
                bounds = [normalized_metric_bounds(row, metric_name, scales) for row in series]
                axis.fill_between(
                    x_values, [bound[0] for bound in bounds], [bound[1] for bound in bounds],
                    color=colors[metric_name], alpha=0.14, linewidth=0,
                )
            if null_row is not None:
                null_value = normalized_metric_value(null_row, metric_name, scales)
                null_lower, null_upper = normalized_metric_bounds(
                    null_row, metric_name, scales,
                )
                plot_null_diamond(
                    axis, null_positions[metric_index], null_value,
                    null_lower, null_upper, colors[metric_name],
                )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        axis.set_title(displayed_corruption_name(corruption))
        axis.set_xlim(-0.04, 1.04)
        axis.set_xticks(np.linspace(0.0, 1.0, 5))
        if panel_index // 2 == n_rows - 1:
            axis.set_xlabel("Relative corruption severity")
        if panel_index % 2 == 0:
            axis.set_ylabel("Normalized divergence")
        axis.grid(True, alpha=0.3)
    legend_axis = axes.ravel()[len(corruptions)]
    handles, legend_labels = axes.ravel()[0].get_legend_handles_labels()
    legend_axis.set_axis_off()
    legend_axis.legend(
        handles, legend_labels, loc="center", frameon=False,
        title="Metrics", fontsize=7, title_fontsize=8,
    )
    for axis in axes.ravel()[len(corruptions) + 1:]:
        axis.set_axis_off()
    fig.suptitle(
        f"Normalized distributional metrics by corruption: "
        f"{displayed_variable_name(variable)}\n"
        "Diamonds: ERA5 test-vs-train null (horizontally offset)"
    )
    output_path = output_root / "plots" / "corruption_by_type_normalized.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_bundle(fig, output_path, plot_type="baseline_plot", dpi=220)
    plt.close(fig)
    print(f"Saved normalized corruption-by-type plot to: {output_path}")


def plot_lead_metrics(rows, metric_names, variables, output_root):
    """Plot each fixed metric against forecast lead time in a compact grid."""
    metric_names = plotted_metric_names(metric_names)
    if not metric_names:
        return
    variable = joint_variable_name(variables)
    variable_rows = [row for row in rows if row["variable"] == variable]
    if not variable_rows:
        return
    labels = sorted({
        row["label"] for row in variable_rows if row["label"] != ERA5_NULL_LABEL
    })
    colors = model_colors(labels)
    n_cols = 2
    n_rows = int(np.ceil((len(metric_names) + 1) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(10.0, 3.0 * n_rows), squeeze=False,
        layout="constrained",
    )
    axes = axes.ravel()
    for metric_idx, metric_name in enumerate(metric_names):
        metric_label = displayed_metric_name(metric_name)
        axis = axes[metric_idx]
        max_lead = 0.0
        for label_index, (color, label) in enumerate(zip(colors, labels)):
            series = sorted(
                [row for row in variable_rows if row["label"] == label],
                key=lambda row: row["lead_hour"],
            )
            x_values = [row["lead_hour"] for row in series]
            max_lead = max(max_lead, max(x_values, default=0.0))
            axis.plot(
                x_values, [display_metric_value(row, metric_name) for row in series],
                marker=series_marker(label_index), linewidth=1.5,
                color=color, label=displayed_model_name(label),
            )
            if series and f"{metric_name}_lower" in series[0]:
                bounds = [display_metric_bounds(row, metric_name) for row in series]
                axis.fill_between(
                    x_values, [bound[0] for bound in bounds], [bound[1] for bound in bounds],
                    color=color, alpha=0.16, linewidth=0,
                )
        era5_rows = [row for row in variable_rows if row["label"] == ERA5_NULL_LABEL]
        if era5_rows:
            axis.plot(
                [0], [display_metric_value(era5_rows[0], metric_name)],
                marker="D", markersize=3.5, linestyle="None",
                color="black", label=ERA5_NULL_LABEL,
            )
        axis.set_title(metric_label)
        axis.set_xlim(-6.0, max_lead * 1.03 if max_lead else 1.0)
        if metric_idx // 2 == n_rows - 1:
            axis.set_xlabel("Lead time (hours)")
        axis.grid(True, alpha=0.3)
    legend_axis = axes[len(metric_names)]
    handles, legend_labels = axes[0].get_legend_handles_labels()
    legend_axis.set_axis_off()
    legend_axis.legend(
        handles, legend_labels, loc="center", frameon=False,
        ncol=2, fontsize=7,
    )
    for axis in axes[len(metric_names) + 1:]:
        axis.set_axis_off()
    fig.suptitle(
        f"Distributional metrics vs. lead time: {displayed_variable_name(variable)}"
    )
    output_path = output_root / "plots" / "lead_time.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_bundle(fig, output_path, plot_type="baseline_plot", dpi=220)
    plt.close(fig)
    print(f"Saved lead-time metric plot to: {output_path}")


def plot_corruption_metrics(rows, metric_names, variables, output_root):
    """Plot standalone and combined fixed metrics against corruption severity."""
    standalone_names = standalone_metric_names(metric_names)
    metric_names = plotted_metric_names(metric_names)
    if not standalone_names:
        return
    variable = joint_variable_name(variables)
    variable_rows = [row for row in rows if row["variable"] == variable]
    if not variable_rows:
        return
    corruptions = sorted({row["corruption"] for row in variable_rows})
    metric_output_dir = output_root / "plots" / "corruption"
    metric_output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name in standalone_names:
        metric_label = displayed_metric_name(metric_name)
        n_cols = 2
        n_rows = int(np.ceil(len(corruptions) / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(10.0, 2.8 * n_rows), squeeze=False,
            layout="constrained",
        )
        axes = axes.ravel()
        for panel_index, corruption_type in enumerate(corruptions):
            axis = axes[panel_index]
            series = sorted(
                [
                    row for row in variable_rows
                    if row["corruption"] == corruption_type and not row_is_null(row)
                ],
                key=lambda row: row["severity"],
            )
            axis.plot(
                [row["severity"] for row in series],
                [display_metric_value(row, metric_name) for row in series],
                marker="o", linewidth=1.5,
            )
            if series and f"{metric_name}_lower" in series[0]:
                bounds = [display_metric_bounds(row, metric_name) for row in series]
                axis.fill_between(
                    [row["severity"] for row in series],
                    [bound[0] for bound in bounds], [bound[1] for bound in bounds],
                    alpha=0.16, linewidth=0,
                )
            null_row = next(
                (
                    row for row in variable_rows
                    if row["corruption"] == corruption_type and row_is_null(row)
                ),
                None,
            )
            if null_row is not None:
                axis.scatter(
                    [0.0], [display_metric_value(null_row, metric_name)],
                    marker="D", s=14, color="black", zorder=3,
                )
            axis.set_title(displayed_corruption_name(corruption_type))
            if panel_index // 2 == n_rows - 1:
                axis.set_xlabel("Corruption severity")
            if panel_index % 2 == 0:
                axis.set_ylabel(metric_label)
            axis.grid(True, alpha=0.3)
        for axis in axes[len(corruptions):]:
            axis.set_axis_off()
        fig.suptitle(
            f"{metric_label} vs. corruption strength: "
            f"{displayed_variable_name(variable)}\n"
            "Diamonds: ERA5 test-vs-train null"
        )
        output_path = metric_output_dir / f"{metric_name}.png"
        save_figure_bundle(fig, output_path, plot_type="baseline_plot", dpi=220)
        plt.close(fig)
        print(f"Saved corruption metric plot to: {output_path}")

    if not metric_names:
        return
    colors = categorical_colors(max(len(corruptions), 1))
    n_cols = 2
    n_rows = int(np.ceil((len(metric_names) + 1) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(10.0, 3.0 * n_rows), squeeze=False,
        layout="constrained",
    )
    axes = axes.ravel()
    for metric_idx, metric_name in enumerate(metric_names):
        metric_label = displayed_metric_name(metric_name)
        axis = axes[metric_idx]
        for corruption_index, (color, corruption_type) in enumerate(
            zip(colors, corruptions)
        ):
            series = sorted(
                [
                    row for row in variable_rows
                    if row["corruption"] == corruption_type and not row_is_null(row)
                ],
                key=lambda row: row["severity"],
            )
            x_values = relative_corruption_coordinates(series, "severity")
            axis.plot(
                x_values, [display_metric_value(row, metric_name) for row in series],
                marker=series_marker(corruption_index), linewidth=1.5,
                color=color, label=corruption_range_label(
                    corruption_type, series, "severity"
                ),
            )
            if series and f"{metric_name}_lower" in series[0]:
                bounds = [display_metric_bounds(row, metric_name) for row in series]
                axis.fill_between(
                    x_values, [bound[0] for bound in bounds], [bound[1] for bound in bounds],
                    color=color, alpha=0.14, linewidth=0,
                )
            null_row = next(
                (
                    row for row in variable_rows
                    if row["corruption"] == corruption_type and row_is_null(row)
                ),
                None,
            )
            if null_row is not None:
                axis.scatter(
                    [0.0], [display_metric_value(null_row, metric_name)],
                    marker="D", s=14, color="black", zorder=3,
                    label=(
                        ERA5_NULL_LABEL
                        if metric_idx == 0 and corruption_index == 0
                        else "_nolegend_"
                    ),
                )
        axis.set_title(metric_label)
        axis.set_xlim(-0.04, 1.04)
        axis.set_xticks(np.linspace(0.0, 1.0, 5))
        if metric_idx // 2 == n_rows - 1:
            axis.set_xlabel("Relative corruption severity")
        axis.grid(True, alpha=0.3)
    legend_axis = axes[len(metric_names)]
    handles, legend_labels = axes[0].get_legend_handles_labels()
    legend_axis.set_axis_off()
    legend_axis.legend(
        handles, legend_labels, loc="center", frameon=False,
        ncol=2, fontsize=6.5,
    )
    for axis in axes[len(metric_names) + 1:]:
        axis.set_axis_off()
    fig.suptitle(
        f"Distributional metrics vs. corruption strength: "
        f"{displayed_variable_name(variable)}\n"
        "Diamonds: ERA5 test-vs-train null"
    )
    output_path = output_root / "plots" / "corruption_strength.png"
    save_figure_bundle(fig, output_path, plot_type="baseline_plot", dpi=220)
    plt.close(fig)
    print(f"Saved combined corruption metric plot to: {output_path}")


def evaluate_corruption_disturbances(cfg, truth_ds, normalization_stats, variables, output_root):
    """Persist physical perturbations for cheap, repeatable gallery plotting."""
    means, stds = normalization_stats
    output_path = output_root / "data" / "corruption_disturbances.nc"
    if not bool(baseline_get(cfg, "plot_corruption_disturbances", True)):
        output_path.unlink(missing_ok=True)
        return
    max_samples = int(
        baseline_get(
            cfg, "corruption_eval_samples", baseline_get(cfg, "eval_samples", cfg_get(cfg, "max_samples", 100))
        )
    )
    eval_ds = select_level(
        select_era5_split(truth_ds, cfg, "test", coverage="corruption"), cfg.get("level")
    )
    time_indices = sample_time_indices(eval_ds, max_samples)
    sample_index = representative_corruption_time_index(
        time_indices, baseline_get(cfg, "corruption_visualization_time_index", None)
    )
    sample = eval_ds.isel(time=sample_index)
    latitudes = np.asarray(sample[variables[0]].latitude.values, dtype=np.float64)
    longitudes = np.asarray(sample[variables[0]].longitude.values, dtype=np.float64)
    clean = np.stack(
        [canonical_latlon(sample[variable].values, latitudes).astype(np.float32) for variable in variables]
    )
    standardized = np.stack(
        [
            (clean[channel] - means[variable]) / stds[variable]
            for channel, variable in enumerate(variables)
        ]
    ).astype(np.float32)
    timestamp = str(np.asarray(sample.time.values))[:19]
    base_seed = int(baseline_get(cfg, "corruption_seed", 0))
    corruption_names = []
    severity_values = []
    disturbance_values = []
    corrupted_field_values = []

    for corruption_type in compatible_corruption_types(corruption_types_from_config(cfg), variables):
        levels = corruption_levels(corruption_type, cfg)
        special_corruption = (
            corruption_type in STRUCTURED_NEAR_NULL_CORRUPTIONS | DATA_DEPENDENT_CORRUPTIONS
        )
        donor_standardized = None
        if corruption_type in DATA_DEPENDENT_CORRUPTIONS:
            sample_position = time_indices.index(sample_index)
            donor_positions = (
                deranged_sample_positions(len(time_indices), base_seed)
                if corruption_type == "hemisphere_splice"
                else fieldwise_deranged_sample_positions(len(time_indices), len(variables), base_seed)
            )
            donor_clean = []
            for channel, variable in enumerate(variables):
                positions = donor_positions if donor_positions.ndim == 1 else donor_positions[channel]
                donor_index = time_indices[int(positions[sample_position])]
                donor_sample = eval_ds.isel(time=donor_index)
                donor_clean.append(
                    canonical_latlon(donor_sample[variable].values, latitudes).astype(np.float32)
                )
            donor_standardized = np.stack([
                (donor_clean[channel] - means[variable]) / stds[variable]
                for channel, variable in enumerate(variables)
            ]).astype(np.float32)
        corrupted_fields = []
        disturbances = []
        for severity in levels:
            if special_corruption:
                corrupted = apply_special_baseline_corruption(
                    standardized, corruption_type, severity, latitudes, cfg, donor_standardized,
                    random_seed=corruption_sample_seed(base_seed, corruption_type, sample_index),
                )
            else:
                seed = corruption_sample_seed(base_seed, corruption_type, sample_index)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    corrupted = apply_configured_corruption(
                        torch.from_numpy(standardized), corruption_type, float(severity)
                    ).detach().cpu().numpy()
            corrupted = match_standardized(
                cfg, corrupted, variables, means, stds, "standard", "corruption",
                corruption_type, severity,
            )
            corrupted_physical = np.stack(
                [
                    corrupted[channel] * stds[variable] + means[variable]
                    for channel, variable in enumerate(variables)
                ]
            )
            corrupted_fields.append(corrupted_physical)
            disturbances.append(
                np.stack(
                    [
                        (corrupted[channel] - standardized[channel]) * stds[variable]
                        for channel, variable in enumerate(variables)
                    ]
                )
            )
        disturbances = np.asarray(disturbances)
        corruption_names.append(corruption_type)
        severity_values.append(np.asarray(levels, dtype=np.float64))
        disturbance_values.append(disturbances.astype(np.float32))
        corrupted_field_values.append(np.asarray(corrupted_fields, dtype=np.float32))

    if not disturbance_values:
        output_path.unlink(missing_ok=True)
        return
    dataset = xr.Dataset(
        data_vars={
            "disturbance": (
                ("corruption", "severity_index", "variable", "latitude", "longitude"),
                np.stack(disturbance_values),
            ),
            "corrupted_field": (
                ("corruption", "severity_index", "variable", "latitude", "longitude"),
                np.stack(corrupted_field_values),
            ),
            "severity": (("corruption", "severity_index"), np.stack(severity_values)),
        },
        coords={
            "corruption": corruption_names,
            "variable": variables,
            "latitude": latitudes,
            "longitude": longitudes,
        },
        attrs={
            "timestamp": timestamp,
            "description": "Physical corrupted fields and their differences from clean ERA5 for one representative sample.",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)
    dataset.close()
    print(f"Saved corruption disturbance data to: {output_path}")


def plot_corruption_field_gallery(
    output_dir, corruption_type, levels, fields, variables, latitudes, longitudes, timestamp,
    title, colorbar_label, filename, symmetric=False, placeholder=False,
):
    """Render one two-row-per-field corruption gallery from persisted physical values."""
    if placeholder:
        lower = np.full(len(variables), -1.0)
        upper = np.full(len(variables), 1.0)
        cmap = "Greys"
    elif symmetric:
        upper = np.maximum(np.nanpercentile(np.abs(fields), 99, axis=(0, 2, 3)), 1e-8)
        lower = -upper
        cmap = "RdBu_r"
    else:
        lower = np.nanpercentile(fields, 1, axis=(0, 2, 3))
        upper = np.nanpercentile(fields, 99, axis=(0, 2, 3))
        equal = np.isclose(lower, upper)
        lower[equal] -= 1e-8
        upper[equal] += 1e-8
        cmap = "viridis"
    severity_rows = 2
    severity_columns = int(np.ceil(len(levels) / severity_rows))
    figure, axes = plt.subplots(
        len(variables) * severity_rows, severity_columns,
        figsize=(3.5 * severity_columns, 3.15 * len(variables) * severity_rows),
        squeeze=False, subplot_kw={"projection": ccrs.PlateCarree()},
    )
    for variable_index, variable in enumerate(variables):
        row_start = variable_index * severity_rows
        image = None
        for severity_index, severity in enumerate(levels):
            row = row_start + severity_index // severity_columns
            column = severity_index % severity_columns
            axis = axes[row, column]
            image = axis.pcolormesh(
                longitudes, latitudes,
                (np.ma.masked_all_like(fields[severity_index, variable_index])
                 if placeholder else fields[severity_index, variable_index]),
                shading="auto",
                cmap=cmap, vmin=float(lower[variable_index]), vmax=float(upper[variable_index]),
                transform=ccrs.PlateCarree(), rasterized=True,
            )
            axis.set_global()
            axis.coastlines(linewidth=0.5)
            axis.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
            if placeholder:
                axis.text(0.5, 0.5, "blank field", transform=axis.transAxes,
                          ha="center", va="center", color="0.45", fontsize=8)
            axis.set_title(f"severity={severity:.3g}", fontsize=9)
            if column == 0:
                axis.set_ylabel(variable.replace("_", " "), fontsize=9)
        for severity_index in range(len(levels), severity_rows * severity_columns):
            row = row_start + severity_index // severity_columns
            column = severity_index % severity_columns
            axes[row, column].set_visible(False)
        figure.colorbar(
            image, ax=axes[row_start:row_start + severity_rows, :].ravel().tolist(),
            fraction=0.035, pad=0.025, label=colorbar_label,
        )
    figure.suptitle(f"{corruption_type}: {title} at each severity\nERA5 {timestamp}", fontsize=14)
    figure.subplots_adjust(left=0.03, right=0.87, bottom=0.03, top=0.88, wspace=0.07, hspace=0.25)
    output_path = output_dir / filename
    save_figure_bundle(figure, output_path, plot_type="baseline_plot", dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved corruption gallery to: {output_path}")



def plot_combined_corruption_gallery(output_dir, corruptions, levels, corrupted_fields, disturbances,
                                     variables, latitudes, longitudes, timestamp, placeholder=False):
    """Render a compact overview with labelled rows and per-corruption scales."""
    n_corruptions, n_severity, n_variables = corrupted_fields.shape[:3]
    endpoint_only = {"hemisphere_splice", "field_splice", "field_replace"}
    for variable_index, variable in enumerate(variables):
        raw = corrupted_fields[:, :, variable_index]
        difference = disturbances[:, :, variable_index]
        if placeholder:
            raw_low = np.full(n_corruptions, -1.0)
            raw_high = np.full(n_corruptions, 1.0)
            difference_limit = np.full(n_corruptions, 1.0)
        else:
            raw_low = np.nanpercentile(raw, 1.0, axis=(1, 2, 3))
            raw_high = np.nanpercentile(raw, 99.0, axis=(1, 2, 3))
            equal = np.isclose(raw_low, raw_high)
            raw_low[equal] -= 1e-8
            raw_high[equal] += 1e-8
            difference_limit = np.maximum(
                np.nanpercentile(np.abs(difference), 99.0, axis=(1, 2, 3)), 1e-8,
            )
        n_rows = 2 * n_corruptions
        figure = plt.figure(figsize=(2.63 * n_severity + 1.25, 1.40 * n_rows + 0.55))
        grid = figure.add_gridspec(
            n_rows, n_severity + 2,
            width_ratios=[0.62] + [1.0] * n_severity + [0.075],
            left=0.015, right=0.965, bottom=0.018, top=0.962,
            wspace=0.055, hspace=0.065,
        )
        for corruption_index, corruption in enumerate(corruptions):
            corruption = str(corruption)
            raw_row = 2 * corruption_index
            difference_row = raw_row + 1
            display_name = textwrap.fill(displayed_corruption_name(corruption), width=13)

            raw_label_axis = figure.add_subplot(grid[raw_row, 0])
            difference_label_axis = figure.add_subplot(grid[difference_row, 0])
            for axis in (raw_label_axis, difference_label_axis):
                axis.set_axis_off()
            raw_label_axis.text(
                0.98, 0.5, f"{display_name}\nField", ha="right", va="center",
                fontsize=8, fontweight="semibold", linespacing=1.15,
            )
            difference_label_axis.text(
                0.98, 0.5, "Difference", ha="right", va="center", fontsize=7.5,
            )

            if corruption in endpoint_only and n_severity > 1:
                severity_layout = (
                    (0, slice(1, 1 + n_severity // 2)),
                    (n_severity - 1, slice(1 + n_severity // 2, 1 + n_severity)),
                )
            else:
                severity_layout = tuple(
                    (severity_index, slice(1 + severity_index, 2 + severity_index))
                    for severity_index in range(n_severity)
                )

            raw_artist = difference_artist = None
            for severity_index, column_slice in severity_layout:
                raw_axis = figure.add_subplot(
                    grid[raw_row, column_slice], projection=ccrs.PlateCarree(),
                )
                difference_axis = figure.add_subplot(
                    grid[difference_row, column_slice], projection=ccrs.PlateCarree(),
                )
                raw_artist = raw_axis.pcolormesh(
                    longitudes, latitudes,
                    (np.ma.masked_all_like(raw[corruption_index, severity_index])
                     if placeholder else raw[corruption_index, severity_index]),
                    shading="auto", cmap="viridis",
                    vmin=float(raw_low[corruption_index]), vmax=float(raw_high[corruption_index]),
                    transform=ccrs.PlateCarree(), rasterized=True,
                )
                difference_artist = difference_axis.pcolormesh(
                    longitudes, latitudes,
                    (np.ma.masked_all_like(difference[corruption_index, severity_index])
                     if placeholder else difference[corruption_index, severity_index]),
                    shading="auto", cmap="RdBu_r",
                    vmin=-float(difference_limit[corruption_index]),
                    vmax=float(difference_limit[corruption_index]),
                    transform=ccrs.PlateCarree(), rasterized=True,
                )
                for axis in (raw_axis, difference_axis):
                    axis.set_global()
                    axis.coastlines(linewidth=0.42)
                    axis.add_feature(cfeature.BORDERS, linewidth=0.25, alpha=0.4)
                    if placeholder:
                        axis.text(0.5, 0.5, "blank field", transform=axis.transAxes,
                                  ha="center", va="center", color="0.45", fontsize=7)
                relative_strength = (
                    severity_index / (n_severity - 1) if n_severity > 1 else 1.0
                )
                raw_axis.set_title(f"Strength {relative_strength:g}", fontsize=7.5, pad=1.5)

            raw_bar = figure.colorbar(
                raw_artist, cax=figure.add_subplot(grid[raw_row, -1]),
                orientation="vertical", format="%.3g",
                ticks=[float(raw_low[corruption_index]),
                       float((raw_low[corruption_index] + raw_high[corruption_index]) / 2.0),
                       float(raw_high[corruption_index])],
            )
            difference_bar = figure.colorbar(
                difference_artist, cax=figure.add_subplot(grid[difference_row, -1]),
                orientation="vertical", format="%.3g",
                ticks=[-float(difference_limit[corruption_index]), 0.0,
                       float(difference_limit[corruption_index])],
            )
            for colorbar in (raw_bar, difference_bar):
                colorbar.ax.tick_params(labelsize=5.5, length=2, pad=1)

        figure.suptitle(
            f"Selected corruption probes: {variable.replace('_', ' ')} — ERA5 {timestamp}",
            fontsize=12.5, y=0.993,
        )
        suffix = "all_corruptions_gallery.png" if n_variables == 1 else f"all_corruptions_gallery_{variable}.png"
        output_path = output_dir / suffix
        save_figure_bundle(
            figure, output_path, plot_type="corruption_gallery", dpi=160,
            bbox_inches="tight", paperize=False,
        )
        plt.close(figure)
        print(f"Saved combined corruption gallery to: {output_path}")

def plot_corruption_disturbances(output_root, cfg=None):
    """Render raw corrupted fields and their ERA5 differences from the saved artifact."""
    input_path = output_root / "data" / "corruption_disturbances.nc"
    if not input_path.exists():
        return
    output_dir = output_root / "plots" / "corruption" / "disturbances"
    output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(input_path) as dataset:
        variables = [str(value) for value in dataset.variable.values]
        latitudes = np.asarray(dataset.latitude.values)
        longitudes = np.asarray(dataset.longitude.values)
        timestamp = str(dataset.attrs.get("timestamp", ""))
        has_corrupted_fields = "corrupted_field" in dataset.variables
        placeholder = bool(dataset.attrs.get("synthetic_layout_preview", 0))
        for corruption_index, corruption_type in enumerate(dataset.corruption.values):
            corruption_type = str(corruption_type)
            levels = np.asarray(dataset.severity.values[corruption_index])
            disturbances = np.asarray(dataset.disturbance.values[corruption_index])
            plot_corruption_field_gallery(
                output_dir, corruption_type, levels, disturbances, variables, latitudes, longitudes, timestamp,
                "physical disturbance", "Corrupted − ERA5", f"{corruption_type}.png", symmetric=True,
                placeholder=placeholder,
            )
            if has_corrupted_fields:
                corrupted_fields = np.asarray(dataset.corrupted_field.values[corruption_index])
                plot_corruption_field_gallery(
                    output_dir, corruption_type, levels, corrupted_fields, variables, latitudes, longitudes, timestamp,
                    "corrupted field", "Corrupted field", f"{corruption_type}_corrupted.png",
                    placeholder=placeholder,
                )
        if has_corrupted_fields:
            gallery_settings = (
                ((cfg.get("plotting", {}) or {}).get("corruption_gallery", {}) or {})
                if cfg is not None else {}
            )
            available_corruptions = [str(value) for value in dataset.corruption.values]
            requested_corruptions = [
                str(value) for value in gallery_settings.get(
                    "corruptions", available_corruptions,
                )
            ]
            corruption_indices = [
                available_corruptions.index(name)
                for name in requested_corruptions if name in available_corruptions
            ]
            if not corruption_indices:
                corruption_indices = list(range(len(available_corruptions)))
            severity_total = int(dataset.sizes["severity_index"])
            severity_count = min(
                max(int(gallery_settings.get("severity_count", severity_total)), 1),
                severity_total,
            )
            severity_indices = np.unique(
                np.rint(np.linspace(0, severity_total - 1, severity_count)).astype(int)
            )
            plot_combined_corruption_gallery(
                output_dir,
                [available_corruptions[index] for index in corruption_indices],
                np.asarray(dataset.severity.values)[np.ix_(corruption_indices, severity_indices)],
                np.asarray(dataset.corrupted_field.values)[np.ix_(corruption_indices, severity_indices)],
                np.asarray(dataset.disturbance.values)[np.ix_(corruption_indices, severity_indices)],
                variables, latitudes, longitudes, timestamp, placeholder=placeholder,
            )
        else:
            print(f"{input_path} lacks raw corrupted fields; rerun standard metric evaluation once.")


def discriminator_baseline_rows(cfg, real_ds, variables, output_root):
    """Re-evaluate separately trained target discriminators into baseline outputs."""
    settings = baseline_get(cfg, "discriminator", {})
    if not bool(settings.get("enabled", False)):
        return []
    try:
        from .train_target_discriminator_baselines import (
            SFNO_VARIABLES, compatible, load_sfno_encoder,
            load_sfno_probe_checkpoint, matched_statistics, data_dependent_donor_positions, indices, logits_for, sfno_context_settings,
        )
    except ImportError:
        from train_target_discriminator_baselines import (
            SFNO_VARIABLES, compatible, load_sfno_encoder,
            load_sfno_probe_checkpoint, matched_statistics, data_dependent_donor_positions, indices, logits_for, sfno_context_settings,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corruption_train = select_era5_split(real_ds, cfg, "train", coverage="corruption")
    corruption_test = select_era5_split(real_ds, cfg, "test", coverage="corruption")
    corruption_null = select_era5_split(real_ds, cfg, "null", coverage="corruption")
    checkpoint_root = Path(str(settings["checkpoint_dir"]))
    maximum = int(settings.get("evaluation_samples", 0))
    batch_size = int(settings.get("evaluation_batch_size", 32))
    discriminator_maximum = float(
        settings.get("corruption_severity_max", corruption_max_severity("hemisphere_splice", cfg))
    )
    discriminator_overrides = settings.get("corruption_severity_max_overrides", {}) or {}
    rows = []
    term_rows = []
    architectures = [("squeezenet", variables, None)]
    masked = settings.get("equator_masked_hemisphere_splice", {}) or {}
    masked_root = checkpoint_root / "squeezenet_equator_mask"
    if bool(masked.get("enabled", True)) and masked_root.exists():
        architectures.append(("squeezenet_equator_mask", variables, None))
    attention = settings.get("attention_squeezenet", {}) or {}
    attention_root = checkpoint_root / "squeezenet_attention"
    if bool(attention.get("enabled", False)) and attention_root.exists():
        architectures.append(("squeezenet_attention", variables, None))
    elif bool(attention.get("enabled", False)):
        print(
            "Skipping attention-SqueezeNet target baselines: no checkpoints under "
            f"{attention_root}."
        )
    sfno = settings.get("sfno", {}) or {}
    if bool(sfno.get("enabled", False)):
        sfno_checkpoints = [
            checkpoint_root / architecture / kind / label.replace(" ", "_") / "model.pth"
            for architecture in ("sfno_linear", "sfno_mlp")
            for kind, label in (
                [("forecast", name) for name in settings.get("forecast_files", {})]
                + [("corruption", name) for name in corruption_types_from_config(cfg)]
            )
        ]
        if any(path.exists() for path in sfno_checkpoints):
            encoder = load_sfno_encoder(cfg, device)
            architectures.extend([
                ("sfno_linear", SFNO_VARIABLES, encoder),
                ("sfno_mlp", SFNO_VARIABLES, encoder),
            ])
        else:
            print(f"Skipping SFNO target baselines: no probe checkpoints under {checkpoint_root}.")
    for architecture, input_variables, encoder in architectures:
        targets = [
            ("forecast", label, None, paths)
            for label, paths in settings.get("forecast_files", {}).items()
        ]
        targets += [
            ("corruption", name, name, None)
            for name in corruption_types_from_config(cfg) if compatible(name, input_variables)
        ]
        if architecture == "squeezenet_equator_mask":
            targets = [("corruption", "hemisphere_splice", "hemisphere_splice", None)]
        for kind, label, corruption, path in tqdm(
            targets, desc=f"Evaluating {architecture} target baselines"
        ):
            checkpoint = (
                checkpoint_root / kind / label.replace(" ", "_") / "model.pth"
                if architecture == "squeezenet" else
                checkpoint_root / architecture / kind / label.replace(" ", "_") / "model.pth"
            )
            if not checkpoint.exists():
                print(f"Skipping {architecture} {kind}/{label}: checkpoint not found at {checkpoint}")
                continue
            validate_binding(checkpoint, cfg)
            checkpoint_hash = file_sha256(checkpoint)
            if architecture in {"squeezenet", "squeezenet_attention", "squeezenet_equator_mask"}:
                model_name = (
                    settings.get("model_name", "squeezenet")
                    if architecture in {"squeezenet", "squeezenet_equator_mask"} else
                    attention.get("model_name", "squeezenet_attention")
                )
                model = WeatherDiscriminator(
                    len(variables), model_name,
                    pretrained_backbone=False,
                ).to(device)
                model.model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                if architecture == "squeezenet_equator_mask":
                    model.equator_mask_degrees = float(masked.get("half_width_degrees", 10.0))
                metadata = {"encoder_pretraining": ""}
                model.eval()
                model.histogram_target = label
            else:
                model, metadata = load_sfno_probe_checkpoint(
                    checkpoint, cfg, device, encoder=encoder,
                )
                model.sfno_use_era5_context, model.sfno_target_variables = sfno_context_settings(cfg)
                model.histogram_target = label
            reference_indices = None
            if corruption:
                means = {v: float(corruption_train[v].mean()) for v in variables}
                stds = {v: max(float(corruption_train[v].std()), 1e-8) for v in variables}
                ep_dataset = corruption_test
                null_dataset = corruption_null
                candidate = corruption_test
                test_pairs = None
            else:
                opened = [
                    normalize_prediction_timedelta(safe_open_dataset(candidate_path))
                    for candidate_path in path
                ]
                candidate = concatenate_forecasts(opened)
                train_pairs = forecast_pairs(candidate, real_ds, cfg, "train", cfg.lead_times)
                test_pairs = forecast_pairs(candidate, real_ds, cfg, "test", cfg.lead_times)
                means, stds = matched_statistics(real_ds, train_pairs, variables)
                ep_dataset = real_ds
                reference_indices = np.unique(
                    [pair.era5_index for pair in test_pairs]
                ).astype(int)
                model_years = np.unique(
                    np.asarray(candidate.time.values).astype("datetime64[Y]").astype(int)
                )
                model_null = select_era5_split(real_ds, cfg, "null", coverage="model")
                null_years = np.asarray(model_null.time.values).astype("datetime64[Y]").astype(int)
                null_dataset = model_null.isel(
                    time=np.flatnonzero(np.isin(null_years, model_years))
                )
            ep_selected = (
                indices(ep_dataset, maximum)
                if reference_indices is None else np.asarray(reference_indices, dtype=int)
            )
            if maximum > 0 and len(ep_selected) > maximum:
                ep_selected = ep_selected[np.linspace(0, len(ep_selected) - 1, maximum, dtype=int)]
            ep_logits = logits_for(
                model, ep_dataset, input_variables, means, stds, device, maximum, batch_size,
                progress_description=f"{architecture} {label}: ERA5 held-out reference",
                selected_indices=ep_selected,
            )
            ep_terms = -np.exp(np.minimum(-ep_logits, 80))
            ep = float(ep_terms.mean())
            epse = float(ep_terms.std(ddof=1) / np.sqrt(len(ep_terms))) if len(ep_terms) > 1 else 0.0
            for position, (source_index, logit, term) in enumerate(zip(ep_selected, ep_logits, ep_terms)):
                term_rows.append({
                    "architecture": architecture, "kind": kind, "target": label,
                    "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                    "role": "ep_reference", "source": "ERA5 test", "x": "",
                    "sample_position": position, "source_index": int(source_index),
                    "time": str(np.asarray(ep_dataset.time.values)[int(source_index)]),
                    "initialization_time": "", "valid_time": str(np.asarray(ep_dataset.time.values)[int(source_index)]),
                    "lead_hour": "", "severity": "", "logit": float(logit),
                    "transformed_term": float(term),
                })
            points = [(0.0, None, None, ERA5_NULL_LABEL, None, None)]
            if corruption:
                points.append((0.0, None, 0.0, label, None, None))
                maximum_severity = float(discriminator_overrides.get(corruption, discriminator_maximum))
                levels = np.linspace(0.0, maximum_severity, int(settings["corruption_steps"]))
                points += [(float(level), None, float(level), label, None, None) for level in levels[1:]]
            else:
                for lead_index, lead in enumerate(lead_hours(candidate)):
                    selected_pairs = [pair for pair in test_pairs if pair.lead_index == lead_index]
                    selected = [pair.forecast_index for pair in selected_pairs]
                    context_selected = [pair.era5_index for pair in selected_pairs]
                    if int(lead) in set(cfg.lead_times) and selected:
                        points.append((float(lead), lead_index, None, label, selected, context_selected))
            for x, lead_index, severity, source, selected, context_selected in points:
                evaluation_ds = null_dataset if source == ERA5_NULL_LABEL else candidate
                eval_selected = indices(evaluation_ds, maximum) if selected is None else np.asarray(selected, dtype=int)
                eval_context = None if context_selected is None else np.asarray(context_selected, dtype=int)
                if maximum > 0 and len(eval_selected) > maximum:
                    keep = np.linspace(0, len(eval_selected) - 1, maximum, dtype=int)
                    eval_selected = eval_selected[keep]
                    if eval_context is not None:
                        eval_context = eval_context[keep]
                eval_logits = logits_for(
                    model, evaluation_ds, input_variables, means, stds, device, maximum, batch_size,
                    lead=lead_index, corruption=corruption if severity is not None else None,
                    severity=severity or 0.0, cfg=cfg,
                    maximum_severity=(maximum_severity if corruption else None),
                    selected_indices=eval_selected,
                    context_ds=(real_ds if eval_context is not None and getattr(model, "sfno_use_era5_context", False) else None),
                    context_indices=eval_context,
                    progress_description=(
                        f"{architecture} {label}: {source} "
                        f"({'severity' if severity is not None else 'lead'}={x:g})"
                    ),
                )
                eval_terms = eval_logits - 1.0
                mean = float(eval_terms.mean())
                stderr = float(eval_terms.std(ddof=1) / np.sqrt(len(eval_terms))) if len(eval_terms) > 1 else 0.0
                count = len(eval_terms)
                pair_lookup = ({pair.forecast_index: pair for pair in test_pairs if pair.lead_index == lead_index}
                               if test_pairs is not None and lead_index is not None else {})
                eval_donors = (data_dependent_donor_positions(
                    corruption, len(eval_selected), len(input_variables), int(cfg.get("seed", 0))
                ) if corruption in DATA_DEPENDENT_CORRUPTIONS else None)
                for position, (source_index, logit, term) in enumerate(zip(eval_selected, eval_logits, eval_terms)):
                    pair = pair_lookup.get(int(source_index))
                    timestamp = str(np.asarray(evaluation_ds.time.values)[int(source_index)])
                    if eval_donors is None:
                        donor_indices = []
                    elif eval_donors.ndim == 1:
                        donor_indices = [int(eval_selected[int(eval_donors[position])])]
                    else:
                        donor_indices = [int(eval_selected[int(eval_donors[field, position])])
                                         for field in range(eval_donors.shape[0])]
                    term_rows.append({
                        "architecture": architecture, "kind": kind, "target": label,
                        "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                        "role": "candidate", "source": source, "x": float(x),
                        "sample_position": position, "source_index": int(source_index),
                        "time": str(pair.valid_time) if pair is not None else timestamp,
                        "initialization_time": str(pair.initialization_time) if pair is not None else "",
                        "valid_time": str(pair.valid_time) if pair is not None else timestamp,
                        "lead_hour": int(pair.lead_hour) if pair is not None else "",
                        "severity": "" if severity is None else float(severity),
                        "corruption_seed": ("" if corruption is None else int(corruption_sample_seed(
                            cfg.get("seed", 0), corruption, int(source_index)
                        ))),
                        "donor_source_indices": json.dumps(donor_indices),
                        "logit": float(logit), "transformed_term": float(term),
                    })
                rows.append({
                    "architecture": architecture,
                    "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                    "input_variables": ",".join(input_variables),
                    "encoder_pretraining": metadata.get("encoder_pretraining", ""),
                    "kind": kind, "target": label, "x": x, "source": source,
                    "is_era5_test_null": source == ERA5_NULL_LABEL, "score": ep - mean,
                    "stderr": float(np.hypot(epse, stderr)), "n_samples": count,
                    "ep_reference": ep, "ep_reference_split": "test",
                    "ep_n_samples": len(ep_terms),
                })
            if not corruption:
                candidate.close()
    return rows, term_rows


def write_discriminator_baselines(rows, cfg, output_root):
    data_path = output_root / "data" / "discriminator_reverse_kl.csv"
    if not rows:
        data_path.unlink(missing_ok=True)
        return
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_discriminator_baselines(output_root):
    """Read target-discriminator scores without loading models or datasets."""
    data_path = output_root / "data" / "discriminator_reverse_kl.csv"
    if not data_path.exists():
        return []
    with open(data_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in (
            "x", "score", "score_lower", "score_upper", "stderr",
            "ep_reference", "ep_train",
        ):
            if field not in row or row[field] == "":
                continue
            row[field] = float(row[field])
        row["n_samples"] = int(float(row["n_samples"]))
        if row.get("ep_n_samples", "") != "":
            row["ep_n_samples"] = int(float(row["ep_n_samples"]))
        row["is_era5_test_null"] = str(row["is_era5_test_null"]).lower() == "true"
    return rows


def read_discriminator_terms(output_root):
    """Read the reconstructible per-sample discriminator terms."""
    data_path = output_root / "data" / "discriminator_terms.csv.gz"
    if not data_path.exists():
        return []
    with gzip.open(data_path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in ("x", "lead_hour", "logit"):
            if row.get(field, "") != "":
                row[field] = float(row[field])
    return rows


def gallery_logit_bin_edges(panels, bins=40, independent_target="ERA5 Forecast"):
    """Use a shared logit axis except for a target with a distinct critic scale."""
    all_values = [reference for _, reference, _ in panels]
    all_values += [candidate for _, _, groups in panels for _, candidate in groups]
    finite = [values[np.isfinite(values)] for values in all_values if values.size]
    common_edges = np.histogram_bin_edges(np.concatenate(finite), bins=int(bins))
    edges_by_model = {}
    for model, reference, groups in panels:
        if displayed_model_name(model).lower() != independent_target.lower():
            edges_by_model[model] = common_edges
            continue
        model_values = [reference, *[candidate for _, candidate in groups]]
        model_finite = [values[np.isfinite(values)] for values in model_values if values.size]
        edges_by_model[model] = np.histogram_bin_edges(
            np.concatenate(model_finite), bins=int(bins),
        )
    return common_edges, edges_by_model


def plot_forecast_logit_histogram_gallery(term_rows, cfg, output_root):
    """Plot every available forecast target's held-out logit densities."""
    settings = (cfg.get("plotting", {}) or {}).get("logit_histogram_gallery", {}) or {}
    if not bool(settings.get("enabled", True)) or not term_rows:
        return
    architecture = str(settings.get("architecture", "squeezenet"))
    candidate_rows = [
        row for row in term_rows
        if row.get("architecture") == architecture
        and row.get("kind") == "forecast"
        and row.get("role") == "candidate"
    ]
    available_models = {
        row.get("target") for row in candidate_rows if row.get("target")
    }
    requested = [str(value) for value in settings.get("models", [])]
    configured = list((baseline_get(cfg, "forecast_files", {}) or {}).keys())
    if bool(settings.get("include_all_available", True)):
        model_order = list(dict.fromkeys([*requested, *configured, *sorted(available_models)]))
    else:
        model_order = requested or configured
    models = [model for model in model_order if model in available_models]

    available_resamples = sorted({
        row.get("resample_id", "") for row in candidate_rows
    })
    preferred = str(settings.get("resample_id", "learned_04"))
    if available_resamples and any(available_resamples):
        selected_resample = (
            preferred if preferred in available_resamples else available_resamples[0]
        )
        candidate_rows = [
            row for row in candidate_rows
            if row.get("resample_id", "") == selected_resample
        ]
    else:
        selected_resample = ""

    panels = []
    for model in models:
        model_rows = [row for row in candidate_rows if row.get("target") == model]
        reference = np.asarray([
            row["logit"] for row in model_rows
            if row.get("source") == ERA5_NULL_LABEL
        ], dtype=float)
        leads = sorted({
            int(row["lead_hour"]) for row in model_rows
            if row.get("source") == model and row.get("lead_hour", "") != ""
        })
        groups = []
        for lead in leads:
            candidate = np.asarray([
                row["logit"] for row in model_rows
                if row.get("source") == model
                and row.get("lead_hour", "") != ""
                and int(row["lead_hour"]) == lead
            ], dtype=float)
            if candidate.size:
                groups.append((lead, candidate))
        if reference.size and groups:
            panels.append((model, reference, groups))
    if not panels:
        print(f"Skipping {architecture} forecast-logit gallery: no matching per-sample terms.")
        return

    edges, edges_by_model = gallery_logit_bin_edges(panels, bins=40)
    all_leads = sorted({lead for _, _, groups in panels for lead, _ in groups})
    colors = dict(zip(all_leads, lead_time_colors(len(all_leads))))
    nrows = int(math.ceil((len(panels) + 1) / 2))
    figure, axes = plt.subplots(
        nrows, 2, figsize=(10.0, 3.0 * nrows), squeeze=False,
        layout="constrained",
    )
    payload = {"bin_edges": edges, "resample_id": np.asarray(selected_resample)}
    for panel_index, (axis, (model, reference, groups)) in enumerate(
        zip(axes.flat, panels)
    ):
        panel_edges = edges_by_model[model]
        axis.hist(
            reference, bins=panel_edges, density=True, histtype="stepfilled",
            color=REFERENCE_COLOR, alpha=0.32, label="ERA5 test",
        )
        payload[f"model_{panel_index}"] = np.asarray(model)
        payload[f"display_scale_{panel_index}"] = np.asarray(1.0)
        payload[f"bin_edges_{panel_index}"] = panel_edges
        payload[f"reference_logits_{panel_index}"] = reference
        for lead, candidate in groups:
            axis.hist(
                candidate, bins=panel_edges, density=True, histtype="step",
                linewidth=1.35, color=colors[lead], label=f"+{lead} h",
            )
            payload[f"candidate_logits_{panel_index}_{lead}h"] = candidate
        axis.axvline(0.0, color="black", linewidth=0.7, alpha=0.45)
        axis.set_title(displayed_model_name(model))
        if panel_index // 2 == nrows - 1:
            axis.set_xlabel("Real-vs-fake logit")
        if panel_index % 2 == 0:
            axis.set_ylabel("Density")
        axis.grid(alpha=0.22)

    legend_index = len(panels)
    legend_axis = axes.flat[legend_index]
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend_axis.set_axis_off()
    legend_axis.legend(
        handles, labels, loc="center", ncol=2, frameon=False,
        title="Lead time", fontsize=7, title_fontsize=8,
    )
    for axis in axes.flat[legend_index + 1:]:
        axis.set_axis_off()
    figure.suptitle(
        f"Held-out forecast logits: {displayed_architecture_name(architecture)}"
    )
    output_path = (
        output_root / "plots" / "target_logit_distributions" / architecture
        / "forecast" / "all_models_all_lead_times.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_bundle(
        figure, output_path, plot_type="forecast_logit_histogram_gallery",
        payload=payload, dpi=220, bbox_inches="tight",
    )
    plt.close(figure)
    print(
        f"Saved two-column forecast-logit gallery with {len(panels)} target(s) "
        f"to: {output_path}"
    )

    if not bool(settings.get("individual_models", True)):
        return
    individual_root = output_path.parent / "by_model"
    for model, reference, groups in panels:
        model_values = [reference, *[candidate for _, candidate in groups]]
        model_finite = [
            values[np.isfinite(values)] for values in model_values if values.size
        ]
        model_edges = np.histogram_bin_edges(np.concatenate(model_finite), bins=40)
        individual, axis = plt.subplots(figsize=(6.0, 4.4))
        axis.hist(
            reference, bins=model_edges, density=True, histtype="stepfilled",
            color=REFERENCE_COLOR, alpha=0.32, label="ERA5 test",
        )
        individual_payload = {
            "bin_edges": model_edges,
            "resample_id": np.asarray(selected_resample),
            "model": np.asarray(model),
            "reference_logits": reference,
        }
        for lead, candidate in groups:
            axis.hist(
                candidate, bins=model_edges, density=True, histtype="step",
                linewidth=1.35, color=colors[lead], label=f"+{lead} h",
            )
            individual_payload[f"candidate_logits_{lead}h"] = candidate
        axis.axvline(0.0, color="black", linewidth=0.7, alpha=0.45)
        axis.set_xlabel("Real-vs-fake logit")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.22)
        handles, labels = axis.get_legend_handles_labels()
        individual.legend(
            handles, labels, loc="lower center", ncol=2, frameon=False,
            title="Lead time", fontsize=7, title_fontsize=8,
        )
        individual.suptitle(
            f"Held-out forecast logits: {displayed_model_name(model)}\n"
            f"{displayed_architecture_name(architecture)}; {selected_resample}"
        )
        individual.subplots_adjust(
            left=0.14, right=0.98, top=0.82, bottom=0.30,
        )
        filename = f"{model.replace(' ', '_')}_all_lead_times.png"
        individual_path = individual_root / filename
        individual_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure_bundle(
            individual, individual_path,
            plot_type="forecast_logit_histogram_by_model",
            payload=individual_payload, dpi=220, bbox_inches="tight",
            paper_width_kind="half",
        )
        plt.close(individual)
        print(f"Saved per-model forecast-logit histogram to: {individual_path}")


def plot_forecast_critic_and_graphcast_histograms(rows, term_rows, cfg, output_root):
    """Pair forecast critic curves with GraphCast held-out logit densities."""
    if not rows or not term_rows:
        return None
    architecture = "squeezenet"
    target = "GraphCast"
    forecast_rows = [
        row for row in rows
        if row.get("architecture") == architecture and row.get("kind") == "forecast"
    ]
    targets = [
        label for label in (baseline_get(cfg, "forecast_files", {}) or {}).keys()
        if any(row.get("target") == label for row in forecast_rows)
    ]
    targets.extend(sorted({
        row.get("target") for row in forecast_rows
        if row.get("target") and row.get("target") not in targets
    }))
    graphcast_terms = [
        row for row in term_rows
        if row.get("architecture") == architecture
        and row.get("kind") == "forecast"
        and row.get("target") == target
    ]
    available_resamples = sorted({
        row.get("resample_id", "") for row in graphcast_terms
        if row.get("resample_id", "")
    })
    preferred = str(
        ((cfg.get("plotting", {}) or {}).get("logit_histogram_gallery", {}) or {}).get(
            "resample_id", "learned_04"
        )
    )
    if available_resamples:
        selected_resample = preferred if preferred in available_resamples else available_resamples[0]
        graphcast_terms = [
            row for row in graphcast_terms
            if row.get("resample_id", "") == selected_resample
        ]
    else:
        selected_resample = ""
    reference = np.asarray([
        row["logit"] for row in graphcast_terms
        if row.get("source") == ERA5_NULL_LABEL
    ], dtype=float)
    leads = sorted({
        int(row["lead_hour"]) for row in graphcast_terms
        if row.get("source") == target and row.get("lead_hour", "") != ""
    })
    groups = [
        (lead, np.asarray([
            row["logit"] for row in graphcast_terms
            if row.get("source") == target
            and row.get("lead_hour", "") != ""
            and int(row["lead_hour"]) == lead
        ], dtype=float))
        for lead in leads
    ]
    groups = [(lead, values) for lead, values in groups if values.size]
    if not targets or not reference.size or not groups:
        print("Skipping paired forecast critic/GraphCast histogram: incomplete persisted data.")
        return None

    figure = plt.figure(figsize=(11.0, 5.8), layout="constrained")
    grid = figure.add_gridspec(
        2, 2, height_ratios=[4.4, 1.7], width_ratios=[1.0, 1.0],
    )
    critic_axis = figure.add_subplot(grid[0, 0])
    histogram_axis = figure.add_subplot(grid[0, 1])
    critic_legend_axis = figure.add_subplot(grid[1, 0])
    histogram_legend_axis = figure.add_subplot(grid[1, 1])
    critic_legend_axis.set_axis_off()
    histogram_legend_axis.set_axis_off()

    target_colors = model_colors(targets)
    maximum_lead = max(
        [float(row["x"]) for row in forecast_rows if not row["is_era5_test_null"]]
        or [1.0]
    )
    null_positions = null_diamond_positions(
        len(targets), 0.03 * max(maximum_lead, 1.0),
    )
    null_target_labels = []
    null_x_positions = []
    for target_index, (color, label) in enumerate(zip(target_colors, targets)):
        target_rows = [row for row in forecast_rows if row.get("target") == label]
        series = sorted(
            [row for row in target_rows if not row["is_era5_test_null"]],
            key=lambda row: row["x"],
        )
        if not series:
            continue
        x_values = np.asarray([row["x"] for row in series], dtype=float)
        critic_axis.plot(
            x_values, [row["score"] for row in series],
            marker=series_marker(target_index), color=color,
            label=displayed_model_name(label),
        )
        if "score_lower" in series[0]:
            critic_axis.fill_between(
                x_values,
                [row["score_lower"] for row in series],
                [row["score_upper"] for row in series],
                color=color, alpha=0.18, linewidth=0,
            )
        null = next((row for row in target_rows if row["is_era5_test_null"]), None)
        if null is not None:
            lower = float(null.get("score_lower", null["score"]))
            upper = float(null.get("score_upper", null["score"]))
            null_x = float(null_positions[target_index])
            critic_axis.errorbar(
                [null_x], [null["score"]],
                yerr=[[null["score"] - lower], [upper - null["score"]]],
                marker="D", markersize=3.0, linestyle="None",
                color=color, capsize=2, zorder=4,
            )
            null_target_labels.append(label)
            null_x_positions.append(null_x)
    critic_axis.set(
        title="Critic score vs. lead time",
        xlabel="Lead time (hours)", ylabel="Critic score",
        xlim=(-0.035 * max(maximum_lead, 1.0), 1.035 * max(maximum_lead, 1.0)),
    )
    critic_axis.grid(alpha=0.3)
    handles, labels = critic_axis.get_legend_handles_labels()
    critic_legend_axis.legend(
        handles, labels, loc="center", ncol=2, frameon=False,
        title="Forecast model", fontsize=6.5, title_fontsize=7,
    )

    finite = [reference[np.isfinite(reference)]] + [
        values[np.isfinite(values)] for _, values in groups
    ]
    edges = np.histogram_bin_edges(np.concatenate(finite), bins=40)
    histogram_axis.hist(
        reference, bins=edges, density=True, histtype="stepfilled",
        color=REFERENCE_COLOR, alpha=0.32, label="ERA5 test",
    )
    lead_colors = lead_time_colors(len(groups))
    for lead_index, (color, (lead, values)) in enumerate(zip(lead_colors, groups)):
        histogram_axis.hist(
            values, bins=edges, density=True, histtype="step",
            linewidth=1.25, linestyle=("-" if lead_index < len(PLOT_PALETTE) - 1 else "--"),
            color=color, label=f"+{lead} h",
        )
    histogram_axis.axvline(0.0, color="black", linewidth=0.7, alpha=0.45)
    histogram_axis.set(
        title="GraphCast held-out logits",
        xlabel="Real-vs-fake logit", ylabel="Density",
    )
    histogram_axis.grid(alpha=0.22)
    handles, labels = histogram_axis.get_legend_handles_labels()
    histogram_legend_axis.legend(
        handles, labels, loc="center", ncol=2, frameon=False,
        title="Lead time", fontsize=6.5, title_fontsize=7,
    )

    figure.suptitle("Forecast realism critic and GraphCast logit distributions")
    output_path = output_root / "plots" / "forecast_critic_and_graphcast_logits.png"
    payload = {
        "histogram_bin_edges": edges,
        "graphcast_reference_logits": reference,
        "resample_id": np.asarray(selected_resample),
        "critic_null_targets": np.asarray(null_target_labels),
        "critic_null_x_positions": np.asarray(null_x_positions, dtype=float),
    }
    for lead, values in groups:
        payload[f"graphcast_logits_{lead}h"] = values
    save_figure_bundle(
        figure, output_path,
        plot_type="forecast_critic_and_graphcast_histograms",
        payload=payload, dpi=220, paper_width_kind="full",
    )
    plt.close(figure)
    print(f"Saved paired forecast critic/GraphCast histogram figure to: {output_path}")
    return output_path

def plot_discriminator_baselines(rows, cfg, output_root):

    """Plot evaluated target-discriminator scores with dedicated legend space."""
    if not rows:
        return
    settings = baseline_get(cfg, "discriminator", {})
    plot_root = output_root / "plots" / "discriminator"
    plot_root.mkdir(parents=True, exist_ok=True)
    for architecture in sorted({row["architecture"] for row in rows}):
        architecture_rows = [row for row in rows if row["architecture"] == architecture]
        architecture_root = plot_root / architecture
        architecture_root.mkdir(parents=True, exist_ok=True)
        for kind, filename in (
            ("forecast", "lead_time_reverse_kl.png"),
            ("corruption", "corruption_strength_reverse_kl.png"),
        ):
            visual_corruption_scale = kind == "corruption"
            targets = sorted({
                row["target"] for row in architecture_rows if row["kind"] == kind
            })
            if not targets:
                continue
            figure = plt.figure(
                figsize=(6.4, 9.0 if visual_corruption_scale else 6.4),
                layout="constrained",
            )
            grid = figure.add_gridspec(
                2, 1,
                height_ratios=([4.2, 3.5] if visual_corruption_scale else [4.2, 1.7]),
            )
            axis = figure.add_subplot(grid[0])
            legend_axis = figure.add_subplot(grid[1])
            legend_axis.set_axis_off()
            colors = (
                model_colors(targets) if kind == "forecast"
                else categorical_colors(max(len(targets), 1))
            )
            for target_index, (color, target) in enumerate(zip(colors, targets)):
                target_rows = [
                    row for row in architecture_rows
                    if row["kind"] == kind and row["target"] == target
                ]
                series = sorted(
                    [row for row in target_rows if not row["is_era5_test_null"]],
                    key=lambda row: row["x"],
                )
                if not series:
                    continue
                x_values = (
                    relative_corruption_coordinates(series, "x")
                    if visual_corruption_scale
                    else [row["x"] for row in series]
                )
                label = (
                    corruption_range_label(target, series, "x")
                    if visual_corruption_scale else displayed_model_name(target)
                )
                axis.plot(
                    x_values, [row["score"] for row in series],
                    marker=series_marker(target_index), color=color, label=label,
                )
                if "score_lower" in series[0]:
                    axis.fill_between(
                        x_values,
                        [row["score_lower"] for row in series],
                        [row["score_upper"] for row in series],
                        color=color, alpha=0.18, linewidth=0,
                    )
                null = next(
                    (row for row in target_rows if row["is_era5_test_null"]), None
                )
                if null is not None:
                    lower = float(null.get("score_lower", null["score"]))
                    upper = float(null.get("score_upper", null["score"]))
                    axis.errorbar(
                        [0], [null["score"]],
                        yerr=[[null["score"] - lower], [upper - null["score"]]],
                        marker="D", markersize=3.5, linestyle="None",
                        color=color, capsize=2, zorder=4,
                    )
            axis.set_ylabel("Critic score")
            axis.set_xlabel(
                "Relative corruption severity"
                if visual_corruption_scale else "Lead time (hours)"
            )
            if visual_corruption_scale:
                axis.set_xlim(-0.035, 1.035)
            if settings.get("plot_yscale", "symlog") == "symlog":
                axis.set_yscale(
                    "symlog", linthresh=float(settings.get("plot_linthresh", 1e-2))
                )
            else:
                axis.set_yscale(str(settings.get("plot_yscale")))
            axis.grid(alpha=0.3, which="both")
            handles, labels = axis.get_legend_handles_labels()
            legend_axis.legend(
                handles, labels, loc="center",
                ncol=(1 if visual_corruption_scale else 2),
                frameon=False, fontsize=6.5,
            )
            comparison = "corruption strength" if kind == "corruption" else "lead time"
            figure.suptitle(
                f"Critic score vs. {comparison}\n"
                f"{displayed_architecture_name(architecture)}; diamonds: ERA5 null"
            )
            save_figure_bundle(
                figure, architecture_root / filename,
                plot_type="discriminator_reverse_kl", dpi=220,
                paper_width_kind="half",
            )
            plt.close(figure)


def evaluate_standard_metrics(cfg):
    """Evaluate non-learned baselines and persist all plot-ready statistics."""
    variables = variables_from_config(cfg)
    metric_names = metric_names_from_config(cfg)
    output_root = baseline_output_dir(cfg, variables)

    real_ds = select_level(safe_open_dataset(cfg.real_nc_file), cfg.get("level"))
    missing = [variable for variable in variables if variable not in real_ds.data_vars]
    if missing:
        raise ValueError(f"Variables missing from ERA5/reference data: {missing}")

    scratch_dir = baseline_get(cfg, "scratch_dir", None)
    if scratch_dir is not None:
        Path(str(scratch_dir)).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="baseline-full-statistics-",
        dir=None if scratch_dir is None else str(scratch_dir),
    ) as temporary_dir:
        model_train = select_level(
            select_era5_split(real_ds, cfg, "train", coverage="model"), cfg.get("level")
        )
        model_null = select_level(
            select_era5_split(real_ds, cfg, "null", coverage="model"), cfg.get("level")
        )
        max_samples = int(baseline_get(cfg, "eval_samples", cfg_get(cfg, "max_samples", 100)))
        train_indices = sample_time_indices(model_train, max_samples)
        null_indices = sample_time_indices(model_null, max_samples)
        if active_schedule(cfg) is not None and len(train_indices) > len(null_indices):
            keep = np.linspace(0, len(train_indices) - 1, len(null_indices), dtype=int)
            train_indices = [train_indices[int(position)] for position in keep]
        null_stats = streaming_reference_stats(cfg, model_train, variables, train_indices)
        train_features = streaming_joint_features(
            cfg, model_train, variables, null_stats, train_indices, metric_names,
            Path(temporary_dir) / "model-train-null-scwd.dat",
            description="ERA5 training complement model-range reference",
        )
        null_features = streaming_joint_features(
            cfg, model_null, variables, null_stats, null_indices, metric_names,
            Path(temporary_dir) / "model-second-half-null-scwd.dat",
            description="ERA5 model-range test null",
        )
        null_row = evaluate_era5_train_shift_metrics(
            null_features, train_features, variables, metric_names, cfg
        )
        null_row.update(label=ERA5_NULL_LABEL, is_null=True)
        null_scwd_diagnostics = []
        null_global_mean_diagnostics = []
        if "global_mean_wasserstein" in metric_names:
            null_global_mean_diagnostics.append(global_mean_wasserstein_diagnostic(
                null_features, train_features, cfg, ERA5_NULL_LABEL, 0,
                null_row["global_mean_wasserstein"], comparison_kind="null", row=null_row,
            ))
        if "scwd" in metric_names:
            null_scwd_diagnostics.append(
                scwd_anchor_diagnostic(
                    null_features, train_features, cfg, ERA5_NULL_LABEL, 0,
                    null_row["scwd"], comparison_kind="null",
                )
            )
        lead_rows = [null_row]
        if bool(baseline_get(cfg, "evaluate_forecasts", True)):
            forecast_rows, scwd_diagnostics, global_mean_diagnostics = evaluate_lead_metrics(
                cfg, real_ds, variables, metric_names, temporary_dir,
            )
        else:
            forecast_rows, scwd_diagnostics, global_mean_diagnostics = [], [], []
            print("Skipping forecast-model standard metrics (baseline.evaluate_forecasts=false).")
        lead_rows.extend(forecast_rows)
        corruption_reference_stats = normalization_stats_for_corruptions(
            cfg, real_ds, variables
        )
        corruption_rows, corruption_scwd_diagnostics, corruption_global_mean_diagnostics = evaluate_corruption_metrics(
            cfg, real_ds, corruption_reference_stats, variables, metric_names, temporary_dir,
            return_scwd_diagnostics=True,
        )
        global_mean_diagnostics = (
            null_global_mean_diagnostics + global_mean_diagnostics + corruption_global_mean_diagnostics
        )
        if "global_mean_wasserstein" in metric_names:
            finalize_global_mean_wasserstein(global_mean_diagnostics, cfg)
        close_feature_memmaps(train_features)
        close_feature_memmaps(null_features)

    output_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=output_root / "resolved_config.yaml", resolve=True)
    lead_path = output_root / "data" / "lead_time.csv"
    corruption_path = output_root / "data" / "corruption_strength.csv"
    write_metric_csv(lead_rows, metric_names, lead_path, "lead_time")
    write_metric_csv(
        corruption_rows, metric_names, corruption_path, "corruption_strength"
    )
    schedule = active_schedule(cfg)
    canonical_ordinal = min(4, int(settings(cfg).get("fixed_replicates", 50)) - 1)
    retain_diagnostics = schedule is None or schedule.ordinal == canonical_ordinal
    if retain_diagnostics:
        write_scwd_anchor_diagnostics(
            null_scwd_diagnostics + scwd_diagnostics + corruption_scwd_diagnostics, output_root
        )
        write_global_mean_wasserstein_diagnostics(global_mean_diagnostics, variables, output_root)
        evaluate_corruption_disturbances(
            cfg, real_ds, corruption_reference_stats, variables, output_root
        )
    split_manifest_path = write_split_membership(cfg, real_ds, output_root)
    real_ds.close()
    paths = [output_root / "resolved_config.yaml", lead_path, corruption_path]
    if split_manifest_path is not None:
        paths.append(split_manifest_path)
    paths += [
        path for path in (
            output_root / "data" / "scwd_anchor_contributions.nc",
            output_root / "data" / "global_mean_wasserstein_distributions.nc",
            output_root / "data" / "corruption_disturbances.nc",
        ) if path.is_file()
    ]
    return paths


def evaluate_discriminator_metrics(cfg):
    """Evaluate trained target critics independently of standard metrics."""
    variables = variables_from_config(cfg)
    output_root = baseline_output_dir(cfg, variables)
    real_ds = select_level(safe_open_dataset(cfg.real_nc_file), cfg.get("level"))
    missing = [variable for variable in variables if variable not in real_ds.data_vars]
    if missing:
        real_ds.close()
        raise ValueError(f"Variables missing from ERA5/reference data: {missing}")
    output_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=output_root / "resolved_config.yaml", resolve=True)
    discriminator_rows, term_rows = discriminator_baseline_rows(cfg, real_ds, variables, output_root)
    schedule = active_schedule(cfg)
    resample_id = schedule.resample_id if schedule is not None else "single"
    for row in discriminator_rows:
        row["resample_id"] = resample_id
    for row in term_rows:
        row["resample_id"] = resample_id
    write_discriminator_baselines(discriminator_rows, cfg, output_root)
    terms_path = output_root / "data" / "discriminator_terms.csv.gz"
    if term_rows:
        write_csv_gz(terms_path, term_rows)
    split_manifest_path = write_split_membership(cfg, real_ds, output_root)
    real_ds.close()
    data_path = output_root / "data" / "discriminator_reverse_kl.csv"
    return [output_root / "resolved_config.yaml"] + ([data_path] if data_path.is_file() else []) + ([terms_path] if terms_path.is_file() else []) + ([split_manifest_path] if split_manifest_path is not None else [])


def evaluate_standard_metric_baselines(cfg):
    """Compatibility wrapper evaluating both standard and learned baselines."""
    return evaluate_standard_metrics(cfg) + evaluate_discriminator_metrics(cfg)


def plot_main_corruption_comparison(
    corruption_rows, discriminator_rows, metric_names, variables, cfg, output_root,
):
    """Plot five fixed-metric corruption panels beside their learned-critic curves."""
    settings = (cfg.get("plotting", {}) or {}).get("main_corruption_figure", {}) or {}
    if not bool(settings.get("enabled", True)):
        return
    metric_names = plotted_metric_names(metric_names)
    if not metric_names or not corruption_rows or not discriminator_rows:
        return

    variable = joint_variable_name(variables)
    available = {
        row["corruption"] for row in corruption_rows if row.get("variable") == variable
    }
    requested = [
        str(value) for value in settings.get(
            "corruptions",
            [
                "gaussian_blur", "grf", "checkerboard_2px",
                "zonal_scanlines", "hemisphere_splice",
            ],
        )
    ]
    architecture = str(settings.get("architecture", "squeezenet"))
    all_critic_rows = [
        row for row in discriminator_rows
        if row.get("architecture") == architecture
        and row.get("kind") == "corruption"
    ]
    critic_targets = {row["target"] for row in all_critic_rows}
    fallback_order = [
        str(value) for value in settings.get(
            "fallback_corruptions",
            ["hf_noise", "field_splice", "wind_patch_shuffle", "pixel_replace"],
        )
    ] + [
        str(value) for value in baseline_get(cfg, "corruptions", [])
    ] + sorted(available)
    candidate_order = list(dict.fromkeys([*requested, *fallback_order]))
    corruptions = [
        name for name in candidate_order
        if name in available and name in critic_targets
    ][:5]
    if len(corruptions) != 5:
        print(
            "Skipping main corruption comparison: expected five corruptions with "
            f"both fixed-metric and valid {architecture} critic rows, found {corruptions}."
        )
        return
    substitutions = [
        name for name in requested[:5] if name not in corruptions
    ]
    if substitutions:
        print(
            "Main corruption comparison substituted targets without valid critic "
            f"rows: requested={requested[:5]}, selected={corruptions}."
        )
    critic_rows = [
        row for row in all_critic_rows if row.get("target") in corruptions
    ]

    selected_rows = [
        row for row in corruption_rows
        if row.get("variable") == variable and row.get("corruption") in corruptions
    ]
    scales = metric_normalization_scales(selected_rows, metric_names)
    critic_values = np.asarray([
        float(row["score"]) for row in critic_rows if np.isfinite(float(row["score"]))
    ])
    critic_positive = critic_values[critic_values > 0.0]
    critic_scale = float(
        critic_positive.max() if critic_positive.size else np.abs(critic_values).max()
    ) if critic_values.size else 1.0
    critic_scale = max(critic_scale, 1e-12)
    colors = metric_colors(metric_names)
    critic_color, critic_marker = CRITIC_COLOR, "X"
    figure, axes = plt.subplots(
        2, 3, figsize=(12.0, 7.8), squeeze=False, layout="constrained",
        sharey="row",
    )
    axes = axes.ravel()
    null_positions = null_diamond_positions(len(metric_names) + 1, 0.035)

    for panel_index, (axis, corruption) in enumerate(zip(axes[:5], corruptions)):
        series = sorted(
            [
                row for row in selected_rows
                if row["corruption"] == corruption and not row_is_null(row)
            ],
            key=lambda row: row["severity"],
        )
        x_values = relative_corruption_coordinates(series, "severity")
        null_row = next(
            (
                row for row in selected_rows
                if row["corruption"] == corruption and row_is_null(row)
            ),
            None,
        )
        for metric_index, metric_name in enumerate(metric_names):
            axis.plot(
                x_values,
                [normalized_metric_value(row, metric_name, scales) for row in series],
                marker=series_marker(metric_index),
                linewidth=1.6,
                color=colors[metric_name],
                label=displayed_metric_name(metric_name),
            )
            if series and f"{metric_name}_lower" in series[0]:
                bounds = [
                    normalized_metric_bounds(row, metric_name, scales) for row in series
                ]
                axis.fill_between(
                    x_values,
                    [bound[0] for bound in bounds],
                    [bound[1] for bound in bounds],
                    facecolor=colors[metric_name],
                    edgecolor=colors[metric_name],
                    alpha=0.22,
                    linewidth=0.55,
                    zorder=1,
                )
            if null_row is not None:
                null_value = normalized_metric_value(null_row, metric_name, scales)
                null_lower, null_upper = normalized_metric_bounds(
                    null_row, metric_name, scales,
                )
                plot_null_diamond(
                    axis, null_positions[metric_index], null_value,
                    null_lower, null_upper, colors[metric_name],
                )
        target_rows = [row for row in critic_rows if row["target"] == corruption]
        critic_series = sorted(
            [row for row in target_rows if not row["is_era5_test_null"]],
            key=lambda row: row["x"],
        )
        critic_x = relative_corruption_coordinates(critic_series, "x")
        axis.plot(
            critic_x,
            [row["score"] / critic_scale for row in critic_series],
            marker=critic_marker, linestyle="--", linewidth=1.6,
            color=critic_color, label="Learned critic",
        )
        if critic_series and "score_lower" in critic_series[0]:
            axis.fill_between(
                critic_x,
                [row["score_lower"] / critic_scale for row in critic_series],
                [row["score_upper"] / critic_scale for row in critic_series],
                color=critic_color, alpha=0.12, linewidth=0,
            )
        critic_null = next(
            (row for row in target_rows if row["is_era5_test_null"]), None
        )
        if critic_null is not None:
            critic_value = critic_null["score"] / critic_scale
            critic_lower = critic_null.get("score_lower", critic_null["score"]) / critic_scale
            critic_upper = critic_null.get("score_upper", critic_null["score"]) / critic_scale
            plot_null_diamond(
                axis, null_positions[-1], critic_value, critic_lower, critic_upper,
                critic_color,
            )
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
        axis.set(
            title=CORRUPTION_DISPLAY_NAMES.get(
                corruption, corruption.replace("_", " ").title(),
            ),
            xlim=(-0.04, 1.04),
        )
        axis.set_xticks(np.linspace(0.0, 1.0, 5))
        # Keep the semantic axis at [0, 1] while reserving enough drawing room
        # for the diamond markers at the endpoints.
        # Outer labels keep the compact 2x3 paper layout readable. Repeating the
        # x label on the top row collides with the lower panel titles after the
        # figure is resized to its final manuscript width.
        if panel_index >= 3:
            axis.set_xlabel("Relative corruption severity")
        if panel_index % 3 == 0:
            axis.set_ylabel("Normalized divergence")
        axis.grid(True, alpha=0.3)

    legend_axis = axes[5]
    legend_axis.set_axis_off()
    metric_handles, metric_labels = axes[0].get_legend_handles_labels()
    legend_axis.legend(
        metric_handles,
        metric_labels,
        loc="center",
        frameon=False,
        fontsize=8,
        title="Metrics",
        title_fontsize=9,
    )

    figure.suptitle(
        "Normalized fixed-metric and learned-critic responses to corruption\n"
        "Bands: fixed metrics 5--95% over 10 splits; learned critic min--max over 5 folds",
        fontsize=14,
    )
    output_path = output_root / "plots" / "corruption_metrics_and_critic.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_bundle(
        figure,
        output_path,
        plot_type="main_corruption_comparison",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)
    print(f"Saved main corruption-comparison figure to: {output_path}")



def plot_saved_standard_metric_baselines(cfg):
    """Render baseline figures exclusively from persisted evaluation artifacts."""
    configure_plot_bundle_saving_from_cfg(cfg)
    variables = variables_from_config(cfg)
    metric_names = metric_names_from_config(cfg)
    output_root = baseline_output_dir(cfg, variables)
    lead_rows = read_metric_csv(
        output_root / "data" / "lead_time.csv", metric_names, "lead_time"
    )
    corruption_rows = read_metric_csv(
        output_root / "data" / "corruption_strength.csv", metric_names, "corruption_strength"
    )
    scwd_diagnostics = read_scwd_anchor_diagnostics(output_root) if "scwd" in metric_names else []
    global_mean_diagnostics = (read_global_mean_wasserstein_diagnostics(output_root)
                               if "global_mean_wasserstein" in metric_names else [])
    discriminator_rows = read_discriminator_baselines(output_root)
    discriminator_terms = read_discriminator_terms(output_root)

    plot_lead_metrics(lead_rows, metric_names, variables, output_root)
    plot_corruption_metrics(corruption_rows, metric_names, variables, output_root)
    plot_normalized_lead_metrics_by_model(
        lead_rows, metric_names, variables, output_root, discriminator_rows,
    )
    plot_normalized_corruption_metrics_by_type(corruption_rows, metric_names, variables, output_root)
    plot_main_corruption_comparison(
        corruption_rows, discriminator_rows, metric_names, variables, cfg, output_root
    )
    repeated_plot_mode = str(
        (cfg.get("plotting", {}) or {}).get("repeated_plot_mode", "all")
    )
    if repeated_plot_mode not in {"all", "representative_only"}:
        raise ValueError(
            "plotting.repeated_plot_mode must be 'all' or 'representative_only'."
        )
    representative_only = repeated_plot_mode == "representative_only"
    plot_corruption_disturbances(output_root, cfg)
    plot_scwd_anchor_diagnostics(
        scwd_diagnostics, output_root, representative_only=representative_only,
    )
    if bool((cfg.get("plotting", {}) or {}).get("scwd_response_histograms", False)):
        plot_scwd_top_wasserstein_distributions(
            scwd_diagnostics, cfg, output_root,
            representative_only=representative_only,
        )
    plot_scwd_mean_response_differences(
        scwd_diagnostics, output_root, representative_only=representative_only,
    )
    plot_global_mean_wasserstein_distributions(
        global_mean_diagnostics, output_root,
        representative_only=representative_only,
    )
    plot_discriminator_baselines(discriminator_rows, cfg, output_root)
    plot_forecast_logit_histogram_gallery(discriminator_terms, cfg, output_root)
    plot_forecast_critic_and_graphcast_histograms(
        discriminator_rows, discriminator_terms, cfg, output_root,
    )
    mmd_root = mmd_global_moment_matching_output_dir(cfg, variables)
    plot_mmd_global_moment_matching(read_mmd_global_moment_matching(mmd_root), mmd_root)


@hydra.main(version_base=None, config_path="../conf", config_name="baseline_config")
def main(cfg: DictConfig):
    """Plot previously evaluated temporal-holdout-aligned baselines."""
    plot_saved_standard_metric_baselines(cfg)


if __name__ == "__main__":
    main()

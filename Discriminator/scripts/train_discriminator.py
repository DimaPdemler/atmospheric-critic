"""Train a ResNet18 or SqueezeNet binary discriminator on ERA5 vs. forecasts."""

import os

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
import xarray as xr

try:
    from .corruptions import (
        apply_gaussian_blur,
        apply_gaussian_field_noise,
        apply_high_freq_noise,
        apply_random_pixel_replace,
        apply_wind_channel_rotation,
        apply_wind_patch_shuffle,
    )
except ImportError:
    from corruptions import (
        apply_gaussian_blur,
        apply_gaussian_field_noise,
        apply_high_freq_noise,
        apply_random_pixel_replace,
        apply_wind_channel_rotation,
        apply_wind_patch_shuffle,
    )

try:
    from .monthly_split import forecast_pairs
except ImportError:
    from monthly_split import forecast_pairs

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Optional W&B credentials live one directory above this experiment folder on
# the cluster.  CSV/no-logger runs should not require python-dotenv.
if load_dotenv is not None:
    load_dotenv("../wandb_info.env")


def variables_from_config(cfg):
    """Return all configured input fields, falling back to one selected field."""
    variables = cfg.get("variables")
    if variables:
        return list(variables)
    return [cfg.selected_variable]


def _as_time_ranges(value):
    """Normalize `[start, end]` or `[[start, end], ...]` into a list of ranges."""
    if not value:
        return []
    value = list(value)
    if len(value) == 2 and all(item is None or isinstance(item, str) for item in value):
        return [value]
    return [list(item) for item in value]


def _time_ranges_overlap(left, right):
    """Inclusive interval-overlap test for optional start/end timestamp strings."""
    left_start, left_end = left
    right_start, right_end = right
    if left_end is not None and right_start is not None and pd.Timestamp(left_end) < pd.Timestamp(right_start):
        return False
    if right_end is not None and left_start is not None and pd.Timestamp(right_end) < pd.Timestamp(left_start):
        return False
    return True


def validate_no_train_test_overlap(cfg):
    """Fail early if train/test time splits leak into each other."""
    if str((cfg.get("monthly_split") or {}).get("strategy", "")) == "monthly_valid_time":
        return
    if cfg.get("skip_train_test_overlap_check", False):
        return

    checks = [
        ("real", cfg.get("train_real_range"), cfg.get("test_real_ranges")),
        ("fake", cfg.get("train_fake_range"), cfg.get("test_fake_range")),
    ]
    for name, train_ranges, test_ranges in checks:
        # K-fold reuses forecast periods as fake train/test data, but ERA5
        # train/test overlap should still fail so real test samples stay masked.
        if name == "fake" and cfg.get("allow_fake_train_test_overlap", False):
            continue
        for train_range in _as_time_ranges(train_ranges):
            for test_range in _as_time_ranges(test_ranges):
                if _time_ranges_overlap(train_range, test_range):
                    raise ValueError(
                        f"Overlapping {name} train/test ranges: "
                        f"train={train_range}, test={test_range}"
                    )


def select_time_ranges(ds, ranges):
    """Select and concatenate one or more time intervals from an xarray dataset."""
    ranges = _as_time_ranges(ranges)
    if not ranges:
        return ds

    selected = []
    for start, end in ranges:
        curr = ds.sel(time=slice(start, end))
        if curr.sizes.get("time", 0) > 0:
            selected.append(curr)

    if not selected:
        return ds.isel(time=slice(0, 0))
    if len(selected) == 1:
        return selected[0]
    return xr.concat(selected, dim="time")


def normalize_prediction_timedelta(ds):
    """Convert timedelta lead-time coordinates to integer hours when present."""
    if "prediction_timedelta" not in ds.coords:
        return ds

    lead_times = ds.prediction_timedelta.values
    if np.issubdtype(lead_times.dtype, np.timedelta64):
        return ds.assign_coords(
            prediction_timedelta=lead_times.astype("timedelta64[h]").astype(int)
        )
    return ds


def cftime_decode_kwargs():
    """Return xarray kwargs for cftime decoding across old/new xarray versions."""
    if hasattr(xr, "coders") and hasattr(xr.coders, "CFDatetimeCoder"):
        return {"decode_times": xr.coders.CFDatetimeCoder(use_cftime=True)}
    return {"use_cftime": True}


def safe_open_dataset(path):
    """Open NetCDF-like data with explicit fallbacks for cluster file variants."""
    print(f"Attempting to open dataset: {path}")
    if not os.path.exists(path):
        print(f"Error: File does not exist at {path}")
        raise FileNotFoundError(f"No such file: {path}")
    
    engines = ["netcdf4", "h5netcdf", "scipy"]
    decode_kwargs = cftime_decode_kwargs()
    for engine in engines:
        try:
            ds = xr.open_dataset(path, engine=engine, **decode_kwargs)
            # Force lazy time decoding now so failures happen before training.
            if "time" in ds.coords:
                _ = ds.time.values[0]
            print(f"Successfully opened {path} with engine {engine}")
            return ds
        except Exception:
            continue
            
    try:
        print(f"Attempting standard open for {path} as last resort...")
        ds = xr.open_dataset(path, **decode_kwargs)
        return ds
    except Exception as e:
        print(f"Manual fallback for {path}...")
        ds = xr.open_dataset(path, decode_times=False)
        if "time" in ds.coords and "units" in ds.time.attrs:
            units = ds.time.attrs["units"]
            try:
                base_time_parts = units.split("since ")
                if len(base_time_parts) > 1:
                    base_time = base_time_parts[1].split(" ")[0]
                    unit_type = units.split(" ")[0].lower()
                    pd_unit = "D"
                    if "hour" in unit_type:
                        pd_unit = "h"
                    elif "minute" in unit_type:
                        pd_unit = "m"
                    elif "second" in unit_type:
                        pd_unit = "s"
                    new_times = pd.to_datetime(base_time) + pd.to_timedelta(ds.time.values, unit=pd_unit)
                    ds = ds.assign_coords(time=new_times)
                    print(f"Successfully decoded time manually for {path}")
            except Exception as e2:
                print(f"Manual decoding also failed: {e2}")
        return ds

CORRUPTION_FNS = {
    "blur": apply_gaussian_blur,
    "gaussian_blur": apply_gaussian_blur,
    "grf": apply_gaussian_field_noise,
    "gaussian_field_noise": apply_gaussian_field_noise,
    "hf_noise": apply_high_freq_noise,
    "high_freq_noise": apply_high_freq_noise,
    "pixel_replace": apply_random_pixel_replace,
    "random_pixel_replace": apply_random_pixel_replace,
    "wind_rotated": apply_wind_channel_rotation,
    "wind_rotation": apply_wind_channel_rotation,
    "wind_shuffled": apply_wind_patch_shuffle,
    "wind_patch_shuffle": apply_wind_patch_shuffle,
}

FIELDWISE_CORRUPTION_FNS = {
    "blur": apply_gaussian_blur,
    "gaussian_blur": apply_gaussian_blur,
    "grf": apply_gaussian_field_noise,
    "gaussian_field_noise": apply_gaussian_field_noise,
    "hf_noise": apply_high_freq_noise,
    "high_freq_noise": apply_high_freq_noise,
    "pixel_replace": apply_random_pixel_replace,
    "random_pixel_replace": apply_random_pixel_replace,
}

def apply_configured_corruption(sample, corruption_type, severity):
    """Apply a corruption to a single CxHxW sample."""
    if severity <= 0:
        return sample
    try:
        fn = CORRUPTION_FNS[corruption_type]
    except KeyError as exc:
        valid = ", ".join(sorted(CORRUPTION_FNS))
        raise ValueError(f"Unknown corruption_type '{corruption_type}'. Valid options: {valid}") from exc
    return fn(sample.unsqueeze(0), severity).squeeze(0)

def apply_fieldwise_corruption(sample, corruption_type, channel_idx, severity):
    """Apply a spatial corruption to one channel while leaving other fields intact."""
    if severity <= 0:
        return sample
    try:
        fn = FIELDWISE_CORRUPTION_FNS[corruption_type]
    except KeyError as exc:
        valid = ", ".join(sorted(FIELDWISE_CORRUPTION_FNS))
        raise ValueError(
            f"Corruption '{corruption_type}' cannot be applied to an individual field. "
            f"Valid fieldwise corruptions: {valid}"
        ) from exc
    corrupted = sample.clone()
    channel = corrupted[channel_idx:channel_idx + 1].unsqueeze(0)
    corrupted[channel_idx:channel_idx + 1] = fn(channel, severity).squeeze(0)
    return corrupted

def random_field_subset(num_fields, field_prob):
    """Sample at least one field index for fieldwise corruption."""
    mask = np.random.random(num_fields) < field_prob
    if not mask.any():
        mask[np.random.randint(num_fields)] = True
    return np.flatnonzero(mask)

def sample_power_law_severity(max_severity, power, min_severity=0.0):
    """Sample from ``min + (max - min) * U**power`` for fake corruptions."""
    max_severity, min_severity = float(max_severity), float(min_severity)
    if min_severity < 0.0 or min_severity > max_severity:
        raise ValueError(
            f"Expected 0 <= min_severity <= max_severity, got "
            f"{min_severity} and {max_severity}."
        )
    if max_severity <= 0:
        return 0.0
    if power <= 0:
        raise ValueError(f"corruption_severity_power must be > 0, got {power}")
    return min_severity + (max_severity - min_severity) * (
        np.random.random() ** float(power)
    )


def build_logger(cfg):
    """Create the configured Lightning logger.

    The default `csv` mode writes local metrics under `output_dir/lightning_logs`
    and needs no W&B account, network, or credentials.  Use `logger=wandb` for
    explicit Weights & Biases logging, or `logger=none` for no experiment logger.
    """
    logger_kind = str(cfg.get("logger", "csv")).lower()
    if logger_kind == "csv":
        return CSVLogger(save_dir=cfg.output_dir, name="lightning_logs")
    if logger_kind == "wandb":
        return WandbLogger(
            project=cfg.project_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    if logger_kind in ("none", "false", "disabled"):
        return False
    raise ValueError("logger must be one of: csv, wandb, none")


class WeatherDiscriminatorDataset(Dataset):
    """Binary dataset for ERA5-vs-forecast discriminator training/evaluation.

    Labels are deliberately simple:
    - `1.0`: ERA5/reference fields.
    - `0.0`: model forecasts, corrupted ERA5, or corrupted forecasts.

    In balanced training mode the first half of an epoch is real samples and
    the second half is fake samples.  Fake samples are drawn from forecasts and,
    when augmentation is enabled, synthetic corruption categories.
    """

    def __init__(self, real_nc_path, fake_nc_path, variables, 
                 real_range=None, fake_range=None, lead_times=None, level=None, 
                 means=None, stds=None, balanced=True,
                 augment=False, augment_prob=0.5, disturb_type=None, disturb_level=0.0,
                 corruption_types=None, corruption_severity_max=1.0,
                 corruption_severity_power=2.0, corruption_severity_min=0.0,
                 corruption_severity_min_overrides=None, field_corruption_prob=0.5,
                 max_samples=0, monthly_split_cfg=None, monthly_split_name="train"):
        self.variables = variables
        self.balanced = balanced
        self.augment = augment
        self.augment_prob = augment_prob
        self.disturb_type = disturb_type
        self.disturb_level = disturb_level
        self.corruption_types = list(corruption_types or ["blur", "grf", "pixel_replace"])
        self.corruption_severity_max = corruption_severity_max
        self.corruption_severity_power = corruption_severity_power
        self.corruption_severity_min = float(corruption_severity_min)
        self.corruption_severity_min_overrides = dict(corruption_severity_min_overrides or {})
        self.field_corruption_prob = field_corruption_prob
        self.monthly_split_cfg = monthly_split_cfg
        max_samples = int(max_samples or 0)
        monthly = monthly_split_cfg is not None

        self.real_ds = normalize_prediction_timedelta(safe_open_dataset(real_nc_path))
        self.real_ds = self._prepare_dataset(
            self.real_ds, level=level, time_range=None if monthly else real_range,
            max_samples=0 if monthly else max_samples,
        )
        
        from omegaconf import ListConfig
        if isinstance(fake_nc_path, (list, ListConfig)): 
            self.fake_sources = []
            for p in fake_nc_path:
                ds = normalize_prediction_timedelta(safe_open_dataset(p))
                ds = self._prepare_dataset(
                    ds, level=level, time_range=None if monthly else fake_range,
                    max_samples=0 if monthly else max_samples,
                )
                if ds.sizes.get("time", 0) > 0:
                    self.fake_sources.append(ds)
            if not self.fake_sources:
                raise ValueError(f"No fake samples found in configured training range: {fake_range}")
            self.fake_range_applied = True
        else:
            self.fake_ds = normalize_prediction_timedelta(safe_open_dataset(fake_nc_path))
            self.fake_ds = self._prepare_dataset(
                self.fake_ds, level=level, time_range=None if monthly else fake_range,
                max_samples=0 if monthly else max_samples,
            )
            self.fake_sources = [self.fake_ds]
            self.fake_range_applied = False

        self.real_times = self.real_ds.time.values
        if len(self.real_times) == 0:
            raise ValueError("No real samples remain after applying train_real_range.")

        missing_real = [v for v in self.variables if v not in self.real_ds.data_vars]
        missing_fake_by_source = [
            [v for v in self.variables if v not in ds.data_vars]
            for ds in self.fake_sources
        ]
        missing_fake = sorted({v for missing in missing_fake_by_source for v in missing})
        if missing_real or missing_fake:
            raise ValueError(
                "Configured variables are missing from the loaded datasets. "
                f"Missing from real: {missing_real}; missing from fake: {missing_fake}"
            )
        
        # Forecast datasets contain multiple lead times.  ERA5/reference data is
        # evaluated at lead zero when that coordinate exists.
        if monthly:
            self.fake_sample_index = []
            for source_index, source in enumerate(self.fake_sources):
                for pair in forecast_pairs(
                    source, self.real_ds, monthly_split_cfg, monthly_split_name, lead_times
                ):
                    self.fake_sample_index.append((
                        source_index, pair.forecast_index, pair.lead_index, pair.era5_index
                    ))
            if max_samples > 0 and len(self.fake_sample_index) > max_samples:
                positions = np.linspace(0, len(self.fake_sample_index) - 1, max_samples, dtype=int)
                self.fake_sample_index = [self.fake_sample_index[int(position)] for position in positions]
        else:
            self.fake_sample_index = self._build_fake_sample_index(self.fake_sources, lead_times)
        if not self.fake_sample_index:
            raise ValueError(f"No fake samples found for configured lead_times={lead_times}")

        if "prediction_timedelta" in self.real_ds.dims:
            lt_h = self.real_ds.prediction_timedelta.values
            zero_idx = np.where(lt_h == 0)[0]
            self.real_lead_idx = int(zero_idx[0]) if len(zero_idx) > 0 else 0
        else:
            self.real_lead_idx = None

        # Normalize all fields using ERA5/reference statistics from the selected
        # training split.  This makes the discriminator output comparable across
        # real and fake samples without using forecast statistics.
        if means is None or stds is None:
            print(f"Calculating stats from REAL dataset...")
            stats_ds = (
                self.real_ds.isel(time=[record[3] for record in self.fake_sample_index])
                if monthly else self.real_ds
            )
            self.means = {}
            self.stds = {}
            for v in self.variables:
                if v in self.real_ds.data_vars:
                    self.means[v] = float(stats_ds[v].mean())
                    self.stds[v] = float(stats_ds[v].std())
                else:
                    self.means[v] = 0.0
                    self.stds[v] = 1.0
        else:
            self.means, self.stds = means, stds

        # Balanced mode defines a stable 50/50 real/fake epoch even when the
        # number of ERA5 times and forecast-times-by-leads differs.
        self.total_fake_samples = len(self.fake_sample_index)
        self.total_real_samples = len(self.fake_sample_index) if monthly else len(self.real_times)
        self.fake_categories = (
            ("forecast", "corrupted_real", "corrupted_forecast")
            if self.augment else
            ("forecast",)
        )

        
        if self.balanced:
            self.samples_per_class = max(self.total_fake_samples, self.total_real_samples)
        
        print(f"Dataset Initialized (Balanced={self.balanced}):")
        print(f"  REAL: {self.total_real_samples} | FAKE: {self.total_fake_samples}")
        if self.balanced:
            print(f"  Per epoch final labels: REAL={self.samples_per_class} | FAKE={self.samples_per_class}")
            fake_category_prob = 1.0 / len(self.fake_categories)
            fake_mix = " | ".join(
                f"{category}={fake_category_prob:.2f}"
                for category in self.fake_categories
            )
            print(f"  Fake multi-sampling: {fake_mix}")

    @classmethod
    def _prepare_dataset(cls, ds, level=None, time_range=None, max_samples=0):
        """Apply cheap selections before expensive multi-file concat or stats."""
        if level is not None:
            if "level" in ds.dims:
                ds = ds.sel(level=level)
            elif "pressure_level" in ds.dims:
                ds = ds.sel(pressure_level=level)
        if time_range:
            ds = select_time_ranges(ds, time_range)
        if max_samples > 0:
            ds = cls._subsample_time(ds, max_samples)
        return ds

    @staticmethod
    def _lead_indices_for_dataset(ds, lead_times):
        """Return lead indices available in one forecast dataset."""
        if "prediction_timedelta" not in ds.dims:
            return [None]
        available = ds.prediction_timedelta.values
        if lead_times is None:
            return np.where(available > 0)[0].tolist()
        missing = [lead for lead in lead_times if lead not in available]
        if missing:
            print(f"Warning: fake dataset is missing lead_times {missing}; using available leads only.")
        return [int(np.where(available == lead)[0][0]) for lead in lead_times if lead in available]

    @classmethod
    def _build_fake_sample_index(cls, fake_sources, lead_times):
        """Flatten valid fake `(source, time, lead)` combinations."""
        sample_index = []
        for source_idx, ds in enumerate(fake_sources):
            lead_indices = cls._lead_indices_for_dataset(ds, lead_times)
            for time_idx in range(ds.sizes.get("time", 0)):
                for lead_idx in lead_indices:
                    sample_index.append((source_idx, time_idx, lead_idx))
        return sample_index

    @staticmethod
    def _subsample_time(ds, max_samples):
        """Select an evenly spaced time subset for smoke tests and small jobs."""
        n_time = ds.sizes.get("time", 0)
        if n_time <= max_samples:
            return ds
        indices = np.linspace(0, n_time - 1, max_samples, dtype=int)
        return ds.isel(time=indices)

    def __len__(self):
        if self.balanced:
            return self.samples_per_class * 2
        else:
            return self.total_real_samples + self.total_fake_samples

    def _real_indices(self, idx):
        if self.monthly_split_cfg is not None:
            return self.fake_sample_index[idx % self.total_real_samples][3], self.real_lead_idx
        return idx % self.total_real_samples, self.real_lead_idx

    def _fake_indices(self, idx):
        return self.fake_sample_index[idx % self.total_fake_samples][:3]

    def _sample_tensor(self, ds, t_idx, l_idx):
        ds_slice = ds.isel(time=t_idx)
        if l_idx is not None:
            ds_slice = ds_slice.isel(prediction_timedelta=l_idx)

        channels = []
        for v in self.variables:
            if v in ds_slice.data_vars:
                raw_val = ds_slice[v].values.astype(np.float32)
                norm = (raw_val - self.means[v]) / (self.stds[v] if self.stds[v] > 1e-8 else 1.0)
                channels.append(np.nan_to_num(norm, nan=0.0))
            else:
                ref_shape = ds_slice.temperature.shape if 'temperature' in ds_slice.data_vars else list(ds_slice.data_vars.values())[0].shape
                channels.append(np.zeros(ref_shape, dtype=np.float32))
        return torch.tensor(np.stack(channels), dtype=torch.float32)

    def _apply_random_field_corruptions(self, sample):
        fieldwise_types = [t for t in self.corruption_types if t in FIELDWISE_CORRUPTION_FNS]
        if not fieldwise_types:
            return sample
        field_corruption_type = str(np.random.choice(fieldwise_types))
        return self._apply_fieldwise_corruption_type(sample, field_corruption_type)

    def _apply_fieldwise_corruption_type(self, sample, field_corruption_type):
        minimum = float(self.corruption_severity_min_overrides.get(
            field_corruption_type, self.corruption_severity_min
        ))
        for channel_idx in random_field_subset(len(self.variables), self.field_corruption_prob):
            field_severity = sample_power_law_severity(
                self.corruption_severity_max,
                self.corruption_severity_power,
                minimum,
            )
            sample = apply_fieldwise_corruption(
                sample,
                field_corruption_type,
                channel_idx,
                field_severity,
            )
        return sample

    def _apply_random_corruption(self, sample):
        valid_types = [t for t in self.corruption_types if t in CORRUPTION_FNS]
        if not valid_types:
            valid = ", ".join(sorted(CORRUPTION_FNS))
            raise ValueError(f"No valid corruption_types configured. Valid options: {valid}")
        corruption_type = str(np.random.choice(valid_types))
        if corruption_type in FIELDWISE_CORRUPTION_FNS:
            return self._apply_fieldwise_corruption_type(sample, corruption_type)
        else:
            minimum = float(self.corruption_severity_min_overrides.get(
                corruption_type, self.corruption_severity_min
            ))
            severity = sample_power_law_severity(
                self.corruption_severity_max,
                self.corruption_severity_power,
                minimum,
            )
            return apply_configured_corruption(sample, corruption_type, severity)

    def _apply_fake_corruption(self, sample):
        if self.disturb_type is not None and self.disturb_level > 0:
            return apply_configured_corruption(sample, self.disturb_type, self.disturb_level)
        return self._apply_random_corruption(sample)

    def __getitem__(self, idx):
        if self.balanced:
            sub_idx = idx % self.samples_per_class
            if idx < self.samples_per_class:
                t_idx, l_idx = self._real_indices(sub_idx)
                sample = self._sample_tensor(self.real_ds, t_idx, l_idx)
                return sample, torch.tensor([1.0], dtype=torch.float32)

            fake_category = str(np.random.choice(self.fake_categories))

            if fake_category == "corrupted_real":
                t_idx, l_idx = self._real_indices(sub_idx)
                sample = self._sample_tensor(self.real_ds, t_idx, l_idx)
                sample = self._apply_fake_corruption(sample)
                return sample, torch.tensor([0.0], dtype=torch.float32)

            source_idx, t_idx, l_idx = self._fake_indices(sub_idx)
            sample = self._sample_tensor(self.fake_sources[source_idx], t_idx, l_idx)
            if fake_category == "corrupted_forecast":
                sample = self._apply_fake_corruption(sample)
            return sample, torch.tensor([0.0], dtype=torch.float32)
        else:
            is_fake = idx >= self.total_real_samples
            if not is_fake:
                t_idx, l_idx = self._real_indices(idx)
                sample = self._sample_tensor(self.real_ds, t_idx, l_idx)
                label = torch.tensor([1.0], dtype=torch.float32)
            else:
                source_idx, t_idx, l_idx = self._fake_indices(idx - self.total_real_samples)
                sample = self._sample_tensor(self.fake_sources[source_idx], t_idx, l_idx)
                label = torch.tensor([0.0], dtype=torch.float32)

        if self.disturb_type is not None and self.disturb_level > 0:
            sample = apply_configured_corruption(sample, self.disturb_type, self.disturb_level)
            label = torch.tensor([0.0], dtype=torch.float32)
        elif self.augment and np.random.random() < self.augment_prob:
            sample = self._apply_random_corruption(sample)
            label = torch.tensor([0.0], dtype=torch.float32)

        return sample, label

class SqueezeNetGlobalAttention(nn.Module):
    """SqueezeNet features followed by lightweight global spatial mixing."""

    def __init__(
        self,
        num_weather_channels,
        pretrained_backbone=True,
        attention_channels=64,
        attention_heads=4,
        attention_ffn_channels=128,
        attention_dropout=0.1,
    ):
        super().__init__()
        if attention_channels % 4 != 0:
            raise ValueError("attention_channels must be divisible by 4 for spherical positions.")
        weights = models.SqueezeNet1_1_Weights.DEFAULT if pretrained_backbone else None
        backbone = models.squeezenet1_1(weights=weights)
        old_conv = backbone.features[0]
        backbone.features[0] = nn.Conv2d(
            num_weather_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
        )
        with torch.no_grad():
            repeat_factor = (num_weather_channels // 3) + 1
            new_weights = old_conv.weight.repeat(1, repeat_factor, 1, 1)[
                :, :num_weather_channels, :, :
            ]
            backbone.features[0].weight = nn.Parameter(
                new_weights * (3.0 / num_weather_channels)
            )
        self.features = backbone.features
        self.projection = nn.Conv2d(512, attention_channels, kernel_size=1)
        layer = nn.TransformerEncoderLayer(
            d_model=attention_channels,
            nhead=attention_heads,
            dim_feedforward=attention_ffn_channels,
            dropout=attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_mixer = layer
        self.global_token = nn.Parameter(torch.zeros(1, 1, attention_channels))
        nn.init.normal_(self.global_token, std=0.02)
        self.output_norm = nn.LayerNorm(attention_channels)
        self.classifier = nn.Linear(attention_channels, 1)
        self.attention_channels = int(attention_channels)

    @staticmethod
    def spherical_position_encoding(height, width, channels, device, dtype):
        """Fixed latitude and exactly periodic longitude Fourier positions."""
        quarter = channels // 4
        frequencies = torch.arange(1, quarter + 1, device=device, dtype=dtype)
        latitudes = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        latitude_angles = latitudes[:, None] * frequencies[None, :] * (torch.pi / 2.0)
        latitude_encoding = torch.cat(
            [latitude_angles.sin(), latitude_angles.cos()], dim=-1
        )
        longitudes = torch.arange(width, device=device, dtype=dtype) * (2.0 * torch.pi / width)
        longitude_angles = longitudes[:, None] * frequencies[None, :]
        longitude_encoding = torch.cat(
            [longitude_angles.sin(), longitude_angles.cos()], dim=-1
        )
        latitude_encoding = latitude_encoding[:, None, :].expand(height, width, -1)
        longitude_encoding = longitude_encoding[None, :, :].expand(height, width, -1)
        return torch.cat([latitude_encoding, longitude_encoding], dim=-1).reshape(
            1, height * width, channels
        )

    def forward(self, x):
        features = self.projection(self.features(x))
        _, _, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self.spherical_position_encoding(
            height, width, self.attention_channels, tokens.device, tokens.dtype
        )
        global_token = self.global_token.expand(tokens.shape[0], -1, -1)
        mixed = self.global_mixer(torch.cat([global_token, tokens], dim=1))
        return self.classifier(self.output_norm(mixed[:, 0]))


class WeatherDiscriminator(L.LightningModule):
    """Torchvision backbone adapted for weather-field binary classification."""

    def __init__(
        self,
        num_weather_channels,
        model_name="resnet18",
        learning_rate=1e-4,
        pretrained_backbone=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model_name = model_name
        
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
            self.model = models.resnet18(weights=weights)
            old_conv = self.model.conv1
            self.model.conv1 = nn.Conv2d(
                num_weather_channels, old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, stride=old_conv.stride, 
                padding=old_conv.padding, bias=old_conv.bias is not None
            )
            with torch.no_grad():
                repeat_factor = (num_weather_channels // 3) + 1
                new_weights = old_conv.weight.repeat(1, repeat_factor, 1, 1)[:, :num_weather_channels, :, :]
                self.model.conv1.weight = nn.Parameter(new_weights * (3.0 / num_weather_channels))
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)
            
        elif model_name == "squeezenet":
            weights = models.SqueezeNet1_1_Weights.DEFAULT if pretrained_backbone else None
            self.model = models.squeezenet1_1(weights=weights)
            old_conv = self.model.features[0]
            self.model.features[0] = nn.Conv2d(
                num_weather_channels, old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, stride=old_conv.stride
            )
            with torch.no_grad():
                repeat_factor = (num_weather_channels // 3) + 1
                new_weights = old_conv.weight.repeat(1, repeat_factor, 1, 1)[:, :num_weather_channels, :, :]
                self.model.features[0].weight = nn.Parameter(new_weights * (3.0 / num_weather_channels))
            
            old_classifier = self.model.classifier[1]
            self.model.classifier[1] = nn.Conv2d(old_classifier.in_channels, 1, kernel_size=(1, 1))
            self.model.classifier[2] = nn.Identity()
        elif model_name == "squeezenet_attention":
            self.model = SqueezeNetGlobalAttention(
                num_weather_channels,
                pretrained_backbone=pretrained_backbone,
                attention_channels=64,
                attention_heads=4,
                attention_ffn_channels=128,
                attention_dropout=0.1,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        out = self.model(x)
        if self.model_name == "squeezenet":
            out = torch.flatten(out, 1)
        return out

    def pre_pool_logit_map(self, x):
        """Return the spatial logit map immediately before global pooling.

        Its spatial mean is the scalar discriminator logit returned by
        :meth:`forward`. For ResNet this is the class-activation map induced by
        the binary fully connected head; for SqueezeNet it is the final 1x1
        classifier-convolution output.
        """
        if self.model_name == "squeezenet":
            features = self.model.features(x)
            return self.model.classifier[:3](features)

        if self.model_name == "resnet18":
            model = self.model
            features = model.conv1(x)
            features = model.bn1(features)
            features = model.relu(features)
            features = model.maxpool(features)
            features = model.layer1(features)
            features = model.layer2(features)
            features = model.layer3(features)
            features = model.layer4(features)
            return torch.nn.functional.conv2d(
                features,
                model.fc.weight[:, :, None, None],
                model.fc.bias,
            )

        if self.model_name == "squeezenet_attention":
            raise NotImplementedError(
                "The attention discriminator uses a global token, so its scalar logit "
                "has no exact additive pre-pooling logit map."
            )

        raise ValueError(f"Unsupported model: {self.model_name}")

    def forward_with_logit_map(self, x):
        """Return scalar logits and their exact pre-pooling spatial maps."""
        logit_map = self.pre_pool_logit_map(x)
        logits = torch.mean(logit_map, dim=(-2, -1))
        return logits, logit_map

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = ((outputs > 0.0).float() == labels).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Train one discriminator from the Hydra configuration."""
    validate_no_train_test_overlap(cfg)
    vars_to_use = variables_from_config(cfg)
    real_path, fake_path = cfg.real_nc_file, cfg.fake_nc_file
    
    dataset = WeatherDiscriminatorDataset(
        real_path, fake_path, vars_to_use,
        real_range=cfg.train_real_range,
        fake_range=cfg.train_fake_range,
        lead_times=cfg.lead_times,
        level=cfg.get("level"),
        balanced=True,
        augment=cfg.get("augment", False),
        augment_prob=cfg.get("augment_prob", 0.5),
        disturb_type=cfg.get("disturb_type"),
        disturb_level=cfg.get("disturb_level", 0.0),
        corruption_types=cfg.get("corruption_types"),
        corruption_severity_max=cfg.get("corruption_severity_max", 1.0),
        corruption_severity_power=cfg.get("corruption_severity_power", 2.0),
        corruption_severity_min=cfg.get("corruption_severity_min", 0.0),
        corruption_severity_min_overrides=cfg.get("corruption_severity_min_overrides", {}),
        field_corruption_prob=cfg.get("field_corruption_prob", 0.5),
        max_samples=cfg.get("max_samples", 0),
        monthly_split_cfg=(
            cfg if str((cfg.get("monthly_split") or {}).get("strategy", "")) == "monthly_valid_time"
            else None
        ),
        monthly_split_name="train",
    )
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    
    model = WeatherDiscriminator(
        num_weather_channels=len(vars_to_use), 
        model_name=cfg.model_name, 
        learning_rate=cfg.learning_rate
    )
    
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        logger=build_logger(cfg),
        accelerator="auto",
        precision=cfg.precision if torch.cuda.is_available() else 32,
    )
    
    trainer.fit(model, dataloader)

    os.makedirs(cfg.output_dir, exist_ok=True)
    variable_tag = cfg.get("selected_variable", "all_fields") if len(vars_to_use) == 1 else "all_fields"
    output_filename = cfg.get("output_filename", f"weather_discriminator_{cfg.model_name}_{variable_tag}_lightning.pth")
    save_path = os.path.join(cfg.output_dir, output_filename)
    torch.save(model.model.state_dict(), save_path)
    print(f"Weights saved to {save_path}")

if __name__ == "__main__":
    main()

"""Canonical monthly valid-time splitting and exact ERA5 pairing utilities."""

from dataclasses import dataclass

import numpy as np
import xarray as xr

try:
    from .temporal_resampling import active_schedule, schedule_mask
except ImportError:
    from temporal_resampling import active_schedule, schedule_mask


@dataclass(frozen=True)
class ForecastPair:
    """One forecast sample and its exact ERA5 valid-time counterpart."""

    forecast_index: int
    lead_index: int
    lead_hour: int
    era5_index: int
    initialization_time: np.datetime64
    valid_time: np.datetime64


def split_settings(cfg):
    settings = cfg.get("monthly_split")
    if settings is None or str(settings.get("strategy", "")) != "monthly_valid_time":
        raise ValueError("Canonical workflow requires monthly_split.strategy=monthly_valid_time.")
    return settings


def day_bounds(cfg, split):
    settings = split_settings(cfg)
    key = {
        "train": "train_days",
        "test": "test_days",
        "null": "null_comparison_days",
    }.get(str(split))
    if key is None:
        raise ValueError(f"Unknown monthly split: {split!r}")
    lower, upper = (int(value) for value in settings[key])
    if not (1 <= lower <= upper <= 31):
        raise ValueError(f"Invalid {key}: {[lower, upper]}")
    return lower, upper


def time_ranges(cfg, coverage):
    settings = split_settings(cfg)
    key = "corruption_time_range" if coverage == "corruption" else "model_valid_time_ranges"
    ranges = settings[key]
    if coverage == "corruption":
        ranges = [ranges]
    return [(np.datetime64(start, "ns"), np.datetime64(end, "ns")) for start, end in ranges]


def datetime_mask(values, ranges, days):
    values = np.asarray(values).astype("datetime64[ns]")
    date_days = values.astype("datetime64[D]")
    month_starts = date_days.astype("datetime64[M]")
    day_of_month = (date_days - month_starts).astype(int) + 1
    in_range = np.zeros(values.shape, dtype=bool)
    for start, end in ranges:
        in_range |= (values >= start) & (values <= end + np.timedelta64(1, "D") - np.timedelta64(1, "ns"))
    return in_range & (day_of_month >= int(days[0])) & (day_of_month <= int(days[1]))


def select_era5_split(dataset, cfg, split, coverage="corruption"):
    """Select an ERA5 calendar split without altering the original dataset."""
    schedule = active_schedule(cfg)
    mask = (
        schedule_mask(dataset.time.values, schedule, split, time_ranges(cfg, coverage))
        if schedule is not None else
        datetime_mask(dataset.time.values, time_ranges(cfg, coverage), day_bounds(cfg, split))
    )
    return dataset.isel(time=np.flatnonzero(mask))


def era5_time_lookup(dataset):
    times = np.asarray(dataset.time.values).astype("datetime64[ns]")
    lookup = {int(value.astype(np.int64)): index for index, value in enumerate(times)}
    if len(lookup) != len(times):
        raise ValueError("ERA5 time coordinate contains duplicate timestamps.")
    return lookup


def lead_hours(dataset):
    if "prediction_timedelta" not in dataset.coords:
        return np.asarray([0], dtype=int)
    values = np.asarray(dataset.prediction_timedelta.values)
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[h]").astype(int)
    return values.astype(int)


def forecast_pairs(dataset, era5, cfg, split, configured_leads=None):
    """Return forecast indices paired to ERA5 by exact forecast valid time."""
    initialization = np.asarray(dataset.time.values).astype("datetime64[ns]")
    hours = lead_hours(dataset)
    permitted_leads = None if configured_leads is None else {int(value) for value in configured_leads}
    ranges = time_ranges(cfg, "model")
    schedule = active_schedule(cfg)
    days = None if schedule is not None else day_bounds(cfg, split)
    lookup = era5_time_lookup(era5)
    pairs = []
    missing = []
    for lead_index, lead_hour in enumerate(hours):
        lead_hour = int(lead_hour)
        if permitted_leads is not None and lead_hour not in permitted_leads:
            continue
        valid = initialization + np.timedelta64(lead_hour, "h")
        positions = np.flatnonzero(
            schedule_mask(valid, schedule, split, ranges)
            if schedule is not None else datetime_mask(valid, ranges, days)
        )
        for forecast_index in positions:
            valid_time = valid[forecast_index]
            era5_index = lookup.get(int(valid_time.astype(np.int64)))
            if era5_index is None:
                missing.append((lead_hour, str(valid_time)))
                continue
            pairs.append(ForecastPair(
                forecast_index=int(forecast_index),
                lead_index=int(lead_index),
                lead_hour=lead_hour,
                era5_index=int(era5_index),
                initialization_time=initialization[forecast_index],
                valid_time=valid_time,
            ))
    if missing and bool(split_settings(cfg).get("require_exact_era5_match", True)):
        preview = ", ".join(f"+{lead}h at {timestamp}" for lead, timestamp in missing[:5])
        raise ValueError(f"Missing {len(missing)} exact ERA5 forecast-valid timestamps: {preview}")
    return pairs


def concatenate_forecasts(datasets):
    """Lazily concatenate same-model forecast files along initialization time."""
    datasets = list(datasets)
    if not datasets:
        raise ValueError("At least one forecast dataset is required.")
    if len(datasets) == 1:
        return datasets[0]
    return xr.concat(datasets, dim="time").sortby("time")


def evenly_spaced_pairs(pairs, maximum):
    maximum = int(maximum or 0)
    if maximum <= 0 or len(pairs) <= maximum:
        return list(pairs)
    positions = np.linspace(0, len(pairs) - 1, maximum, dtype=int)
    return [pairs[int(position)] for position in positions]


def coverage_metadata(pairs):
    if not pairs:
        return {"n_pairs": 0, "initialization_start": "", "initialization_end": "", "valid_start": "", "valid_end": ""}
    initialization = np.asarray([pair.initialization_time for pair in pairs])
    valid = np.asarray([pair.valid_time for pair in pairs])
    return {
        "n_pairs": len(pairs),
        "initialization_start": str(initialization.min()),
        "initialization_end": str(initialization.max()),
        "valid_start": str(valid.min()),
        "valid_end": str(valid.max()),
    }

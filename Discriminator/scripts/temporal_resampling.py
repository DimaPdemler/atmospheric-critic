"""Deterministic within-month temporal resampling and auditable result tables."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
LEARNED_WINDOWS = ((5, 11), (9, 15), (13, 19), (17, 23), (20, 26))


@dataclass(frozen=True)
class TemporalSchedule:
    resample_id: str
    family: str
    ordinal: int
    test_days: tuple[int, int] | None = None
    monthly_test_starts: tuple[tuple[str, int], ...] = ()
    buffer_days: int = 4

    def test_window(self, month: str) -> tuple[int, int]:
        if self.test_days is not None:
            return self.test_days
        starts = dict(self.monthly_test_starts)
        start = int(starts[month])
        return start, start + 6

    def to_dict(self):
        payload = asdict(self)
        payload["test_days"] = list(self.test_days) if self.test_days else None
        payload["monthly_test_starts"] = dict(self.monthly_test_starts)
        return payload


def settings(cfg):
    return cfg.get("temporal_resampling", {}) or {}


def enabled(cfg) -> bool:
    return bool(settings(cfg).get("enabled", False))


def active_schedule(cfg) -> TemporalSchedule | None:
    payload = settings(cfg).get("active_schedule")
    if not payload:
        return None
    monthly = payload.get("monthly_test_starts", {}) or {}
    days = payload.get("test_days")
    return TemporalSchedule(
        resample_id=str(payload["resample_id"]), family=str(payload["family"]),
        ordinal=int(payload["ordinal"]),
        test_days=None if days is None else (int(days[0]), int(days[1])),
        monthly_test_starts=tuple(sorted((str(key), int(value)) for key, value in monthly.items())),
        buffer_days=int(payload.get("buffer_days", 4)),
    )


def _month_strings(ranges) -> list[str]:
    months = set()
    for start, end in ranges:
        cursor = np.datetime64(start, "M")
        final = np.datetime64(end, "M")
        while cursor <= final:
            months.add(str(cursor))
            cursor += np.timedelta64(1, "M")
    return sorted(months)


def learned_schedules(cfg) -> list[TemporalSchedule]:
    configured = settings(cfg).get("learned_test_windows", LEARNED_WINDOWS)
    buffer_days = int(settings(cfg).get("buffer_days", 4))
    count = int(settings(cfg).get("learned_replicates", 5))
    if count != len(configured):
        raise ValueError("learned_replicates must equal the number of learned_test_windows")
    return [TemporalSchedule(f"learned_{index:02d}", "learned", index,
                             (int(days[0]), int(days[1])), buffer_days=buffer_days)
            for index, days in enumerate(configured)]


def fixed_schedules(cfg, ranges) -> list[TemporalSchedule]:
    learned = learned_schedules(cfg)
    count = int(settings(cfg).get("fixed_replicates", 10))
    if count < len(learned):
        raise ValueError("fixed_replicates cannot be smaller than learned_replicates")
    schedules = [TemporalSchedule(f"fixed_{s.ordinal:03d}", "fixed", s.ordinal,
                                  s.test_days, buffer_days=s.buffer_days) for s in learned]
    months = _month_strings(ranges)
    seed = int(settings(cfg).get("seed", 0))
    low, high = (int(v) for v in settings(cfg).get("random_test_start_range", [5, 20]))
    for ordinal in range(len(schedules), count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, ordinal]))
        starts = tuple((month, int(rng.integers(low, high + 1))) for month in months)
        schedules.append(TemporalSchedule(
            f"fixed_{ordinal:03d}", "fixed", ordinal,
            monthly_test_starts=starts, buffer_days=int(settings(cfg).get("buffer_days", 4)),
        ))
    return schedules


def schedule_mask(values, schedule: TemporalSchedule, split: str, ranges) -> np.ndarray:
    """Return test or buffered-training membership for a schedule."""
    values = np.asarray(values).astype("datetime64[ns]")
    dates = values.astype("datetime64[D]")
    months = dates.astype("datetime64[M]")
    in_coverage = np.zeros(values.shape, dtype=bool)
    for start, end in ranges:
        start = np.datetime64(start, "ns"); end = np.datetime64(end, "ns")
        in_coverage |= (values >= start) & (values <= end + np.timedelta64(1, "D") - np.timedelta64(1, "ns"))
    test = np.zeros(values.shape, dtype=bool)
    exclusion = np.zeros(values.shape, dtype=bool)
    for month in np.unique(months[in_coverage]):
        month_key = str(month)
        lower, upper = schedule.test_window(month_key)
        month_start = month.astype("datetime64[D]")
        test_start = month_start + np.timedelta64(lower - 1, "D")
        test_end = month_start + np.timedelta64(upper - 1, "D")
        test |= in_coverage & (dates >= test_start) & (dates <= test_end)
        exclusion |= in_coverage & (dates >= test_start - np.timedelta64(schedule.buffer_days, "D")) & (
            dates <= test_end + np.timedelta64(schedule.buffer_days, "D"))
    if split in {"test", "null"}:
        return test
    if split == "train":
        return in_coverage & ~exclusion
    raise ValueError(f"Unknown temporal-resampling split: {split!r}")


def write_csv_gz(path: Path, rows, fieldnames=None):
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or (rows[0].keys() if rows else []))
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with gzip.open(temporary, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_draws(rows, value_fields, group_fields, bounds: str):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(field) for field in group_fields), []).append(row)
    output = []
    for key, group in grouped.items():
        row = {field: value for field, value in zip(group_fields, key)}
        row["n_resamples"] = len(group)
        row["draw_ids"] = ",".join(str(item["resample_id"]) for item in group)
        for field in value_fields:
            values = np.asarray([float(item[field]) for item in group], dtype=np.float64)
            row[field] = float(np.mean(values))
            if bounds == "minmax":
                row[f"{field}_lower"], row[f"{field}_upper"] = float(values.min()), float(values.max())
            elif bounds == "p05_p95":
                row[f"{field}_lower"], row[f"{field}_upper"] = (float(v) for v in np.quantile(values, [0.05, 0.95]))
            else:
                raise ValueError(f"Unknown aggregation bounds: {bounds}")
        output.append(row)
    return output


def write_manifest(root: Path, payload: dict, paths):
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": SCHEMA_VERSION, **payload, "files": {}}
    for path in paths:
        path = Path(path)
        if path.is_file():
            try:
                relative = path.relative_to(root)
            except ValueError:
                relative = Path("..") / path.name
            manifest["files"][str(relative)] = {
                "bytes": path.stat().st_size, "sha256": file_sha256(path),
            }
    target = root / "manifest.json"
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with open(temporary, "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target



def _selected_positions(length: int, maximum: int) -> set[int]:
    if maximum <= 0 or length <= maximum:
        return set(range(length))
    return set(np.linspace(0, length - 1, maximum, dtype=int).tolist())


def write_split_membership(cfg, dataset, output_root: Path):
    """Persist every eligible ERA5 timestamp and deterministic cap membership."""
    try:
        from .monthly_split import select_era5_split
    except ImportError:
        from monthly_split import select_era5_split
    schedule = active_schedule(cfg)
    if schedule is None:
        return None
    original_times = np.asarray(dataset.time.values).astype("datetime64[ns]")
    lookup = {int(value.astype(np.int64)): index for index, value in enumerate(original_times)}
    baseline = cfg.get("baseline", {}) or {}
    discriminator = cfg.get("target_discriminator", {}) or {}
    standard_default = int(baseline.get("eval_samples", cfg.get("max_samples", 0)))
    rows = []
    for coverage in ("model", "corruption"):
        standard_maximum = int(
            baseline.get("corruption_eval_samples", standard_default)
            if coverage == "corruption" else standard_default
        )
        comparison_size = len(select_era5_split(dataset, cfg, "test", coverage=coverage).time)
        for split in ("train", "test", "null"):
            subset = select_era5_split(dataset, cfg, split, coverage=coverage)
            times = np.asarray(subset.time.values).astype("datetime64[ns]")
            effective_standard_maximum = (comparison_size if split == "train" and standard_maximum <= 0
                                          else standard_maximum)
            standard_selected = _selected_positions(len(times), effective_standard_maximum)
            discriminator_maximum = int(
                discriminator.get("max_train_samples", 0) if split == "train"
                else discriminator.get("max_eval_samples", 0)
            )
            discriminator_selected = _selected_positions(len(times), discriminator_maximum)
            for position, timestamp in enumerate(times):
                rows.append({
                    "resample_id": schedule.resample_id, "family": schedule.family,
                    "coverage": coverage, "role": split, "source": "ERA5",
                    "source_path": str(cfg.real_nc_file),
                    "source_index": lookup[int(timestamp.astype(np.int64))],
                    "subset_position": position, "timestamp": str(timestamp),
                    "selected_standard": position in standard_selected,
                    "selected_discriminator": position in discriminator_selected,
                })
    forecast_files = baseline.get("forecast_files", {}) or {}
    if forecast_files:
        try:
            from .monthly_split import concatenate_forecasts, evenly_spaced_pairs, forecast_pairs
            from .train_discriminator import normalize_prediction_timedelta, safe_open_dataset
        except ImportError:
            from monthly_split import concatenate_forecasts, evenly_spaced_pairs, forecast_pairs
            from train_discriminator import normalize_prediction_timedelta, safe_open_dataset
        for label, configured_paths in forecast_files.items():
            paths = ([str(configured_paths)] if isinstance(configured_paths, str)
                     else [str(value) for value in configured_paths])
            if not paths or any(not Path(value).is_file() for value in paths):
                continue
            forecast = concatenate_forecasts([
                normalize_prediction_timedelta(safe_open_dataset(value)) for value in paths
            ])
            for split in ("train", "test"):
                pairs = forecast_pairs(forecast, dataset, cfg, split, cfg.lead_times)
                standard_selected = set()
                by_lead = {}
                for pair in pairs:
                    by_lead.setdefault(pair.lead_index, []).append(pair)
                for lead_pairs in by_lead.values():
                    standard_selected.update(
                        (pair.lead_index, pair.forecast_index)
                        for pair in evenly_spaced_pairs(lead_pairs, standard_default)
                    )
                discriminator_maximum = int(
                    discriminator.get("max_train_samples", 0) if split == "train"
                    else discriminator.get("max_eval_samples", 0)
                )
                discriminator_selected = {
                    (pair.lead_index, pair.forecast_index)
                    for pair in evenly_spaced_pairs(pairs, discriminator_maximum)
                }
                for position, pair in enumerate(pairs):
                    identity = (pair.lead_index, pair.forecast_index)
                    rows.append({
                        "resample_id": schedule.resample_id, "family": schedule.family,
                        "coverage": "model", "role": split, "source": str(label),
                        "source_path": ";".join(paths), "source_index": pair.forecast_index,
                        "subset_position": position, "timestamp": str(pair.valid_time),
                        "initialization_time": str(pair.initialization_time),
                        "valid_time": str(pair.valid_time), "lead_hour": pair.lead_hour,
                        "era5_index": pair.era5_index,
                        "selected_standard": identity in standard_selected,
                        "selected_discriminator": identity in discriminator_selected,
                    })
            forecast.close()
    path = Path(output_root) / "data" / "split_manifest.csv.gz"
    write_csv_gz(path, rows)
    coordinate_hash = hashlib.sha256(original_times.tobytes()).hexdigest()
    metadata = {
        "schema_version": SCHEMA_VERSION, "schedule": schedule.to_dict(),
        "era5_path": str(cfg.real_nc_file), "era5_time_coordinate_sha256": coordinate_hash,
        "rows": len(rows),
    }
    metadata_path = path.parent / "split_manifest.json"
    temporary = metadata_path.with_name(f".{metadata_path.name}.tmp")
    try:
        with open(temporary, "w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.flush(); os.fsync(handle.fileno())
        temporary.replace(metadata_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path

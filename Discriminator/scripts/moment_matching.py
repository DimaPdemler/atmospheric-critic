"""Training-only pooled scalar mean/standard-deviation matching."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def settings(cfg):
    return cfg.get("moment_matching", {}) or {}


def enabled(cfg):
    return bool(settings(cfg).get("enabled", False))


def artifact_dir(cfg, *, writing=False):
    if not writing:
        configured = (cfg.get("pipeline", {}) or {}).get("input_moment_matching_dir")
        if configured:
            return Path(str(configured))
    pipeline = cfg.get("pipeline", {}) or {}
    root = Path(str(pipeline.get("output_root") or cfg.baseline.output_dir))
    return root / "data" / "preprocessing" / "moment_matching"


def coordinate_token(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return format(float(value), ".12g")


def map_id(family, kind, target, coordinate, variable):
    return "|".join(map(str, (family, kind, target, coordinate_token(coordinate), variable)))


class RunningMoments:
    """Float64 streaming scalar moments over finite values."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        self.count += int(finite.size)
        self.total += float(np.sum(finite, dtype=np.float64))
        self.total_sq += float(np.sum(finite * finite, dtype=np.float64))

    def result(self):
        if self.count <= 0:
            raise ValueError("Moment matching requires non-empty finite values.")
        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean * mean, 0.0)
        return float(mean), float(np.sqrt(variance)), int(self.count)


@dataclass
class MomentMap:
    source_mean: float
    source_std: float
    target_mean: float
    target_std: float

    @property
    def scale(self):
        return self.target_std / self.source_std

    @property
    def shift(self):
        return self.target_mean - self.source_mean * self.scale

    def apply(self, values):
        array = np.asarray(values)
        result = np.array(array, dtype=np.float32, copy=True)
        finite = np.isfinite(array)
        result[finite] = (
            (np.asarray(array[finite], dtype=np.float64) - self.source_mean) * self.scale
            + self.target_mean
        ).astype(np.float32)
        return result


class MomentMapStore:
    def __init__(self, maps=None, metadata=None):
        self.maps = dict(maps or {})
        self.metadata = list(metadata or [])

    def add_moments(self, key, source, target, *, metadata=None):
        source_mean, source_std, source_count = source.result()
        target_mean, target_std, target_count = target.result()
        if not np.isfinite(source_std) or source_std <= 1e-8:
            raise ValueError(f"Cannot moment-match {key}: source standard deviation is {source_std!r}.")
        if not np.isfinite(target_std) or target_std <= 1e-8:
            raise ValueError(f"Cannot moment-match {key}: target standard deviation is {target_std!r}.")
        mapping = MomentMap(source_mean, source_std, target_mean, target_std)
        self.maps[str(key)] = mapping
        row = dict(metadata or {})
        row.update(
            map_id=str(key), source_count=source_count, target_count=target_count,
            source_mean=source_mean, source_std=source_std,
            target_mean=target_mean, target_std=target_std,
            scale=mapping.scale, shift=mapping.shift,
            fitted_mean_residual=0.0,
            fitted_std_ratio_error=0.0,
        )
        self.metadata.append(row)

    def require(self, family, kind, target, coordinate, variable):
        key = map_id(family, kind, target, coordinate, variable)
        if key not in self.maps:
            raise KeyError(f"Moment-matching artifact is missing map {key!r}.")
        return self.maps[key]

    def apply_channels(self, values, variables, family, kind, target, coordinate):
        array = np.asarray(values)
        if array.shape[0] != len(variables):
            raise ValueError("Moment-matching channel count does not match variables.")
        return np.stack([
            self.require(family, kind, target, coordinate, variable).apply(array[index])
            for index, variable in enumerate(variables)
        ]).astype(np.float32)

    def save(self, directory, config=None):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.maps)
        arrays = {"keys": np.asarray(keys)}
        for index, key in enumerate(keys):
            mapping = self.maps[key]
            arrays[f"moments_{index}"] = np.asarray([
                mapping.source_mean, mapping.source_std,
                mapping.target_mean, mapping.target_std,
            ], dtype=np.float64)
        np.savez_compressed(directory / "moments.npz", **arrays)
        manifest = {"version": 1, "maps": self.metadata, "config": config or {}}
        payload = json.dumps(manifest, sort_keys=True, indent=2)
        manifest["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return directory / "moments.npz", directory / "manifest.json"

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        archive_path, manifest_path = directory / "moments.npz", directory / "manifest.json"
        if not archive_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Moment matching requires moments.npz and manifest.json under {directory}."
            )
        manifest = json.loads(manifest_path.read_text())
        with np.load(archive_path, allow_pickle=False) as archive:
            keys = [str(value) for value in archive["keys"]]
            maps = {}
            for index, key in enumerate(keys):
                values = np.asarray(archive[f"moments_{index}"], dtype=np.float64)
                maps[key] = MomentMap(*map(float, values))
        return cls(maps, manifest.get("maps", []))


_CACHE = {}


def store_for(cfg):
    if not enabled(cfg):
        return None
    directory = artifact_dir(cfg).resolve()
    if directory not in _CACHE:
        _CACHE[directory] = MomentMapStore.load(directory)
    return _CACHE[directory]

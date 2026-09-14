"""Leakage-free pooled scalar histogram matching for baseline experiments."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def settings(cfg):
    return cfg.get("histogram_matching", {}) or {}


def enabled(cfg):
    return bool(settings(cfg).get("enabled", False))


def artifact_dir(cfg, *, writing=False):
    if not writing:
        configured = (cfg.get("pipeline", {}) or {}).get("input_histogram_matching_dir")
        if configured:
            return Path(str(configured))
    pipeline = cfg.get("pipeline", {}) or {}
    root = Path(str(pipeline.get("output_root") or cfg.baseline.output_dir))
    return root / "data" / "preprocessing" / "histogram_matching"


def coordinate_token(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return format(float(value), ".12g")


def map_id(family, kind, target, coordinate, variable):
    return "|".join(map(str, (family, kind, target, coordinate_token(coordinate), variable)))


def _finite(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def deterministic_subsample(values, maximum, seed):
    values = _finite(values)
    maximum = int(maximum)
    if maximum <= 0 or values.size <= maximum:
        return values
    return values[np.random.default_rng(int(seed)).choice(values.size, maximum, replace=False)]


def quantile_table(source, target, knots):
    source, target = _finite(source), _finite(target)
    if not source.size or not target.size:
        raise ValueError("Histogram matching requires non-empty finite source and target values.")
    probabilities = np.linspace(0.0, 1.0, max(2, int(knots)), dtype=np.float64)
    source_q = np.quantile(source, probabilities)
    target_q = np.quantile(target, probabilities)
    # np.interp requires increasing abscissae. Keep the last target quantile for
    # each repeated source value, which also preserves the upper edge of atoms.
    reversed_unique, reversed_index = np.unique(source_q[::-1], return_index=True)
    keep = len(source_q) - 1 - reversed_index
    order = np.argsort(source_q[keep], kind="stable")
    return source_q[keep][order], target_q[keep][order]


def apply_quantile_table(values, source_quantiles, target_quantiles):
    array = np.asarray(values)
    result = np.array(array, dtype=np.float32, copy=True)
    finite = np.isfinite(array)
    result[finite] = np.interp(
        np.asarray(array[finite], dtype=np.float64), source_quantiles, target_quantiles,
        left=float(target_quantiles[0]), right=float(target_quantiles[-1]),
    ).astype(np.float32)
    return result


@dataclass
class HistogramMap:
    source_quantiles: np.ndarray
    target_quantiles: np.ndarray

    def apply(self, values):
        return apply_quantile_table(values, self.source_quantiles, self.target_quantiles)


class HistogramMapStore:
    def __init__(self, maps=None, metadata=None):
        self.maps = dict(maps or {})
        self.metadata = list(metadata or [])

    def add(self, key, source, target, *, knots=2049, metadata=None):
        source_q, target_q = quantile_table(source, target, knots)
        self.maps[str(key)] = HistogramMap(source_q, target_q)
        row = dict(metadata or {})
        row.update(map_id=str(key), source_count=int(np.isfinite(source).sum()),
                   target_count=int(np.isfinite(target).sum()))
        self.metadata.append(row)

    def require(self, family, kind, target, coordinate, variable):
        key = map_id(family, kind, target, coordinate, variable)
        if key not in self.maps:
            raise KeyError(f"Histogram matching artifact is missing map {key!r}.")
        return self.maps[key]

    def apply_channels(self, values, variables, family, kind, target, coordinate):
        array = np.asarray(values)
        if array.shape[0] != len(variables):
            raise ValueError("Histogram-matching channel count does not match variables.")
        return np.stack([
            self.require(family, kind, target, coordinate, variable).apply(array[index])
            for index, variable in enumerate(variables)
        ]).astype(np.float32)

    def save(self, directory, config=None):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        keys = sorted(self.maps)
        arrays = {"keys": np.asarray(keys)}
        for index, key in enumerate(keys):
            arrays[f"source_{index}"] = self.maps[key].source_quantiles
            arrays[f"target_{index}"] = self.maps[key].target_quantiles
        np.savez_compressed(directory / "maps.npz", **arrays)
        manifest = {"version": 1, "maps": self.metadata, "config": config or {}}
        payload = json.dumps(manifest, sort_keys=True, indent=2)
        manifest["sha256"] = hashlib.sha256(payload.encode()).hexdigest()
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return directory / "maps.npz", directory / "manifest.json"

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        archive_path, manifest_path = directory / "maps.npz", directory / "manifest.json"
        if not archive_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Histogram matching requires maps.npz and manifest.json under {directory}."
            )
        manifest = json.loads(manifest_path.read_text())
        with np.load(archive_path, allow_pickle=False) as archive:
            keys = [str(value) for value in archive["keys"]]
            maps = {
                key: HistogramMap(np.asarray(archive[f"source_{index}"]),
                                  np.asarray(archive[f"target_{index}"]))
                for index, key in enumerate(keys)
            }
        return cls(maps, manifest.get("maps", []))


_CACHE = {}


def store_for(cfg):
    if not enabled(cfg):
        return None
    directory = artifact_dir(cfg).resolve()
    if directory not in _CACHE:
        _CACHE[directory] = HistogramMapStore.load(directory)
    return _CACHE[directory]

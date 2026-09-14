"""Runtime adapter for mutually exclusive fake-distribution matching modes."""

import numpy as np
import torch

try:
    from .histogram_matching import enabled as histogram_enabled, store_for as histogram_store
    from .moment_matching import enabled as moment_enabled, store_for as moment_store
except ImportError:
    from histogram_matching import enabled as histogram_enabled, store_for as histogram_store
    from moment_matching import enabled as moment_enabled, store_for as moment_store


def matching_mode(cfg):
    histogram = histogram_enabled(cfg)
    moments = moment_enabled(cfg)
    if histogram and moments:
        raise ValueError("Histogram matching and moment matching are mutually exclusive.")
    return "histogram" if histogram else "moments" if moments else "none"


def _store(cfg):
    mode = matching_mode(cfg)
    if mode == "histogram":
        return histogram_store(cfg)
    if mode == "moments":
        return moment_store(cfg)
    return None


def match_raw(cfg, values, variables, family, kind, target, coordinate):
    """Match channel-first physical fake fields, preserving tensor/array type."""
    if matching_mode(cfg) == "none" or (kind == "corruption" and float(coordinate) == 0.0):
        return values
    tensor = isinstance(values, torch.Tensor)
    array = values.detach().cpu().numpy() if tensor else np.asarray(values)
    matched = _store(cfg).apply_channels(array, variables, family, kind, target, coordinate)
    if tensor:
        return torch.from_numpy(matched).to(device=values.device, dtype=values.dtype)
    return matched


def match_standardized(cfg, values, variables, means, stds, family, kind, target, coordinate):
    """Round-trip standardized fake fields through physical-space matching."""
    if matching_mode(cfg) == "none" or (kind == "corruption" and float(coordinate) == 0.0):
        return values
    tensor = isinstance(values, torch.Tensor)
    array = values.detach().cpu().numpy() if tensor else np.asarray(values)
    raw = np.stack([
        array[index] * float(stds[variable]) + float(means[variable])
        for index, variable in enumerate(variables)
    ])
    raw = _store(cfg).apply_channels(raw, variables, family, kind, target, coordinate)
    matched = np.stack([
        (raw[index] - float(means[variable])) / float(stds[variable])
        for index, variable in enumerate(variables)
    ]).astype(np.float32)
    if tensor:
        return torch.from_numpy(matched).to(device=values.device, dtype=values.dtype)
    return matched

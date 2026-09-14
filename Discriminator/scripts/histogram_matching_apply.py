"""Small runtime adapters for applying fitted histogram maps."""

import numpy as np
import torch

try:
    from .histogram_matching import enabled, store_for
except ImportError:
    from histogram_matching import enabled, store_for


def match_raw(cfg, values, variables, family, kind, target, coordinate):
    """Match channel-first physical fields, preserving tensor/array type."""
    if not enabled(cfg) or float(coordinate) == 0.0 and kind == "corruption":
        return values
    tensor = isinstance(values, torch.Tensor)
    array = values.detach().cpu().numpy() if tensor else np.asarray(values)
    matched = store_for(cfg).apply_channels(
        array, variables, family, kind, target, coordinate,
    )
    return torch.from_numpy(matched).to(values.device) if tensor else matched


def match_standardized(cfg, values, variables, means, stds, family, kind, target, coordinate):
    """Round-trip normalized fields through physical-space histogram matching."""
    if not enabled(cfg) or float(coordinate) == 0.0 and kind == "corruption":
        return values
    tensor = isinstance(values, torch.Tensor)
    array = values.detach().cpu().numpy() if tensor else np.asarray(values)
    raw = np.stack([
        array[index] * float(stds[variable]) + float(means[variable])
        for index, variable in enumerate(variables)
    ])
    raw = store_for(cfg).apply_channels(raw, variables, family, kind, target, coordinate)
    matched = np.stack([
        (raw[index] - float(means[variable])) / float(stds[variable])
        for index, variable in enumerate(variables)
    ]).astype(np.float32)
    return torch.from_numpy(matched).to(values.device) if tensor else matched

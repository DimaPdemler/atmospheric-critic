"""Temporal-pair composition for the realism encoders.

Single source of truth used by both training (`utils/dataset.py`) and evaluation
(`eval/eval_real_vs_forecast.py`) so the two paths cannot drift on channel
ordering, normalization choice, or clip range.

Modes:
  none   : 4ch — abs-normalized X_t
  diff   : 4ch — diff-normalized (X_t - X_{t-Δt})
  concat : 8ch — [abs-normalized X_{t-Δt}, abs-normalized X_t]
  phase  : 8ch — [abs-normalized X_t, diff-normalized (X_t - X_{t-Δt})]

Diff channels are clipped to [-DIFF_CLIP, +DIFF_CLIP] after normalization so a
hallucinating forecast (e.g. 100 K temperature jump) cannot blow up the FID
covariance or PCA projection.
"""

import numpy as np


IN_CHANS_BY_MODE = {"none": 4, "diff": 4, "concat": 8, "phase": 8}

DIFF_CLIP = 15.0

PAD_H = (4, 3)
PAD_W = (8, 8)


def _normalize(arr, mean, std):
    """arr: (..., 4, H, W) raw. mean/std: any shape that flattens to per-channel (4,).

    Reshapes mean/std to (4, 1, 1) so broadcasting is correct for both 3-D
    (C, H, W) and 4-D (T, C, H, W) inputs — without numpy promoting the rank
    of the result.
    """
    m = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
    s = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
    return (arr - m) / s


def _pad(arr):
    """Pad the spatial axes only. Works for any leading shape (..., C, H, W)."""
    pad_width = [(0, 0)] * (arr.ndim - 2) + [PAD_H, (0, 0)]
    arr = np.pad(arr, pad_width=pad_width, mode="constant", constant_values=0)
    pad_width = [(0, 0)] * (arr.ndim - 2) + [(0, 0), PAD_W]
    arr = np.pad(arr, pad_width=pad_width, mode="wrap")
    return np.asarray(arr, dtype=np.float32)


def compose_temporal_input(
    present_raw,
    prior_raw,
    mode,
    abs_mean,
    abs_std,
    diff_mean=None,
    diff_std=None,
):
    """Build the encoder input from a (raw, raw) timestep pair.

    Args:
        present_raw: ndarray (..., 4, H, W) of X_t, un-normalized.
        prior_raw:   ndarray (..., 4, H, W) of X_{t-Δt}, un-normalized.
                     Required for diff/concat/phase; ignored for `none`.
        mode: one of "none", "diff", "concat", "phase".
        abs_mean, abs_std: (1, 4, 1, 1) absolute-field stats.
        diff_mean, diff_std: (1, 4, 1, 1) diff-field stats. Required when mode
                             is "diff" or "phase".

    Returns:
        ndarray (..., C, H+7, W+16) float32 with C = IN_CHANS_BY_MODE[mode].
        Padded (H 4/3, W wrap 8/8) — same scheme as the original dataset.
    """
    present_raw = np.asarray(present_raw, dtype=np.float32)

    if mode == "none":
        out = _normalize(present_raw, abs_mean, abs_std)

    elif mode == "diff":
        if diff_mean is None or diff_std is None:
            raise ValueError("mode='diff' requires diff_mean and diff_std")
        prior_raw = np.asarray(prior_raw, dtype=np.float32)
        diff = present_raw - prior_raw
        out = _normalize(diff, diff_mean, diff_std)
        out = np.clip(out, -DIFF_CLIP, DIFF_CLIP)

    elif mode == "concat":
        prior_raw = np.asarray(prior_raw, dtype=np.float32)
        prior_norm = _normalize(prior_raw, abs_mean, abs_std)
        present_norm = _normalize(present_raw, abs_mean, abs_std)
        out = np.concatenate([prior_norm, present_norm], axis=-3)

    elif mode == "phase":
        if diff_mean is None or diff_std is None:
            raise ValueError("mode='phase' requires diff_mean and diff_std")
        prior_raw = np.asarray(prior_raw, dtype=np.float32)
        present_norm = _normalize(present_raw, abs_mean, abs_std)
        diff_norm = _normalize(present_raw - prior_raw, diff_mean, diff_std)
        diff_norm = np.clip(diff_norm, -DIFF_CLIP, DIFF_CLIP)
        out = np.concatenate([present_norm, diff_norm], axis=-3)

    else:
        raise ValueError(f"Unknown temporal mode: {mode!r}")

    # Sanity check before padding: catch a future refactor that drops or
    # duplicates a channel.
    expected_c = IN_CHANS_BY_MODE[mode]
    if out.shape[-3] != expected_c:
        raise AssertionError(
            f"compose_temporal_input: mode={mode!r} expected {expected_c} channels, "
            f"got {out.shape[-3]} (shape={out.shape})"
        )

    return _pad(out)


def derive_delta_steps(time_coord, delta_hours):
    """Convert a target Δt in hours to an integer index offset.

    Asserts the file's time spacing is uniform and exactly divides delta_hours.
    Fails loudly rather than silently using a wrong stride.

    Args:
        time_coord: 1-D array of np.datetime64 timestamps.
        delta_hours: integer/float hours.

    Returns:
        int: index offset such that time_coord[i] - time_coord[i - offset] = delta_hours.
    """
    times = np.asarray(time_coord)
    if times.size < 2:
        raise ValueError("Need at least 2 timestamps to derive Δt step.")
    dts = np.diff(times)
    dt0 = dts[0]
    if not np.all(dts == dt0):
        raise ValueError(
            f"Non-uniform time spacing detected (first: {dt0}, "
            f"sample of differing: {dts[dts != dt0][:3]})."
        )
    dt_hours = dt0.astype("timedelta64[s]").astype(np.int64) / 3600.0
    if dt_hours <= 0:
        raise ValueError(f"Non-positive time spacing: {dt_hours}h")
    steps_f = delta_hours / dt_hours
    steps = int(round(steps_f))
    if abs(steps_f - steps) > 1e-6:
        raise ValueError(
            f"delta_hours={delta_hours} is not an integer multiple of "
            f"time spacing {dt_hours}h (would need {steps_f} steps)."
        )
    return steps

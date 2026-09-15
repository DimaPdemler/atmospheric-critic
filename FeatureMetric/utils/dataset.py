import torch
from torch.utils.data import Dataset
import numpy as np
from netCDF4 import Dataset as NetCDFDataset

from utils.temporal import (
    IN_CHANS_BY_MODE,
    compose_temporal_input,
)


class AtmosphereDataset(Dataset):
    """PyTorch dataset for ERA5 surface fields with per-channel normalization.

    Loads the 4 atmospheric variables from a NetCDF file, normalizes each channel,
    and pads to ``(128, 256)`` (wrap in longitude, zero-pad in latitude). Supports
    eager (in-memory) or lazy (per-item) loading, an 80/20 time-based train/val
    split, and temporal-pair modes (diff/concat/phase) via ``utils.temporal``.
    """

    def __init__(
        self,
        data_path,
        split="train",
        train_ratio=0.8,
        stats=None,
        lazy=False,
        stats_chunk_size=256,
        temporal_mode="none",
        delta_steps=0,
        diff_stats=None,
    ):
        """
        Loads ERA5 data.
        If lazy=False, eagerly loads into memory.
        If lazy=True, keeps xarray pointer and loads slice-by-slice per __getitem__.

        Temporal modes ("diff", "concat", "phase") build paired inputs (X_{t-Δt}, X_t)
        via `utils.temporal.compose_temporal_input`. The pair is read in raw units,
        then normalized + clipped + padded by the composer. `delta_steps` is the
        integer index offset corresponding to the desired Δt in hours.
        """
        super().__init__()
        if temporal_mode not in IN_CHANS_BY_MODE:
            raise ValueError(f"Unknown temporal_mode: {temporal_mode!r}")
        if temporal_mode != "none" and delta_steps <= 0:
            raise ValueError(f"temporal_mode={temporal_mode!r} requires delta_steps > 0")
        if temporal_mode in ("diff", "phase") and diff_stats is None:
            raise ValueError(f"temporal_mode={temporal_mode!r} requires diff_stats")

        self.data_path = data_path
        self.split = split
        self.lazy = lazy
        self.stats_chunk_size = stats_chunk_size
        self.temporal_mode = temporal_mode
        self.delta_steps = int(delta_steps)
        self.ds = None

        self.vars = [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure"
        ]

        with NetCDFDataset(data_path, mode="r") as ds:
            self.total_samples = len(ds.dimensions["time"])

        split_idx = int(self.total_samples * train_ratio)

        if split == "train":
            self.start_idx = 0
            self.end_idx = split_idx
        elif split == "val":
            self.start_idx = split_idx
            self.end_idx = self.total_samples
        else:
            self.start_idx = 0
            self.end_idx = self.total_samples

        # Shift the start of each split forward by delta_steps so every sample has
        # a prior inside the same split (no train→val leakage of priors).
        if self.temporal_mode != "none":
            self.start_idx += self.delta_steps

        self.n_samples = self.end_idx - self.start_idx
        print(
            f"[{split}] Dataset initialized with {self.n_samples} samples. "
            f"Lazy={lazy}, temporal_mode={self.temporal_mode}, delta_steps={self.delta_steps}"
        )

        # Absolute-field stats (always needed; "none" uses them directly,
        # "concat"/"phase" use them for the absolute halves).
        if stats is None:
            if split != "train":
                print("Warning: Creating val/test dataset without train stats!")
            print(f"[{split}] Computing normalization stats...")
            self.mean, self.std = self._compute_stats()
        else:
            self.mean = np.asarray(stats[0], dtype=np.float32)
            self.std = np.asarray(stats[1], dtype=np.float32)

        self.mean_1d = self.mean.reshape(-1).astype(np.float32, copy=False)
        self.std_1d = self.std.reshape(-1).astype(np.float32, copy=False)
        self.std_1d[self.std_1d == 0] = 1.0

        # Diff-field stats (only needed for "diff"/"phase").
        self.diff_mean = None
        self.diff_std = None
        if self.temporal_mode in ("diff", "phase"):
            if diff_stats is None:
                # Should not reach here — ctor checks above — but guard anyway.
                raise ValueError("diff_stats required but missing.")
            self.diff_mean = np.asarray(diff_stats[0], dtype=np.float32)
            self.diff_std = np.asarray(diff_stats[1], dtype=np.float32).copy()
            # Avoid divide-by-zero on any flat channel.
            flat = self.diff_std.reshape(-1)
            flat[flat == 0] = 1.0
            self.diff_std = flat.reshape(self.diff_mean.shape)

        if not self.lazy:
            print(f"[{split}] Eagerly loading {self.n_samples} samples into memory...")
            self.data_tensor = self._build_eager_tensor()

    def _get_dataset(self):
        """Lazily open (and cache) the NetCDF file handle."""
        if self.ds is None:
            self.ds = NetCDFDataset(self.data_path, mode="r")
        return self.ds

    def _read_time_slice(self, start_idx, end_idx):
        """Read variables for time rows ``[start_idx, end_idx)`` as ``(T, 4, H, W)``.

        Transposes the NetCDF ``(time, lon, lat)`` layout to ``(time, lat, lon)``.
        """
        ds = self._get_dataset()
        arrays = []
        for var_name in self.vars:
            # NetCDF variable layout is (time, longitude, latitude); transpose to (time, latitude, longitude).
            arr = np.asarray(ds.variables[var_name][start_idx:end_idx, :, :], dtype=np.float32)
            arr = np.transpose(arr, (0, 2, 1))
            arrays.append(arr)
        return np.stack(arrays, axis=1).astype(np.float32, copy=False)

    def _build_eager_tensor(self):
        """Load the whole split into a single composed, padded tensor in memory.

        For temporal modes it reads an extended window so each sample has its
        prior, then builds the paired/normalized input via ``compose_temporal_input``.
        """
        if self.temporal_mode == "none":
            data = self._read_time_slice(self.start_idx, self.end_idx)
            composed = compose_temporal_input(
                data, None, "none", self.mean, self.std,
            )
            return torch.from_numpy(composed)

        # Temporal modes: read the full window including priors, then build pairs.
        prior_start = self.start_idx - self.delta_steps
        if prior_start < 0:
            raise RuntimeError(
                f"prior_start={prior_start} is negative — start_idx={self.start_idx} "
                f"must be ≥ delta_steps={self.delta_steps}."
            )
        raw = self._read_time_slice(prior_start, self.end_idx)
        present = raw[self.delta_steps:]
        prior   = raw[:-self.delta_steps]
        composed = compose_temporal_input(
            present, prior, self.temporal_mode,
            self.mean, self.std, self.diff_mean, self.diff_std,
        )
        return torch.from_numpy(composed)

    def _compute_stats(self):
        """Stream the split in chunks to compute per-channel mean/std.

        Returns ``(mean, std)`` each shaped ``(1, 4, 1, 1)`` for broadcasting.
        """
        total_count = 0
        total_sum = np.zeros(4, dtype=np.float64)
        total_sumsq = np.zeros(4, dtype=np.float64)

        for chunk_start in range(self.start_idx, self.end_idx, self.stats_chunk_size):
            chunk_end = min(chunk_start + self.stats_chunk_size, self.end_idx)
            chunk = self._read_time_slice(chunk_start, chunk_end)
            total_sum += chunk.sum(axis=(0, 2, 3), dtype=np.float64)
            total_sumsq += np.square(chunk, dtype=np.float32).sum(axis=(0, 2, 3), dtype=np.float64)
            total_count += chunk.shape[0] * chunk.shape[2] * chunk.shape[3]

        mean = total_sum / total_count
        var = (total_sumsq / total_count) - np.square(mean)
        var = np.maximum(var, 1e-12)
        std = np.sqrt(var)

        mean = mean.reshape(1, 4, 1, 1).astype(np.float32)
        std = std.reshape(1, 4, 1, 1).astype(np.float32)
        return mean, std

    def compute_diff_stats(self, delta_steps=None):
        """Compute mean/std of (X_t − X_{t-Δt}) over this dataset's index range.

        Streaming + chunked: each chunk reads `[chunk_start − delta_steps, chunk_end]`
        from the file so the in-chunk diff doesn't need a cross-chunk buffer.

        If `delta_steps` is None, uses `self.delta_steps`. Works both at bootstrap
        time (called on a `temporal_mode="none"` dataset before stats exist) and on
        an already-configured temporal-mode dataset:

          - Bootstrap case: start_idx=0 and delta_steps=Δ → diffs computed over
            [Δ, end_idx), priors over [0, end_idx − Δ).
          - Configured case: start_idx already shifted by Δ → diffs computed over
            [start_idx, end_idx), priors over [start_idx − Δ, end_idx − Δ).

        Returns (diff_mean, diff_std), both shaped (1, 4, 1, 1).
        """
        ds = self.delta_steps if delta_steps is None else int(delta_steps)
        if ds <= 0:
            raise ValueError("compute_diff_stats requires delta_steps > 0")

        # First diff is at index `max(start_idx, ds)`; before that, no prior exists.
        eff_start = max(self.start_idx, ds)
        if eff_start >= self.end_idx:
            raise RuntimeError(
                f"compute_diff_stats: effective_start={eff_start} >= end_idx={self.end_idx} "
                f"— split too short for delta_steps={ds}."
            )

        total_count = 0
        total_sum = np.zeros(4, dtype=np.float64)
        total_sumsq = np.zeros(4, dtype=np.float64)

        for chunk_start in range(eff_start, self.end_idx, self.stats_chunk_size):
            chunk_end = min(chunk_start + self.stats_chunk_size, self.end_idx)
            prior_start = chunk_start - ds
            window = self._read_time_slice(prior_start, chunk_end)
            present = window[ds:]                    # rows [chunk_start, chunk_end)
            prior   = window[:present.shape[0]]      # rows [chunk_start − ds, chunk_end − ds)
            diff = present - prior
            total_sum += diff.sum(axis=(0, 2, 3), dtype=np.float64)
            total_sumsq += np.square(diff, dtype=np.float32).sum(axis=(0, 2, 3), dtype=np.float64)
            total_count += diff.shape[0] * diff.shape[2] * diff.shape[3]

        mean = total_sum / total_count
        var = (total_sumsq / total_count) - np.square(mean)
        var = np.maximum(var, 1e-12)
        std = np.sqrt(var)

        mean = mean.reshape(1, 4, 1, 1).astype(np.float32)
        std = std.reshape(1, 4, 1, 1).astype(np.float32)
        return mean, std

    def get_stats(self):
        """Return the per-channel absolute-field ``(mean, std)`` used for normalization."""
        return self.mean, self.std

    def get_diff_stats(self):
        """Return the per-channel diff-field ``(mean, std)`` (None unless diff/phase mode)."""
        return self.diff_mean, self.diff_std

    def read_raw(self, absolute_idx):
        """Read one raw (un-normalized, un-padded) 4-channel sample at the given
        absolute file index (NOT split-relative). Used by eval-time
        TemporalPairDataset to fetch X_t and X_{t-Δt} from possibly different
        source files at matched absolute indices.
        """
        return self._read_time_slice(absolute_idx, absolute_idx + 1)[0]

    def __len__(self):
        """Number of samples in this split."""
        return self.n_samples

    def __getitem__(self, idx):
        """Return the composed, normalized, padded tensor for sample ``idx``.

        Eager mode indexes the preloaded tensor; lazy mode reads the sample (and
        its prior, for temporal modes) from disk on demand.
        """
        if not self.lazy:
            return self.data_tensor[idx]

        actual_idx = self.start_idx + idx

        if self.temporal_mode == "none":
            present = self._read_time_slice(actual_idx, actual_idx + 1)
            sample = compose_temporal_input(
                present[0], None, "none", self.mean, self.std,
            )
            return torch.from_numpy(sample)

        prior_idx = actual_idx - self.delta_steps
        prior_arr   = self._read_time_slice(prior_idx, prior_idx + 1)[0]
        present_arr = self._read_time_slice(actual_idx, actual_idx + 1)[0]
        sample = compose_temporal_input(
            present_arr, prior_arr, self.temporal_mode,
            self.mean, self.std, self.diff_mean, self.diff_std,
        )
        return torch.from_numpy(sample)


if __name__ == "__main__":
    from pathlib import Path
    data_path = Path("data/test_data_local.nc")
    if data_path.exists():
        ds_lazy = AtmosphereDataset(data_path, split="train", lazy=True)
        print("Lazy dataloader shape:", ds_lazy[0].shape)

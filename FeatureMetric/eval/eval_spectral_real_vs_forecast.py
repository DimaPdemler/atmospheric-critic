"""Spectral (2D PSD) comparison of ERA5, Pangu-Weather, and GraphCast fields.

Operates on raw (un-normalized) fields — no encoder weights needed.
Produces two plots:
  - spectra_<tag>.png       : 4×3 grid of mean log-PSD heatmaps per channel × source
  - spectra_radial_<tag>.png: radially-averaged 1D PSD per channel, three curves each

PSD utilities adapted from the atmo-discriminator codebase (ETH PMLR team07 teammates).

Usage:
    /opt/miniconda3/envs/pmlr/bin/python eval/eval_spectral_real_vs_forecast.py --n-samples 100
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ERA5_PATH      = Path("data/test_data_local.nc")
PANGU_PATH     = Path("data/pangu_surface_2020_lead24h.nc")
GRAPHCAST_PATH = Path("data/graphcast_surface_2020_lead24h.nc")

VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
VAR_LABELS = ["T2M (K)", "U10 (m/s)", "V10 (m/s)", "MSL (Pa)"]

SOURCE_COLORS = {"era5": "#4CAF50", "pangu": "#9C27B0", "graphcast": "#FF9800"}
SOURCE_LABELS = {"era5": "ERA5", "pangu": "Pangu-24h", "graphcast": "GraphCast-24h"}

# ---------------------------------------------------------------------------
# PSD helpers — adapted from atmo-discriminator/Discriminator/plot_spectrograms.py
# and plot_psd_histograms.py (ETH PMLR team07)
# ---------------------------------------------------------------------------

def compute_2d_psd(data):
    """2D log10-PSD of a 2D array (zero-mean before transform)."""
    data = data - np.mean(data)
    fft_shifted = np.fft.fftshift(np.fft.fft2(data))
    return np.log10(np.abs(fft_shifted) ** 2 + 1e-8)


def compute_psd_components(data):
    """Absolute FFT components of a 2D array (flattened)."""
    data = data - np.mean(data)
    return np.abs(np.fft.fft2(data)).flatten()


def compute_radial_psd(data):
    """Radially-averaged 1D log-PSD from a 2D array."""
    data = data - np.mean(data)
    fft_shifted = np.fft.fftshift(np.fft.fft2(data))
    psd_2d = np.abs(fft_shifted) ** 2
    H, W = psd_2d.shape
    cy, cx = H // 2, W // 2
    y_idx, x_idx = np.ogrid[:H, :W]
    r = np.round(np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)).astype(int)
    max_r = min(cy, cx)
    radial = np.full(max_r, np.nan)
    for ri in range(max_r):
        mask = r == ri
        if mask.any():
            radial[ri] = psd_2d[mask].mean()
    return np.log10(radial + 1e-10)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_times(nc_path):
    """Read the 'time' dimension of a NetCDF file as a pd.DatetimeIndex."""
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path, decode_times=True)
        raw = ds.time.values
        ds.close()
        if hasattr(raw[0], "strftime"):
            return pd.DatetimeIndex([pd.Timestamp(str(t)) for t in raw])
        return pd.DatetimeIndex(raw)
    except Exception:
        from netCDF4 import Dataset as NC4, num2date
        with NC4(nc_path) as ds:
            tvar = ds.variables["time"]
            cft  = num2date(tvar[:], tvar.units, getattr(tvar, "calendar", "standard"))
        return pd.DatetimeIndex([
            pd.Timestamp(t.year, t.month, t.day, t.hour, t.minute, t.second) for t in cft
        ])


def load_raw_fields(nc_path, time_indices):
    """Load un-normalized fields for given integer time indices.

    Returns (N, 4, lat, lon) float32 numpy array.
    """
    from netCDF4 import Dataset as NC4
    with NC4(nc_path) as ds:
        arrays = []
        for var in VARS:
            v = ds.variables[var]
            # raw dims are (time, longitude, latitude); transpose → (time, lat, lon)
            arr = np.asarray(v[time_indices, :, :], dtype=np.float32)
            arr = np.transpose(arr, (0, 2, 1))
            arrays.append(arr)
    return np.stack(arrays, axis=1)  # (N, 4, lat, lon)


def build_overlapping_indices(era5_path, forecast_path, n, seed=0):
    """Find up to n time indices that exist in both ERA5 and the forecast.

    Returns (era5_indices, forecast_indices) lists of integers.
    """
    era5_times = _read_times(era5_path)
    fc_times   = _read_times(forecast_path)
    max_dt = pd.Timedelta(hours=6)

    era5_idx_list, fc_idx_list = [], []
    for fi, ft in enumerate(fc_times):
        delta = np.abs(era5_times - ft)
        ei = int(delta.argmin())
        if delta[ei] <= max_dt:
            era5_idx_list.append(ei)
            fc_idx_list.append(fi)

    # Deterministic subsample
    total = len(era5_idx_list)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)[:min(n, total)]
    return [era5_idx_list[i] for i in perm], [fc_idx_list[i] for i in perm]

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_psd_heatmaps(fields_by_source, plots_dir, run_tag):
    """4 rows (variables) × 3 cols (ERA5, Pangu, GraphCast) mean log-PSD heatmaps."""
    sources = ["era5", "pangu", "graphcast"]
    n_vars  = len(VARS)

    # Pre-compute mean 2D PSDs
    mean_psds = {}  # {source: {vi: (H, W) mean log-PSD}}
    for src, fields in fields_by_source.items():
        mean_psds[src] = {}
        for vi in range(n_vars):
            psds = np.stack([compute_2d_psd(fields[ti, vi]) for ti in range(len(fields))])
            mean_psds[src][vi] = psds.mean(axis=0)

    fig, axes = plt.subplots(n_vars, len(sources), figsize=(5 * len(sources), 4 * n_vars))

    for vi in range(n_vars):
        # Shared color range across sources for this variable
        all_vals = np.concatenate([mean_psds[s][vi].flatten() for s in sources])
        vmin, vmax = np.nanpercentile(all_vals, 5), np.nanpercentile(all_vals, 95)

        for si, src in enumerate(sources):
            ax = axes[vi, si]
            im = ax.imshow(mean_psds[src][vi], cmap="magma", vmin=vmin, vmax=vmax,
                           aspect="auto", origin="lower")
            ax.set_title(f"{SOURCE_LABELS[src]}\n{VAR_LABELS[vi]}", fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Mean 2D Log-PSD: ERA5 vs 24h Forecasts\nRun: {run_tag}", fontsize=13)
    fig.tight_layout()
    out = plots_dir / f"spectra_{run_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_radial_psd(fields_by_source, plots_dir, run_tag):
    """Per-channel radially-averaged 1D PSD curves for all three sources."""
    sources = ["era5", "pangu", "graphcast"]
    n_vars  = len(VARS)

    # Pre-compute mean radial PSD for each source × variable
    mean_radials = {}
    for src, fields in fields_by_source.items():
        mean_radials[src] = {}
        for vi in range(n_vars):
            radials = np.stack([compute_radial_psd(fields[ti, vi]) for ti in range(len(fields))])
            mean_radials[src][vi] = np.nanmean(radials, axis=0)

    n_freqs = len(next(iter(mean_radials.values()))[0])
    k = np.arange(n_freqs)

    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4))

    for vi, ax in enumerate(axes):
        for src in sources:
            ax.plot(k, mean_radials[src][vi],
                    color=SOURCE_COLORS[src], label=SOURCE_LABELS[src], linewidth=1.5)
        ax.set_xlabel("Radial wavenumber k")
        ax.set_ylabel("Mean log10 PSD")
        ax.set_title(VAR_LABELS[vi])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(f"Radially-Averaged PSD: ERA5 vs 24h Forecasts\nRun: {run_tag}", fontsize=12)
    fig.tight_layout()
    out = plots_dir / f"spectra_radial_{run_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """CLI entry point: load matched ERA5/forecast fields and save 2D + radial PSD plots."""
    parser = argparse.ArgumentParser(description="Spectral PSD comparison: ERA5 vs forecasts.")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of time steps to average over (default: 100).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for p in [ERA5_PATH, PANGU_PATH, GRAPHCAST_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    rng_seed = args.seed

    print("Building overlapping time indices...")
    print("  Pangu:")
    era5_p_idx, pangu_idx = build_overlapping_indices(
        ERA5_PATH, PANGU_PATH, args.n_samples, rng_seed)
    print(f"    Using {len(pangu_idx)} samples.")
    print("  GraphCast:")
    era5_g_idx, gc_idx = build_overlapping_indices(
        ERA5_PATH, GRAPHCAST_PATH, args.n_samples, rng_seed)
    print(f"    Using {len(gc_idx)} samples.")

    # Use Pangu-aligned ERA5 as the representative ERA5 source
    n = min(len(pangu_idx), len(gc_idx), len(era5_p_idx))
    era5_p_idx, pangu_idx = era5_p_idx[:n], pangu_idx[:n]
    era5_g_idx, gc_idx    = era5_g_idx[:n], gc_idx[:n]

    print(f"\nLoading raw fields for {n} time steps per source...")
    fields_by_source = {
        "era5":      load_raw_fields(ERA5_PATH,      era5_p_idx),
        "pangu":     load_raw_fields(PANGU_PATH,     pangu_idx),
        "graphcast": load_raw_fields(GRAPHCAST_PATH, gc_idx),
    }

    run_tag   = f"n{n}_seed{rng_seed}"
    plots_dir = Path("plots/real_vs_forecast")
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating spectral plots...")
    plot_psd_heatmaps(fields_by_source, plots_dir, run_tag)
    plot_radial_psd(fields_by_source, plots_dir, run_tag)
    print("Done.")


if __name__ == "__main__":
    main()

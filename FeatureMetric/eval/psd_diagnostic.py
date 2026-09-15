"""Radially-averaged power spectrum: ERA5 vs forecasts (classical baseline).

ML weather forecasts systematically lose high-wavenumber power relative to ERA5
(they are smoothed). This script measures that deficit directly:
  - per channel, radially-averaged 2-D power spectrum E(k) for ERA5/Pangu/GraphCast,
  - the ratio E_forecast(k) / E_era5(k), and the wavenumber where it drops below a
    threshold (informs the apply_spectral_lowpass cutoff used in training),
  - an optional per-sample high-k band-power AUC as a baseline number to beat.

It uses RAW (un-normalized, un-padded) interior fields read via AtmosphereDataset.

Usage (local):
    /opt/miniconda3/envs/pmlr/bin/python eval/psd_diagnostic.py --local
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.dataset import AtmosphereDataset

CHANNELS = ["T2M", "U10", "V10", "MSL"]

LOCAL_ERA5_PATH = Path("data/test_data_local.nc")
LOCAL_PANGU_PATH = Path("data/pangu_surface_2020_lead24h.nc")
LOCAL_GRAPHCAST_PATH = Path("data/graphcast_surface_2020_lead24h.nc")

CLUSTER_ERA5_PATH = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
CLUSTER_PANGU_PATH = Path("/cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/pangu_surface_2020_lead24h.nc")
CLUSTER_GRAPHCAST_PATH = Path("/cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/graphcast_surface_2020_lead24h.nc")

SOURCE_COLORS = {"era5": "#4CAF50", "pangu": "#9C27B0", "graphcast": "#FF9800"}


def radial_psd(fields):
    """Radially-averaged 2-D power spectrum, per channel.

    fields: (N, C, H, W) raw interior arrays. Applies a Hann window in latitude
    (non-periodic) before the FFT; longitude is left periodic. Returns
    ``(k_bin_centers, E)`` where E has shape (C, n_bins) — mean over samples.
    """
    N, C, H, W = fields.shape
    win = np.hanning(H)[None, None, :, None]  # latitude window
    f = fields * win

    spec = np.fft.fft2(f, axes=(-2, -1))
    power = (np.abs(spec) ** 2) / (H * W)

    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    kr = np.sqrt(fy ** 2 + fx ** 2)

    n_bins = min(H, W) // 2
    bins = np.linspace(0, kr.max(), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    which = np.digitize(kr.ravel(), bins) - 1
    which = np.clip(which, 0, n_bins - 1)

    E = np.zeros((C, n_bins))
    counts = np.bincount(which, minlength=n_bins).astype(float)
    counts[counts == 0] = 1.0
    pmean = power.mean(axis=0)  # (C, H, W)
    for c in range(C):
        sums = np.bincount(which, weights=pmean[c].ravel(), minlength=n_bins)
        E[c] = sums / counts
    return centers, E


def read_interior(ds_path, stats, indices):
    """Read RAW interior (un-normalized, un-padded) fields at given indices.

    Returns (len(indices), 4, H, W). Uses AtmosphereDataset.read_raw, which yields
    the native (4, H, W) field in real units.
    """
    ds = AtmosphereDataset(ds_path, split="all", stats=stats, lazy=True)
    arrs = [ds.read_raw(i) for i in indices]
    return np.stack(arrs, axis=0).astype(np.float32)


def _read_times(nc_path):
    import xarray as xr
    with xr.open_dataset(nc_path, decode_times=True) as ds:
        raw = ds.time.values
    if hasattr(raw[0], "strftime"):
        return pd.DatetimeIndex([pd.Timestamp(str(t)) for t in raw])
    return pd.DatetimeIndex(raw)


def paired_indices(era5_path, fc_path):
    """Exact valid-time match: returns aligned (era5_idx, fc_idx)."""
    era5_times, fc_times = _read_times(era5_path), _read_times(fc_path)
    era5_map = {t: i for i, t in enumerate(era5_times)}
    e, f = [], []
    for fi, ft in enumerate(fc_times):
        ei = era5_map.get(ft)
        if ei is not None:
            e.append(ei)
            f.append(fi)
    return e, f


def main():
    """CLI entry: compute and plot PSDs and report the high-k deficit + baseline AUC."""
    parser = argparse.ArgumentParser(description="ERA5 vs forecast radial PSD diagnostic.")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratio-threshold", type=float, default=0.9,
                        help="Report the wavenumber where E_fc/E_era5 first drops below this.")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Holds data_mean/std.npy; plots go under <dir>/plots/psd.")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Prefix added to the output plot filename (e.g. june29).")
    parser.add_argument("--era5-path", type=str, default=None)
    parser.add_argument("--pangu-path", type=str, default=None)
    parser.add_argument("--graphcast-path", type=str, default=None)
    args = parser.parse_args()

    era5_path = Path(args.era5_path) if args.era5_path else (LOCAL_ERA5_PATH if args.local else CLUSTER_ERA5_PATH)
    pangu_path = Path(args.pangu_path) if args.pangu_path else (LOCAL_PANGU_PATH if args.local else CLUSTER_PANGU_PATH)
    graphcast_path = Path(args.graphcast_path) if args.graphcast_path else (LOCAL_GRAPHCAST_PATH if args.local else CLUSTER_GRAPHCAST_PATH)

    stats_dir = Path(args.output_dir)
    stats = (np.load(stats_dir / "data_mean.npy"), np.load(stats_dir / "data_std.npy"))
    rng = np.random.default_rng(args.seed)

    fields = {}
    for source, fc_path in [("pangu", pangu_path), ("graphcast", graphcast_path)]:
        e_idx, f_idx = paired_indices(era5_path, fc_path)
        if len(f_idx) > args.n_samples:
            sel = rng.permutation(len(f_idx))[:args.n_samples]
            e_idx = [e_idx[i] for i in sel]
            f_idx = [f_idx[i] for i in sel]
        if "era5" not in fields:  # reuse a single ERA5 draw (paired to pangu)
            fields["era5"] = read_interior(era5_path, stats, e_idx)
        fields[source] = read_interior(fc_path, stats, f_idx)
        print(f"{source}: {len(f_idx)} paired samples")

    k, E_era5 = radial_psd(fields["era5"])
    psd = {"era5": E_era5}
    for source in ("pangu", "graphcast"):
        _, psd[source] = radial_psd(fields[source])

    # Report the high-k deficit per channel.
    print("\n" + "=" * 70)
    print("HIGH-WAVENUMBER DEFICIT (E_forecast / E_era5)")
    print("=" * 70)
    for source in ("pangu", "graphcast"):
        ratio = psd[source] / (E_era5 + 1e-30)
        print(f"  {source}:")
        for c, ch in enumerate(CHANNELS):
            below = np.where(ratio[c] < args.ratio_threshold)[0]
            kc = k[below[0]] if below.size else float("nan")
            print(f"    {ch:<4}  ratio<{args.ratio_threshold} first at k={kc:.4f} cyc/px "
                  f"(k_nyq=0.5); ratio at k_max={ratio[c, -1]:.3f}")

    # Baseline: per-sample high-k band power AUC (real vs forecast), T2M channel.
    def per_sample_highk(fields_arr, channel=0, k_lo=0.25):
        out = []
        for n in range(fields_arr.shape[0]):
            kk, E = radial_psd(fields_arr[n:n + 1, channel:channel + 1])
            out.append(E[0, kk >= k_lo].sum())
        return np.array(out)

    print("\n  Baseline separability (per-sample high-k band power, k>=0.25, T2M):")
    print("    AUC = P(ERA5 high-k power > forecast high-k power); >0.5 = separable.")
    e_bp = per_sample_highk(fields["era5"])
    for source in ("pangu", "graphcast"):
        f_bp = per_sample_highk(fields[source])
        # forecasts have LESS high-k power; orient AUC so >0.5 means separable.
        u, _ = mannwhitneyu(e_bp, f_bp, alternative="greater")
        a = float(u) / (len(e_bp) * len(f_bp))
        print(f"    {source}: AUC={a:.3f}")
    print("=" * 70)

    # Plots: log-log PSD per channel + ratio.
    plots_dir = stats_dir / "plots" / "psd"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for c, ch in enumerate(CHANNELS):
        ax = axes[0, c]
        for source in ("era5", "pangu", "graphcast"):
            ax.loglog(k[1:], psd[source][c, 1:], color=SOURCE_COLORS[source], label=source)
        ax.set_title(f"PSD — {ch}")
        ax.set_xlabel("wavenumber k (cyc/px)")
        ax.set_ylabel("E(k)")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)

        axr = axes[1, c]
        for source in ("pangu", "graphcast"):
            axr.semilogx(k[1:], (psd[source][c] / (E_era5[c] + 1e-30))[1:],
                         color=SOURCE_COLORS[source], label=f"{source}/era5")
        axr.axhline(1.0, color="k", lw=1, ls="--")
        axr.axhline(args.ratio_threshold, color="r", lw=0.8, ls=":")
        axr.set_title(f"ratio — {ch}")
        axr.set_xlabel("wavenumber k (cyc/px)")
        axr.set_ylabel("E_fc / E_era5")
        axr.set_ylim(0, 1.3)
        axr.grid(True, which="both", alpha=0.2)
        axr.legend(fontsize=8)
    fig.suptitle("Radial power spectrum: ERA5 vs 24h forecasts", fontsize=14)
    fig.tight_layout()
    prefix = f"{args.run_tag}_" if args.run_tag else ""
    out = plots_dir / f"{prefix}psd_era5_vs_forecast.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

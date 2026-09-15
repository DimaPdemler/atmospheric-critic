"""Evaluate the contrastive realism metric on ERA5 vs 24h forecasts (valid-time PAIRED).

Unlike the unpaired MMD/FID eval (eval_real_vs_forecast.py), this matches each
forecast valid time to the exact ERA5 timestamp, so the weather state cancels and
a paired test has far more power. For ERA5 and each forecast it computes:
  - realism score r = realism_head(embed(x))   (reference-free; higher = less real)
  - center distance d = ||normalize(embed(x)) - manifold_center||

Reports paired Wilcoxon (r_forecast - r_era5), AUC (Mann-Whitney) for r and d, and
saves a histogram + paired-difference plot.

Usage (local):
    /opt/miniconda3/envs/pmlr/bin/python eval/eval_realism_forecast.py --local
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.dataset import AtmosphereDataset
from utils.model_io import checkpoint_path
from utils.realism import RealismMetric

LOCAL_ERA5_PATH = Path("data/test_data_local.nc")
LOCAL_PANGU_PATH = Path("data/pangu_surface_2020_lead24h.nc")
LOCAL_GRAPHCAST_PATH = Path("data/graphcast_surface_2020_lead24h.nc")

CLUSTER_ERA5_PATH = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
CLUSTER_PANGU_PATH = Path("/cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/pangu_surface_2020_lead24h.nc")
CLUSTER_GRAPHCAST_PATH = Path("/cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/graphcast_surface_2020_lead24h.nc")

SOURCE_COLORS = {"era5": "#4CAF50", "pangu": "#9C27B0", "graphcast": "#FF9800"}


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
            cftimes = num2date(tvar[:], tvar.units, getattr(tvar, "calendar", "standard"))
        return pd.DatetimeIndex([
            pd.Timestamp(t.year, t.month, t.day, t.hour, t.minute, t.second)
            for t in cftimes
        ])


def build_paired_indices(era5_path, forecast_path):
    """Match each forecast valid time to the exact ERA5 timestamp.

    Returns (era5_idx, forecast_idx) aligned element-wise. Forecasts whose valid
    time is absent from ERA5 are dropped (reported).
    """
    era5_times = _read_times(era5_path)
    fc_times = _read_times(forecast_path)
    era5_map = {t: i for i, t in enumerate(era5_times)}
    era5_idx, fc_idx, dropped = [], [], 0
    for fi, ft in enumerate(fc_times):
        ei = era5_map.get(ft)
        if ei is None:
            dropped += 1
        else:
            era5_idx.append(ei)
            fc_idx.append(fi)
    print(f"  paired {len(fc_idx)} valid times ({dropped} forecast times had no exact ERA5 match)")
    return era5_idx, fc_idx


@torch.no_grad()
def score_loader(model, loader, device):
    """Return (realism_scores, center_distances) as 1-D numpy arrays for a loader."""
    r_all, d_all = [], []
    for img in loader:
        img = img.to(device, non_blocking=device.type == "cuda")
        z = model.embed(img)
        r_all.append(model.predict_realism(z).cpu())
        d_all.append(model.center_distance(z).cpu())
    return torch.cat(r_all).numpy(), torch.cat(d_all).numpy()


def auc(neg, pos):
    """AUC = P(score_pos > score_neg) via Mann-Whitney U (pos = forecast)."""
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    u, _ = mannwhitneyu(pos, neg, alternative="greater")
    return float(u) / (len(pos) * len(neg))


def _ci(vals, ci):
    """(lo, hi) percentile interval at confidence level ``ci`` (percent)."""
    lo_q, hi_q = (100 - ci) / 2.0, 100 - (100 - ci) / 2.0
    return float(np.percentile(vals, lo_q)), float(np.percentile(vals, hi_q))


def boot_mean_ci(x, n_boot, rng, ci):
    """Bootstrap CI on the mean of ``x`` (resample rows with replacement)."""
    if n_boot <= 0 or len(x) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return _ci(x[idx].mean(axis=1), ci)


def boot_auc_ci(neg, pos, n_boot, rng, ci):
    """Bootstrap CI on AUC (resample the two pools independently)."""
    if n_boot <= 0 or len(neg) == 0 or len(pos) == 0:
        return (float("nan"), float("nan"))
    vals = [
        auc(neg[rng.integers(0, len(neg), len(neg))], pos[rng.integers(0, len(pos), len(pos))])
        for _ in range(n_boot)
    ]
    return _ci(np.array(vals), ci)


def main():
    """CLI entry: load the realism model and score ERA5 vs forecasts (paired)."""
    parser = argparse.ArgumentParser(description="Paired ERA5-vs-forecast realism eval.")
    parser.add_argument("--local", action="store_true", help="Use local data paths.")
    parser.add_argument("--n-samples", type=int, default=500, help="Max paired samples per forecast source.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=200,
                        help="Bootstrap resamples for CIs on Δr / mean-score / AUC (0 disables).")
    parser.add_argument("--ci", type=float, default=95.0, help="Confidence level (percent).")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Prefix added to output plot filenames (e.g. june29).")
    parser.add_argument("--model-size", choices=["default", "twin"], default="twin")
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Dir holding the checkpoint + stats; plots go under <dir>/plots/realism_forecast.")
    parser.add_argument("--era5-path", type=str, default=None)
    parser.add_argument("--pangu-path", type=str, default=None)
    parser.add_argument("--graphcast-path", type=str, default=None)
    args = parser.parse_args()

    era5_path = Path(args.era5_path) if args.era5_path else (LOCAL_ERA5_PATH if args.local else CLUSTER_ERA5_PATH)
    pangu_path = Path(args.pangu_path) if args.pangu_path else (LOCAL_PANGU_PATH if args.local else CLUSTER_PANGU_PATH)
    graphcast_path = Path(args.graphcast_path) if args.graphcast_path else (LOCAL_GRAPHCAST_PATH if args.local else CLUSTER_GRAPHCAST_PATH)

    stats_dir = Path(args.output_dir)
    ckpt = checkpoint_path("realism", args.model_size, stats_dir, embed_dim=args.embed_dim)
    for p in [era5_path, pangu_path, graphcast_path, ckpt,
              stats_dir / "data_mean.npy", stats_dir / "data_std.npy"]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    model = RealismMetric(model_size=args.model_size, in_chans=4,
                          embed_dim=args.embed_dim, proj_dim=args.proj_dim).to(device)
    blob = torch.load(ckpt, map_location=device)
    model.load_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob else blob)
    model.eval()
    print(f"Loaded realism checkpoint: {ckpt} (embed_dim={model.embed_dim})")

    stats = (np.load(stats_dir / "data_mean.npy"), np.load(stats_dir / "data_std.npy"))
    era5_lazy = not args.local
    era5_ds = AtmosphereDataset(era5_path, split="all", stats=stats, lazy=era5_lazy)
    fc_datasets = {
        "pangu": AtmosphereDataset(pangu_path, split="all", stats=stats, lazy=False),
        "graphcast": AtmosphereDataset(graphcast_path, split="all", stats=stats, lazy=False),
    }
    fc_paths = {"pangu": pangu_path, "graphcast": graphcast_path}

    loader_kw = dict(batch_size=args.batch_size, shuffle=False,
                     num_workers=args.num_workers, pin_memory=device.type == "cuda")

    scores = {}  # source -> dict(r, d) ; plus paired era5 ref per source
    for source, fc_ds in fc_datasets.items():
        print(f"\n=== {source} ===")
        era5_idx, fc_idx = build_paired_indices(era5_path, fc_paths[source])
        # Cap to n_samples while keeping the pairing aligned.
        if len(fc_idx) > args.n_samples:
            sel = rng.permutation(len(fc_idx))[:args.n_samples]
            era5_idx = [era5_idx[i] for i in sel]
            fc_idx = [fc_idx[i] for i in sel]

        era5_loader = DataLoader(Subset(era5_ds, era5_idx), **loader_kw)
        fc_loader = DataLoader(Subset(fc_ds, fc_idx), **loader_kw)
        r_era5, d_era5 = score_loader(model, era5_loader, device)
        r_fc, d_fc = score_loader(model, fc_loader, device)
        scores[source] = dict(r_era5=r_era5, d_era5=d_era5, r_fc=r_fc, d_fc=d_fc)

    # ---- Report (with bootstrap uncertainties) ------------------------------
    for source, s in scores.items():
        dr = s["r_fc"] - s["r_era5"]
        s["dr"] = dr
        s["dr_mean"] = float(dr.mean())
        try:
            _, s["p"] = wilcoxon(dr)
        except ValueError:
            s["p"] = float("nan")
        s["auc_r"] = auc(s["r_era5"], s["r_fc"])
        s["auc_d"] = auc(s["d_era5"], s["d_fc"])
        s["ci_dr"] = boot_mean_ci(dr, args.n_boot, rng, args.ci)
        s["ci_rfc"] = boot_mean_ci(s["r_fc"], args.n_boot, rng, args.ci)
        s["ci_dfc"] = boot_mean_ci(s["d_fc"], args.n_boot, rng, args.ci)
        s["ci_auc_r"] = boot_auc_ci(s["r_era5"], s["r_fc"], args.n_boot, rng, args.ci)
        s["ci_auc_d"] = boot_auc_ci(s["d_era5"], s["d_fc"], args.n_boot, rng, args.ci)

    print("\n" + "=" * 78)
    print("REALISM METRIC — ERA5 vs 24h FORECAST (valid-time PAIRED)")
    print("=" * 78)
    print(f"  {'Source':<12} {'N':>5} {'r(ERA5)':>10} {'r(fcst)':>10} "
          f"{'Δr':>9} {'Wilcoxon p':>12} {'AUC(r)':>8} {'AUC(d)':>8}")
    print("  " + "-" * 80)
    for source, s in scores.items():
        print(f"  {source:<12} {len(s['dr']):>5} {s['r_era5'].mean():>10.4f} {s['r_fc'].mean():>10.4f} "
              f"{s['dr_mean']:>9.4f} {s['p']:>12.2e} {s['auc_r']:>8.3f} {s['auc_d']:>8.3f}")
    if args.n_boot > 0:
        print(f"\n  {args.ci:g}% bootstrap CIs ({args.n_boot} resamples):")
        for source, s in scores.items():
            print(f"    {source:<10} Δr [{s['ci_dr'][0]:+.4f}, {s['ci_dr'][1]:+.4f}]   "
                  f"AUC(r) [{s['ci_auc_r'][0]:.3f}, {s['ci_auc_r'][1]:.3f}]   "
                  f"AUC(d) [{s['ci_auc_d'][0]:.3f}, {s['ci_auc_d'][1]:.3f}]")
    print("=" * 78)
    print("  Success: Δr CI above 0, AUC CI above 0.5  →  forecasts score as less realistic.")

    # ---- Plots --------------------------------------------------------------
    plots_dir = stats_dir / "plots" / "realism_forecast"
    plots_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.run_tag}_" if args.run_tag else ""
    tag = f"{prefix}{args.model_size}{f'_d{args.embed_dim}' if args.embed_dim else ''}_n{args.n_samples}_seed{args.seed}"
    fc_sources = list(scores.keys())
    ref = next(iter(scores.values()))  # representative ERA5 pool for baseline

    # Figure 1: distributions (score histograms + paired Δr).
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(ref["r_era5"], bins=40, alpha=0.6, color=SOURCE_COLORS["era5"], label="ERA5", density=True)
    for source in fc_sources:
        axes[0].hist(scores[source]["r_fc"], bins=40, alpha=0.5,
                     color=SOURCE_COLORS[source], label=source, density=True)
    axes[0].set_xlabel("realism score r (higher = less realistic)")
    axes[0].set_ylabel("density"); axes[0].set_title("Realism-score distributions"); axes[0].legend()
    for source in fc_sources:
        axes[1].hist(scores[source]["dr"], bins=40, alpha=0.5,
                     color=SOURCE_COLORS[source], label=f"{source} Δr", density=True)
    axes[1].axvline(0, color="k", lw=1, ls="--")
    axes[1].set_xlabel("paired Δr = r(forecast) − r(ERA5, same valid time)")
    axes[1].set_ylabel("density"); axes[1].set_title("Paired realism difference"); axes[1].legend()
    fig.suptitle(f"Realism metric: ERA5 vs forecast (paired)\n{tag}")
    fig.tight_layout()
    out1 = plots_dir / f"realism_dist_{tag}.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out1}")

    # Figure 2: mean-score bar charts WITH bootstrap CIs + ERA5 baseline band.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(fc_sources))
    for ax, key_fc, key_era5, ci_key, title in [
        (axes[0], "r_fc", "r_era5", "ci_rfc", "Realism score r"),
        (axes[1], "d_fc", "d_era5", "ci_dfc", "Center distance d"),
    ]:
        means = [scores[s][key_fc].mean() for s in fc_sources]
        lo = [max(0.0, scores[s][key_fc].mean() - scores[s][ci_key][0]) for s in fc_sources]
        hi = [max(0.0, scores[s][ci_key][1] - scores[s][key_fc].mean()) for s in fc_sources]
        ax.bar(x, means, width=0.5, color=[SOURCE_COLORS[s] for s in fc_sources],
               yerr=np.array([lo, hi]), capsize=4, edgecolor="white")
        # ERA5 baseline: dashed mean line + shaded bootstrap band.
        e_mean = ref[key_era5].mean()
        e_ci = boot_mean_ci(ref[key_era5], args.n_boot, rng, args.ci)
        if np.isfinite(e_ci[0]):
            ax.axhspan(e_ci[0], e_ci[1], color=SOURCE_COLORS["era5"], alpha=0.15)
        ax.axhline(e_mean, color=SOURCE_COLORS["era5"], ls="--", lw=1.5, label="ERA5 baseline")
        for xi, m in zip(x, means):
            ax.text(xi, m, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(fc_sources)
        ax.set_ylabel(title); ax.set_title(f"{title}: ERA5 vs forecast")
        ax.grid(True, axis="y", alpha=0.3); ax.legend()
    fig.suptitle(f"Realism metric — mean score ± {args.ci:g}% bootstrap CI\n{tag}")
    fig.tight_layout()
    out2 = plots_dir / f"realism_bars_{tag}.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()

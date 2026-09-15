"""Evaluate latent-space FID and MMD between ERA5 reanalysis and 24h forecast products.

For each of two self-supervised encoders (MAE-twin, I-JEPA-twin) this script:
  1. Pairs ERA5 samples to forecast valid times (nearest-neighbour in time).
  2. Extracts latent features from each encoder.
  3. Computes Fréchet Distance and MMD between ERA5 and each forecast (Pangu, GraphCast).
  4. Saves a metric bar chart and a joint-PCA scatter plot.

Usage (local, smoke test):
    /opt/miniconda3/envs/pmlr/bin/python eval/eval_real_vs_forecast.py --local --n-samples 50

Usage (local, full):
    /opt/miniconda3/envs/pmlr/bin/python eval/eval_real_vs_forecast.py --local --n-samples 500
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import scipy.linalg
import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from utils.dataset import AtmosphereDataset
from utils.model_io import build_model, checkpoint_path, load_model_checkpoint
from utils.features import extract_features_for_loader
from utils.temporal import IN_CHANS_BY_MODE, compose_temporal_input, derive_delta_steps

# ---------------------------------------------------------------------------
# Default paths (overridable via CLI flags below)
# ---------------------------------------------------------------------------

LOCAL_ERA5_PATH      = Path("data/test_data_local.nc")
LOCAL_PANGU_PATH     = Path("data/pangu_surface_2020_lead24h.nc")
LOCAL_GRAPHCAST_PATH = Path("data/graphcast_surface_2020_lead24h.nc")

CLUSTER_ERA5_PATH      = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
CLUSTER_PANGU_PATH     = Path("/cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/pangu_surface_2020_lead24h.nc")
CLUSTER_GRAPHCAST_PATH = Path("/cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/graphcast_surface_2020_lead24h.nc")

MODEL_COLORS  = {"mae": "#2196F3", "ijepa": "#FF5722"}
MODEL_LABELS  = {"mae": "MAE (twin)", "ijepa": "I-JEPA (twin)"}
SOURCE_COLORS = {"era5": "#4CAF50", "pangu": "#9C27B0", "graphcast": "#FF9800"}
SOURCE_LABELS = {"era5": "ERA5", "pangu": "Pangu-24h", "graphcast": "GraphCast-24h"}
SOURCE_HATCHES = {"pangu": "", "graphcast": "//"}

# ---------------------------------------------------------------------------
# Metric functions (mirrors eval_distances.py to keep this script self-contained)
# ---------------------------------------------------------------------------

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet (FID-style) distance between two Gaussians given their means/covariances."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print(f"Warning: Imaginary FD component {np.max(np.abs(covmean.imag)):.4f}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def mmd_rbf(X, Y, gamma=None):
    """Unbiased squared MMD between feature sets ``X`` and ``Y`` with an RBF kernel.

    Uses the median-distance heuristic for the bandwidth when ``gamma`` is None.
    """
    XX = torch.cdist(X, X, p=2) ** 2
    YY = torch.cdist(Y, Y, p=2) ** 2
    XY = torch.cdist(X, Y, p=2) ** 2
    if gamma is None:
        diag_mask = ~torch.eye(XX.shape[0], dtype=torch.bool)
        dists = torch.cat([XX[diag_mask], YY[diag_mask], XY.flatten()])
        gamma = 1.0 / (2.0 * torch.median(dists).item() + 1e-6)
    K_XX = torch.exp(-gamma * XX)
    K_YY = torch.exp(-gamma * YY)
    K_XY = torch.exp(-gamma * XY)
    N, M = X.shape[0], Y.shape[0]
    K_XX.fill_diagonal_(0)
    K_YY.fill_diagonal_(0)
    return float(K_XX.sum() / (N * (N - 1)) + K_YY.sum() / (M * (M - 1)) - 2 * K_XY.mean())

# ---------------------------------------------------------------------------
# Time-aligned index building
# ---------------------------------------------------------------------------

def _read_times(nc_path):
    """Read the 'time' dimension of a NetCDF file as a pd.DatetimeIndex."""
    try:
        import xarray as xr
        ds = xr.open_dataset(nc_path, decode_times=True)
        raw = ds.time.values
        ds.close()
        if hasattr(raw[0], "strftime"):  # cftime objects
            return pd.DatetimeIndex([pd.Timestamp(str(t)) for t in raw])
        return pd.DatetimeIndex(raw)
    except Exception:
        # Fallback: netCDF4 directly
        from netCDF4 import Dataset as NC4, num2date
        with NC4(nc_path) as ds:
            tvar = ds.variables["time"]
            cftimes = num2date(tvar[:], tvar.units, getattr(tvar, "calendar", "standard"))
        return pd.DatetimeIndex([
            pd.Timestamp(t.year, t.month, t.day, t.hour, t.minute, t.second)
            for t in cftimes
        ])


def build_era5_ref_pool(era5_path, forecast_path, max_dt_hours=6, prior_hours=None,
                        restrict_date_range=True):
    """Build an ERA5 pool decoupled from forecast valid-time pairing.

    Returns ERA5 indices for ERA5 timestamps at the forecast's hours-of-day. If
    `restrict_date_range` is True (default — used for the forecast-side reference),
    the pool is further restricted to the forecast date range. If False (used for
    the baseline when --baseline-pool=all-years), the pool spans every year in
    the ERA5 file, giving a much larger sample for a tighter noise-floor estimate.

    When `prior_hours` is set (temporal eval), each ref index t is also matched
    to the nearest ERA5 index at t − prior_hours. Refs whose prior is > max_dt
    from any ERA5 timestamp are dropped.

    Returns:
        (ref_idx, prior_idx)  — prior_idx is None when prior_hours is None.
    """
    era5_times = _read_times(era5_path)
    fc_times   = _read_times(forecast_path)

    fc_hours   = sorted({int(t.hour) for t in fc_times})
    hour_match = np.array([int(t.hour) in fc_hours for t in era5_times])

    if restrict_date_range:
        fc_start, fc_end = fc_times.min(), fc_times.max()
        in_range = (era5_times >= fc_start) & (era5_times <= fc_end)
        mask     = in_range & hour_match
        scope    = f"{fc_times[0].date()}–{fc_times[-1].date()}"
    else:
        mask  = hour_match
        scope = f"{era5_times[0].date()}–{era5_times[-1].date()} (all years)"

    ref_idx = np.where(mask)[0].tolist()

    if prior_hours is None:
        print(f"  ERA5 pool: {len(ref_idx)} samples (range {scope}, hours {fc_hours})")
        return ref_idx, None

    max_dt = pd.Timedelta(hours=max_dt_hours)
    prior_offset = pd.Timedelta(hours=prior_hours)
    keep_ref, keep_prior = [], []
    skipped = 0
    for i in ref_idx:
        target = era5_times[i] - prior_offset
        delta = np.abs(era5_times - target)
        j = int(delta.argmin())
        if delta[j] <= max_dt:
            keep_ref.append(i)
            keep_prior.append(j)
        else:
            skipped += 1
    print(f"  ERA5 pool: {len(keep_ref)} samples "
          f"(range {scope}, hours {fc_hours}, Δt={prior_hours}h prior; {skipped} dropped)")
    return keep_ref, keep_prior


def build_forecast_indices(era5_path, forecast_path, max_dt_hours=6, prior_hours=None):
    """All forecast indices, with matched ERA5 prior indices for temporal mode.

    The forecast pool itself never depends on per-sample ERA5 pairing — it's
    the full set of forecast samples. For temporal mode, the prior side is
    still ERA5 at valid_time − Δt, since Pangu/GraphCast take ERA5 as input.
    Forecasts whose prior is > max_dt from any ERA5 timestamp are dropped.
    """
    fc_times = _read_times(forecast_path)
    if prior_hours is None:
        print(f"  Forecast pool: {len(fc_times)} samples")
        return list(range(len(fc_times))), None

    era5_times = _read_times(era5_path)
    max_dt = pd.Timedelta(hours=max_dt_hours)
    prior_offset = pd.Timedelta(hours=prior_hours)
    keep_fc, keep_prior = [], []
    skipped = 0
    for fi, ft in enumerate(fc_times):
        target = ft - prior_offset
        delta = np.abs(era5_times - target)
        ei = int(delta.argmin())
        if delta[ei] <= max_dt:
            keep_fc.append(fi)
            keep_prior.append(ei)
        else:
            skipped += 1
    print(f"  Forecast pool: {len(keep_fc)} samples "
          f"(Δt={prior_hours}h prior; {skipped} dropped)")
    return keep_fc, keep_prior


def cap_indices(idx, prior_idx, n, rng):
    """Shuffle and cap an index list (plus optional matched prior list) to min(n, len)."""
    total = len(idx)
    if n >= total:
        return idx, prior_idx
    perm = rng.permutation(total)[:n].tolist()
    capped_idx   = [idx[i] for i in perm]
    capped_prior = [prior_idx[i] for i in perm] if prior_idx is not None else None
    return capped_idx, capped_prior


def split_indices(idx, prior_idx, rng):
    """Random 50/50 disjoint split of an index list (plus optional matched prior list).

    Returns ((a_idx, a_prior), (b_idx, b_prior)). Each half has size len(idx)//2.
    """
    total = len(idx)
    half  = total // 2
    perm  = rng.permutation(total).tolist()
    a, b  = perm[:half], perm[half:2 * half]
    a_idx = [idx[i] for i in a]
    b_idx = [idx[i] for i in b]
    if prior_idx is None:
        return (a_idx, None), (b_idx, None)
    return (a_idx, [prior_idx[i] for i in a]), (b_idx, [prior_idx[i] for i in b])


class TemporalPairDataset(torch.utils.data.Dataset):
    """Eval-time wrapper that builds (X_{t-Δt}, X_t) composite inputs from two
    source datasets at matched absolute indices.

    The prior side always comes from ERA5 (`prior_ds`). The present side comes
    from `present_ds`, which may be ERA5 (real) or a forecast file (fake).
    Both source datasets are constructed with `split="all"` so absolute indices
    are valid.
    """

    def __init__(self, prior_ds, present_ds, prior_idx, present_idx,
                 mode, abs_stats, diff_stats=None):
        if len(prior_idx) != len(present_idx):
            raise ValueError(
                f"prior_idx (len {len(prior_idx)}) and present_idx (len {len(present_idx)}) must match"
            )
        if mode == "none":
            raise ValueError("TemporalPairDataset is only for temporal modes (diff/concat/phase).")
        self.prior_ds = prior_ds
        self.present_ds = present_ds
        self.prior_idx = list(prior_idx)
        self.present_idx = list(present_idx)
        self.mode = mode
        self.abs_mean = abs_stats[0]
        self.abs_std  = abs_stats[1]
        self.diff_mean = diff_stats[0] if diff_stats is not None else None
        self.diff_std  = diff_stats[1] if diff_stats is not None else None

    def __len__(self):
        """Number of (prior, present) pairs in this dataset."""
        return len(self.prior_idx)

    def __getitem__(self, i):
        """Read the raw prior/present pair at index ``i`` and compose the temporal input."""
        prior_raw   = self.prior_ds.read_raw(self.prior_idx[i])
        present_raw = self.present_ds.read_raw(self.present_idx[i])
        sample = compose_temporal_input(
            present_raw, prior_raw, self.mode,
            self.abs_mean, self.abs_std, self.diff_mean, self.diff_std,
        )
        return torch.from_numpy(sample)

# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def compute_distances(z_real, z_fake):
    """z_real, z_fake: (N, D) torch tensors (on CPU).
    Returns dict with 'fid' and 'mmd' keys."""
    z_real_np = z_real.numpy()
    z_fake_np = z_fake.numpy()
    mu_r, sigma_r = np.mean(z_real_np, axis=0), np.cov(z_real_np, rowvar=False)
    mu_f, sigma_f = np.mean(z_fake_np, axis=0), np.cov(z_fake_np, rowvar=False)
    fid = calculate_frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
    mmd = mmd_rbf(z_real, z_fake)
    return {"fid": fid, "mmd": mmd}


# ---------------------------------------------------------------------------
# Uncertainty (bootstrap CIs on forecast bars, repeated-split baseline band).
# Shared home — the SFNO eval imports these.
# ---------------------------------------------------------------------------

def _nanmean(vals):
    """Mean over finite entries, or NaN if none are finite (avoids RuntimeWarnings)."""
    vals = np.asarray(vals, dtype=float)
    finite = vals[np.isfinite(vals)]
    return float(finite.mean()) if finite.size else float("nan")


def _percentile_ci(vals, ci):
    """(lo, hi) percentile interval over finite entries; (nan, nan) if all NaN."""
    vals = np.asarray(vals, dtype=float)
    if not np.isfinite(vals).any():
        return (float("nan"), float("nan"))
    lo_q, hi_q = (100 - ci) / 2.0, 100 - (100 - ci) / 2.0
    return (float(np.nanpercentile(vals, lo_q)), float(np.nanpercentile(vals, hi_q)))


def bootstrap_distances(a, b, dist_fn, n_boot, rng):
    """Sampling distribution of the distance between feature pools ``a`` and ``b``.

    Each iteration resamples the rows of ``a`` and ``b`` *independently, with
    replacement* (sizes preserved) and recomputes the distance — the standard
    bootstrap estimate of how much the FID/MMD point estimate would vary under a
    different draw of the same size. Returns ``{metric: [values]}`` of length
    ``n_boot``.
    """
    na, nb = a.shape[0], b.shape[0]
    out = {"fid": [], "mmd": []}
    for _ in range(n_boot):
        ia = torch.from_numpy(rng.integers(0, na, size=na))
        ib = torch.from_numpy(rng.integers(0, nb, size=nb))
        d = dist_fn(a[ia], b[ib])
        out["fid"].append(d["fid"])
        out["mmd"].append(d["mmd"])
    return out


def split_distances(features, dist_fn, n_splits, rng):
    """Noise-floor distribution from repeated disjoint 50/50 splits of one pool.

    Both halves are genuinely real (ERA5), so this traces out the distance you
    get purely from finite-sample noise when the two sides share a distribution —
    the null band each forecast distance should clear. Splits are *without*
    replacement (a fresh random partition each time). Returns ``{metric:
    [values]}`` of length ``n_splits``.
    """
    n = features.shape[0]
    half = n // 2
    out = {"fid": [], "mmd": []}
    for _ in range(n_splits):
        perm = rng.permutation(n)
        ia = torch.from_numpy(perm[:half])
        ib = torch.from_numpy(perm[half:2 * half])
        d = dist_fn(features[ia], features[ib])
        out["fid"].append(d["fid"])
        out["mmd"].append(d["mmd"])
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_metrics_table(results, models, sizes, device, label_suffix=""):
    """sizes: dict with n_ref_p, n_ref_g, n_pangu, n_gc, n_base."""
    hdr = "=" * 80
    print(f"\n{hdr}")
    print("REAL vs 24h FORECAST — LATENT DISTANCES (un-paired)")
    print(hdr)
    print(f"N ERA5 ref (Pangu/GraphCast): {sizes['n_ref_p']}/{sizes['n_ref_g']}    "
          f"N forecast (Pangu/GraphCast): {sizes['n_pangu']}/{sizes['n_gc']}    "
          f"N baseline halves: {sizes['n_base']}    Device: {device}")
    print()
    print(f"  {'Model':<40} {'Forecast':<14} {'FID':>10} {'MMD':>12}")
    print("  " + "-" * 78)
    for model_name in models:
        label = f"{MODEL_LABELS[model_name]}{label_suffix}"
        baseline = results[model_name].get("era5_self")
        if baseline:
            print(f"  {label:<40} {'ERA5 vs ERA5':<14} {baseline['fid']:>10.3f} {baseline['mmd']:>12.6f}  ← baseline")
        for source in ["pangu", "graphcast"]:
            d = results[model_name][source]
            print(f"  {label:<40} {SOURCE_LABELS[source]:<14} {d['fid']:>10.3f} {d['mmd']:>12.6f}")
        print()
    print(hdr)


def plot_metric_bars(results, models, plots_dir, run_tag, cis=None, ci_level=95):
    """Bar chart of FID and MMD per model/forecast source, with the ERA5-self baseline.

    When ``cis`` is given (``{model: {key: {metric: (lo, hi)}}}``), each forecast
    bar carries a bootstrap CI whisker and each model's ERA5-self baseline is drawn
    as a shaded null band (its split-distribution CI) rather than a bare line. A
    forecast bar whose lower whisker clears the top of that model's band is
    distinguishable from real fields beyond finite-sample noise.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sources = ["pangu", "graphcast"]
    x = np.arange(len(models))
    width = 0.35

    def _yerr(source, metric_key):
        """Asymmetric (2, n_models) error array for one source/metric, or None."""
        if not cis:
            return None
        lo, hi = [], []
        for m in models:
            v = results[m][source][metric_key]
            ci = cis.get(m, {}).get(source, {}).get(metric_key)
            if ci is None or not np.isfinite(ci[0]) or not np.isfinite(v):
                lo.append(0.0); hi.append(0.0)
            else:
                lo.append(max(0.0, v - ci[0])); hi.append(max(0.0, ci[1] - v))
        return np.array([lo, hi])

    for ax, metric_key, metric_label in [
        (axes[0], "fid", "Fréchet Distance"),
        (axes[1], "mmd", "MMD"),
    ]:
        for si, source in enumerate(sources):
            vals = [results[m][source][metric_key] for m in models]
            yerr = _yerr(source, metric_key)
            bars = ax.bar(
                x + (si - 0.5) * width, vals, width=width,
                label=SOURCE_LABELS[source],
                color=[MODEL_COLORS[m] for m in models],
                hatch=SOURCE_HATCHES[source],
                edgecolor="white", linewidth=0.5,
                yerr=yerr, capsize=3, error_kw=dict(ecolor="#333333", elinewidth=1.1),
            )
            ups = yerr[1] if yerr is not None else [0.0] * len(bars)
            for bar, val, up in zip(bars, vals, ups):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + up,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                )

        # ERA5 self-baseline: a shaded null band (CI) + dashed mean line per model.
        for mi, model_name in enumerate(models):
            baseline = results[model_name].get("era5_self")
            if not baseline:
                continue
            bval = baseline[metric_key]
            x_left  = x[mi] - width - width / 2
            x_right = x[mi] + width + width / 2
            base_ci = cis.get(model_name, {}).get("era5_self", {}).get(metric_key) if cis else None
            if base_ci is not None and np.isfinite(base_ci[0]):
                ax.fill_between([x_left, x_right], base_ci[0], base_ci[1],
                                color=MODEL_COLORS[model_name], alpha=0.15, zorder=0)
            ax.hlines(bval, x_left, x_right,
                      colors=MODEL_COLORS[model_name], linestyles="dashed",
                      linewidth=1.5, label=f"ERA5 self ({MODEL_LABELS[model_name]})")
            ax.text(x_right + 0.03, bval, f"{bval:.3f}",
                    va="center", ha="left", fontsize=7,
                    color=MODEL_COLORS[model_name])

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models])
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label}: ERA5 vs 24h Forecast")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(title="Forecast / Baseline", loc="upper right", fontsize=8)

    if len(models) > 1:
        color_handles = [Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m]) for m in models]
        fig.legend(handles=color_handles, title="Model", loc="lower center",
                   ncol=2, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(f"ERA5 vs 24h Forecast — Latent Distance\nRun: {run_tag}", fontsize=12)
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / f"metric_bars_{run_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_pca_scatter(features_by_model, results, models, plots_dir, run_tag, model_label_suffix=""):
    """features_by_model: {model: {"era5": ndarray, "pangu": ndarray, "graphcast": ndarray}}"""
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        feats = features_by_model[model_name]
        all_f = np.vstack([feats["era5"], feats["pangu"], feats["graphcast"]])
        n_e, n_p = len(feats["era5"]), len(feats["pangu"])

        # Joint PCA via SVD
        mean = all_f.mean(axis=0, keepdims=True)
        Xc = all_f - mean
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Xc @ Vt[:2].T  # (N_total, 2)

        era5_p  = proj[:n_e]
        pangu_p = proj[n_e:n_e + n_p]
        gc_p    = proj[n_e + n_p:]

        for pts, src in [(era5_p, "era5"), (pangu_p, "pangu"), (gc_p, "graphcast")]:
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=SOURCE_COLORS[src], alpha=0.4, s=12, label=SOURCE_LABELS[src])

        fid_p = results[model_name]["pangu"]["fid"]
        fid_g = results[model_name]["graphcast"]["fid"]
        ax.text(0.02, 0.98,
                f"FID Pangu: {fid_p:.2f}\nFID GCast: {fid_g:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_title(f"{MODEL_LABELS[model_name]}{model_label_suffix}")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.grid(True, alpha=0.2)
        ax.legend(markerscale=2, fontsize=9)

    fig.suptitle(f"Latent PCA: ERA5 vs 24h Forecasts\nRun: {run_tag}", fontsize=12)
    fig.tight_layout()
    out = plots_dir / f"pca_scatter_{run_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """CLI entry point: extract encoder latents for ERA5/forecasts and report FID/MMD + plots."""
    parser = argparse.ArgumentParser(
        description="Compare ERA5 vs forecast latent distributions (MAE + I-JEPA).")
    parser.add_argument("--local", action="store_true",
                        help="Use local data paths (data/*.nc). Default = cluster paths.")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Max paired samples per (model, source) pair (default: 500).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=200,
                        help="Resamples for uncertainty bars: bootstrap CIs on each "
                             "forecast bar and the repeated-split baseline null band. "
                             "0 disables (single-split baseline, no error bars).")
    parser.add_argument("--ci", type=float, default=95.0,
                        help="Confidence level (percent) for the bootstrap/split intervals.")
    parser.add_argument("--model-size", choices=["default", "twin", "tiny", "small"], default="twin",
                        help="Model size config to build for both MAE and IJEPA (default: twin).")
    parser.add_argument("--embed-dim", type=int, default=None,
                        help="Override encoder embed_dim for both models (matches train flag).")
    parser.add_argument("--variant", type=str, default=None,
                        help="Checkpoint variant suffix (e.g. 'shared-targets').")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory holding checkpoints + stats; plots saved under <dir>/plots/real_vs_forecast. "
                             "If unset, falls back to checkpoints/ + plots/real_vs_forecast/.")
    parser.add_argument("--era5-path", type=str, default=None,
                        help="Override ERA5 NetCDF path.")
    parser.add_argument("--pangu-path", type=str, default=None,
                        help="Override Pangu NetCDF path.")
    parser.add_argument("--graphcast-path", type=str, default=None,
                        help="Override GraphCast NetCDF path.")
    parser.add_argument("--pooling", choices=["mean", "max", "concat"], default=None,
                        help="Pooling mode for extract_features. Sets EXTRACT_FEATURES_POOLING. "
                             "Defaults to the env var if set, otherwise 'mean'.")
    parser.add_argument("--mae-only", action="store_true",
                        help="Only run MAE (skip I-JEPA checkpoint requirement).")
    parser.add_argument("--temporal-mode", choices=["none", "diff", "concat", "phase"], default="none",
                        help="Temporal-pair eval mode. Must match the trained checkpoint's mode.")
    parser.add_argument("--delta-hours", type=int, default=24,
                        help="Δt in hours for the temporal pair (default 24).")
    parser.add_argument("--baseline-pool", choices=["forecast-range", "all-years"],
                        default="forecast-range",
                        help="ERA5 pool used for the ERA5-vs-ERA5 baseline split. "
                             "'forecast-range' = same year(s) as the forecast file. "
                             "'all-years' = entire ERA5 file at the forecast hours-of-day "
                             "(much larger pool → tighter noise-floor estimate; only "
                             "useful on the cluster where ERA5 spans many years).")
    parser.add_argument("--baseline-n-per-half", type=int, default=None,
                        help="Override the per-half size of the baseline split. "
                             "Default = --n-samples (matches forecast comparison N).")
    args = parser.parse_args()

    if args.pooling is not None:
        os.environ["EXTRACT_FEATURES_POOLING"] = args.pooling

    era5_path = Path(args.era5_path) if args.era5_path else (
        LOCAL_ERA5_PATH if args.local else CLUSTER_ERA5_PATH)
    pangu_path = Path(args.pangu_path) if args.pangu_path else (
        LOCAL_PANGU_PATH if args.local else CLUSTER_PANGU_PATH)
    graphcast_path = Path(args.graphcast_path) if args.graphcast_path else (
        LOCAL_GRAPHCAST_PATH if args.local else CLUSTER_GRAPHCAST_PATH)

    stats_dir = Path(args.output_dir) if args.output_dir else Path("checkpoints")
    models_to_run = ["mae"] if args.mae_only else ["mae", "ijepa"]

    # Compose the checkpoint variant: a CLI-supplied --variant takes precedence,
    # otherwise temporal modes fall back to tm-<mode>. This must match what the
    # training scripts wrote.
    effective_variant = args.variant
    if effective_variant is None and args.temporal_mode != "none":
        effective_variant = f"tm-{args.temporal_mode}"

    ckpt_paths = {
        m: checkpoint_path(m, args.model_size, stats_dir, variant=effective_variant, embed_dim=args.embed_dim)
        for m in models_to_run
    }

    for p in [era5_path, pangu_path, graphcast_path] + [ckpt_paths[m] for m in models_to_run]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    temporal = args.temporal_mode != "none"
    prior_hours = args.delta_hours if temporal else None

    # Build pools without per-sample pairing between ERA5 and forecast.
    # The ERA5 reference pool is "ERA5 in the forecast date range at the forecast
    # hours-of-day" — same seasonal/diurnal coverage as forecasts, but treated
    # as an independent distribution (no per-time matching).
    print("Building ERA5 reference and forecast pools (no per-sample pairing)...")
    print("  Pangu ERA5 ref:")
    era5_p_ref_idx, era5_p_ref_prior = build_era5_ref_pool(
        era5_path, pangu_path, prior_hours=prior_hours)
    print("  GraphCast ERA5 ref:")
    era5_g_ref_idx, era5_g_ref_prior = build_era5_ref_pool(
        era5_path, graphcast_path, prior_hours=prior_hours)
    print("  Pangu forecasts:")
    pangu_idx, pangu_prior = build_forecast_indices(
        era5_path, pangu_path, prior_hours=prior_hours)
    print("  GraphCast forecasts:")
    gc_idx, gc_prior = build_forecast_indices(
        era5_path, graphcast_path, prior_hours=prior_hours)

    era5_p_ref_idx, era5_p_ref_prior = cap_indices(
        era5_p_ref_idx, era5_p_ref_prior, args.n_samples, rng)
    era5_g_ref_idx, era5_g_ref_prior = cap_indices(
        era5_g_ref_idx, era5_g_ref_prior, args.n_samples, rng)
    pangu_idx, pangu_prior = cap_indices(pangu_idx, pangu_prior, args.n_samples, rng)
    gc_idx,    gc_prior    = cap_indices(gc_idx,    gc_prior,    args.n_samples, rng)

    # Baseline pool: either reuse the forecast-range Pangu ref pool, or build a
    # larger pool that spans every year in the ERA5 file at the forecast hours.
    if args.baseline_pool == "all-years":
        print("  Baseline ERA5 pool (all-years):")
        base_pool_idx, base_pool_prior = build_era5_ref_pool(
            era5_path, pangu_path, prior_hours=prior_hours, restrict_date_range=False)
    else:
        base_pool_idx, base_pool_prior = era5_p_ref_idx, era5_p_ref_prior

    # Cap the baseline pool to 2 * baseline_n_per_half so each half can reach
    # the requested size. Defaults to n_samples per half (matches forecast N).
    half_n = args.baseline_n_per_half if args.baseline_n_per_half is not None else args.n_samples
    base_pool_idx, base_pool_prior = cap_indices(
        base_pool_idx, base_pool_prior, 2 * half_n, rng)

    (base_a_idx, base_a_prior), (base_b_idx, base_b_prior) = split_indices(
        base_pool_idx, base_pool_prior, rng)

    n_pangu = len(pangu_idx)
    n_gc    = len(gc_idx)
    n_ref_p = len(era5_p_ref_idx)
    n_ref_g = len(era5_g_ref_idx)
    n_base  = len(base_a_idx)

    print(f"\nERA5 ref (Pangu pool): {n_ref_p}    ERA5 ref (GraphCast pool): {n_ref_g}")
    print(f"Pangu forecasts: {n_pangu}    GraphCast forecasts: {n_gc}")
    print(f"Baseline halves: {n_base} each (pool: {args.baseline_pool}).")

    stats = (
        np.load(stats_dir / "data_mean.npy"),
        np.load(stats_dir / "data_std.npy"),
    )
    diff_stats = None
    if args.temporal_mode in ("diff", "phase"):
        dm_path = stats_dir / f"diff_mean_dt{args.delta_hours}h.npy"
        ds_path = stats_dir / f"diff_std_dt{args.delta_hours}h.npy"
        if not (dm_path.exists() and ds_path.exists()):
            raise FileNotFoundError(
                f"Temporal mode '{args.temporal_mode}' requires {dm_path.name} + {ds_path.name} "
                f"in {stats_dir}. Re-run training with this temporal mode to produce them."
            )
        diff_stats = (np.load(dm_path), np.load(ds_path))

    # On the cluster, ERA5 has ~27800 samples (~13GB) and we only need ~500 — eager
    # loading would OOM. Locally the dataset is small enough to load eagerly.
    era5_lazy = not args.local
    print(f"\nLoading datasets (ERA5 lazy={era5_lazy}, forecasts eager; using ERA5 normalization stats)...")
    # For temporal eval we use the source datasets only as raw-readers
    # (via .read_raw), so we set temporal_mode='none' here — the composite
    # happens inside TemporalPairDataset.
    era5_ds  = AtmosphereDataset(era5_path,      split="all", stats=stats, lazy=era5_lazy)
    pangu_ds = AtmosphereDataset(pangu_path,     split="all", stats=stats, lazy=False)
    gc_ds    = AtmosphereDataset(graphcast_path, split="all", stats=stats, lazy=False)

    loader_kw = dict(batch_size=args.batch_size, shuffle=False,
                     num_workers=args.num_workers, pin_memory=device.type == "cuda")

    # Feature dict: {model: {source: tensor (N, D)}}
    all_feats = {}
    results   = {}
    cis       = {}  # {model: {key: {metric: (lo, hi)}}} — bootstrap/baseline CIs

    label_suffix = ""
    if args.embed_dim is not None:
        label_suffix += f" d{args.embed_dim}"
    if args.variant:
        label_suffix += f" [{args.variant}]"

    in_chans = IN_CHANS_BY_MODE[args.temporal_mode]

    for model_name in models_to_run:
        print(f"\n{'='*60}")
        print(f"Model: {MODEL_LABELS[model_name]}{label_suffix}")
        print(f"{'='*60}")

        model = build_model(model_name, device, args.model_size, embed_dim=args.embed_dim,
                            in_chans=in_chans)
        ckpt  = ckpt_paths[model_name]
        print(f"Loading checkpoint: {ckpt}")
        model = load_model_checkpoint(model_name, model, ckpt, device)
        model.eval()

        # Verify the checkpoint's training config matches the requested eval mode.
        # Some early/static checkpoints predate this field — skip the check then.
        try:
            ckpt_blob = torch.load(ckpt, map_location="cpu")
            ckpt_cfg = ckpt_blob.get("config", {}) if isinstance(ckpt_blob, dict) else {}
            ckpt_tm = ckpt_cfg.get("temporal_mode", "none")
            ckpt_dh = ckpt_cfg.get("delta_hours", None)
        except Exception:
            ckpt_tm, ckpt_dh = "none", None
        if ckpt_tm != args.temporal_mode:
            raise RuntimeError(
                f"Checkpoint temporal_mode={ckpt_tm!r} but eval requested {args.temporal_mode!r}. "
                f"Use the matching checkpoint or pass the right --temporal-mode."
            )
        if args.temporal_mode != "none" and ckpt_dh is not None and ckpt_dh != args.delta_hours:
            raise RuntimeError(
                f"Checkpoint delta_hours={ckpt_dh} but eval requested {args.delta_hours}. "
                f"Diff stats and pair offset must match training."
            )

        if args.temporal_mode == "none":
            loaders = {
                "era5_p_ref": DataLoader(Subset(era5_ds,  era5_p_ref_idx), **loader_kw),
                "era5_g_ref": DataLoader(Subset(era5_ds,  era5_g_ref_idx), **loader_kw),
                "pangu":      DataLoader(Subset(pangu_ds, pangu_idx),      **loader_kw),
                "graphcast":  DataLoader(Subset(gc_ds,    gc_idx),         **loader_kw),
                "base_a":     DataLoader(Subset(era5_ds,  base_a_idx),     **loader_kw),
                "base_b":     DataLoader(Subset(era5_ds,  base_b_idx),     **loader_kw),
            }
        else:
            # Temporal eval: present-side may be ERA5 (real) or a forecast.
            # Prior side is always ERA5 at the present timestamp − Δt.
            def _tpd(prior_ds, present_ds, prior_idx, present_idx):
                return TemporalPairDataset(
                    prior_ds, present_ds, prior_idx, present_idx,
                    args.temporal_mode, abs_stats=stats, diff_stats=diff_stats,
                )
            loaders = {
                "era5_p_ref": DataLoader(_tpd(era5_ds, era5_ds,  era5_p_ref_prior, era5_p_ref_idx), **loader_kw),
                "era5_g_ref": DataLoader(_tpd(era5_ds, era5_ds,  era5_g_ref_prior, era5_g_ref_idx), **loader_kw),
                "pangu":      DataLoader(_tpd(era5_ds, pangu_ds, pangu_prior,      pangu_idx),      **loader_kw),
                "graphcast":  DataLoader(_tpd(era5_ds, gc_ds,    gc_prior,         gc_idx),         **loader_kw),
                "base_a":     DataLoader(_tpd(era5_ds, era5_ds,  base_a_prior,     base_a_idx),     **loader_kw),
                "base_b":     DataLoader(_tpd(era5_ds, era5_ds,  base_b_prior,     base_b_idx),     **loader_kw),
            }

        feats = {}
        for name, loader in loaders.items():
            print(f"  Extracting features: {name} ({len(loader.dataset)} samples)...")
            feats[name] = extract_features_for_loader(model, loader, device)

        # Baseline = random 50/50 split of the ERA5 reference pool (real-vs-real
        # noise floor). The forecast comparison uses the full ref pool on the ERA5
        # side and the full forecast pool on the forecast side — no per-sample
        # pairing. With --n-boot>0 the baseline is the mean over many disjoint
        # splits (stable point + a shaded band); n_boot=0 keeps a single split.
        print(f"  Computing FID/MMD: forecast (N_ref={n_ref_p}, N_fc={n_pangu}) and "
              f"baseline (N={n_base} vs {n_base})"
              f"{f'  + {args.n_boot}-resample uncertainty' if args.n_boot > 0 else ''}...")
        base_full = torch.cat([feats["base_a"], feats["base_b"]], dim=0)
        if args.n_boot > 0:
            base_samp = split_distances(base_full, compute_distances, args.n_boot, rng)
            era5_self = {k: _nanmean(base_samp[k]) for k in ("fid", "mmd")}
        else:
            base_samp = None
            era5_self = compute_distances(feats["base_a"], feats["base_b"])
        results[model_name] = {
            "era5_self": era5_self,
            "pangu":     compute_distances(feats["era5_p_ref"], feats["pangu"]),
            "graphcast": compute_distances(feats["era5_g_ref"], feats["graphcast"]),
        }
        if args.n_boot > 0:
            p_samp = bootstrap_distances(feats["era5_p_ref"], feats["pangu"],
                                         compute_distances, args.n_boot, rng)
            g_samp = bootstrap_distances(feats["era5_g_ref"], feats["graphcast"],
                                         compute_distances, args.n_boot, rng)
            cis[model_name] = {
                key: {k: _percentile_ci(samp[k], args.ci) for k in ("fid", "mmd")}
                for key, samp in (("era5_self", base_samp), ("pangu", p_samp), ("graphcast", g_samp))
            }
        all_feats[model_name] = {
            "era5":      feats["era5_p_ref"].numpy(),   # representative ERA5 distribution
            "pangu":     feats["pangu"].numpy(),
            "graphcast": feats["graphcast"].numpy(),
        }

    sizes = {"n_ref_p": n_ref_p, "n_ref_g": n_ref_g,
             "n_pangu": n_pangu, "n_gc": n_gc, "n_base": n_base}
    print_metrics_table(results, models_to_run, sizes, device, label_suffix=label_suffix)

    # Uncertainty summary: does each forecast CI clear that model's baseline band?
    if args.n_boot > 0:
        print(f"  Uncertainty ({args.n_boot} resamples, {args.ci:g}% intervals):")
        for model_name in models_to_run:
            for metric in ("fid", "mmd"):
                bl, bh = cis[model_name]["era5_self"][metric]
                print(f"  {MODEL_LABELS[model_name]} {metric.upper()}  "
                      f"baseline band [{bl:.4g}, {bh:.4g}]")
                for source in ("pangu", "graphcast"):
                    lo, hi = cis[model_name][source][metric]
                    sep = "clears band" if lo > bh else "OVERLAPS band"
                    print(f"      {SOURCE_LABELS[source]:<14} CI [{lo:.4g}, {hi:.4g}] — {sep}")
        print("  " + "=" * 78)

    tag_parts = [args.model_size]
    if args.embed_dim is not None:
        tag_parts.append(f"d{args.embed_dim}")
    if args.temporal_mode != "none":
        tag_parts.append(f"tm-{args.temporal_mode}")
    if args.variant:
        tag_parts.append(args.variant)
    tag_parts.append(f"n{n_pangu}_seed{args.seed}")
    tag_parts.append(f"pool-{os.environ.get('EXTRACT_FEATURES_POOLING', 'mean').lower()}")
    tag_parts.append(f"base-{args.baseline_pool}")
    run_tag = "_".join(tag_parts)

    if args.output_dir:
        plots_dir = Path(args.output_dir) / "plots" / "real_vs_forecast"
    else:
        plots_dir = Path("plots/real_vs_forecast")
    plot_metric_bars(results, models_to_run, plots_dir, run_tag,
                     cis=cis or None, ci_level=args.ci)
    plot_pca_scatter(all_feats, results, models_to_run, plots_dir, run_tag, model_label_suffix=label_suffix)


if __name__ == "__main__":
    main()

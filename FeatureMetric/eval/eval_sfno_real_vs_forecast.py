"""Real-vs-forecast latent distances using a colleague's pretrained SFNO encoder.

Measures Fréchet (FID-style) distance and RBF-MMD between ERA5 (real) and 24h
forecasts (Pangu and GraphCast) surface fields, embedded by the frozen 4-field
SFNO autoencoder from the sibling ``SFNO-Embedding`` repo. This is the SFNO
analogue of ``eval_real_vs_forecast.py`` and produces the same plots (FID/MMD
bars + joint-PCA scatter) so the SFNO results sit alongside MAE/I-JEPA.

Because the 4-field checkpoints drop precipitation (and exclude 2020 from
training), SFNO now reads the *same* standard 4-var ERA5/Pangu/GraphCast files as
MAE/I-JEPA — Pangu (no precip) is no longer excluded.

It reuses the MAE/I-JEPA eval's distance + time-alignment helpers, and adds an
ERA5-vs-ERA5 baseline (random 50/50 split of the reference pool) — the noise
floor each forecast distance should be compared against.

Example (local smoke test)::

    /opt/miniconda3/envs/pmlr/bin/python eval/eval_sfno_real_vs_forecast.py \
        --local --channels 8 --res 31 --n-samples 60
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.features import extract_features_for_loader
from utils.sfno_embedding import SFNOEmbedding, RawFourVarDataset
# Reuse the exact distance + index-alignment + uncertainty logic from the
# MAE/I-JEPA eval (single source of truth for these helpers).
from eval.eval_real_vs_forecast import (
    compute_distances,
    mmd_rbf,
    build_era5_ref_pool,
    build_forecast_indices,
    cap_indices,
    bootstrap_distances,
    split_distances,
    _percentile_ci,
    _nanmean,
    SOURCE_COLORS,
    SOURCE_LABELS,
)

RES_BY_FIRST = {31: (31, 60), 15: (15, 28)}

# Default data paths (mirror eval_real_vs_forecast.py — same 4-var files).
LOCAL_ERA5_PATH      = Path("data/test_data_local.nc")
LOCAL_PANGU_PATH     = Path("data/pangu_surface_2020_lead24h.nc")
LOCAL_GRAPHCAST_PATH = Path("data/graphcast_surface_2020_lead24h.nc")

CLUSTER_ERA5_PATH      = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
CLUSTER_PANGU_PATH     = Path("/cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/pangu_surface_2020_lead24h.nc")
CLUSTER_GRAPHCAST_PATH = Path("/cluster/courses/pmlr/teams/team07/data/sealevel_forcasts/graphcast_surface_2020_lead24h.nc")

SOURCES = ["pangu", "graphcast"]


def plot_metric_bars(results, plots_dir, run_tag, mmd_only, cis=None, ci_level=95):
    """Grouped bars of FID/MMD: ERA5-self baseline vs each forecast source.

    When ``cis`` is given (``{key: {metric: (lo, hi)}}``), each bar gets a
    bootstrap confidence interval as an error bar, and the ERA5-self baseline is
    drawn as a shaded null band (its split-distribution CI) rather than a bare
    line. A forecast bar whose lower whisker clears the top of that band is
    distinguishable from real fields beyond finite-sample noise.

    Drops the FID panel when FID is NaN (MMD-only runs).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ["era5_self"] + SOURCES
    labels = ["ERA5 vs ERA5\n(baseline)"] + [SOURCE_LABELS[s] for s in SOURCES]
    colors = ["#9e9e9e"] + [SOURCE_COLORS[s] for s in SOURCES]

    panels = [("fid", "Fréchet Distance"), ("mmd", "MMD (RBF)")]
    if mmd_only:
        panels = [p for p in panels if p[0] != "fid"]

    def _yerr(metric):
        """Asymmetric (2, N) error array: distance from each point to its CI ends."""
        lo, hi = [], []
        for k, v in zip(keys, [results[k][metric] for k in keys]):
            ci = cis.get(k, {}).get(metric) if cis else None
            if ci is None or not np.isfinite(ci[0]) or not np.isfinite(v):
                lo.append(0.0); hi.append(0.0)
            else:
                lo.append(max(0.0, v - ci[0])); hi.append(max(0.0, ci[1] - v))
        return np.array([lo, hi])

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5), squeeze=False)
    for ax, (metric, title) in zip(axes[0], panels):
        vals = [results[k][metric] for k in keys]
        yerr = _yerr(metric) if cis else None
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5,
                      yerr=yerr, capsize=4,
                      error_kw=dict(ecolor="#333333", elinewidth=1.2))
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        # Baseline null band (shaded CI of the ERA5-vs-ERA5 split distribution)
        # plus the mean as a dashed reference line.
        base_ci = cis.get("era5_self", {}).get(metric) if cis else None
        if base_ci is not None and np.isfinite(base_ci[0]):
            ax.axhspan(base_ci[0], base_ci[1], color="#9e9e9e", alpha=0.18, zorder=0,
                       label=f"baseline {ci_level}% band")
            ax.legend(fontsize=8, loc="upper left")
        ax.axhline(results["era5_self"][metric], color="#9e9e9e",
                   linestyle="dashed", linewidth=1.2)
        # Annotate above the upper whisker so the text doesn't sit on the cap.
        ups = yerr[1] if yerr is not None else [0.0] * len(bars)
        for b, v, up in zip(bars, vals, ups):
            ax.annotate(f"{v:.4g}", (b.get_x() + b.get_width() / 2, v + up),
                        ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"SFNO embedding — real vs 24h forecast\nRun: {run_tag}", y=1.02)
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / f"sfno_metric_bars_{run_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {out}")


def plot_pca_scatter(feats, results, plots_dir, run_tag, mmd_only):
    """Joint-PCA scatter of ERA5 vs Pangu vs GraphCast SFNO features."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_f = np.vstack([feats["era5"], feats["pangu"], feats["graphcast"]])
    n_e, n_p = len(feats["era5"]), len(feats["pangu"])

    mean = all_f.mean(axis=0, keepdims=True)
    Xc = all_f - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:2].T

    chunks = {"era5": proj[:n_e], "pangu": proj[n_e:n_e + n_p], "graphcast": proj[n_e + n_p:]}

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for src in ["era5", "pangu", "graphcast"]:
        pts = chunks[src]
        ax.scatter(pts[:, 0], pts[:, 1], c=SOURCE_COLORS[src], alpha=0.4, s=12,
                   label=SOURCE_LABELS[src])
    metric = "mmd" if mmd_only else "fid"
    mlabel = "MMD" if mmd_only else "FID"
    ax.text(0.02, 0.98,
            f"{mlabel} Pangu: {results['pangu'][metric]:.3g}\n"
            f"{mlabel} GCast: {results['graphcast'][metric]:.3g}\n"
            f"{mlabel} base : {results['era5_self'][metric]:.3g}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_title("SFNO latent PCA: ERA5 vs 24h Forecasts")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.grid(True, alpha=0.2)
    ax.legend(markerscale=2, fontsize=9)
    fig.suptitle(f"Run: {run_tag}", fontsize=10, y=1.01)
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / f"sfno_pca_scatter_{run_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local", action="store_true",
                        help="Use local data paths (data/*.nc). Default = cluster paths.")
    parser.add_argument("--era5-path", type=Path, default=None, help="Override ERA5 NetCDF path.")
    parser.add_argument("--pangu-path", type=Path, default=None, help="Override Pangu NetCDF path.")
    parser.add_argument("--graphcast-path", type=Path, default=None, help="Override GraphCast NetCDF path.")
    parser.add_argument("--channels", type=int, default=8, choices=[4, 8, 16],
                        help="SFNO embedding channels (default: 8). 31x60 only has 8c.")
    parser.add_argument("--res", type=int, default=31, choices=[15, 31],
                        help="SFNO embedding resolution (first dim): 15 -> 15x28, 31 -> 31x60 (default 31)")
    parser.add_argument("--pooling", choices=["mean", "max", "meanstd", "grid", "flatten"],
                        default="mean",
                        help="How to turn the (C,h,w) SFNO embedding into a feature vector. "
                             "mean/max=C dims, meanstd=2C, grid=C*gh*gw (adaptive-pooled, keeps "
                             "spatial structure), flatten=C*h*w (MMD only — FID is singular). "
                             "Default: mean")
    parser.add_argument("--pool-grid", type=int, nargs=2, default=(7, 8), metavar=("GH", "GW"),
                        help="Target grid for --pooling grid (default: 7 8)")
    parser.add_argument("--mmd-only", action="store_true",
                        help="Compute only MMD (skip FID). Use with high-dim features "
                             "(e.g. --pooling flatten) where FID's covariance is singular.")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Cap on samples per pool (real and each forecast)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-dt-hours", type=int, default=6,
                        help="Tolerance for matching ERA5 hours-of-day to forecast valid times")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=200,
                        help="Resamples for uncertainty bars: bootstrap CIs on each "
                             "forecast bar and the repeated-split baseline null band. "
                             "0 disables (single-split baseline, no error bars).")
    parser.add_argument("--ci", type=float, default=95.0,
                        help="Confidence level (percent) for the bootstrap/split intervals.")
    parser.add_argument("--sfno-repo", type=str, default=None,
                        help="Path to the SFNO-Embedding checkout (else $SFNO_REPO or sibling dir)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write plots/ (default: ./plots)")
    args = parser.parse_args()

    era5_path = args.era5_path or (LOCAL_ERA5_PATH if args.local else CLUSTER_ERA5_PATH)
    pangu_path = args.pangu_path or (LOCAL_PANGU_PATH if args.local else CLUSTER_PANGU_PATH)
    graphcast_path = args.graphcast_path or (LOCAL_GRAPHCAST_PATH if args.local else CLUSTER_GRAPHCAST_PATH)
    fc_paths = {"pangu": pangu_path, "graphcast": graphcast_path}

    for p in (era5_path, pangu_path, graphcast_path):
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    res = RES_BY_FIRST[args.res]

    # --- Time-aligned index pools (same logic as the MAE/I-JEPA eval) ---
    print("Building time-aligned index pools (ERA5 ref vs each forecast)...")
    ref_idx, fc_idx = {}, {}
    for src in SOURCES:
        print(f"  {src} ERA5 ref:")
        ri, _ = build_era5_ref_pool(
            era5_path, fc_paths[src], max_dt_hours=args.max_dt_hours,
            prior_hours=None, restrict_date_range=True,
        )
        print(f"  {src} forecasts:")
        fi, _ = build_forecast_indices(
            era5_path, fc_paths[src], max_dt_hours=args.max_dt_hours, prior_hours=None,
        )
        ref_idx[src], _ = cap_indices(ri, None, args.n_samples, rng)
        fc_idx[src], _ = cap_indices(fi, None, args.n_samples, rng)

    # ERA5-self baseline: repeated disjoint 50/50 splits of the Pangu reference
    # pool (done in feature space below, after extraction). Each half is half the
    # pool — that size sets the FID-validity floor.
    n_base = len(ref_idx["pangu"]) // 2

    for src in SOURCES:
        print(f"  {src}: ERA5 ref={len(ref_idx[src])}  forecast={len(fc_idx[src])}")
    print(f"  baseline halves: {n_base} each")

    # --- Model + raw 4-var datasets ---
    print(f"\nLoading 4-field SFNO encoder ({args.channels}c, {res[0]}x{res[1]}, pool={args.pooling})...")
    model = SFNOEmbedding(
        embedding_channels=args.channels, embedding_resolution=res,
        repo_root=args.sfno_repo, pooling=args.pooling, pool_grid=tuple(args.pool_grid),
    ).to(device)
    model.eval()
    print(f"  feature_dim = {model.feature_dim}")

    # FID's covariance is singular once feature_dim >= N — warn so the number isn't
    # silently meaningless. MMD stays valid at any dimension.
    n_min = min([len(ref_idx[s]) for s in SOURCES] + [len(fc_idx[s]) for s in SOURCES] + [2 * n_base])
    fid_valid = model.feature_dim < n_min
    if not fid_valid and not args.mmd_only:
        print(f"  WARNING: feature_dim ({model.feature_dim}) >= N — FID is singular/"
              f"unreliable here; trust MMD or pass --mmd-only. Use --pooling grid "
              f"(smaller --pool-grid) or raise --n-samples for a valid FID.")

    era5_ds = RawFourVarDataset(era5_path)
    fc_ds = {src: RawFourVarDataset(fc_paths[src]) for src in SOURCES}

    loader_kw = dict(batch_size=args.batch_size, shuffle=False,
                     num_workers=args.num_workers, pin_memory=device.type == "cuda")

    def feats(ds, idx):
        return extract_features_for_loader(model, DataLoader(Subset(ds, idx), **loader_kw), device)

    print("\nExtracting features...")
    f_ref = {src: feats(era5_ds, ref_idx[src]) for src in SOURCES}
    f_fc = {src: feats(fc_ds[src], fc_idx[src]) for src in SOURCES}

    def dist(a, b):
        if args.mmd_only:
            return {"fid": float("nan"), "mmd": mmd_rbf(a, b)}
        return compute_distances(a, b)

    # --- Point estimates + uncertainty (bootstrap CIs / baseline null band) ---
    # Forecast bars: full-sample point estimate, bootstrap CI for the whiskers.
    # Baseline: repeated disjoint splits of the real pool define both the point
    # (mean) and the shaded null band. n_boot=0 falls back to a single split.
    results, cis = {}, {}
    n_splits = args.n_boot if args.n_boot > 0 else 1
    print(f"\nUncertainty: {args.n_boot} resamples per bar "
          f"({'bootstrap CIs + baseline band' if args.n_boot > 0 else 'disabled — single-split baseline'})...")
    base_samp = split_distances(f_ref["pangu"], dist, n_splits, rng)
    results["era5_self"] = {m: _nanmean(base_samp[m]) for m in ("fid", "mmd")}
    for src in SOURCES:
        results[src] = dist(f_ref[src], f_fc[src])

    if args.n_boot > 0:
        cis["era5_self"] = {m: _percentile_ci(base_samp[m], args.ci) for m in ("fid", "mmd")}
        for src in SOURCES:
            samp = bootstrap_distances(f_ref[src], f_fc[src], dist, args.n_boot, rng)
            cis[src] = {m: _percentile_ci(samp[m], args.ci) for m in ("fid", "mmd")}

    # --- Report ---
    def _fmt(v):
        return "      n/a" if v != v else f"{v:>12.4f}"  # v != v ↔ NaN

    print("\n" + "=" * 72)
    print("SFNO EMBEDDING — REAL vs 24h FORECAST (Pangu + GraphCast)")
    print("=" * 72)
    print(f"  config: {args.channels}c {res[0]}x{res[1]}  pool={args.pooling}  "
          f"feature_dim={model.feature_dim}  device={device}"
          f"{'   (MMD only)' if args.mmd_only else ''}")
    print(f"  {'Comparison':<28} {'FID':>12} {'MMD':>14}")
    print("  " + "-" * 56)
    b = results["era5_self"]
    print(f"  {'ERA5 vs ERA5 (baseline)':<28} {_fmt(b['fid'])} {b['mmd']:>14.6f}")
    for src in SOURCES:
        d = results[src]
        print(f"  {('ERA5 vs ' + SOURCE_LABELS[src]):<28} {_fmt(d['fid'])} {d['mmd']:>14.6f}")
    print()
    metric = "mmd" if args.mmd_only else "fid"
    if args.n_boot > 0:
        bl, bh = cis["era5_self"][metric]
        print(f"  baseline {metric.upper()} {args.ci:g}% band: [{bl:.4g}, {bh:.4g}]  "
              f"(from {args.n_boot} disjoint splits)")
    for src in SOURCES:
        d = results[src]
        if not args.mmd_only and b['fid']:
            print(f"  {SOURCE_LABELS[src]}: FID/baseline = {d['fid'] / b['fid']:.2f}   "
                  f"MMD-baseline = {d['mmd'] - b['mmd']:.6f}  (>1 / >0 ⇒ separated)")
        else:
            print(f"  {SOURCE_LABELS[src]}: MMD-baseline = {d['mmd'] - b['mmd']:.6f}  (>0 ⇒ separated)")
        if args.n_boot > 0:
            lo, hi = cis[src][metric]
            sep = "clears baseline band" if lo > cis["era5_self"][metric][1] else "OVERLAPS baseline band"
            print(f"           {metric.upper()} {args.ci:g}% CI [{lo:.4g}, {hi:.4g}] — {sep}")
    print("=" * 72)

    run_tag = f"{args.channels}c{res[0]}x{res[1]}_pool-{args.pooling}_n{args.n_samples}_seed{args.seed}"
    if args.mmd_only:
        run_tag += "_mmd"
    plots_dir = (Path(args.output_dir) / "plots" if args.output_dir else Path("plots")) / "sfno_real_vs_forecast"
    plot_metric_bars(results, plots_dir, run_tag, args.mmd_only,
                     cis=cis or None, ci_level=args.ci)
    plot_pca_scatter(
        {"era5": f_ref["pangu"].numpy(),
         "pangu": f_fc["pangu"].numpy(),
         "graphcast": f_fc["graphcast"].numpy()},
        results, plots_dir, run_tag, args.mmd_only,
    )


if __name__ == "__main__":
    main()

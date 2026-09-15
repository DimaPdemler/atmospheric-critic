"""Corruption-severity distances for the pretrained SFNO embedding (Protocol 2).

The SFNO analogue of ``eval_distances.py``: builds a clean ERA5 reference latent
distribution, then measures Fréchet (FID) distance and RBF-MMD against each of the
six physically-motivated corruptions across a severity ladder.

Differences from the MAE/I-JEPA version:
  * raw 4-var 121x240 input (SFNO standardizes internally);
  * corruptions are applied in SFNO's *standardized* space (after per-channel
    normalization, before the encoder) — that's the space the severities are
    calibrated for, and it matches how the MAE/I-JEPA eval corrupts. The existing
    corruption functions already handle un-padded input, so they're reused as-is;
    the wind corruptions leave T2M and MSL untouched.

Only ERA5 is needed (no forecast). Example::

    python eval/eval_sfno_distances.py \
        --era5-path data/test_data_local.nc \
        --channels 4 --res 15 --pooling flatten --mmd-only --n-samples 500
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.sfno_embedding import SFNOEmbedding, RawFourVarDataset
from utils.corruptions import (
    apply_gaussian_blur,
    apply_high_freq_noise,
    apply_gaussian_field_noise,
    apply_random_pixel_replace,
    apply_wind_patch_shuffle,
    apply_wind_channel_rotation,
    get_corruption_ladder,
)
from eval.eval_real_vs_forecast import calculate_frechet_distance, mmd_rbf

RES_BY_FIRST = {31: (31, 60), 15: (15, 28)}

COLORS = {
    "Gaussian Blur": "blue", "High-Freq Noise": "red", "GRF Noise": "green",
    "Random Pixel Replace": "orange", "Spatial Shuffle (Wind Only)": "purple",
    "Channel Rotation": "brown",
}
MARKERS = {
    "Gaussian Blur": "o", "High-Freq Noise": "s", "GRF Noise": "^",
    "Random Pixel Replace": "D", "Spatial Shuffle (Wind Only)": "P",
    "Channel Rotation": "X",
}


def extract(model, loader, device, corruption_fn=None):
    """(N, D) CPU features; ``corruption_fn`` applied in standardized space."""
    feats = []
    with torch.no_grad():
        for img in loader:
            img = img.to(device, non_blocking=device.type == "cuda")
            feats.append(model.extract_features(img, corruption_fn=corruption_fn).cpu())
    return torch.cat(feats, dim=0)


def plot_curves(results, plots_dir, run_tag, mmd_only):
    """FID and/or MMD vs severity, one line per corruption."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("mmd", "MMD")] if mmd_only else [("fid", "Fréchet Distance"), ("mmd", "MMD")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5), squeeze=False)
    for ax, (key, ylabel) in zip(axes[0], metrics):
        for cond, data in results.items():
            ax.plot(data["severities"], data[key], label=cond,
                    color=COLORS[cond], marker=MARKERS[cond], linewidth=2)
        ax.set_xlabel("Severity Level")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. Corruption Severity")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][-1].set_yscale("symlog", linthresh=1e-3)
    fig.suptitle(f"SFNO embedding — corruption sweep\nRun: {run_tag}", y=1.02)
    fig.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / f"sfno_distances_{run_tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot → {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era5-path", required=True, type=Path)
    parser.add_argument("--channels", type=int, default=4, choices=[4, 8, 16],
                        help="SFNO embedding channels. 31x60 only has 8c.")
    parser.add_argument("--res", type=int, default=15, choices=[15, 31],
                        help="SFNO embedding resolution (first dim): 15 -> 15x28, 31 -> 31x60")
    parser.add_argument("--pooling", choices=["mean", "max", "meanstd", "grid", "flatten"],
                        default="flatten",
                        help="Embedding→vector reduction (see eval_sfno_real_vs_forecast). "
                             "Default: flatten (use with --mmd-only).")
    parser.add_argument("--pool-grid", type=int, nargs=2, default=(7, 8), metavar=("GH", "GW"))
    parser.add_argument("--mmd-only", action="store_true",
                        help="Skip FID (use when feature_dim is large, e.g. flatten).")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-severity-steps", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sfno-repo", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if not args.era5_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.era5_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    res = RES_BY_FIRST[args.res]

    print(f"Loading SFNO encoder ({args.channels}c, {res[0]}x{res[1]}, pool={args.pooling})...")
    model = SFNOEmbedding(
        embedding_channels=args.channels, embedding_resolution=res,
        repo_root=args.sfno_repo, pooling=args.pooling, pool_grid=tuple(args.pool_grid),
    ).to(device)
    model.eval()
    print(f"  feature_dim = {model.feature_dim}")

    dataset = RawFourVarDataset(args.era5_path)
    n = min(args.n_samples, len(dataset))
    idx = rng.choice(len(dataset), n, replace=False).tolist()
    loader = DataLoader(Subset(dataset, idx), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    print(f"Reference distribution: {n} samples on {device}.")

    if not args.mmd_only and model.feature_dim >= n:
        print(f"  WARNING: feature_dim ({model.feature_dim}) >= N ({n}); FID covariance "
              f"is singular. Use --mmd-only, a smaller --pool-grid, or more samples.")

    ns = args.n_severity_steps
    ladders = {
        "Gaussian Blur": (get_corruption_ladder("blur", ns), apply_gaussian_blur),
        "High-Freq Noise": (get_corruption_ladder("noise", ns), apply_high_freq_noise),
        "GRF Noise": (get_corruption_ladder("grf", ns), apply_gaussian_field_noise),
        "Random Pixel Replace": (get_corruption_ladder("pixel_replace", ns), apply_random_pixel_replace),
        "Spatial Shuffle (Wind Only)": (get_corruption_ladder("wind_patch_shuffle", ns),
                                        partial(apply_wind_patch_shuffle, patch_size=model.patch_size)),
        "Channel Rotation": (get_corruption_ladder("wind_rotation", ns), apply_wind_channel_rotation),
    }

    print("\nExtracting clean reference features...")
    z_ref = extract(model, loader, device)
    mu_ref = np.mean(z_ref.numpy(), axis=0)
    sigma_ref = np.cov(z_ref.numpy(), rowvar=False)

    results = {}
    for cond, (severities, apply_fn) in ladders.items():
        print(f"\n  Corruption: {cond}")
        results[cond] = {"severities": severities, "fid": [], "mmd": []}
        for sev in severities:
            z_cor = extract(model, loader, device,
                            corruption_fn=lambda img, s=sev: apply_fn(img, s))
            mmd = mmd_rbf(z_ref, z_cor)
            if args.mmd_only:
                fid = float("nan")
            else:
                z = z_cor.numpy()
                fid = calculate_frechet_distance(mu_ref, sigma_ref,
                                                 np.mean(z, axis=0), np.cov(z, rowvar=False))
            results[cond]["fid"].append(fid)
            results[cond]["mmd"].append(mmd)
            print(f"    Severity {sev:.2f} | FID: {fid:8.3f} | MMD: {mmd:9.6f}")

    run_tag = (f"{args.channels}c{res[0]}x{res[1]}_pool-{args.pooling}_n{n}_steps{ns}_seed{args.seed}"
               + ("_mmd" if args.mmd_only else ""))
    plots_dir = (Path(args.output_dir) / "plots" if args.output_dir else Path("plots")) / "sfno_distances"
    plot_curves(results, plots_dir, run_tag, args.mmd_only)


if __name__ == "__main__":
    main()

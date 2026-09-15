"""Frechet Distance and MMD vs. corruption severity in encoder latent space."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
from functools import partial
import torch
import numpy as np
import scipy.linalg
from torch.utils.data import DataLoader, Subset

from utils.dataset import AtmosphereDataset
from utils.features import extract_features_for_loader
from utils.model_io import build_model, checkpoint_path, load_model_checkpoint
from utils.corruptions import (
    apply_gaussian_blur,
    apply_high_freq_noise,
    apply_gaussian_field_noise,
    apply_random_pixel_replace,
    apply_wind_patch_shuffle,
    apply_wind_channel_rotation,
    get_corruption_ladder,
)

CLUSTER_DATA_PATH = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local.nc"
LARGE_LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local_5y.nc"


def format_model_label(model_name, model_size, model_variant=None):
    """Build a human-readable model label like ``MAE (twin) [variant]`` for plots/logs."""
    label = f"{model_name.upper()} ({model_size})"
    if model_variant:
        label += f" [{model_variant}]"
    return label


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet (FID-style) distance between two Gaussians from their means/covariances.

    Handles numerically imaginary matrix-sqrt results by adding a small diagonal
    offset, matching the standard FID implementation.
    """
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print(f"Warning: Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def mmd_rbf(X, Y, gamma=None):
    """Unbiased squared MMD between feature sets ``X`` and ``Y`` with an RBF kernel.

    When ``gamma`` is None it uses the median-distance heuristic for the kernel bandwidth.
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
    return (K_XX.sum() / (N * (N - 1)) + K_YY.sum() / (M * (M - 1)) - 2 * K_XY.mean()).item()


def evaluate_model(model_name, model_size, model_variant, device, stats_dir, dataset, batch_size,
                   num_workers, local_flags, n_severity_steps=5, embed_dim=None):
    """Protocol 2 for one model: build a clean reference latent distribution, then
    measure Fréchet distance and MMD against each corruption across severity levels.

    Returns ``(results, ladders)`` where ``results`` maps each corruption to its
    severity list and per-severity FID/MMD curves.
    """
    print(f"Loading {format_model_label(model_name, model_size, model_variant)} model...")
    model = build_model(model_name, device=device, model_size=model_size, embed_dim=embed_dim)
    ckpt_path = checkpoint_path(model_name, model_size, stats_dir, variant=model_variant, embed_dim=embed_dim)
    model = load_model_checkpoint(model_name, model, ckpt_path, device)
    model.eval()

    ladders = {
        "Gaussian Blur": (get_corruption_ladder("blur", n_severity_steps), apply_gaussian_blur),
        "High-Freq Noise": (get_corruption_ladder("noise", n_severity_steps), apply_high_freq_noise),
        "GRF Noise": (get_corruption_ladder("grf", n_severity_steps), apply_gaussian_field_noise),
        "Random Pixel Replace": (get_corruption_ladder("pixel_replace", n_severity_steps), apply_random_pixel_replace),
        "Spatial Shuffle (Wind Only)": (get_corruption_ladder("wind_patch_shuffle", n_severity_steps),
                                        partial(apply_wind_patch_shuffle, patch_size=model.patch_size)),
        "Channel Rotation": (get_corruption_ladder("wind_rotation", n_severity_steps), apply_wind_channel_rotation),
    }

    n_samples = 200 if local_flags["local"] else (400 if local_flags["large_local"] else 1000)
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices.tolist())
    loader_kwargs = {
        "batch_size": batch_size, "shuffle": False,
        "num_workers": num_workers, "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    subset_loader = DataLoader(subset, **loader_kwargs)

    print(f"\n--- Fréchet & MMD Distances ({model_name.upper()}, {model_size}) ---")
    print(f"Reference distribution: {len(indices)} samples.")

    z_ref = extract_features_for_loader(model, subset_loader, device)
    z_ref_np = z_ref.numpy()
    mu_ref = np.mean(z_ref_np, axis=0)
    sigma_ref = np.cov(z_ref_np, rowvar=False)

    results = {}
    for cond_name, (severities, apply_fn) in ladders.items():
        print(f"\n  Corruption: {cond_name}")
        results[cond_name] = {"severities": severities, "fid": [], "mmd": []}
        for sev in severities:
            z_cor = extract_features_for_loader(
                model, subset_loader, device,
                transform_fn=lambda img, s=sev: apply_fn(img, s),
            )
            z_cor_np = z_cor.numpy()
            mu_cor = np.mean(z_cor_np, axis=0)
            sigma_cor = np.cov(z_cor_np, rowvar=False)
            fid = calculate_frechet_distance(mu_ref, sigma_ref, mu_cor, sigma_cor)
            mmd = mmd_rbf(z_ref, z_cor)
            print(f"    Severity {sev:.2f} | FID: {fid:8.2f} | MMD: {mmd:8.5f}")
            results[cond_name]["fid"].append(fid)
            results[cond_name]["mmd"].append(mmd)

    return results, ladders


def plot_distances(all_results, models_to_run, model_sizes, model_variants, ladders, plots_dir, run_tag):
    """Render and save the distance-vs-severity plots (raw curves + normalised subplots)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {
        "Gaussian Blur": "blue", "High-Freq Noise": "red", "GRF Noise": "green",
        "Random Pixel Replace": "orange", "Spatial Shuffle (Wind Only)": "purple",
        "Channel Rotation": "brown",
    }
    markers = {
        "Gaussian Blur": "o", "High-Freq Noise": "s", "GRF Noise": "^",
        "Random Pixel Replace": "D", "Spatial Shuffle (Wind Only)": "P",
        "Channel Rotation": "X",
    }
    model_styles = ["-", "-."]
    model_widths = [2.0, 3.0]
    model_alphas = [0.8, 1.0]

    labels = [format_model_label(m, model_sizes[m], model_variants[m]) for m in models_to_run]
    title_model = " vs ".join(labels)
    is_comparison = len(models_to_run) == 2

    # Combined FID + MMD overview
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for m_idx, model_name in enumerate(models_to_run):
        results = all_results[model_name]
        for cond_name in ladders.keys():
            sevs = results[cond_name]["severities"]
            single_model = not is_comparison
            ax1.plot(sevs, results[cond_name]["fid"],
                     label=cond_name if single_model else None,
                     marker=markers[cond_name], color=colors[cond_name],
                     linewidth=model_widths[m_idx], linestyle=model_styles[m_idx],
                     alpha=model_alphas[m_idx])
            ax2.plot(sevs, results[cond_name]["mmd"],
                     label=cond_name if single_model else None,
                     marker=markers[cond_name], color=colors[cond_name],
                     linewidth=model_widths[m_idx], linestyle=model_styles[m_idx],
                     alpha=model_alphas[m_idx])

    for ax, ylabel, title in [
        (ax1, "Fréchet Distance", f"Fréchet Distance vs. Corruption Severity\n({title_model})"),
        (ax2, "MMD", f"MMD vs. Corruption Severity\n({title_model})"),
    ]:
        ax.set_xlabel("Severity Level")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    ax1.set_yscale("symlog", linthresh=1.0)
    ax2.set_yscale("symlog", linthresh=1e-3)

    if is_comparison:
        corruption_handles = [
            Line2D([0], [0], color=colors[n], marker=markers[n], lw=2, ls="-", label=n)
            for n in ladders.keys()
        ]
        model_handles = [
            Line2D([0], [0], color="black", lw=2.5, ls=model_styles[i], label=labels[i])
            for i in range(len(models_to_run))
        ]
        for ax in (ax1, ax2):
            leg1 = ax.legend(handles=corruption_handles, title="Corruption", loc="upper left")
            ax.add_artist(leg1)
            ax.legend(handles=model_handles, title="Model", loc="upper right")
    else:
        ax1.legend(); ax2.legend()

    fig.suptitle(f"Run: {run_tag}", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(plots_dir / f"distances_vs_severity_{run_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Normalised per-corruption subplots
    model_colors = {"mae": "#2196F3", "ijepa": "#FF5722"}
    corruption_names = list(ladders.keys())
    ncols = 3
    nrows = (len(corruption_names) + ncols - 1) // ncols

    for metric_key, metric_label in [("fid", "FID"), ("mmd", "MMD")]:
        fig2, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()
        for ax_idx, cond_name in enumerate(corruption_names):
            ax = axes[ax_idx]
            for model_name in models_to_run:
                sevs = np.array(all_results[model_name][cond_name]["severities"])
                vals = np.array(all_results[model_name][cond_name][metric_key])
                vals_norm = (vals - vals[0]) / (np.abs(vals[0]) + 1e-8)
                ax.plot(sevs, vals_norm,
                        color=model_colors.get(model_name, "gray"),
                        linestyle="-", marker="o", markersize=4,
                        label=format_model_label(model_name, model_sizes[model_name], model_variants[model_name]))
            ax.axhline(0, color="gray", lw=0.8, ls=":")
            ax.set_title(cond_name, fontsize=11)
            ax.set_xlabel("Severity")
            ax.set_ylabel(f"Normalised {metric_label} increase")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        for ax_idx in range(len(corruption_names), len(axes)):
            axes[ax_idx].set_visible(False)
        fig2.suptitle(
            f"Normalised {metric_label} Increase vs. Corruption Severity\n"
            f"({title_model})\n"
            f"Run: {run_tag}\n" + r"$(d - d_0)\,/\,|d_0|$",
            fontsize=13, y=1.01,
        )
        fig2.tight_layout()
        fig2.savefig(plots_dir / f"distances_normalised_{metric_key}_{run_tag}.png",
                     dpi=150, bbox_inches="tight")
        plt.close(fig2)

    print(f"Saved distance plots with tag '{run_tag}' to {plots_dir}")


def print_summary(all_results, models_to_run, model_sizes):
    """Print a formatted table of per-corruption FID/MMD values for each model."""
    print("\n" + "=" * 80)
    print("DISTANCE EVALUATION SUMMARY")
    print("=" * 80)
    for model_name in models_to_run:
        print(f"\nModel: {model_name.upper()} ({model_sizes[model_name]})")
        header = f"  {'Corruption':<30} {'Sev':>5} {'FID':>10} {'MMD':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for cond_name, data in all_results[model_name].items():
            for sev, fid, mmd in zip(data["severities"], data["fid"], data["mmd"]):
                print(f"  {cond_name:<30} {sev:5.2f} {fid:10.2f} {mmd:10.5f}")
    print("=" * 80)


def main():
    """CLI entry point for Protocol 2 (Fréchet/MMD distance vs. corruption severity)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mae", "ijepa", "both"], default="mae")
    parser.add_argument("--mae-size", choices=["default", "twin"], default="twin")
    parser.add_argument("--ijepa-size", choices=["tiny", "small", "twin"], default="twin")
    parser.add_argument("--mae-variant", type=str, default=None)
    parser.add_argument("--ijepa-variant", type=str, default=None)
    parser.add_argument("--mae-embed-dim", type=int, default=None,
                        help="Override MAE embed_dim (matches the train flag).")
    parser.add_argument("--ijepa-embed-dim", type=int, default=None,
                        help="Override IJEPA embed_dim (matches the train flag).")
    parser.add_argument("--twin", action="store_true",
                        help="Shorthand for --mae-size twin --ijepa-size twin")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--large-local", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lazy", dest="lazy", action="store_true")
    parser.add_argument("--eager", dest="lazy", action="store_false")
    parser.add_argument("--n-severity-steps", type=int, default=9)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory holding checkpoints + stats; plots saved under <dir>/plots. "
                             "If unset, falls back to checkpoints/ + plots/ (or /work/scratch/$USER/plots on cluster).")
    parser.set_defaults(lazy=None)
    args = parser.parse_args()

    if args.twin:
        args.mae_size = "twin"
        args.ijepa_size = "twin"

    model_sizes = {"mae": args.mae_size, "ijepa": args.ijepa_size}
    model_variants = {"mae": args.mae_variant, "ijepa": args.ijepa_variant}
    model_embed_dims = {"mae": args.mae_embed_dim, "ijepa": args.ijepa_embed_dim}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and device.type != "cuda":
        device = torch.device("cpu")

    if args.local and args.large_local:
        raise ValueError("Use only one of --local or --large-local.")
    data_path = LOCAL_DATA_PATH if args.local else (
        LARGE_LOCAL_DATA_PATH if args.large_local else CLUSTER_DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    stats_dir = Path(args.output_dir) if args.output_dir else Path("checkpoints")
    stats = (np.load(stats_dir / "data_mean.npy"), np.load(stats_dir / "data_std.npy"))
    lazy_load = (not args.local) if args.lazy is None else args.lazy
    num_workers = 0 if args.num_workers is None else args.num_workers
    dataset = AtmosphereDataset(data_path, split="val", stats=stats, lazy=lazy_load)

    if args.output_dir:
        plots_dir = Path(args.output_dir) / "plots"
    else:
        plots_dir = Path("plots") if (args.local or args.large_local) else Path(f"/work/scratch/{os.environ['USER']}/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = ["mae", "ijepa"] if args.model == "both" else [args.model]
    all_results, ladders = {}, None

    for model_name in models_to_run:
        results, ladders = evaluate_model(
            model_name=model_name,
            model_size=model_sizes[model_name],
            model_variant=model_variants[model_name],
            device=device,
            stats_dir=stats_dir,
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=num_workers,
            local_flags={"local": args.local, "large_local": args.large_local},
            n_severity_steps=args.n_severity_steps,
            embed_dim=model_embed_dims[model_name],
        )
        all_results[model_name] = results

    if args.model == "both":
        mae_tag = args.mae_size if not args.mae_variant else f"{args.mae_size}-{args.mae_variant}"
        ijepa_tag = args.ijepa_size if not args.ijepa_variant else f"{args.ijepa_size}-{args.ijepa_variant}"
        run_tag = f"both_mae-{mae_tag}_ijepa-{ijepa_tag}"
    else:
        variant = model_variants[args.model]
        run_tag = f"{args.model}_{model_sizes[args.model]}" if not variant else f"{args.model}_{model_sizes[args.model]}-{variant}"
    run_tag += f"_pool-{os.environ.get('EXTRACT_FEATURES_POOLING', 'mean').lower()}"

    plot_distances(all_results, models_to_run, model_sizes, model_variants, ladders, plots_dir, run_tag)
    print_summary(all_results, models_to_run, model_sizes)


if __name__ == "__main__":
    main()

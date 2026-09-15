"""Linear probe: regress corruption severity from frozen encoder latents."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
from functools import partial
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from utils.dataset import AtmosphereDataset
from utils.model_io import build_model, checkpoint_path, load_model_checkpoint
from utils.corruptions import (
    apply_gaussian_blur,
    apply_high_freq_noise,
    apply_gaussian_field_noise,
    apply_random_pixel_replace,
    apply_wind_patch_shuffle,
    apply_wind_channel_rotation,
    MAX_SEVERITY,
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


def evaluate_model(model_name, model_size, model_variant, device, stats_dir, dataset, batch_size,
                   num_workers, local_flags, n_probe_samples=None, embed_dim=None):
    """Protocol 1 for one model: train a linear probe to regress corruption severity.

    For each corruption, applies a random severity per sample, extracts latent
    features, fits a linear probe (80/20 split), and reports MSE/R². Returns
    ``(results, scatters, corruption_fns)`` for downstream plotting.
    """
    print(f"Loading {format_model_label(model_name, model_size, model_variant)} model...")
    model = build_model(model_name, device=device, model_size=model_size, embed_dim=embed_dim)
    ckpt_path = checkpoint_path(model_name, model_size, stats_dir, variant=model_variant, embed_dim=embed_dim)
    model = load_model_checkpoint(model_name, model, ckpt_path, device)
    model.eval()

    corruption_fns = {
        "Gaussian Blur": apply_gaussian_blur,
        "High-Freq Noise": apply_high_freq_noise,
        "GRF Noise": apply_gaussian_field_noise,
        "Random Pixel Replace": apply_random_pixel_replace,
        "Spatial Shuffle (Wind Only)": partial(apply_wind_patch_shuffle, patch_size=model.patch_size),
        "Channel Rotation": apply_wind_channel_rotation,
    }

    print(f"\n--- Linear Probe ({model_name.upper()}, {model_size}) ---")

    default_n = 50 if local_flags["local"] else (250 if local_flags["large_local"] else 1000)
    n_samples = n_probe_samples if n_probe_samples is not None else default_n
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

    results = {}
    scatters = {}
    for corr_name, apply_fn in corruption_fns.items():
        print(f"\n  Corruption: {corr_name}")
        x_z, y_severity = [], []
        with torch.no_grad():
            for base_img in subset_loader:
                base_img = base_img.to(device, non_blocking=device.type == "cuda")
                for j in range(base_img.shape[0]):
                    sev = np.random.uniform(0.0, MAX_SEVERITY)
                    corrupted = apply_fn(base_img[j:j+1], severity=sev)
                    x_z.append(model.extract_features(corrupted).cpu())
                    y_severity.append(sev)

        x_z = torch.cat(x_z, dim=0)
        y_severity = torch.tensor(y_severity, dtype=torch.float32).unsqueeze(1)

        split_idx = max(1, int(0.8 * x_z.shape[0]))
        perm = torch.randperm(x_z.shape[0])
        train_idx, test_idx = perm[:split_idx], perm[split_idx:]
        if test_idx.numel() == 0:
            test_idx = train_idx

        x_train, y_train = x_z[train_idx], y_severity[train_idx]
        x_test, y_test = x_z[test_idx], y_severity[test_idx]

        probe = torch.nn.Linear(x_train.shape[1], 1).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        probe_batch_size = min(64, x_train.shape[0])
        x_train_d, y_train_d = x_train.to(device), y_train.to(device)
        x_test_d = x_test.to(device)

        probe.train()
        for epoch in range(200):
            perm_e = torch.randperm(x_train_d.shape[0], device=device)
            epoch_loss, n_batches = 0.0, 0
            for i in range(0, x_train_d.shape[0], probe_batch_size):
                idx = perm_e[i:i+probe_batch_size]
                loss = torch.nn.functional.mse_loss(probe(x_train_d[idx]), y_train_d[idx])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += loss.item(); n_batches += 1
            if (epoch + 1) % 50 == 0:
                print(f"    Probe epoch {epoch+1}/200, train MSE: {epoch_loss/n_batches:.4f}")

        probe.eval()
        with torch.no_grad():
            y_pred = probe(x_test_d).cpu()
        mse = torch.mean((y_pred - y_test) ** 2).item()
        y_var = torch.var(y_test).item()
        r2 = 1 - mse / y_var if y_var > 0 else 0.0
        print(f"    MSE: {mse:.4f} | R²: {r2:.4f}")
        results[corr_name] = {"mse": mse, "r2": r2}
        scatters[corr_name] = {
            "y_true": y_test.numpy().flatten(),
            "y_pred": y_pred.numpy().flatten(),
        }

    return results, scatters, corruption_fns


def plot_comparison(all_results, all_scatters, models_to_run, model_sizes,
                    model_variants, corruption_fns, plots_dir, run_tag):
    """Render and save the probe R² bar chart, comparison delta, and prediction scatters."""
    import matplotlib.pyplot as plt

    corr_names = list(corruption_fns.keys())
    x = np.arange(len(corr_names))
    width = 0.35
    labels = [format_model_label(m, model_sizes[m], model_variants[m]) for m in models_to_run]
    colors = ["steelblue", "darkorange"]

    # R² bar chart + delta
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    r2_values = [[all_results[m][c]["r2"] for c in corr_names] for m in models_to_run]
    for i, (r2s, label, color) in enumerate(zip(r2_values, labels, colors)):
        offset = (i - 0.5) * width
        axes[0].bar(x + offset, r2s, width=width, label=label, color=color)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(corr_names, rotation=30, ha="right")
    axes[0].set_ylabel("$R^2$")
    axes[0].set_title("Linear Probe Severity Prediction")
    axes[0].legend(); axes[0].grid(True, axis="y", alpha=0.3)

    if len(models_to_run) == 2:
        m0, m1 = models_to_run
        improvement = np.array(r2_values[1]) - np.array(r2_values[0])
        axes[1].bar(x, improvement,
                    color=["darkorange" if v >= 0 else "firebrick" for v in improvement])
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(corr_names, rotation=30, ha="right")
        axes[1].set_ylabel(f"{labels[1]} $R^2$ − {labels[0]} $R^2$")
        axes[1].set_title("Probe Comparison Delta"); axes[1].grid(True, axis="y", alpha=0.3)
    else:
        axes[1].axis("off")

    fig.suptitle(f"Run: {run_tag}", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(plots_dir / f"probe_comparison_{run_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Scatter plots
    fig, axes = plt.subplots(
        len(models_to_run), len(corr_names),
        figsize=(7 * len(corr_names), 6 * len(models_to_run))
    )
    if len(models_to_run) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(corr_names) == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, m in enumerate(models_to_run):
        for col_idx, corr_name in enumerate(corr_names):
            ax = axes[row_idx, col_idx]
            y_true = all_scatters[m][corr_name]["y_true"]
            y_pred = all_scatters[m][corr_name]["y_pred"]
            r2 = all_results[m][corr_name]["r2"]
            ax.scatter(y_true, y_pred, alpha=0.3, s=10, color=colors[row_idx])
            lims = [0, MAX_SEVERITY]
            ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect")
            fit = np.polyfit(y_true, y_pred, 1)
            fit_x = np.linspace(0, MAX_SEVERITY, 100)
            ax.plot(fit_x, np.polyval(fit, fit_x), "r-", lw=2,
                    label=f"slope={fit[0]:.2f}")
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_title(f"{labels[row_idx]} | {corr_name}\n$R^2$={r2:.4f}")
            ax.set_xlabel("True Severity"); ax.set_ylabel("Predicted Severity")
            ax.grid(True, alpha=0.3)
            if col_idx == 0:
                ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(plots_dir / f"probe_scatter_{run_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plots with tag '{run_tag}' to {plots_dir}")


def print_summary(all_results, models_to_run, model_sizes, corruption_fns):
    """Print a formatted table of per-corruption probe R² (and model delta) values."""
    labels = [f"{m.upper()} ({model_sizes[m]})" for m in models_to_run]
    col_w = 12
    print("\n" + "=" * 80)
    print("PROBE EVALUATION SUMMARY")
    print("=" * 80)
    header = f"  {'Corruption':<30}" + "".join(f"{l:>{col_w}}" for l in labels)
    if len(models_to_run) == 2:
        header += f"  {'Delta':>{col_w}}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for corr_name in corruption_fns.keys():
        r2s = [all_results[m][corr_name]["r2"] for m in models_to_run]
        row = f"  {corr_name:<30}" + "".join(f"{r:>{col_w}.4f}" for r in r2s)
        if len(models_to_run) == 2:
            row += f"  {r2s[1] - r2s[0]:>+{col_w}.4f}"
        print(row)
    print("=" * 80)


def main():
    """CLI entry point for Protocol 1 (linear-probe severity regression from latents)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mae", "ijepa", "both"], default="mae")
    # Per-model size flags for the mixed comparison (mae default vs ijepa small)
    parser.add_argument("--mae-size", choices=["default", "twin"], default="twin")
    parser.add_argument("--ijepa-size", choices=["tiny", "small", "twin"], default="twin")
    parser.add_argument("--mae-variant", type=str, default=None)
    parser.add_argument("--ijepa-variant", type=str, default=None)
    parser.add_argument("--mae-embed-dim", type=int, default=None,
                        help="Override MAE embed_dim (matches the train flag).")
    parser.add_argument("--ijepa-embed-dim", type=int, default=None,
                        help="Override IJEPA embed_dim (matches the train flag).")
    # Convenience flag: sets both to twin
    parser.add_argument("--twin", action="store_true",
                        help="Shorthand for --mae-size twin --ijepa-size twin")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--large-local", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lazy", dest="lazy", action="store_true")
    parser.add_argument("--eager", dest="lazy", action="store_false")
    parser.set_defaults(lazy=None)
    parser.add_argument("--n-probe-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory holding checkpoints + stats; plots saved under <dir>/plots. "
                             "If unset, falls back to checkpoints/ + plots/ (or /work/scratch/$USER/plots on cluster).")
    args = parser.parse_args()

    if args.twin:
        args.mae_size = "twin"
        args.ijepa_size = "twin"

    # Map model name -> size for easy lookup
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
    all_results, all_scatters, corruption_fns = {}, {}, None

    for model_name in models_to_run:
        results, scatters, corruption_fns = evaluate_model(
            model_name=model_name,
            model_size=model_sizes[model_name],
            model_variant=model_variants[model_name],
            device=device,
            stats_dir=stats_dir,
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=num_workers,
            local_flags={"local": args.local, "large_local": args.large_local},
            n_probe_samples=args.n_probe_samples,
            embed_dim=model_embed_dims[model_name],
        )
        all_results[model_name] = results
        all_scatters[model_name] = scatters

    # Build a tag for filenames, e.g. "both_mae-default_ijepa-twin" or "mae_default"
    if args.model == "both":
        mae_tag = args.mae_size if not args.mae_variant else f"{args.mae_size}-{args.mae_variant}"
        ijepa_tag = args.ijepa_size if not args.ijepa_variant else f"{args.ijepa_size}-{args.ijepa_variant}"
        run_tag = f"both_mae-{mae_tag}_ijepa-{ijepa_tag}"
    else:
        variant = model_variants[args.model]
        run_tag = f"{args.model}_{model_sizes[args.model]}" if not variant else f"{args.model}_{model_sizes[args.model]}-{variant}"
    run_tag += f"_pool-{os.environ.get('EXTRACT_FEATURES_POOLING', 'mean').lower()}"

    plot_comparison(all_results, all_scatters, models_to_run, model_sizes,
                    model_variants, corruption_fns, plots_dir, run_tag)
    print_summary(all_results, models_to_run, model_sizes, corruption_fns)


if __name__ == "__main__":
    main()

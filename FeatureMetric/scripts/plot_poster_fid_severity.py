"""Poster figure: I-JEPA latent FID vs corruption severity.

Two-row, one-column layout designed to sit as the right-hand column of a 2x2
results figure on the poster (the other team's GAN plots will fill the left
column). Row 1 shows I-JEPA latent FID for two contrasting corruptions
(High-Freq Noise and Wind Channel Rotation) on a log y-axis since their
magnitudes differ by orders of magnitude. Row 2 is a placeholder for the
forecast lead-time experiment (Pangu / GraphCast vs ERA5) that is still in
progress.

Note: in `utils/corruptions.py`, channel rotation maps `severity * pi/2`, so
the meaningful corruption range is severity in [0, 1] (= 0 to 90 degrees).
High-freq noise also lives in [0, 1] (std up to 0.25). We use a shared
severity axis [0, 1] for both.

Run:
    /opt/miniconda3/envs/pmlr/bin/python scripts/plot_poster_fid_severity.py
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import torch
from torch.utils.data import DataLoader, Subset

from utils.corruptions import (
    apply_high_freq_noise,
    apply_wind_channel_rotation,
)
from utils.dataset import AtmosphereDataset
from utils.features import extract_features_for_loader
from utils.model_io import build_model, load_model_checkpoint


# -- Poster palette ---------------------------------------------------------
INK_BLACK = "#0D1821"
YALE_BLUE = "#344966"
BLUSH_PINK = "#E6AACE"
PORCELAIN = "#F0F4EF"

mpl.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42


# -- Paths ------------------------------------------------------------------
DATA = REPO / "data"
ERA5_LOCAL = DATA / "test_data_local.nc"
ERA5_LARGE_LOCAL = DATA / "test_data_local_5y.nc"
ERA5_CLUSTER = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")

MODEL_DIR = REPO / "results" / "may_07_512_encoder"
MODEL_PATH = MODEL_DIR / "best_ijepa_model_twin_d512.pth"
MEAN_PATH = MODEL_DIR / "data_mean.npy"
STD_PATH = MODEL_DIR / "data_std.npy"


# -- FID --------------------------------------------------------------------
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet (FID-style) distance between two Gaussians given their means/covariances."""
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def fid_curve(model, loader, device, apply_fn, severities, label):
    """Compute latent FID(clean, corrupted) across ``severities`` for one corruption.

    Returns an array of FID values (clamped ≥ 0) parallel to ``severities``.
    """
    print(f"\n  {label}")
    fids = []
    for sev in severities:
        if sev == 0.0:
            transform = None
        else:
            transform = lambda img, s=sev: apply_fn(img, s)
        z_ref = extract_features_for_loader(model, loader, device)
        z_cor = extract_features_for_loader(model, loader, device, transform_fn=transform)
        z_ref_np = z_ref.numpy()
        z_cor_np = z_cor.numpy()
        mu_r = z_ref_np.mean(axis=0); sig_r = np.cov(z_ref_np, rowvar=False)
        mu_c = z_cor_np.mean(axis=0); sig_c = np.cov(z_cor_np, rowvar=False)
        fid = frechet_distance(mu_r, sig_r, mu_c, sig_c)
        # FID can be tiny-negative from numerical jitter; clamp to >=0 for log axis.
        fid = max(fid, 0.0)
        print(f"    severity {sev:.2f} -> FID {fid:.4f}")
        fids.append(fid)
    return np.array(fids)


# -- Plotting ---------------------------------------------------------------
def style_axis(ax):
    """Apply the shared poster styling (colors, ticks, spines, grid) to an axis."""
    ax.set_facecolor("white")
    ax.tick_params(colors=INK_BLACK, labelsize=18, width=1.0)
    for spine in ax.spines.values():
        spine.set_edgecolor(INK_BLACK)
        spine.set_linewidth(1.2)
    ax.grid(True, which="both", alpha=0.25, color=INK_BLACK, linewidth=0.6)


def plot_figure(severities, fid_noise, fid_rotation, output_path):
    """Render the two-row poster figure (FID-vs-severity + lead-time placeholder) and save it."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.5, 11), constrained_layout=True
    )
    fig.patch.set_facecolor("white")

    # --- Row 1: I-JEPA latent FID vs severity ---
    style_axis(ax_top)

    ax_top.plot(
        severities, fid_noise,
        marker="o", markersize=11, lw=3.0,
        color=YALE_BLUE, label="High-Freq Noise",
        markeredgecolor=INK_BLACK, markeredgewidth=0.8,
    )
    ax_top.plot(
        severities, fid_rotation,
        marker="s", markersize=11, lw=3.0,
        color=BLUSH_PINK, label="Wind-Vector Rotation",
        markeredgecolor=INK_BLACK, markeredgewidth=0.8,
    )

    # symlog handles the FID=0 point at severity=0 cleanly: linear region
    # below `linthresh`, log scale above. Picked just below the smallest
    # positive FID so the linear band is barely visible.
    ax_top.set_yscale("symlog", linthresh=1e-4, linscale=0.5)
    ax_top.set_ylim(bottom=0)
    ax_top.tick_params(axis="y", labelleft=False)  # no negative region — FID is non-negative
    ax_top.set_xlim(0.0, 1.05)
    ax_top.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_top.set_xlabel("Corruption Severity", fontsize=22, color=INK_BLACK, labelpad=10)
    ax_top.set_ylabel("Latent FID (symlog)", fontsize=22, color=INK_BLACK, labelpad=10)
    ax_top.set_title(
        "I-JEPA Sensitivity to Synthetic Corruptions",
        fontsize=26, color=INK_BLACK, fontweight="bold", pad=14,
    )
    legend = ax_top.legend(
        fontsize=20, loc="lower right",
        frameon=True, facecolor="white", edgecolor=INK_BLACK,
        labelcolor=INK_BLACK,
    )
    legend.get_frame().set_linewidth(1.0)

    # --- Row 2: placeholder for forecast lead-time plot ---
    style_axis(ax_bot)
    ax_bot.set_xlim(0, 240)
    ax_bot.set_ylim(1e-1, 1e3)
    ax_bot.set_yscale("log")
    ax_bot.set_xticks([0, 24, 48, 72, 120, 168, 240])
    ax_bot.set_xlabel("Forecast Lead Time (hours)", fontsize=22, color=INK_BLACK, labelpad=10)
    ax_bot.set_ylabel("Latent FID (log scale)", fontsize=22, color=INK_BLACK, labelpad=10)
    ax_bot.set_title(
        "I-JEPA: Forecast Realism vs Lead Time",
        fontsize=26, color=INK_BLACK, fontweight="bold", pad=14,
    )
    ax_bot.text(
        0.5, 0.5,
        "Work in Progress",
        ha="center", va="center",
        fontsize=36, color=INK_BLACK, fontweight="bold",
        transform=ax_bot.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor=PORCELAIN, edgecolor=INK_BLACK, linewidth=1.5,
        ),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    if output_path.suffix.lower() == ".pdf":
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved -> {output_path}\nSaved -> {png_path}")
    else:
        print(f"Saved -> {output_path}")
    plt.close(fig)


# -- Main -------------------------------------------------------------------
def compute_and_cache(args, cache_path):
    """Run the heavy FID sweep and persist the curves to `cache_path` (.npz)."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cluster:
        data_path = ERA5_CLUSTER
    elif args.large_local:
        data_path = ERA5_LARGE_LOCAL
    else:
        data_path = ERA5_LOCAL
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data:   {data_path}")
    print(f"Model:  {MODEL_PATH}")

    stats = (np.load(MEAN_PATH), np.load(STD_PATH))
    dataset = AtmosphereDataset(data_path, split="val", stats=stats, lazy=False)
    n_samples = min(args.n_samples, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False).tolist()
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    print(f"Reference samples: {n_samples}")

    model = build_model("ijepa", device=device, model_size="twin", embed_dim=512)
    model = load_model_checkpoint("ijepa", model, MODEL_PATH, device)
    model.eval()

    severities = np.linspace(0.0, 1.0, args.n_severity_steps)
    print(f"Severities: {severities.tolist()}")

    fid_noise = fid_curve(model, loader, device, apply_high_freq_noise, severities, "High-Freq Noise")
    fid_rotation = fid_curve(model, loader, device, apply_wind_channel_rotation, severities, "Channel Rotation")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        severities=severities,
        fid_noise=fid_noise,
        fid_rotation=fid_rotation,
        # bookkeeping so we can sanity-check loaded caches later
        n_samples=np.array(n_samples),
        seed=np.array(args.seed),
        model_path=np.array(str(MODEL_PATH)),
        data_path=np.array(str(data_path)),
    )
    print(f"Cached FID curves -> {cache_path}")
    return severities, fid_noise, fid_rotation


def load_cache(cache_path):
    """Load previously cached FID curves from ``cache_path`` and print provenance info."""
    data = np.load(cache_path, allow_pickle=True)
    print(f"Loaded cached FID curves <- {cache_path}")
    print(f"  n_samples={int(data['n_samples'])}  seed={int(data['seed'])}")
    print(f"  model={data['model_path']}")
    print(f"  data={data['data_path']}")
    return data["severities"], data["fid_noise"], data["fid_rotation"]


def main():
    """Compute (or load cached) FID-vs-severity curves and render the poster figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-severity-steps", type=int, default=5,
                        help="Number of severity levels evenly spaced in [0, 1].")
    parser.add_argument("--n-samples", type=int, default=400,
                        help="Number of ERA5 val samples used as the reference distribution.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--large-local", action="store_true",
                        help="Use the 5-year local ERA5 file.")
    parser.add_argument("--cluster", action="store_true",
                        help="Use the cluster ERA5 path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=str(REPO / "plots" / "poster_fid_severity_nontemporal.pdf"))
    parser.add_argument("--cache", default=str(REPO / "plots" / "poster_fid_severity_data_nontemporal.npz"),
                        help="Path to .npz cache for the computed FID curves.")
    parser.add_argument("--recompute", action="store_true",
                        help="Force re-evaluation; overwrite the cache if present.")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if cache_path.exists() and not args.recompute:
        severities, fid_noise, fid_rotation = load_cache(cache_path)
    else:
        if cache_path.exists():
            print(f"--recompute set; overwriting {cache_path}")
        severities, fid_noise, fid_rotation = compute_and_cache(args, cache_path)

    plot_figure(severities, fid_noise, fid_rotation, args.output)


if __name__ == "__main__":
    main()

"""2x3 grid of synthetic corruptions applied to one ERA5 snapshot, for the poster.

Mirrors the look of `plot_forecast_comparison.py` (cartopy + coastlines + big
captions below each panel). The four all-channel corruptions are shown on T2M;
the two wind-only corruptions are shown on U10 (T2M would be unchanged, and
U10 is signed so it shares the same diverging colormap).
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils.corruptions import (
    apply_gaussian_blur,
    apply_high_freq_noise,
    apply_gaussian_field_noise,
    apply_random_pixel_replace,
    apply_wind_patch_shuffle,
    apply_wind_channel_rotation,
)

mpl.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42

DATA = REPO / "data"
ERA5_PATH = DATA / "test_data_local.nc"
STATS_DIR = REPO / "checkpoints"

VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
T2M, U10, V10, _MSL = 0, 1, 2, 3


def load_sample(path: Path, time: np.datetime64):
    """Returns (raw_tensor[1,4,H,W], lat, lon) in physical units."""
    ds = xr.open_dataset(path)
    arrs = []
    for v in VARS:
        a = ds[v].sel(time=time)
        if a.dims[0] == "longitude":
            a = a.transpose("latitude", "longitude")
        arrs.append(a.values.astype(np.float32))
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    x = torch.from_numpy(np.stack(arrs, axis=0))[None]  # (1, 4, H, W)
    return x, lat, lon


def normalize(x: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    """Per-channel normalize a ``(1, 4, H, W)`` tensor with the dataset mean/std."""
    m = torch.from_numpy(mean.reshape(1, -1, 1, 1).astype(np.float32))
    s = torch.from_numpy(std.reshape(1, -1, 1, 1).astype(np.float32))
    return (x - m) / s


def main():
    """Render the poster's 2×3 corruption grid for one ERA5 snapshot and save as PDF."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2020-07-15T12:00:00")
    parser.add_argument("--severity", type=float, default=1.0,
                        help="Severity in [0, 2]; default 1.0 is mid-ladder.")
    parser.add_argument("--output", default=str(REPO / "plots" / "corruption_grid_2x3.pdf"))
    args = parser.parse_args()

    target = np.datetime64(args.date)
    x_raw, lat, lon = load_sample(ERA5_PATH, target)

    mean = np.load(STATS_DIR / "data_mean.npy")
    std = np.load(STATS_DIR / "data_std.npy")
    x_norm = normalize(x_raw, mean, std)  # corruption severity scales are calibrated for normalized inputs

    sev = args.severity
    panels = [
        ("Gaussian Blur",       apply_gaussian_blur(x_norm, sev),          T2M),
        ("High-Freq Noise",     apply_high_freq_noise(x_norm, sev),        T2M),
        ("GRF Noise",           apply_gaussian_field_noise(x_norm, sev),   T2M),
        ("Pixel Replace",       apply_random_pixel_replace(x_norm, sev),   T2M),
        ("Wind Patch Shuffle",  apply_wind_patch_shuffle(x_norm, sev,
                                                         patch_size=16),   U10),
        ("Wind Rotation",       apply_wind_channel_rotation(x_norm, sev),  U10),
    ]

    # Shared color limits per channel so all panels are directly comparable.
    t2m_stack = np.concatenate([p[1][0, T2M].numpy().ravel() for p in panels if p[2] == T2M])
    u10_stack = np.concatenate([p[1][0, U10].numpy().ravel() for p in panels if p[2] == U10])
    t2m_vmin, t2m_vmax = np.percentile(t2m_stack, [1, 99])
    u10_vmin, u10_vmax = np.percentile(u10_stack, [1, 99])
    sym = max(abs(u10_vmin), abs(u10_vmax))
    u10_vmin, u10_vmax = -sym, sym

    proj = ccrs.PlateCarree(central_longitude=0)
    data_crs = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        2, 3,
        figsize=(18, 8.4),
        subplot_kw={"projection": proj},
        constrained_layout=False,
    )
    axes = axes.ravel()

    lon_shifted = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon_shifted)
    lon_sorted = lon_shifted[order]

    for ax, (name, x_cor, ch) in zip(axes, panels):
        field = x_cor[0, ch].numpy()[:, order]
        if ch == T2M:
            vmin, vmax = t2m_vmin, t2m_vmax
        else:
            vmin, vmax = u10_vmin, u10_vmax

        ax.pcolormesh(
            lon_sorted, lat, field,
            cmap="RdBu_r", vmin=vmin, vmax=vmax,
            shading="auto", transform=data_crs,
            rasterized=True, antialiased=False, linewidth=0,
        )
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"),
                       edgecolor="black", linewidth=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(0.8)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.10,
                        wspace=0.04, hspace=0.30)

    for ax, (name, _, ch) in zip(axes, panels):
        bbox = ax.get_position()
        x_center = (bbox.x0 + bbox.x1) / 2
        var_tag = "T2M" if ch == T2M else "U10"
        fig.text(x_center, bbox.y0 - 0.025, f"{name}  ({var_tag})",
                 ha="center", va="top",
                 fontsize=22, fontweight="bold", color="black")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

"""2x3 grid of the same ERA5 snapshot, captioned for I-JEPA block illustration.

Mirrors the look of `plot_corruption_grid.py` (cartopy + coastlines + big
captions below each panel). All six panels show the identical original T2M
sample; downstream tooling will overlay context/target rectangles per panel.

Captions:
  top-left = Original, top-mid = Target 1, top-right = Target 2,
  bot-left = Context,  bot-mid = Target 3, bot-right = Target 4.
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

PANEL_LABELS = [
    "Original", "Target 1", "Target 2",
    "Context",  "Target 3", "Target 4",
]


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
    """Render the 2×3 base grid (identical ERA5 snapshot per panel) for I-JEPA block overlays."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2020-07-15T12:00:00")
    parser.add_argument("--channel", choices=["T2M", "U10"], default="T2M",
                        help="Which channel to plot in every panel.")
    parser.add_argument("--output", default=str(REPO / "plots" / "ijepa_blocks_base_2x3.pdf"))
    args = parser.parse_args()

    target = np.datetime64(args.date)
    x_raw, lat, lon = load_sample(ERA5_PATH, target)

    mean = np.load(STATS_DIR / "data_mean.npy")
    std = np.load(STATS_DIR / "data_std.npy")
    x_norm = normalize(x_raw, mean, std)

    ch = T2M if args.channel == "T2M" else U10
    field_full = x_norm[0, ch].numpy()

    # Color limits identical to the corruption grid style (1/99 percentile, symmetric for wind).
    vmin, vmax = np.percentile(field_full, [1, 99])
    if ch == U10:
        sym = max(abs(vmin), abs(vmax))
        vmin, vmax = -sym, sym

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
    field = field_full[:, order]

    for ax in axes:
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

    for ax, label in zip(axes, PANEL_LABELS):
        bbox = ax.get_position()
        x_center = (bbox.x0 + bbox.x1) / 2
        fig.text(x_center, bbox.y0 - 0.025, label,
                 ha="center", va="top",
                 fontsize=22, fontweight="bold", color="black")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

"""Side-by-side temperature comparison for poster: GraphCast | ERA5.

Renders the same valid day from each source with continent overlays. Big
captions sit directly below each subplot so a sticky note can cover them for
the guessing game.
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Poster uses TheSans (humanist sans-serif, custom .otf). Helvetica Neue is the
# closest match available locally; Avenir Next is a humanist alternative.
mpl.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType so text stays selectable


REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

SOURCES = [
    ("(A) GraphCast", DATA / "graphcast_surface_2020_lead24h.nc"),
    ("(B) ERA5",      DATA / "test_data_local.nc"),
]


def load_temperature(path: Path, time: np.datetime64) -> xr.DataArray:
    """Load the T2M field at the given valid time as a ``(latitude, longitude)`` DataArray."""
    ds = xr.open_dataset(path)
    t2m = ds["2m_temperature"].sel(time=time)
    # dataset stores (longitude, latitude) — transpose to (latitude, longitude)
    if t2m.dims[0] == "longitude":
        t2m = t2m.transpose("latitude", "longitude")
    return t2m


def main():
    """Render the side-by-side GraphCast vs ERA5 temperature comparison and save as PDF."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2020-07-15T12:00:00",
                        help="ISO timestamp present in all three datasets")
    parser.add_argument("--output", default=str(REPO / "plots" / "forecast_comparison_pair.pdf"))
    parser.add_argument("--cmap", default="RdBu_r",
                        help="Matplotlib colormap (blue=cold, red=hot)")
    args = parser.parse_args()

    target = np.datetime64(args.date)

    fields = [(name, load_temperature(path, target)) for name, path in SOURCES]

    # Shared color limits across both so they are directly comparable.
    stacked = np.concatenate([f.values.ravel() for _, f in fields])
    vmin, vmax = np.percentile(stacked, [1, 99])

    proj = ccrs.PlateCarree(central_longitude=0)
    data_crs = ccrs.PlateCarree()

    fig, axes = plt.subplots(
        1, 2,
        figsize=(18, 5.5),
        subplot_kw={"projection": proj},
        constrained_layout=False,
    )

    for ax, (name, field) in zip(axes, fields):
        lon = field["longitude"].values
        lat = field["latitude"].values

        # cartopy expects longitudes in [-180, 180] for nicest seams; shift the
        # data and roll the array so 0° appears in the middle.
        lon_shifted = np.where(lon > 180, lon - 360, lon)
        order = np.argsort(lon_shifted)
        lon_sorted = lon_shifted[order]
        data = field.values[:, order]

        im = ax.pcolormesh(
            lon_sorted, lat, data,
            cmap=args.cmap, vmin=vmin, vmax=vmax,
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

    # Layout: leave generous space under each panel for the big caption.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.18, wspace=0.04)

    # Big captions directly below each subplot — these are what the sticky
    # notes cover.
    for ax, (name, _) in zip(axes, fields):
        bbox = ax.get_position()
        x_center = (bbox.x0 + bbox.x1) / 2
        fig.text(x_center, bbox.y0 - 0.08, name,
                 ha="center", va="top",
                 fontsize=34, fontweight="bold", color="black")

    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()

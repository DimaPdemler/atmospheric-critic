"""Poster figure: 2-panel I-JEPA latent-FID summary.

Top panel:    FID vs corruption severity   (default: plots/poster_fid_severity_data_nontemporal.npz)
Bottom panel: FID vs forecast lead time    (default: plots/poster_fid_leadtime_data_nontemporal.npz)

Pass --severity-npz / --leadtime-npz to point at the _temporal variants.

Pure plotting — no torch, no project models. Edit freely to tweak style
without re-running the heavy compute scripts.
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# -- Poster palette (mirror scripts/plot_poster_fid_severity.py) ------------
INK_BLACK = "#0D1821"
YALE_BLUE = "#344966"
BLUSH_PINK = "#E6AACE"
PORCELAIN = "#F0F4EF"
AMBER = "#C28F2C"  # third forecast line (Pangu)

mpl.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42


def style_axis(ax):
    """Apply the shared poster styling (colors, ticks, spines, grid) to an axis."""
    ax.set_facecolor("white")
    ax.tick_params(colors=INK_BLACK, labelsize=18, width=1.0)
    for spine in ax.spines.values():
        spine.set_edgecolor(INK_BLACK)
        spine.set_linewidth(1.2)
    ax.grid(True, which="both", alpha=0.25, color=INK_BLACK, linewidth=0.6)


def draw_severity_panel(ax, sev_npz_path):
    """Draw the FID-vs-corruption-severity panel from a precomputed .npz file."""
    d = np.load(sev_npz_path)
    severities   = d["severities"]
    fid_noise    = np.maximum(d["fid_noise"], 0.0)
    fid_rotation = np.maximum(d["fid_rotation"], 0.0)

    style_axis(ax)
    ax.plot(
        severities, fid_noise,
        marker="o", markersize=11, lw=3.0,
        color=YALE_BLUE, label="High-Freq Noise",
        markeredgecolor=INK_BLACK, markeredgewidth=0.8,
    )
    ax.plot(
        severities, fid_rotation,
        marker="s", markersize=11, lw=3.0,
        color=BLUSH_PINK, label="Wind-Vector Rotation",
        markeredgecolor=INK_BLACK, markeredgewidth=0.8,
    )
    ax.set_yscale("symlog", linthresh=1e-4, linscale=0.5)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelleft=False)
    ax.set_xlim(0.0, 1.05)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Corruption Severity", fontsize=22, color=INK_BLACK, labelpad=10)
    ax.set_ylabel("Latent FID (symlog)", fontsize=22, color=INK_BLACK, labelpad=10)
    ax.set_title(
        "I-JEPA Sensitivity to Synthetic Corruptions",
        fontsize=26, color=INK_BLACK, fontweight="bold", pad=14,
    )
    legend = ax.legend(
        fontsize=20, loc="lower right",
        frameon=True, facecolor="white", edgecolor=INK_BLACK,
        labelcolor=INK_BLACK,
    )
    legend.get_frame().set_linewidth(1.0)


def draw_leadtime_panel(ax, lt_npz_path):
    """Draw the FID-vs-forecast-lead-time panel from a precomputed .npz file."""
    d = np.load(lt_npz_path)
    leads          = d["lead_times_hours"]
    fid_ifs_hres   = np.maximum(d["fid_ifs_hres"], 0.0)
    fid_graphcast  = np.maximum(d["fid_graphcast"], 0.0)
    fid_pangu      = np.maximum(d["fid_pangu"], 0.0) if "fid_pangu" in d.files else None

    style_axis(ax)
    ax.plot(
        leads, fid_ifs_hres,
        marker="^", markersize=11, lw=3.0,
        color=YALE_BLUE, label="IFS HRES",
        markeredgecolor=INK_BLACK, markeredgewidth=0.8,
    )
    ax.plot(
        leads, fid_graphcast,
        marker="D", markersize=11, lw=3.0,
        color=BLUSH_PINK, label="GraphCast",
        markeredgecolor=INK_BLACK, markeredgewidth=0.8,
    )
    if fid_pangu is not None:
        ax.plot(
            leads, fid_pangu,
            marker="v", markersize=11, lw=3.0,
            color=AMBER, label="Pangu-Weather",
            markeredgecolor=INK_BLACK, markeredgewidth=0.8,
        )
    ax.set_yscale("symlog", linthresh=1e-4, linscale=0.5)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelcolor="#999999", labelsize=14)
    ax.set_xscale("log")
    ax.set_xticks(list(leads))
    ax.set_xticklabels([str(int(x)) for x in leads])
    ax.minorticks_off()
    ax.set_xlim(leads.min() * 0.85, leads.max() * 1.15)
    ax.set_xlabel("Forecast Lead Time (hours)", fontsize=22, color=INK_BLACK, labelpad=10)
    ax.set_ylabel("Latent FID (symlog)", fontsize=22, color=INK_BLACK, labelpad=10)
    ax.set_title(
        "Held-Out Forecast Divergence from ERA5",
        fontsize=26, color=INK_BLACK, fontweight="bold", pad=14,
    )
    legend = ax.legend(
        fontsize=20, loc="lower right",
        frameon=True, facecolor="white", edgecolor=INK_BLACK,
        labelcolor=INK_BLACK,
    )
    legend.get_frame().set_linewidth(1.0)


def parse_args():
    """Parse CLI arguments (input .npz paths and output figure path)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--severity-npz", type=Path,
        default=Path("plots/poster_fid_severity_data_nontemporal.npz"),
    )
    parser.add_argument(
        "--leadtime-npz", type=Path,
        default=Path("plots/poster_fid_leadtime_data_nontemporal.npz"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("plots/poster_fid_combined_nontemporal.pdf"),
    )
    return parser.parse_args()


def main():
    """Assemble the two-panel poster FID figure from the cached .npz data and save it."""
    args = parse_args()

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.5, 11), constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    draw_severity_panel(ax_top, args.severity_npz)
    draw_leadtime_panel(ax_bot, args.leadtime_npz)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"Saved: {args.output}")
    if args.output.suffix.lower() == ".pdf":
        png_path = args.output.with_suffix(".png")
        fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()

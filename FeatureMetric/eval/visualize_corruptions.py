"""Visualize blur, high-freq noise & GRF noise corruption ladders."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from utils.dataset import AtmosphereDataset
from utils.models import MaskedAutoencoderViT
from utils.corruptions import (
    apply_gaussian_blur,
    apply_high_freq_noise,
    apply_gaussian_field_noise,
    apply_random_pixel_replace,
    apply_wind_patch_shuffle,
    apply_wind_channel_rotation,
    get_corruption_ladder,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local.nc"
STATS_DIR = Path("checkpoints")

VAR_NAMES = ["2m Temperature", "10m U-Wind", "10m V-Wind", "Mean Sea Level Pressure"]
CMAPS = ["RdBu_r", "coolwarm", "coolwarm", "viridis"]
U10_CHANNEL = 1
V10_CHANNEL = 2
MSL_CHANNEL = 3


def _orient_for_display(field):
    """Flip a field to geographic orientation (north up, longitudes mirrored)."""
    # Match the geographic orientation verified by the reference map:
    # north up and longitudes mirrored relative to the raw tensor layout.
    return np.fliplr(field)


def _show_scalar_panel(ax, field, var_idx, vmin, vmax):
    """Draw a single scalar field as a heatmap on ``ax`` with the variable's colormap."""
    ax.imshow(
        _orient_for_display(field),
        cmap=CMAPS[var_idx],
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        origin="lower",
    )


def _show_msl_with_wind(ax, msl_field, u_field, v_field, u_ref=None, v_ref=None):
    """Draw MSL as a heatmap with wind vectors overlaid as a quiver plot.

    If ``u_ref``/``v_ref`` are given, the un-corrupted wind is drawn faintly
    underneath so the rotation/shuffle effect is visible.
    """
    _show_scalar_panel(ax, msl_field, MSL_CHANNEL, float(msl_field.min()), float(msl_field.max()))
    u_field = _orient_for_display(u_field)
    v_field = _orient_for_display(v_field)
    step = 12
    y = np.arange(0, u_field.shape[0], step)
    x = np.arange(0, u_field.shape[1], step)
    xx, yy = np.meshgrid(x, y)
    if u_ref is not None and v_ref is not None:
        u_ref = _orient_for_display(u_ref)
        v_ref = _orient_for_display(v_ref)
        ref_quiver = ax.quiver(
            xx,
            yy,
            u_ref[::step, ::step],
            v_ref[::step, ::step],
            color="gray",
            alpha=0.55,
            scale=40,
            width=0.0015,
        )
        ref_quiver.set_linestyle(":")
    uu = u_field[::step, ::step]
    vv = v_field[::step, ::step]
    ax.quiver(xx, yy, uu, vv, color="black", scale=40, width=0.002)

def main():
    """Render corruption-ladder grids (per variable) showing each corruption at rising severity."""
    mean = np.load(STATS_DIR / "data_mean.npy")
    std = np.load(STATS_DIR / "data_std.npy")
    stats = (mean, std)
    model_patch_size = MaskedAutoencoderViT().patch_size

    corruption_groups = {
        "standard_corruptions": {
            "title": "Standard Corruptions",
            "show_vars": [0, 3],
            "specs": [
                ("Gaussian Blur", "blur", apply_gaussian_blur),
                ("High-Freq Noise", "noise", apply_high_freq_noise),
                ("GRF Noise", "grf", apply_gaussian_field_noise),
                ("Random Pixel Replace", "pixel_replace", apply_random_pixel_replace),
            ],
        },
        "physical_decoupling": {
            "title": "Physical Decoupling",
            "show_vars": [1, 2, 3],
            "specs": [
                (
                    "Spatial Shuffle (Wind Only)",
                    "wind_patch_shuffle",
                    partial(apply_wind_patch_shuffle, patch_size=model_patch_size),
                ),
                ("Channel Rotation", "wind_rotation", apply_wind_channel_rotation),
            ],
        },
    }

    ds = AtmosphereDataset(DATA_PATH, split="val", stats=stats, lazy=False)
    sample = ds[0].unsqueeze(0)  # (1, 4, H, W)

    base_plots_dir = Path("plots")
    base_plots_dir.mkdir(exist_ok=True)

    for group_key, group in corruption_groups.items():
        group_plots_dir = base_plots_dir / group_key
        group_plots_dir.mkdir(parents=True, exist_ok=True)

        for var_idx in group["show_vars"]:
            sevs = get_corruption_ladder(group["specs"][0][1])
            n_cols = len(sevs)
            n_rows = len(group["specs"])
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
            if n_rows == 1:
                axes = np.expand_dims(axes, axis=0)
            sample_var = sample[0, var_idx].numpy()
            vmin = float(sample_var.min())
            vmax = float(sample_var.max())

            for col, sev in enumerate(sevs):
                for row, (corr_name, _, apply_fn) in enumerate(group["specs"]):
                    corrupted = apply_fn(sample, sev)
                    if (
                        group_key == "physical_decoupling"
                        and corr_name == "Channel Rotation"
                        and var_idx == MSL_CHANNEL
                    ):
                        _show_msl_with_wind(
                            axes[row, col],
                            sample[0, MSL_CHANNEL].numpy(),
                            corrupted[0, U10_CHANNEL].numpy(),
                            corrupted[0, V10_CHANNEL].numpy(),
                            u_ref=sample[0, U10_CHANNEL].numpy(),
                            v_ref=sample[0, V10_CHANNEL].numpy(),
                        )
                    else:
                        _show_scalar_panel(axes[row, col], corrupted[0, var_idx].numpy(), var_idx, vmin, vmax)
                    if row == 0:
                        axes[row, col].set_title(f"sev={sev:.4f}", fontsize=13)
                    axes[row, col].axis("off")

            for row, (corr_name, _, _) in enumerate(group["specs"]):
                axes[row, 0].axis("on")
                axes[row, 0].set_xticks([])
                axes[row, 0].set_yticks([])
                axes[row, 0].set_ylabel(corr_name, fontsize=14, rotation=90, labelpad=10)

            fig.suptitle(f"{group['title']} — {VAR_NAMES[var_idx]}", fontsize=16)
            fig.tight_layout()

            safe_var = VAR_NAMES[var_idx].lower().replace(" ", "_").replace("-", "_")
            fname = f"corruption_ladder_combined_{safe_var}.png"
            fig.savefig(group_plots_dir / fname, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {group_plots_dir / fname}")


if __name__ == "__main__":
    main()

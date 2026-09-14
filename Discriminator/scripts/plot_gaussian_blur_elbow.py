"""Locate the practical low-severity elbow of training-time Gaussian blur."""

import csv
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

try:
    from .corruptions import (
        GAUSSIAN_BLUR_MAX_SIGMA_PIXELS,
        apply_gaussian_blur,
        gaussian_blur_effective_severity,
    )
    from .monthly_split import select_era5_split
    from .train_discriminator import safe_open_dataset
except ImportError:
    from corruptions import (
        GAUSSIAN_BLUR_MAX_SIGMA_PIXELS,
        apply_gaussian_blur,
        gaussian_blur_effective_severity,
    )
    from monthly_split import select_era5_split
    from train_discriminator import safe_open_dataset


def severity_grid(minimum, maximum, count):
    """Return an inclusive, validated severity grid."""
    minimum, maximum, count = float(minimum), float(maximum), int(count)
    if not (0.0 <= minimum <= maximum <= 1.0):
        raise ValueError("Blur severities must satisfy 0 <= min <= max <= 1.")
    if count < 2:
        raise ValueError("gaussian_blur_elbow.num_severities must be at least 2.")
    return np.linspace(minimum, maximum, count)


def evenly_spaced_indices(size, maximum):
    """Return deterministic time positions, as used by other baseline probes."""
    size, maximum = int(size), int(maximum)
    if size <= 0:
        return []
    if maximum <= 0 or size <= maximum:
        return list(range(size))
    return np.linspace(0, size - 1, maximum, dtype=int).tolist()


def normalized_training_batch(dataset, variables, means, stds, indices):
    """Build the same normalized CxHxW ERA5 tensors used by the trainer."""
    samples = []
    for index in indices:
        sample = dataset.isel(time=int(index))
        channels = []
        for variable in variables:
            values = np.asarray(sample[variable].values, dtype=np.float32)
            normalized = (values - means[variable]) / stds[variable]
            channels.append(np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0))
        samples.append(np.stack(channels))
    return torch.from_numpy(np.stack(samples)).float()


def blur_change_statistics(clean, severity):
    """Return per-sample/channel absolute and relative RMS blur changes."""
    blurred = apply_gaussian_blur(clean, float(severity))
    difference = (blurred - clean).flatten(start_dim=2)
    source = clean.flatten(start_dim=2)
    absolute_rms = difference.square().mean(dim=-1).sqrt()
    source_rms = source.square().mean(dim=-1).sqrt().clamp_min(1e-12)
    return absolute_rms.cpu().numpy().ravel(), (absolute_rms / source_rms).cpu().numpy().ravel()


def quantiles(values):
    return np.quantile(np.asarray(values, dtype=np.float64), [0.1, 0.5, 0.9])


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows, output_path):
    severity = np.asarray([row["severity"] for row in rows])
    sigma = np.asarray([row["sigma_pixels"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for axis, prefix, label in (
        (axes[0], "absolute_rms", "RMS(field$_{blur}$ − field)"),
        (axes[1], "relative_rms", "RMS change / RMS(field)"),
    ):
        low = np.asarray([row[f"{prefix}_p10"] for row in rows])
        median = np.asarray([row[f"{prefix}_p50"] for row in rows])
        high = np.asarray([row[f"{prefix}_p90"] for row in rows])
        axis.fill_between(severity, low, high, alpha=0.22, label="10–90% over samples/channels")
        axis.plot(severity, median, color="C0", linewidth=2, label="median")
        axis.set_xlabel("Gaussian blur severity")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_title("Absolute change in training-standardized units")
    axes[1].set_title("Relative spatial change")
    figure.suptitle(
        f"Gaussian blur low-severity sweep (σ={sigma[0]:.3f}–{sigma[-1]:.3f} pixels)",
        y=1.03,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(cfg):
    settings = cfg.gaussian_blur_elbow
    variables = list(cfg.variables) if cfg.variables else [str(cfg.selected_variable)]
    dataset = safe_open_dataset(cfg.real_nc_file)
    if cfg.get("level") is not None:
        level_name = "level" if "level" in dataset.dims else "pressure_level"
        dataset = dataset.sel({level_name: cfg.level})
    dataset = select_era5_split(dataset, cfg, "train", coverage="corruption")
    if dataset.sizes.get("time", 0) == 0:
        raise ValueError("No ERA5 samples remain in the monthly corruption training split.")
    missing = [variable for variable in variables if variable not in dataset.data_vars]
    if missing:
        raise ValueError(f"ERA5 data is missing configured variable(s): {missing}")

    means = {variable: float(dataset[variable].mean()) for variable in variables}
    stds = {variable: max(float(dataset[variable].std()), 1e-8) for variable in variables}
    indices = evenly_spaced_indices(dataset.sizes["time"], settings.max_samples)
    severities = severity_grid(settings.min_severity, settings.max_severity, settings.num_severities)
    values = {float(severity): [[], []] for severity in severities}

    for start in range(0, len(indices), int(settings.batch_size)):
        clean = normalized_training_batch(
            dataset, variables, means, stds, indices[start:start + int(settings.batch_size)]
        )
        for severity in severities:
            absolute, relative = blur_change_statistics(clean, severity)
            values[float(severity)][0].append(absolute)
            values[float(severity)][1].append(relative)

    rows = []
    for severity in severities:
        absolute = np.concatenate(values[float(severity)][0])
        relative = np.concatenate(values[float(severity)][1])
        absolute_q, relative_q = quantiles(absolute), quantiles(relative)
        rows.append({
            "severity": float(severity),
            "sigma_pixels": float(
                gaussian_blur_effective_severity(severity)
                * GAUSSIAN_BLUR_MAX_SIGMA_PIXELS
            ),
            "absolute_rms_p10": float(absolute_q[0]),
            "absolute_rms_p50": float(absolute_q[1]),
            "absolute_rms_p90": float(absolute_q[2]),
            "relative_rms_p10": float(relative_q[0]),
            "relative_rms_p50": float(relative_q[1]),
            "relative_rms_p90": float(relative_q[2]),
            "n_sample_channels": int(absolute.size),
        })
    output_csv, output_plot = Path(str(settings.output_csv)), Path(str(settings.output_plot))
    write_csv(rows, output_csv)
    plot_rows(rows, output_plot)
    dataset.close()
    print(f"Saved Gaussian-blur elbow data to: {output_csv}")
    print(f"Saved Gaussian-blur elbow plot to: {output_plot}")


@hydra.main(version_base=None, config_path="../conf", config_name="gaussian_blur_elbow")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()

"""Render the real paper plot layouts from deterministic synthetic CSV data."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

try:
    from .baseline_pipeline_tracking import PipelineTracker
except ImportError:
    from baseline_pipeline_tracking import PipelineTracker

try:
    from .train_target_discriminator_baselines import (
        _plot_logit_histogram, _plot_logit_histogram_overlay, safe_target_name,
    )
except ImportError:
    from train_target_discriminator_baselines import (
        _plot_logit_histogram, _plot_logit_histogram_overlay, safe_target_name,
    )

try:
    from .plot_standard_metric_baselines import (
        ERA5_NULL_LABEL,
        baseline_output_dir,
        corruption_levels,
        joint_variable_name,
        metric_names_from_config,
        plot_saved_standard_metric_baselines,
        plot_forecast_logit_histogram_gallery,
        write_scwd_anchor_diagnostics,
        write_global_mean_wasserstein_diagnostics,
        variables_from_config,
    )
except ImportError:
    from plot_standard_metric_baselines import (
        ERA5_NULL_LABEL,
        baseline_output_dir,
        corruption_levels,
        joint_variable_name,
        metric_names_from_config,
        plot_saved_standard_metric_baselines,
        plot_forecast_logit_histogram_gallery,
        write_scwd_anchor_diagnostics,
        write_global_mean_wasserstein_diagnostics,
        variables_from_config,
    )


METRIC_SCALES = {
    "mean_bias": 0.08,
    "std_ratio_error": 0.12,
    "crps_like_field_energy": 0.25,
    "zonal_energy_spectrum_l2": 0.4,
    "zonal_energy_spectrum_log_l2": 1.8,
    "sliced_wasserstein": 0.3,
    "sliced_wasserstein_lon_corrected": 0.3,
    "global_mean_wasserstein": 0.7,
    "mmd_rbf": 0.15,
    "scwd": 0.5,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Preview-run root. By default, create a timestamped directory below "
            "results/paper_layout_previews/."
        ),
    )
    parser.add_argument("--text-width-inches", type=float, default=5.5)
    parser.add_argument(
        "--repeated-plot-mode", choices=("all", "representative_only"), default="all",
        help=("Use representative_only to render one example of repeated single-target "
              "templates while retaining galleries and multi-target figures."),
    )
    parser.add_argument(
        "--pdf", action=argparse.BooleanOptionalAction, default=True,
        help="Save PDF companions (enabled by default for manuscript layout checks).",
    )
    parser.add_argument(
        "--include", action="append", default=[],
        help="Optional output filename/path glob; repeat to restrict plot families.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=False,
        help="Upload preview images, data, config, and complete bundles to W&B.",
    )
    parser.add_argument(
        "--wandb-project", default="weather-discriminator-baselines",
        help="W&B project used when --wandb is enabled.",
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument(
        "--wandb-name", default=None,
        help="Preview group/name prefix (default: generated local run-directory name).",
    )
    return parser.parse_args()


def load_config():
    config_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="baseline_pipeline")


def default_run_dir():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("results") / "paper_layout_previews" / f"dummy-{timestamp}"


def metric_values(metric_names, magnitude, phase):
    values = {}
    for metric_index, metric_name in enumerate(metric_names):
        scale = METRIC_SCALES.get(metric_name, 1.0)
        value = scale * max(0.0, magnitude + 0.025 * np.sin(phase + metric_index))
        spread = max(scale * 0.025, abs(value) * 0.09)
        values[metric_name] = float(value)
        values[f"{metric_name}_lower"] = float(value - spread)
        values[f"{metric_name}_upper"] = float(value + spread)
    return values


def dummy_standard_rows(cfg, metric_names, variables, seed=2026):
    rng = np.random.default_rng(seed)
    variable = joint_variable_name(variables)
    leads = [int(value) for value in cfg.lead_times]
    models = list(cfg.baseline.forecast_files)
    corruptions = [str(value) for value in cfg.baseline.corruptions]

    null = {
        "label": ERA5_NULL_LABEL, "variable": variable, "lead_hour": 0,
        "is_null": True, "n_samples": 1000, "pairwise_n_samples": 256,
        "n_resamples": 10,
    }
    null.update(metric_values(metric_names, 0.08, 0.0))
    lead_rows = [null]
    maximum_lead = max(leads)
    for model_index, model in enumerate(models):
        model_offset = 0.035 * model_index
        for lead_index, lead in enumerate(leads):
            magnitude = 0.12 + model_offset + 0.72 * (lead / maximum_lead) ** 0.72
            magnitude += float(rng.normal(0.0, 0.008))
            row = {
                "label": model, "variable": variable, "lead_hour": lead,
                "is_null": False, "n_samples": 1000, "pairwise_n_samples": 256,
                "n_resamples": 10,
            }
            row.update(metric_values(metric_names, magnitude, model_index + lead_index / 3))
            lead_rows.append(row)

    corruption_rows = []
    for corruption_index, corruption in enumerate(corruptions):
        null_row = {
            "label": ERA5_NULL_LABEL, "variable": variable,
            "corruption": corruption, "severity": 0.0, "is_null": True,
            "n_samples": 1000, "pairwise_n_samples": 256, "n_resamples": 10,
        }
        null_row.update(metric_values(metric_names, 0.075, corruption_index))
        corruption_rows.append(null_row)
        levels = np.asarray(corruption_levels(corruption, cfg), dtype=float)
        maximum = max(float(levels.max()), 1e-12)
        for level_index, severity in enumerate(levels):
            relative = float(severity) / maximum
            magnitude = 0.02 + 0.78 * relative ** (0.7 + 0.08 * (corruption_index % 4))
            row = {
                "label": "ERA5 test corrupted vs matched test", "variable": variable,
                "corruption": corruption, "severity": float(severity), "is_null": False,
                "n_samples": 1000, "pairwise_n_samples": 256, "n_resamples": 10,
            }
            row.update(metric_values(metric_names, magnitude, corruption_index + level_index / 2))
            corruption_rows.append(row)
    return lead_rows, corruption_rows


def dummy_discriminator_rows(cfg, variables):
    rows = []
    input_variables = ",".join(variables)
    leads = [int(value) for value in cfg.lead_times]
    models = list(cfg.baseline.forecast_files)
    corruptions = [str(value) for value in cfg.baseline.corruptions]
    targets = [
        ("forecast", model, np.asarray(leads, dtype=float)) for model in models
    ] + [
        ("corruption", corruption, np.asarray(corruption_levels(corruption, cfg), dtype=float))
        for corruption in corruptions
    ]
    for target_index, (kind, target, coordinates) in enumerate(targets):
        null_score = 0.04 + 0.008 * (target_index % 4)
        common = {
            "architecture": "squeezenet", "checkpoint_path": "SYNTHETIC_PREVIEW_ONLY",
            "checkpoint_sha256": "", "input_variables": input_variables,
            "encoder_pretraining": "", "kind": kind, "target": target,
            "n_samples": 1000, "ep_train": -1.0, "ep_n_samples": 1000,
            "n_resamples": 5,
        }
        rows.append({
            **common, "x": 0.0, "source": ERA5_NULL_LABEL,
            "is_era5_test_null": True, "score": null_score,
            "score_lower": null_score - 0.025, "score_upper": null_score + 0.025,
            "stderr": 0.01,
        })
        maximum = max(float(coordinates.max()), 1e-12)
        for coordinate_index, coordinate in enumerate(coordinates):
            relative = float(coordinate) / maximum
            score = 0.03 + (0.55 + 0.05 * (target_index % 5)) * relative ** 0.8
            score += 0.018 * np.sin(target_index + coordinate_index)
            rows.append({
                **common, "x": float(coordinate), "source": target,
                "is_era5_test_null": False, "score": float(score),
                "score_lower": float(score - 0.04), "score_upper": float(score + 0.04),
                "stderr": 0.015,
            })
    return rows


def include_requests(patterns, keywords):
    """Return true when an unrestricted or matching preview family was requested."""
    if not patterns:
        return True
    lowered = [str(pattern).lower() for pattern in patterns]
    return any(keyword in pattern for pattern in lowered for keyword in keywords)


def write_blank_corruption_disturbances(cfg, variables, output_root):
    """Persist layout-only blank fields in the production corruption-gallery schema."""
    corruptions = [str(value) for value in cfg.baseline.corruptions]
    levels = np.stack([
        np.asarray(corruption_levels(corruption, cfg), dtype=np.float64)
        for corruption in corruptions
    ])
    latitudes = np.linspace(-90.0, 90.0, 31, dtype=np.float64)
    longitudes = np.linspace(0.0, 360.0, 60, endpoint=False, dtype=np.float64)
    shape = (
        len(corruptions), levels.shape[1], len(variables),
        len(latitudes), len(longitudes),
    )
    blank = np.zeros(shape, dtype=np.float32)
    dataset = xr.Dataset(
        data_vars={
            "disturbance": (
                ("corruption", "severity_index", "variable", "latitude", "longitude"),
                blank,
            ),
            "corrupted_field": (
                ("corruption", "severity_index", "variable", "latitude", "longitude"),
                blank.copy(),
            ),
            "severity": (("corruption", "severity_index"), levels),
        },
        coords={
            "corruption": corruptions,
            "variable": variables,
            "latitude": latitudes,
            "longitude": longitudes,
        },
        attrs={
            "timestamp": "SYNTHETIC BLANK PLACEHOLDER",
            "description": "Blank layout placeholders; these are not atmospheric fields.",
            "synthetic_layout_preview": 1,
        },
    )
    path = output_root / "data" / "corruption_disturbances.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)
    dataset.close()
    return path


def write_blank_scwd_diagnostics(cfg, variables, output_root):
    """Persist zero-valued maps in the production SCWD diagnostic schema."""
    latitudes = np.linspace(-82.5, 82.5, 12, dtype=np.float64)
    longitudes = np.linspace(0.0, 360.0, 24, endpoint=False, dtype=np.float64)
    blank = np.zeros((len(latitudes), len(longitudes)), dtype=np.float32)
    blank_fields = np.zeros(
        (len(variables), len(latitudes), len(longitudes)), dtype=np.float32
    )

    def diagnostic(label, comparison_kind, lead_hour=0, severity=np.nan):
        return {
            "label": str(label),
            "comparison_kind": str(comparison_kind),
            "severity": float(severity),
            "lead_hour": int(lead_hour),
            "field_names": list(variables),
            "anchor_latitudes": latitudes,
            "anchor_longitudes": longitudes,
            "anchor_transport_cost": blank.copy(),
            "anchor_local_wasserstein": blank.copy(),
            "anchor_mean_response_difference": blank_fields.copy(),
            "top_wasserstein_distributions": [{
                "anchor_index": 0, "wasserstein": 0.0,
                "latitude": float(latitudes[0]), "longitude": float(longitudes[0]),
                "candidate": np.zeros((16, len(variables)), dtype=np.float32),
                "reference": np.zeros((16, len(variables)), dtype=np.float32),
            }],
            "scwd": 0.0,
            "scwd_order": 2.0,
        }

    diagnostics = [diagnostic(ERA5_NULL_LABEL, "null")]
    diagnostics.extend(
        diagnostic(model, "forecast", lead_hour=lead)
        for model in cfg.baseline.forecast_files
        for lead in cfg.lead_times
    )
    diagnostics.extend(
        diagnostic(
            corruption,
            "corruption",
            severity=float(np.max(corruption_levels(corruption, cfg))),
        )
        for corruption in cfg.baseline.corruptions
    )
    write_scwd_anchor_diagnostics(diagnostics, output_root)
    return output_root / "data" / "scwd_anchor_contributions.nc"


def write_dummy_global_mean_diagnostics(cfg, variables, output_root, seed):
    """Persist synthetic distributions in the production global-mean WD schema."""
    rng = np.random.default_rng(seed)
    n_fields = len(variables)
    grid = {
        "lower": np.full(n_fields, -3.0, dtype=np.float64),
        "upper": np.full(n_fields, 3.0, dtype=np.float64),
        "n_bins": int(cfg.baseline.global_mean_wd_bins),
    }

    def diagnostic(label, comparison_kind, coordinate, maximum):
        relative = 0.0 if maximum <= 0.0 else float(coordinate) / float(maximum)
        reference = np.clip(
            rng.normal(0.0, 0.7, size=(80, n_fields)), -2.9, 2.9
        ).astype(np.float32)
        candidate = np.clip(
            rng.normal(0.12 * relative, 0.7 + 0.08 * relative, size=(80, n_fields)),
            -2.9,
            2.9,
        ).astype(np.float32)
        return {
            "label": str(label),
            "comparison_kind": str(comparison_kind),
            "severity": float(coordinate) if comparison_kind == "corruption" else np.nan,
            "lead_hour": int(coordinate) if comparison_kind == "forecast" else 0,
            "candidate": candidate,
            "reference": reference,
            "distance": float(0.18 * relative),
            "n_bins": int(grid["n_bins"]),
            "grid": grid,
        }

    leads = [int(value) for value in cfg.lead_times]
    diagnostics = [
        {
            **diagnostic(ERA5_NULL_LABEL, "null", 0.0, 1.0),
            "distance": 0.02,
        }
    ]
    diagnostics.extend(
        diagnostic(model, "forecast", lead, max(leads))
        for model in cfg.baseline.forecast_files
        for lead in leads
    )
    for corruption in cfg.baseline.corruptions:
        levels = np.asarray(corruption_levels(corruption, cfg), dtype=float)
        diagnostics.extend(
            diagnostic(corruption, "corruption", severity, float(levels.max()))
            for severity in levels
        )
    write_global_mean_wasserstein_diagnostics(
        diagnostics, variables, output_root
    )
    return output_root / "data" / "global_mean_wasserstein_distributions.nc"



def dummy_logit_groups(coordinates, coordinate_labels, seed, shift_scale):
    rng = np.random.default_rng(seed)
    groups = []
    maximum = max(float(np.max(coordinates)), 1e-12)
    for index, (coordinate, label) in enumerate(zip(coordinates, coordinate_labels)):
        reference = rng.normal(loc=1.0, scale=0.9, size=800)
        relative = float(coordinate) / maximum
        candidate = rng.normal(
            loc=0.95 - shift_scale * relative,
            scale=0.9 + 0.12 * relative,
            size=800,
        )
        groups.append({
            "label": str(label),
            "reference": reference,
            "candidate": candidate,
            "index": index,
        })
    return groups


def plot_dummy_logit_histograms(cfg, output_root, seed):
    """Render representative per-point and overlay production logit histograms."""
    architecture = "squeezenet"
    paths = []
    forecast = next(iter(cfg.baseline.forecast_files))
    leads = np.asarray([int(value) for value in cfg.lead_times], dtype=float)
    forecast_groups = dummy_logit_groups(
        leads, [f"+{int(value)}h" for value in leads], seed, shift_scale=2.0,
    )
    forecast_root = (
        output_root / "plots" / "target_logit_distributions" / architecture
        / "forecast" / safe_target_name(forecast)
    )
    for group, lead in zip(forecast_groups, leads):
        path = forecast_root / f"lead_{int(lead):03d}h.png"
        _plot_logit_histogram(
            group["reference"], group["candidate"],
            f"{architecture}: {forecast} ({group['label']})",
            forecast, path,
        )
        paths.append(path)
    overlay = forecast_root / "all_lead_times.png"
    _plot_logit_histogram_overlay(
        forecast_groups,
        f"{architecture}: {forecast} — all lead times",
        overlay,
    )
    paths.append(overlay)

    gallery_rows = []
    selected_models = list(
        ((cfg.get("plotting", {}) or {}).get("logit_histogram_gallery", {}) or {}).get(
            "models", list(cfg.baseline.forecast_files)
        )
    )
    for model_index, model in enumerate(selected_models):
        groups = dummy_logit_groups(
            leads,
            [f"+{int(value)}h" for value in leads],
            seed + 20 + model_index,
            shift_scale=1.5 + 0.25 * model_index,
        )
        for value in groups[0]["reference"]:
            gallery_rows.append({
                "architecture": architecture, "kind": "forecast",
                "target": model, "role": "candidate", "source": ERA5_NULL_LABEL,
                "lead_hour": "", "logit": float(value), "resample_id": "learned_04",
            })
        for group, lead in zip(groups, leads):
            for value in group["candidate"]:
                gallery_rows.append({
                    "architecture": architecture, "kind": "forecast",
                    "target": model, "role": "candidate", "source": model,
                    "lead_hour": int(lead), "logit": float(value),
                    "resample_id": "learned_04",
                })
    plot_forecast_logit_histogram_gallery(gallery_rows, cfg, output_root)
    paths.append(
        output_root / "plots" / "target_logit_distributions" / architecture
        / "forecast" / "all_models_all_lead_times.png"
    )

    corruption = "hemisphere_splice"
    levels = np.asarray(corruption_levels(corruption, cfg), dtype=float)
    corruption_groups = dummy_logit_groups(
        levels,
        [f"severity={value:.3g}" for value in levels],
        seed + 1,
        shift_scale=2.4,
    )
    corruption_root = (
        output_root / "plots" / "target_logit_distributions" / architecture
        / "corruption" / safe_target_name(corruption)
    )
    for group, severity in zip(corruption_groups, levels):
        path = corruption_root / f"severity_{severity:.3g}.png"
        _plot_logit_histogram(
            group["reference"], group["candidate"],
            f"{architecture}: {corruption} ({group['label']})",
            corruption, path,
        )
        paths.append(path)
    overlay = corruption_root / "all_corruption_strengths.png"
    _plot_logit_histogram_overlay(
        corruption_groups,
        f"{architecture}: {corruption} — all corruption strengths",
        overlay,
    )
    paths.append(overlay)
    return paths



def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def preview_artifact_paths(run_dir):
    """Return preview outputs without transient or offline W&B working files."""
    run_dir = Path(run_dir)
    return sorted(
        path for path in run_dir.rglob("*")
        if path.is_file() and "wandb" not in path.relative_to(run_dir).parts
    )


def run(args):
    cfg = load_config()
    run_dir = (args.output_dir or default_run_dir()).expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(
            f"Preview directory already exists: {run_dir}. Choose a fresh --output-dir."
        )
    run_dir.mkdir(parents=True)
    variables = variables_from_config(cfg)
    metric_names = metric_names_from_config(cfg)
    cfg.baseline.output_dir = str(run_dir)
    cfg.target_discriminator.output_dir = str(run_dir)
    cfg.plotting.profile = "paper"
    cfg.plotting.repeated_plot_mode = str(args.repeated_plot_mode)
    cfg.plotting.save_pdf = bool(args.pdf)
    cfg.plotting.paper.text_width_inches = float(args.text_width_inches)
    cfg.plotting.paper.width = "full"
    cfg.plotting.paper.include = list(args.include)
    preview_id = str(args.wandb_name or run_dir.name)
    cfg.pipeline.id = preview_id
    OmegaConf.update(cfg, "pipeline.run_dir", str(run_dir), force_add=True)
    cfg.pipeline.wandb.enabled = bool(args.wandb)
    cfg.pipeline.wandb.project = str(args.wandb_project)
    cfg.pipeline.wandb.entity = args.wandb_entity
    cfg.pipeline.wandb.mode = str(args.wandb_mode)
    cfg.pipeline.wandb.tags = ["synthetic", "paper-layout-preview"]

    output_root = baseline_output_dir(cfg, variables)
    data_root = output_root / "data"
    lead_rows, corruption_rows = dummy_standard_rows(
        cfg, metric_names, variables, seed=args.seed
    )
    write_rows(data_root / "lead_time.csv", lead_rows)
    write_rows(data_root / "corruption_strength.csv", corruption_rows)
    write_rows(
        data_root / "discriminator_reverse_kl.csv",
        dummy_discriminator_rows(cfg, variables),
    )
    if include_requests(
        args.include, ["disturbance", "gallery", "_corrupted", "all_corruptions"]
    ):
        write_blank_corruption_disturbances(cfg, variables, output_root)
    if include_requests(args.include, ["scwd"]):
        write_blank_scwd_diagnostics(cfg, variables, output_root)
    if include_requests(args.include, ["global_mean_wasserstein"]):
        write_dummy_global_mean_diagnostics(cfg, variables, output_root, args.seed)
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml", resolve=True)
    (run_dir / "SYNTHETIC_LAYOUT_PREVIEW.txt").write_text(
        "All numerical values in this run are deterministic synthetic placeholders.\n"
        "These figures are only for checking manuscript layout and typography.\n"
    )

    tracker = PipelineTracker(cfg, preview_id)
    try:
        with tracker.run(
            "plotting/paper-layout-preview", "synthetic-paper-layout-preview", cfg,
            metadata={"synthetic": True, "purpose": "manuscript layout only"},
            tags=["synthetic", "layout-preview", "paper"],
        ) as wandb_run:
            plot_saved_standard_metric_baselines(cfg)
            if include_requests(
                args.include,
                ["logit", "all_lead_times", "all_corruption_strengths", "lead_", "severity_"],
            ):
                plot_dummy_logit_histograms(cfg, output_root, args.seed)
            paper_root = output_root / "plots" / "paper"
            pngs = sorted(path for path in paper_root.rglob("*.png"))
            tracker.log_images(wandb_run, pngs, output_root / "plots")
            tracker.log_csv_table(
                wandb_run, "synthetic/lead_time", data_root / "lead_time.csv"
            )
            tracker.log_csv_table(
                wandb_run, "synthetic/corruption_strength",
                data_root / "corruption_strength.csv",
            )
            tracker.log_csv_table(
                wandb_run, "synthetic/discriminator_reverse_kl",
                data_root / "discriminator_reverse_kl.csv",
            )
            wandb_run.summary["synthetic"] = True
            wandb_run.summary["plots/count"] = len(pngs)
            tracker.log_artifact(
                wandb_run, "synthetic-paper-layout-preview", "layout-preview",
                preview_artifact_paths(run_dir),
                metadata={"synthetic": True, "reportable_results": False},
            )
    finally:
        tracker.close()
    print(f"Rendered {len(pngs)} synthetic paper-layout PNG(s) beneath: {paper_root}")
    print("WARNING: all plotted values are synthetic and must not be reported as results.")
    return paper_root


def main():
    run(parse_args())


if __name__ == "__main__":
    main()

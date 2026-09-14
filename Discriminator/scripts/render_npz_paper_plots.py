"""Render schema-v2 plot NPZ bundles without the original experiment run."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from omegaconf import OmegaConf

try:
    from .plot_bundles import (
        REFERENCE_COLOR, categorical_colors,
        configure_plot_bundle_saving, save_figure_bundle,
    )
except ImportError:
    from plot_bundles import (
        REFERENCE_COLOR, categorical_colors,
        configure_plot_bundle_saving, save_figure_bundle,
    )

try:
    from .plot_standard_metric_baselines import (
        metric_names_from_config,
        plot_corruption_metrics,
        plot_forecast_critic_and_graphcast_histograms,
        plot_discriminator_baselines,
        plot_forecast_logit_histogram_gallery,
        plot_lead_metrics,
        plot_main_corruption_comparison,
        plot_normalized_corruption_metrics_by_type,
        plot_normalized_lead_metrics_by_model,
        plot_corruption_disturbances,
        plot_global_mean_wasserstein_distributions,
        plot_scwd_anchor_diagnostics,
        plot_scwd_mean_response_differences,
        read_discriminator_baselines,
        read_discriminator_terms,
        read_global_mean_wasserstein_diagnostics,
        read_metric_csv,
        read_scwd_anchor_diagnostics,
        variables_from_config,
    )
except ImportError:
    from plot_standard_metric_baselines import (
        metric_names_from_config,
        plot_corruption_metrics,
        plot_forecast_critic_and_graphcast_histograms,
        plot_discriminator_baselines,
        plot_forecast_logit_histogram_gallery,
        plot_lead_metrics,
        plot_main_corruption_comparison,
        plot_normalized_corruption_metrics_by_type,
        plot_normalized_lead_metrics_by_model,
        plot_corruption_disturbances,
        plot_global_mean_wasserstein_distributions,
        plot_scwd_anchor_diagnostics,
        plot_scwd_mean_response_differences,
        read_discriminator_baselines,
        read_discriminator_terms,
        read_global_mean_wasserstein_diagnostics,
        read_metric_csv,
        read_scwd_anchor_diagnostics,
        variables_from_config,
    )


def load_bundle(path):
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"]))
    if int(metadata.get("schema_version", 1)) < 2:
        archive.close()
        raise ValueError(
            f"{path} uses legacy schema {metadata.get('schema_version', 1)}. "
            "Render its complete run with render_paper_plots.py instead; legacy "
            "bundles lack reliable labels, scales, and map normalization."
        )
    return archive, metadata


def make_axis(figure, metadata):
    projection = None
    if "GeoAxes" in str(metadata.get("projection", "")):
        try:
            import cartopy.crs as ccrs
            projection = ccrs.PlateCarree()
        except ImportError:
            pass
    kwargs = {"projection": projection} if projection is not None else {}
    return figure.add_axes(metadata.get("position", [0.12, 0.12, 0.8, 0.8]), **kwargs)


def _pretty_name(value):
    value = str(value)
    known = {
        "squeezenet": "SqueezeNet",
        "squeezenet_equator_mask": "Equator-masked SqueezeNet",
        "GraphCast": "GraphCast",
        "Pangu-Weather": "Pangu-Weather",
        "FuXi": "FuXi",
        "IFS_HRES": "IFS HRES",
        "ERA5_Forecast": "ERA5 Forecast",
        "UCast_member_0": "UCast member 0",
        "SWIFT": "SWIFT",
        "grf": "GRF noise",
        "hf_noise": "High-frequency noise",
        "checkerboard_2px": "2-pixel checkerboard",
        "equatorial_checker_texture": "Equatorial checkerboard",
        "hemisphere_splice": "Hemisphere splice",
        "pixel_replace": "Pixel replacement",
        "zonal_scanlines": "Zonal scanlines",
        "meridional_scanlines": "Meridional scanlines",
    }
    return known.get(value, value.replace("_", " ").title())


def _path_context(path, family):
    parts = list(Path(path).parts)
    try:
        index = parts.index(family)
        return parts[index + 1], parts[index + 2], parts[index + 3]
    except (ValueError, IndexError):
        return "", "", Path(path).stem


def _histogram_title(path, overlay):
    architecture, kind, target = _path_context(path, "target_logit_distributions")
    target = _pretty_name(target)
    if overlay:
        suffix = "all lead times" if kind == "forecast" else "all corruption strengths"
    else:
        stem = Path(path).stem
        suffix = stem.replace("lead_", "+").replace("severity_", "severity ").replace("h", " h")
    return f"{target}: {suffix} ({_pretty_name(architecture)})"


def reconstruct_histogram(archive, metadata, width, source_path):
    overlay = metadata.get("plot_type") == "target_logit_histogram_overlay"
    figure, axis = plt.subplots(figsize=(width, width * 0.82), layout="constrained")
    edges = np.asarray(archive["input_bin_edges"], dtype=float)
    if overlay:
        reference = np.asarray(archive["input_pooled_reference_logits"], dtype=float)
        groups = []
        label_keys = sorted(
            [key for key in archive.files if key.startswith("input_label_")],
            key=lambda key: int(key.rsplit("_", 1)[1]),
        )
        for key in label_keys:
            index = int(key.rsplit("_", 1)[1])
            groups.append((
                str(archive[key]),
                np.asarray(archive[f"input_candidate_logits_{index}"], dtype=float),
            ))
    else:
        reference = np.asarray(archive["input_reference_logits"], dtype=float)
        candidate = np.asarray(archive["input_candidate_logits"], dtype=float)
        stem = Path(source_path).stem
        label = stem.replace("lead_", "+").replace("severity_", "severity=").replace("h", " h")
        groups = [(label, candidate)]
    axis.hist(
        reference, bins=edges, density=True, histtype="stepfilled",
        color=REFERENCE_COLOR, alpha=0.30, label="ERA5 test",
    )
    colors = categorical_colors(max(len(groups), 1), offset=1)
    for color, (label, values) in zip(colors, groups):
        axis.hist(
            values, bins=edges, density=True, histtype="step",
            color=color, linewidth=1.3, label=label.replace("h", " h"),
        )
    axis.axvline(0.0, color="black", linewidth=0.7, alpha=0.45)
    axis.set(
        title=_histogram_title(source_path, overlay),
        xlabel="Real-vs-fake logit", ylabel="Density",
    )
    axis.grid(alpha=0.22)
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.20),
        ncol=2, frameon=False, fontsize=7,
    )
    return figure


def reconstruct_interpretability(archive, metadata, width, source_path):
    latitudes = np.asarray(archive["input_latitudes"], dtype=float)
    longitudes = np.asarray(archive["input_longitudes"], dtype=float)
    physical = np.asarray(archive["input_physical_inputs"], dtype=float)
    attributions = np.asarray(archive["input_integrated_gradients"], dtype=float)
    logits = np.asarray(archive["input_logits"], dtype=float)
    true_classes = np.asarray(archive["input_true_classes"]).astype(str)
    selections = np.asarray(archive["input_selections"]).astype(str)
    architecture, kind, target_file = _path_context(source_path, "target_interpretability")
    target = target_file.replace("_integrated_gradients.npz", "")
    try:
        import cartopy.crs as ccrs
        projection = ccrs.PlateCarree()
        subplot_kw = {"projection": projection}
    except ImportError:
        projection = None
        subplot_kw = {}
    n_cases = len(logits)
    figure, axes = plt.subplots(
        n_cases, 2, figsize=(10.0, max(2.15 * n_cases, 5.0)),
        subplot_kw=subplot_kw, squeeze=False, layout="constrained",
    )
    physical_min = float(np.nanmin(physical[:, 0]))
    physical_max = float(np.nanmax(physical[:, 0]))
    aggregate = attributions.sum(axis=1)
    relevance_limit = max(
        float(np.nanpercentile(np.abs(aggregate), 99.0)),
        np.finfo(np.float32).eps,
    )
    physical_artist = relevance_artist = None
    for index in range(n_cases):
        left, right = axes[index]
        if projection is not None:
            physical_artist = left.pcolormesh(
                longitudes, latitudes, physical[index, 0], shading="auto",
                cmap="coolwarm", vmin=physical_min, vmax=physical_max,
                transform=projection, rasterized=True,
            )
            relevance_artist = right.pcolormesh(
                longitudes, latitudes, aggregate[index], shading="auto",
                cmap="RdBu_r", vmin=-relevance_limit, vmax=relevance_limit,
                transform=projection, rasterized=True,
            )
            left.coastlines(linewidth=0.4)
            right.coastlines(linewidth=0.4)
            left.set_global()
            right.set_global()
        else:
            extent = [longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()]
            physical_artist = left.imshow(
                physical[index, 0], extent=extent, origin="lower", aspect="auto",
                cmap="coolwarm", vmin=physical_min, vmax=physical_max, rasterized=True,
            )
            relevance_artist = right.imshow(
                aggregate[index], extent=extent, origin="lower", aspect="auto",
                cmap="RdBu_r", vmin=-relevance_limit, vmax=relevance_limit,
                rasterized=True,
            )
        predicted = "real" if logits[index] >= 0.0 else "fake"
        left.set_title(
            f"{true_classes[index].upper()} / {selections[index].replace('_', ' ')}"
            f" | logit {logits[index]:+.3f} | predicted {predicted.upper()}",
            loc="left", fontsize=7,
        )
    figure.colorbar(
        physical_artist, ax=list(axes[:, 0]), orientation="horizontal",
        fraction=0.025, pad=0.025, aspect=45, label="2 m temperature",
    )
    figure.colorbar(
        relevance_artist, ax=list(axes[:, 1]), orientation="horizontal",
        fraction=0.025, pad=0.025, aspect=45, label="Integrated-gradient attribution",
    )
    figure.suptitle(
        f"{_pretty_name(architecture)}: {kind} / {_pretty_name(target)}\n"
        "Signed IG: positive values support ERA5"
    )
    return figure


def bundle_width_kind(metadata, fallback="full"):
    original = metadata.get("figure_size_inches", [5.5, 3.5])
    return "half" if float(original[0]) <= 3.2 else fallback



def reconstruct(archive, metadata, width, source_path=None):
    plot_type = metadata.get("plot_type", "")
    source_path = Path(source_path or metadata.get("png", "plot.png"))
    if plot_type in {"target_logit_histogram", "target_logit_histogram_overlay"}:
        return reconstruct_histogram(archive, metadata, width, source_path)
    if (
        plot_type == "target_integrated_gradients"
        and "input_physical_inputs" in archive
        and "input_integrated_gradients" in archive
    ):
        return reconstruct_interpretability(archive, metadata, width, source_path)

    original = metadata.get("figure_size_inches", [width, width * 0.65])
    height = width * float(original[1]) / max(float(original[0]), 1e-12)
    figure = plt.figure(figsize=(width, height))
    for axis_metadata in metadata.get("axes", []):
        axis = make_axis(figure, axis_metadata)
        for line in axis_metadata.get("lines", []):
            prefix = line["prefix"]
            marker = line.get("marker")
            axis.plot(
                archive[f"{prefix}_x"], archive[f"{prefix}_y"],
                label=line.get("label"), color=line.get("color"),
                linestyle=line.get("linestyle", "-"),
                marker=None if marker in {"None", "none", ""} else marker,
            )
        for collection in axis_metadata.get("collections", []):
            prefix = collection["prefix"]
            values_key, coordinates_key = f"{prefix}_values", f"{prefix}_coordinates"
            offsets_key = f"{prefix}_offsets"
            style = {
                "cmap": collection.get("cmap", "viridis"),
                "vmin": collection.get("vmin"), "vmax": collection.get("vmax"),
            }
            if coordinates_key in archive and values_key in archive:
                coordinates, values = archive[coordinates_key], archive[values_key]
                expected = (coordinates.shape[0] - 1, coordinates.shape[1] - 1)
                if values.size == int(np.prod(expected)):
                    values = values.reshape(expected)
                axis.pcolormesh(
                    coordinates[..., 0], coordinates[..., 1], values,
                    shading="auto", rasterized=True, **style,
                )
            elif offsets_key in archive:
                offsets = archive[offsets_key]
                values = archive[values_key] if values_key in archive else None
                colors = values if values is not None and values.size == len(offsets) else None
                scatter_style = style if colors is not None else {}
                axis.scatter(
                    offsets[:, 0], offsets[:, 1], c=colors, s=8,
                    rasterized=True, **scatter_style,
                )
        for image in axis_metadata.get("images", []):
            prefix = image["prefix"]
            axis.imshow(
                archive[prefix], extent=archive[f"{prefix}_extent"],
                cmap=image.get("cmap", "viridis"), vmin=image.get("vmin"),
                vmax=image.get("vmax"), origin=image.get("origin", "upper"),
                rasterized=True,
            )
        for patch in axis_metadata.get("patches", []):
            style = {
                "label": patch.get("label"),
                "facecolor": patch.get("facecolor", "none"),
                "edgecolor": patch.get("edgecolor", "none"),
                "linewidth": patch.get("linewidth", 1.0),
                "alpha": patch.get("alpha"),
            }
            if patch.get("geometry", "rectangle") == "path":
                artist = Polygon(archive[f"{patch['prefix']}_vertices"], closed=True, **style)
            else:
                x, y, patch_width, patch_height = map(float, archive[patch["prefix"]])
                artist = Rectangle((x, y), patch_width, patch_height, **style)
            axis.add_patch(artist)
        axis.set_xscale(axis_metadata.get("xscale", "linear"))
        axis.set_yscale(axis_metadata.get("yscale", "linear"))
        axis.set_xlim(axis_metadata.get("xlim", axis.get_xlim()))
        axis.set_ylim(axis_metadata.get("ylim", axis.get_ylim()))
        axis.set(
            title=axis_metadata.get("title", ""),
            xlabel=axis_metadata.get("xlabel", ""),
            ylabel=axis_metadata.get("ylabel", ""),
        )
        if hasattr(axis, "coastlines") and "GeoAxes" in str(axis_metadata.get("projection", "")):
            axis.coastlines(linewidth=0.45)
        handles, labels = axis.get_legend_handles_labels()
        if any(label and not label.startswith("_") for label in labels):
            axis.legend(fontsize=7)
    title = (metadata.get("figure") or {}).get("suptitle")
    if title:
        figure.suptitle(title)
    return figure


def find_saved_experiment(source):
    """Locate a variable-scoped data directory and its resolved run config."""
    start = source if source.is_dir() else source.parent
    artifact_root = next(
        (
            candidate for candidate in (start, *start.parents)
            if (candidate / "data" / "lead_time.csv").is_file()
            and (candidate / "data" / "corruption_strength.csv").is_file()
        ),
        None,
    )
    if artifact_root is None:
        return None, None
    config_path = next(
        (
            candidate / "resolved_config.yaml"
            for candidate in (artifact_root, *artifact_root.parents)
            if (candidate / "resolved_config.yaml").is_file()
        ),
        None,
    )
    return artifact_root, config_path


def render_derived_gallery(source, output, derived_source=None):
    """Build current composite figures from saved numerical evaluation rows."""
    artifact_root, config_path = find_saved_experiment(derived_source or source)
    if artifact_root is None or config_path is None:
        print(
            "No saved evaluation CSV/config pair found above the NPZ source; "
            "skipping derived gallery figures."
        )
        return []
    cfg = OmegaConf.load(config_path)
    variables = variables_from_config(cfg)
    metric_names = metric_names_from_config(cfg)
    lead_rows = read_metric_csv(
        artifact_root / "data" / "lead_time.csv", metric_names, "lead_time"
    )
    corruption_rows = read_metric_csv(
        artifact_root / "data" / "corruption_strength.csv",
        metric_names,
        "corruption_strength",
    )
    discriminator_rows = read_discriminator_baselines(artifact_root)
    discriminator_terms = read_discriminator_terms(artifact_root)
    plot_root = output / "plots"
    before = {
        path: path.stat().st_mtime_ns for path in plot_root.rglob("*.png")
    } if plot_root.exists() else {}

    # Re-run every current aggregate plot that can be reconstructed exactly from
    # the persisted numerical rows. Dense maps and per-target histograms remain
    # covered by their semantic NPZ bundles above.
    plot_lead_metrics(lead_rows, metric_names, variables, output)
    plot_corruption_metrics(corruption_rows, metric_names, variables, output)
    plot_normalized_lead_metrics_by_model(
        lead_rows, metric_names, variables, output,
    )
    plot_normalized_corruption_metrics_by_type(
        corruption_rows, metric_names, variables, output,
    )
    plot_main_corruption_comparison(
        corruption_rows, discriminator_rows, metric_names, variables, cfg, output,
    )
    plot_discriminator_baselines(discriminator_rows, cfg, output)
    plot_forecast_logit_histogram_gallery(
        discriminator_terms, cfg, output,
    )
    plot_forecast_critic_and_graphcast_histograms(
        discriminator_rows, discriminator_terms, cfg, output,
    )
    # A composed source may carry replacement GRF diagnostics absent from the
    # original paper run. Render them without modifying either source.
    for name in (
        "scwd_anchor_contributions.nc",
        "global_mean_wasserstein_distributions.nc",
        "corruption_disturbances.nc",
    ):
        diagnostic = artifact_root / "data" / name
        if diagnostic.is_file():
            (output / "data").mkdir(parents=True, exist_ok=True)
            target = output / "data" / name
            if diagnostic.resolve() != target.resolve():
                shutil.copy2(diagnostic, target)
    scwd = read_scwd_anchor_diagnostics(output)
    if scwd:
        plot_scwd_anchor_diagnostics(scwd, output)
        plot_scwd_mean_response_differences(scwd, output)
    global_mean = read_global_mean_wasserstein_diagnostics(output)
    if global_mean:
        plot_global_mean_wasserstein_distributions(global_mean, output)
    if (output / "data" / "corruption_disturbances.nc").is_file():
        plot_corruption_disturbances(output, cfg)
    return sorted(
        path for path in plot_root.rglob("*.png")
        if before.get(path) != path.stat().st_mtime_ns
    )


def upload_gallery_to_wandb(source, output, args):
    """Upload rendered images for browsing and complete bundles as one artifact."""
    import re
    import tempfile
    from contextlib import nullcontext

    import wandb

    _, config_path = find_saved_experiment(source)
    default_name = (
        f"{config_path.parent.name}-paper-gallery"
        if config_path is not None else f"{source.name}-paper-gallery"
    )
    run_name = args.wandb_name or default_name
    image_root = output / "plots" / "paper"
    image_paths = sorted(image_root.rglob("*.png"))
    bundle_paths = sorted(
        path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".npz", ".json"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No rendered PNG gallery found beneath {image_root}.")
    if args.wandb_mode == "offline":
        offline_dir = output / "wandb_offline"
        offline_dir.mkdir(parents=True, exist_ok=True)
        wandb_directory = nullcontext(str(offline_dir))
    else:
        wandb_directory = tempfile.TemporaryDirectory(
            prefix="wandb-paper-gallery-"
        )
    with wandb_directory as wandb_dir:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            name=run_name,
            job_type="paper-gallery",
            tags=["paper", "plotting", "reconstructed-gallery"],
            dir=wandb_dir,
            config={
                "source": str(source),
                "derived_source": None if args.derived_source is None else str(args.derived_source),
                "output_dir": str(output),
                "text_width_inches": args.text_width_inches,
                "width": args.width,
                "existing_only": args.existing_only,
            },
        )
        try:
            for image_path in image_paths:
                relative = image_path.relative_to(image_root)
                key = f"plots/{relative.with_suffix('').as_posix()}"
                run.log({key: wandb.Image(str(image_path))})
            artifact_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-")
            artifact = wandb.Artifact(
                artifact_name or "paper-gallery",
                type="paper-plots",
                metadata={
                    "source": str(source),
                    "image_count": len(image_paths),
                    "bundle_file_count": len(bundle_paths),
                },
            )
            artifact.add_dir(str(image_root), name="plots/paper")
            provenance = output / "grf_composition.json"
            if provenance.is_file():
                artifact.add_file(str(provenance), name="grf_composition.json")
            run.log_artifact(artifact)
            run.summary["paper_gallery/image_count"] = len(image_paths)
            run.summary["paper_gallery/bundle_file_count"] = len(bundle_paths)
            run.summary["paper_gallery/source"] = str(source)
            run.summary["paper_gallery/output_dir"] = str(output)
            run_url = getattr(run, "url", None)
        finally:
            run.finish()
    print(
        f"Uploaded {len(image_paths)} paper-gallery image(s) and "
        f"{len(bundle_paths)} bundle file(s) to W&B"
        + (f": {run_url}" if run_url else ".")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="A schema-v2 NPZ or directory.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--derived-source", type=Path, default=None,
        help="Use this composed run for CSV/NetCDF-derived plots while replaying NPZs from source.",
    )
    parser.add_argument(
        "--exclude-grf-bundles", action="store_true",
        help="Do not replay stale GRF-specific NPZs from the base run.",
    )
    parser.add_argument("--text-width-inches", type=float, default=5.5)
    parser.add_argument("--width", choices=("full", "half"), default="full")
    parser.add_argument(
        "--existing-only", action="store_true",
        help=("Replay only existing NPZs; do not build current composite "
              "figures from saved CSVs."),
    )
    parser.add_argument(
        "--wandb", action=argparse.BooleanOptionalAction, default=False,
        help="Log every rendered PNG and upload the complete gallery bundle to W&B.",
    )
    parser.add_argument(
        "--wandb-project", default="weather-discriminator-baselines",
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online",
    )
    parser.add_argument(
        "--wandb-name", default=None,
        help="Run/artifact name (default: <source pipeline id>-paper-gallery).",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    paths = [source] if source.is_file() else sorted(
        path for path in source.rglob("*.npz") if not path.name.endswith("_notitle.npz")
    )
    if args.exclude_grf_bundles:
        paths = [path for path in paths if "grf" not in path.as_posix().lower()]
    if not paths:
        raise FileNotFoundError(f"No plot NPZ bundles found beneath {source}.")
    common = source.parent if source.is_file() else source
    output = (args.output_dir or (common / "paper_from_npz")).resolve()
    configure_plot_bundle_saving(
        profile="paper", paper_width_inches=args.text_width_inches,
        paper_width_kind=args.width,
    )
    for path in paths:
        archive, metadata = load_bundle(path)
        try:
            width_kind = bundle_width_kind(metadata, fallback=args.width)
            render_width = (
                (args.text_width_inches - 0.12) / 2.0
                if width_kind == "half" else args.text_width_inches
            )
            figure = reconstruct(archive, metadata, render_width, source_path=path)
            relative = Path(path.name) if source.is_file() else path.relative_to(common)
            save_figure_bundle(
                figure, output / "plots" / relative.with_suffix(".png"),
                plot_type=f"paper_replot:{metadata.get('plot_type', 'unknown')}",
                metadata={"source_npz": str(path)},
                paper_width_kind=width_kind,
            )
            plt.close(figure)
        finally:
            archive.close()
    derived_source = None if args.derived_source is None else args.derived_source.resolve()
    derived = [] if args.existing_only else render_derived_gallery(
        source, output, derived_source=derived_source,
    )
    print(f"Rendered {len(paths)} semantic NPZ bundle(s) beneath: {output / 'plots' / 'paper'}")
    if derived:
        print(
            f"Rendered {len(derived)} derived gallery PNG(s), including current "
            "composite layouts."
        )
    if args.wandb:
        upload_gallery_to_wandb(source, output, args)


if __name__ == "__main__":
    main()

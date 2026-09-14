"""Regenerate manuscript-sized figures from an evaluated baseline run."""

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

try:
    from .plot_standard_metric_baselines import (
        baseline_output_dir,
        plot_saved_standard_metric_baselines,
        variables_from_config,
    )
except ImportError:
    from plot_standard_metric_baselines import (
        baseline_output_dir,
        plot_saved_standard_metric_baselines,
        variables_from_config,
    )


def parser():
    result = argparse.ArgumentParser(
        description=(
            "Render paper-ready PDF/PNG figures from an existing run's CSV, "
            "NetCDF, and resolved configuration; evaluation is never rerun."
        )
    )
    result.add_argument("run_dir", type=Path, help="Existing pipeline run directory.")
    result.add_argument(
        "--config", type=Path, default=None,
        help="Resolved config to use (default: RUN_DIR/resolved_config.yaml).",
    )
    result.add_argument(
        "--text-width-inches", type=float, default=5.5,
        help="Physical neurips_2026 text width measured from LaTeX.",
    )
    result.add_argument(
        "--width", choices=("full", "half"), default="full",
        help="Render at the full text width or one half-width panel.",
    )
    result.add_argument("--column-gap-inches", type=float, default=0.12)
    result.add_argument(
        "--include", action="append", default=[],
        help="Optional filename/path glob; repeat to select multiple plot families.",
    )
    return result


def legacy_npz_message(output_root):
    bundles = sorted((output_root / "plots").rglob("*.npz"))
    if not bundles:
        return ""
    versions = set()
    for path in bundles[:16]:
        try:
            with np.load(path, allow_pickle=False) as archive:
                versions.add(int(json.loads(str(archive["metadata_json"])).get("schema_version", 1)))
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            versions.add(1)
    return (
        f" Found {len(bundles)} plot NPZ bundle(s) with schema version(s) "
        f"{sorted(versions)}, but legacy artist snapshots do not contain enough "
        "labels, scales, projections, and normalization metadata for a reliable replot."
    )


def configure_existing_run(cfg, run_dir, args):
    """Bind a resolved config to its existing immutable evaluation directory."""
    variables = variables_from_config(cfg)
    expected = baseline_output_dir(cfg, variables)
    variable_leaf = expected.name
    cfg.baseline.output_dir = str(run_dir)
    cfg.target_discriminator.output_dir = str(run_dir / variable_leaf)
    cfg.target_discriminator.checkpoint_dir = str(
        run_dir / variable_leaf / "models" / "target_discriminators"
    )
    cfg.plotting.profile = "paper"
    cfg.plotting.save_pdf = True
    if "paper" not in cfg.plotting:
        cfg.plotting.paper = {}
    cfg.plotting.paper.text_width_inches = float(args.text_width_inches)
    cfg.plotting.paper.column_gap_inches = float(args.column_gap_inches)
    cfg.plotting.paper.width = str(args.width)
    cfg.plotting.paper.include = list(args.include)
    return run_dir / variable_leaf


def render(args):
    run_dir = args.run_dir.expanduser().resolve()
    config_path = (args.config.expanduser().resolve() if args.config is not None
                   else run_dir / "resolved_config.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"No resolved configuration found at {config_path}.")
    cfg = OmegaConf.load(config_path)
    output_root = configure_existing_run(cfg, run_dir, args)
    required = [
        output_root / "data" / "lead_time.csv",
        output_root / "data" / "corruption_strength.csv",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Paper rendering requires the canonical evaluated data artifact(s): "
            + ", ".join(map(str, missing))
            + "."
            + legacy_npz_message(output_root)
        )
    plot_saved_standard_metric_baselines(cfg)
    paper_root = output_root / "plots" / "paper"
    outputs = sorted(path for path in paper_root.rglob("*") if path.is_file())
    print(f"Saved {len(outputs)} paper plot bundle file(s) beneath: {paper_root}")
    return outputs



def main():
    render(parser().parse_args())


if __name__ == "__main__":
    main()

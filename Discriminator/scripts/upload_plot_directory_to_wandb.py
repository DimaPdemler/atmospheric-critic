#!/usr/bin/env python3
"""Upload an existing plot directory to W&B without rerendering it."""

import argparse
from pathlib import Path

import wandb
from dotenv import load_dotenv


def main():
    # Match the established pipeline tracker without exposing credential values.
    load_dotenv(Path(__file__).resolve().parents[2] / "wandb_info.env")
    parser = argparse.ArgumentParser()
    parser.add_argument("plot_root", type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--tags", nargs="*", default=[])
    args = parser.parse_args()
    root = args.plot_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    pngs = sorted(root.rglob("*.png"))
    pdfs = sorted(root.rglob("*.pdf"))
    npzs = sorted(root.rglob("*.npz"))
    if not pngs or not pdfs:
        raise RuntimeError(f"Expected PNG and PDF files beneath {root}")

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        tags=args.tags,
        config={
            "source_directory": str(root),
            "png_count": len(pngs),
            "pdf_count": len(pdfs),
            "npz_count": len(npzs),
        },
    )
    for path in pngs:
        key = "plots/" + path.relative_to(root).with_suffix("").as_posix()
        run.log({key: wandb.Image(str(path))})
    featured = {
        "featured/corruption_metrics_and_critic_notitle": root / "corruption_metrics_and_critic_notitle.png",
        "featured/all_models_all_lead_times_notitle": (
            root / "target_logit_distributions" / "squeezenet" / "forecast"
            / "all_models_all_lead_times_notitle.png"
        ),
        "featured/lead_time_by_model_normalized_notitle": (
            root / "lead_time_by_model_normalized_notitle.png"
        ),
    }
    for key, path in featured.items():
        if path.is_file():
            run.log({key: wandb.Image(str(path))})
    artifact = wandb.Artifact(
        args.artifact_name,
        type="baseline-plots",
        metadata={"contains_pdf": True, "source_directory": str(root)},
    )
    artifact.add_dir(str(root), name="plots/paper")
    run.log_artifact(artifact, aliases=["latest", "paper-gallery-lead-palette-fix-20260829"])
    run.summary["plots/count"] = len(pngs)
    run.summary["pdfs/count"] = len(pdfs)
    run.summary["npz/count"] = len(npzs)
    run.summary["status"] = "completed"
    run.finish()


if __name__ == "__main__":
    main()

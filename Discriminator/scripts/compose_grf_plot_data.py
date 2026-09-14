"""Compose plot-ready data with GRF rows replaced by dedicated reruns."""

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path


VARIABLE_TAG = (
    "2m_temperature__10m_u_component_of_wind__10m_v_component_of_wind__"
    "mean_sea_level_pressure"
)


def artifact_root(run: Path) -> Path:
    direct = run / VARIABLE_TAG
    if (direct / "data").is_dir():
        return direct
    matches = [path.parent.parent for path in run.rglob("data/corruption_strength.csv")]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one variable artifact beneath {run}, found {matches}")
    return matches[0]


def read_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def replace_rows(base_path, replacement_path, predicate):
    if not replacement_path.is_file():
        raise FileNotFoundError(replacement_path)
    base = read_rows(base_path) if base_path.is_file() else []
    replacement = [row for row in read_rows(replacement_path) if predicate(row)]
    write_rows(base_path, [row for row in base if not predicate(row)] + replacement)
    return len(replacement)


def is_grf_standard(row):
    return row.get("corruption") == "grf"


def is_grf_critic(row):
    return row.get("kind") == "corruption" and row.get("target") == "grf"


def is_grf_metric_draw(row):
    return is_grf_standard(row) or is_grf_critic(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--grf-standard-run", type=Path, required=True)
    parser.add_argument("--grf-discriminator-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = artifact_root(args.base_run.resolve())
    standard = artifact_root(args.grf_standard_run.resolve())
    learned = artifact_root(args.grf_discriminator_run.resolve())
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Composite output already exists: {output}")
    (output / "data").mkdir(parents=True)
    shutil.copy2(args.base_run.resolve() / "resolved_config.yaml", output / "resolved_config.yaml")
    shutil.copytree(base / "data", output / "data", dirs_exist_ok=True)

    counts = {}
    for name in ("corruption_strength.csv", "fixed_metric_draws.csv"):
        counts[name] = replace_rows(output / "data" / name, standard / "data" / name, is_grf_standard)
    counts["metric_draws.standard"] = replace_rows(
        output / "data" / "metric_draws.csv", standard / "data" / "metric_draws.csv", is_grf_standard,
    )
    for name in ("discriminator_reverse_kl.csv", "discriminator_metric_draws.csv", "discriminator_terms.csv.gz"):
        counts[name] = replace_rows(output / "data" / name, learned / "data" / name, is_grf_critic)
    counts["metric_draws.learned"] = replace_rows(
        output / "data" / "metric_draws.csv", learned / "data" / "metric_draws.csv", is_grf_critic,
    )
    excluded = output / "data" / "excluded_critics.csv"
    learned_excluded = learned / "data" / "excluded_critics.csv"
    if excluded.is_file() or learned_excluded.is_file():
        base_rows = read_rows(excluded) if excluded.is_file() else []
        replacement = [row for row in read_rows(learned_excluded) if is_grf_critic(row)] if learned_excluded.is_file() else []
        write_rows(excluded, [row for row in base_rows if not is_grf_critic(row)] + replacement)

    for name in ("scwd_anchor_contributions.nc", "global_mean_wasserstein_distributions.nc", "corruption_disturbances.nc"):
        source = standard / "data" / name
        if source.is_file():
            shutil.copy2(source, output / "data" / name)

    provenance = {
        "base_run": str(args.base_run.resolve()),
        "grf_standard_run": str(args.grf_standard_run.resolve()),
        "grf_discriminator_run": str(args.grf_discriminator_run.resolve()),
        "policy": "replace all GRF aggregate and draw rows; retain every non-GRF base row",
        "replacement_counts": counts,
    }
    (output / "grf_composition.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Composed GRF replacement plotting data beneath {output}")


if __name__ == "__main__":
    main()

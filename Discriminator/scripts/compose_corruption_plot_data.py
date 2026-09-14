"""Compose plot-ready data with one corruption supplied by dedicated reruns."""

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import xarray as xr


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
        raise FileNotFoundError(
            f"Expected one variable artifact beneath {run}, found {matches}"
        )
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
        writer.writeheader()
        writer.writerows(rows)


def replace_rows(base_path: Path, replacement_path: Path, predicate):
    if not replacement_path.is_file():
        raise FileNotFoundError(replacement_path)
    base = read_rows(base_path) if base_path.is_file() else []
    replacement = [row for row in read_rows(replacement_path) if predicate(row)]
    if not replacement:
        raise ValueError(
            f"No matching replacement rows in {replacement_path}"
        )
    write_rows(base_path, [row for row in base if not predicate(row)] + replacement)
    return len(replacement)


def replace_dataset_slice(base_path: Path, replacement_path: Path, corruption: str):
    """Replace one corruption while retaining every unrelated diagnostic slice."""
    if not replacement_path.is_file():
        return 0
    with xr.open_dataset(base_path) as source:
        base = source.load()
    with xr.open_dataset(replacement_path) as source:
        replacement = source.load()
    if "corruption" in base.dims:
        replacement = replacement.sel(corruption=[corruption])
        keep = np.asarray(base.corruption.values).astype(str) != corruption
        base = base.isel(corruption=np.flatnonzero(keep))
        merged = xr.concat([base, replacement], dim="corruption", join="outer")
        count = int(replacement.sizes["corruption"])
    elif "comparison" in base.dims and "label" in base.coords:
        replacement_keep = np.asarray(replacement.label.values).astype(str) == corruption
        if not replacement_keep.any():
            raise ValueError(f"No {corruption!r} comparison in {replacement_path}")
        base_keep = np.asarray(base.label.values).astype(str) != corruption
        replacement = replacement.isel(comparison=np.flatnonzero(replacement_keep))
        base = base.isel(comparison=np.flatnonzero(base_keep))
        # Support arrays are stored as unindexed, variable-length dimensions.
        # Give them positional coordinates so an outer concat can retain the
        # longer support from either run and pad the shorter one with NaNs.
        for dimension in (set(base.dims) & set(replacement.dims)) - {"comparison"}:
            if dimension not in base.coords:
                base = base.assign_coords({dimension: np.arange(base.sizes[dimension])})
            if dimension not in replacement.coords:
                replacement = replacement.assign_coords(
                    {dimension: np.arange(replacement.sizes[dimension])}
                )
        merged = xr.concat([base, replacement], dim="comparison", join="outer")
        count = int(replacement.sizes["comparison"])
    else:
        raise ValueError(f"Cannot identify corruption dimension in {base_path}")
    for variable in merged.variables:
        merged[variable].encoding = {}
    temporary = base_path.with_name(f".{base_path.stem}.tmp{base_path.suffix}")
    merged.to_netcdf(temporary)
    temporary.replace(base_path)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--standard-run", type=Path, required=True)
    parser.add_argument("--discriminator-run", type=Path, required=True)
    parser.add_argument("--corruption", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    corruption = str(args.corruption)
    standard_predicate = lambda row: row.get("corruption") == corruption
    critic_predicate = lambda row: (
        row.get("kind") == "corruption" and row.get("target") == corruption
    )

    base = artifact_root(args.base_run.resolve())
    standard = artifact_root(args.standard_run.resolve())
    learned = artifact_root(args.discriminator_run.resolve())
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Composite output already exists: {output}")
    (output / "data").mkdir(parents=True)
    shutil.copy2(
        args.base_run.resolve() / "resolved_config.yaml",
        output / "resolved_config.yaml",
    )
    shutil.copytree(base / "data", output / "data", dirs_exist_ok=True)

    counts = {}
    for name in ("corruption_strength.csv", "fixed_metric_draws.csv"):
        counts[name] = replace_rows(
            output / "data" / name,
            standard / "data" / name,
            standard_predicate,
        )
    counts["metric_draws.standard"] = replace_rows(
        output / "data" / "metric_draws.csv",
        standard / "data" / "metric_draws.csv",
        standard_predicate,
    )

    for name in (
        "discriminator_reverse_kl.csv",
        "discriminator_metric_draws.csv",
        "discriminator_terms.csv.gz",
    ):
        counts[name] = replace_rows(
            output / "data" / name,
            learned / "data" / name,
            critic_predicate,
        )
    counts["metric_draws.learned"] = replace_rows(
        output / "data" / "metric_draws.csv",
        learned / "data" / "metric_draws.csv",
        critic_predicate,
    )

    excluded = output / "data" / "excluded_critics.csv"
    learned_excluded = learned / "data" / "excluded_critics.csv"
    if excluded.is_file() or learned_excluded.is_file():
        base_rows = read_rows(excluded) if excluded.is_file() else []
        replacement = (
            [row for row in read_rows(learned_excluded) if critic_predicate(row)]
            if learned_excluded.is_file()
            else []
        )
        write_rows(
            excluded,
            [row for row in base_rows if not critic_predicate(row)] + replacement,
        )

    for name in (
        "scwd_anchor_contributions.nc",
        "global_mean_wasserstein_distributions.nc",
        "corruption_disturbances.nc",
    ):
        source = standard / "data" / name
        counts[name] = replace_dataset_slice(
            output / "data" / name, source, corruption,
        )

    provenance = {
        "base_run": str(args.base_run.resolve()),
        "standard_run": str(args.standard_run.resolve()),
        "discriminator_run": str(args.discriminator_run.resolve()),
        "corruption": corruption,
        "policy": (
            f"replace all {corruption} aggregate and draw rows; retain every "
            f"non-{corruption} base row"
        ),
        "replacement_counts": counts,
    }
    provenance_path = output / "corruption_composition.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Composed {corruption} replacement plotting data beneath {output}")


if __name__ == "__main__":
    main()

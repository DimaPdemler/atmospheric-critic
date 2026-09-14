"""Turn temporal-resampling draws into the bootstrap-null deliverable schema.

The blind-spot figure and the N/M table were built against
`evaluate_bootstrap_null.py`, which estimates the ERA5 null from random
day-of-month half/half partitions in a separate cluster job. The temporal
resampling pipeline already produces the same quantities as a by-product of the
run that draws the curves: every fixed resample contributes one null score
(its test window against its buffered training complement) and one score per
corruption and severity.

Reading the table off those draws keeps one null definition for both
deliverables and costs no extra compute. This module only reshapes; every
figure and table is still rendered by `plot_bootstrap_blindspots`, so band
definitions and styling stay changeable from the CSVs alone.

The trade against the 200-replicate sweep is resolution: an upper quantile of
50 draws rests on its top two or three values. `null_replicate_floor` refuses
to emit a threshold from fewer draws than it can support.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from . import plot_standard_metric_baselines as P
    from .evaluate_bootstrap_null import displayed
except ImportError:
    import plot_standard_metric_baselines as P
    from evaluate_bootstrap_null import displayed


DEFAULT_QUANTILE = 0.95
# Below this the requested quantile interpolates between too few order
# statistics to mean anything. 50 draws puts p95 between the 47th and 48th.
NULL_REPLICATE_FLOOR = 20


def read_draws(path):
    opener = __import__("gzip").open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_true(value):
    return str(value).strip().lower() == "true"


def wide_rows(draws, experiment="corruption_strength"):
    """Collapse the long draw table into one record per (resample, point)."""
    grouped = {}
    for row in draws:
        if row.get("experiment") != experiment:
            continue
        key = (row["resample_id"], row["corruption"], float(row["severity"]), _is_true(row["is_null"]))
        entry = grouped.setdefault(key, {})
        entry[row["metric"]] = float(row["value"])
    return grouped


def metric_names(grouped):
    names = set()
    for values in grouped.values():
        names.update(values)
    return sorted(names)


# SCWD goes through a float32 sparse matmul that is not bit-reproducible, so
# repeats of one comparison agree to float32 precision rather than exactly.
NULL_DUPLICATE_RTOL = 1e-5


def null_draws(grouped, metrics):
    """One null score per resample and metric.

    The pipeline recomputes the identical null comparison once per corruption
    from the same two feature sets, so the repeats are collapsed to their mean
    rather than counted as extra replicates. A spread wider than float32 noise
    means they are not actually the same comparison, which would silently
    inflate the replicate count, so it raises instead.
    """
    per_resample = {}
    for (resample_id, _corruption, _severity, is_null), values in grouped.items():
        if not is_null:
            continue
        entry = per_resample.setdefault(resample_id, {})
        for name, value in values.items():
            entry.setdefault(name, []).append(value)

    resample_ids = sorted(per_resample)
    if not resample_ids:
        raise ValueError("No null rows found in the draw table.")
    collapsed = {}
    for resample_id in resample_ids:
        collapsed[resample_id] = {}
        for name, repeats in per_resample[resample_id].items():
            values = np.asarray(repeats, dtype=np.float64)
            spread = float(values.max() - values.min())
            if spread > NULL_DUPLICATE_RTOL * max(abs(float(values.mean())), 1e-30):
                raise ValueError(
                    f"Null draws for {resample_id}/{name} disagree across corruptions "
                    f"(spread {spread:.3g} over {values.size} repeats, "
                    f"min {values.min():.8g}, max {values.max():.8g}). The null is one "
                    "comparison per resample; a real spread means the draw table mixes "
                    "different comparisons and the replicate count would be inflated."
                )
            collapsed[resample_id][name] = float(values.mean())
    missing = [
        (resample_id, name) for resample_id in resample_ids
        for name in metrics if name not in collapsed[resample_id]
    ]
    if missing:
        raise ValueError(f"Null draws missing for {missing[:5]}")
    return resample_ids, {
        name: displayed(name, [collapsed[rid][name] for rid in resample_ids])
        for name in metrics
    }


def curve_records(grouped, metrics, resample_ids):
    """Per-replicate curve rows in `bootstrap_curves.csv` shape."""
    replicate_of = {rid: index for index, rid in enumerate(sorted(
        {rid for (rid, _c, _s, _n) in grouped}
    ))}
    rows = []
    for (resample_id, corruption, severity, is_null), values in sorted(grouped.items()):
        if is_null or severity <= 0.0:
            continue
        record = {
            "corruption": corruption, "severity": severity,
            "replicate": replicate_of[resample_id], "resample_id": resample_id,
        }
        record.update({name: values.get(name, float("nan")) for name in metrics})
        rows.append(record)
    return rows


def verdict_records(curves, nulls, metrics, quantile):
    """N and M per corruption, matching `evaluate_bootstrap_null`'s definitions."""
    thresholds = {name: float(np.quantile(nulls[name], quantile)) for name in metrics}
    by_point = {}
    for row in curves:
        by_point.setdefault((row["corruption"], float(row["severity"])), []).append(row)

    records = []
    for corruption in sorted({name for (name, _) in by_point}):
        ladder = sorted({s for (c, s) in by_point if c == corruption})
        record = {"corruption": corruption, "severity": ladder[-1]}
        for name in metrics:
            columns = [
                displayed(name, [float(r[name]) for r in sorted(
                    by_point[(corruption, s)], key=lambda r: int(r["replicate"])
                )])
                for s in ladder
            ]
            curve = [float(column.mean()) for column in columns]
            value, threshold = curve[-1], thresholds[name]
            monotone_per_replicate = [
                all(b >= a for a, b in zip(sequence, sequence[1:]))
                for sequence in np.stack(columns, axis=1)
            ]
            record.update({
                f"{name}__value": value,
                f"{name}__sd": float(columns[-1].std(ddof=1)) if columns[-1].size > 1 else 0.0,
                f"{name}__threshold": threshold,
                f"{name}__margin": value / threshold if threshold > 0 else float("inf"),
                f"{name}__detected": bool(value > threshold),
                f"{name}__null_draws_exceeded": int((nulls[name] < value).sum()),
                f"{name}__p_value": float((nulls[name] >= value).mean()),
                f"{name}__monotone": bool(all(b >= a for a, b in zip(curve, curve[1:]))),
                f"{name}__monotone_fraction": float(np.mean(monotone_per_replicate)),
            })
        records.append(record)
    return records, thresholds


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def convert(draw_path, output_dir, quantile=DEFAULT_QUANTILE, scheme="temporal_resample"):
    """Write the four files `plot_bootstrap_blindspots` expects."""
    grouped = wide_rows(read_draws(draw_path))
    if not grouped:
        raise ValueError(f"No corruption_strength draws found in {draw_path}.")
    metrics = metric_names(grouped)
    resample_ids, nulls = null_draws(grouped, metrics)
    replicates = len(resample_ids)
    if replicates < NULL_REPLICATE_FLOOR:
        raise ValueError(
            f"Only {replicates} null draws available; a p{quantile*100:g} threshold needs at "
            f"least {NULL_REPLICATE_FLOOR}. Raise temporal_resampling.fixed_replicates."
        )
    curves = curve_records(grouped, metrics, resample_ids)
    verdicts, thresholds = verdict_records(curves, nulls, metrics, quantile)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "bootstrap_curves.csv", curves)
    _write_csv(output_dir / "bootstrap_null_draws.csv", [
        {"replicate": index, "resample_id": resample_id,
         **{f"{scheme}__{name}": float(nulls[name][index]) for name in metrics}}
        for index, resample_id in enumerate(resample_ids)
    ])
    _write_csv(output_dir / "bootstrap_null_verdicts.csv", verdicts)
    summary = {
        "metrics": metrics,
        "replicates": replicates,
        "curve_replicates": len({row["replicate"] for row in curves}),
        "threshold_quantile": quantile,
        "thresholds": thresholds,
        "null_scheme": scheme,
        "source_draws": str(draw_path),
        "null_definition": (
            "ERA5 test window vs its buffered training complement, one draw per "
            "temporal resample"
        ),
        f"{scheme}_mean": {name: float(nulls[name].mean()) for name in metrics},
        f"{scheme}_sd": {
            name: float(nulls[name].std(ddof=1)) if replicates > 1 else 0.0
            for name in metrics
        },
    }
    (output_dir / "bootstrap_null_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("draws", type=Path,
                        help="fixed_metric_draws.csv from a temporal-resampling run")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="directory to write the bootstrap-null-shaped files into")
    parser.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    parser.add_argument("--scheme", default="temporal_resample")
    args = parser.parse_args()
    summary = convert(args.draws, args.output_dir, args.quantile, args.scheme)
    print(f"{summary['replicates']} null draws, {summary['curve_replicates']} curve replicates, "
          f"{len(summary['metrics'])} metrics -> {args.output_dir}")
    print(f"Render with: python scripts/plot_bootstrap_blindspots.py "
          f"bootstrap_null.data_dir={args.output_dir}")


if __name__ == "__main__":
    main()

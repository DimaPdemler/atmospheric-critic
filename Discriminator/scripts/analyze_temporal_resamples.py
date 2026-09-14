"""Query unaggregated temporal-resampling draws."""

import argparse
import csv
from pathlib import Path

import numpy as np


def rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def reconstruct_scores_from_terms(path):
    opener = __import__("gzip").open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        records = list(csv.DictReader(handle))
    ep = {}
    candidates = {}
    for row in records:
        base = (row["resample_id"], row["architecture"], row["kind"], row["target"])
        if row["role"] in {"ep_reference", "ep_train"}:
            ep.setdefault(base, []).append(float(row["transformed_term"]))
        else:
            key = (*base, row["source"], float(row["x"]))
            candidates.setdefault(key, []).append(float(row["transformed_term"]))
    return {key: float(np.mean(ep[key[:4]]) - np.mean(values))
            for key, values in candidates.items()}


def compare_null_to_coordinate(path, architecture, target, coordinate, kind="forecast"):
    selected = [row for row in rows(path)
                if row["architecture"] == architecture and row["target"] == target
                and row["kind"] == kind]
    null = [row for row in selected if row["is_era5_test_null"].lower() == "true"]
    candidate = [row for row in selected
                 if row["is_era5_test_null"].lower() != "true"
                 and np.isclose(float(row["x"]), float(coordinate))]
    if not null or not candidate:
        raise ValueError(
            f"No matching draws: null={len(null)}, candidate={len(candidate)} for "
            f"{architecture}/{kind}/{target} at {coordinate:g}."
        )
    candidate_mean = float(np.mean([float(row["score"]) for row in candidate]))
    exceeding = [row for row in null if float(row["score"]) > candidate_mean]
    return {
        "architecture": architecture, "kind": kind, "target": target,
        "coordinate": float(coordinate), "candidate_mean": candidate_mean,
        "null_count": len(null), "null_exceeding_count": len(exceeding),
        "null_exceeding_fraction": len(exceeding) / len(null),
        "candidate_draws": [(row["resample_id"], float(row["score"])) for row in candidate],
        "null_draws": [(row["resample_id"], float(row["score"])) for row in null],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("draws", type=Path, help="discriminator_metric_draws.csv")
    parser.add_argument("--architecture", default="squeezenet")
    parser.add_argument("--target", required=True)
    parser.add_argument("--kind", default="forecast", choices=("forecast", "corruption"))
    parser.add_argument("--coordinate", type=float, default=12.0,
                        help="lead hour or native corruption severity")
    args = parser.parse_args()
    result = compare_null_to_coordinate(
        args.draws, args.architecture, args.target, args.coordinate, args.kind,
    )
    print(f"Candidate mean: {result['candidate_mean']:.9g}")
    print(f"Nulls above mean: {result['null_exceeding_count']}/{result['null_count']} "
          f"({result['null_exceeding_fraction']:.1%})")
    print("Candidate draws:")
    for resample_id, value in result["candidate_draws"]:
        print(f"  {resample_id}: {value:.9g}")
    print("Null draws:")
    for resample_id, value in result["null_draws"]:
        print(f"  {resample_id}: {value:.9g}")


if __name__ == "__main__":
    main()

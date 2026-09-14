import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Discriminator.scripts.blindspots_from_temporal_draws import (
    convert, curve_records, null_draws, verdict_records, wide_rows,
)

METRICS = ["mmd_rbf", "global_mean_wasserstein"]
SEVERITIES = [0.05, 0.1, 0.2]


def draw_rows(n_resamples=50, null_value=0.03, curves=None, null_jitter=0.0):
    """Long-format draws shaped like the pipeline's fixed_metric_draws.csv.

    `curves` maps a corruption name to its value at each entry of SEVERITIES.
    """
    curves = curves or {
        "strong": [0.25, 0.5, 1.0],     # rises well past the null
        "flat": [0.001, 0.001, 0.001],  # below the null, and non-decreasing
        "falling": [0.9, 0.5, 0.1],     # above the null at first, then decreases
    }
    rows = []
    for index in range(n_resamples):
        resample_id = f"fixed_{index:03d}"
        nulls = {name: null_value + null_jitter * index for name in METRICS}
        for corruption, ladder in curves.items():
            for name in METRICS:
                rows.append({
                    "family": "fixed", "resample_id": resample_id,
                    "experiment": "corruption_strength", "metric": name,
                    "label": "ERA5 test-vs-train null", "variable": "T2M",
                    "corruption": corruption, "severity": 0.0, "is_null": "True",
                    "value": nulls[name], "n_samples": 1000, "pairwise_n_samples": 256,
                })
                for severity, value in zip(SEVERITIES, ladder):
                    rows.append({
                        "family": "fixed", "resample_id": resample_id,
                        "experiment": "corruption_strength", "metric": name,
                        "label": "corrupted", "variable": "T2M",
                        "corruption": corruption, "severity": severity, "is_null": "False",
                        "value": value, "n_samples": 1000, "pairwise_n_samples": 256,
                    })
    return rows


def write_draws(directory, rows):
    path = Path(directory) / "fixed_metric_draws.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


class TemporalBlindspotTests(unittest.TestCase):
    def test_repeated_null_rows_collapse_to_one_draw_per_resample(self):
        grouped = wide_rows(draw_rows(n_resamples=7))
        resample_ids, nulls = null_draws(grouped, METRICS)
        # Two corruptions repeat the same null; the replicate count must not double.
        self.assertEqual(len(resample_ids), 7)
        self.assertEqual(nulls["mmd_rbf"].size, 7)
        self.assertTrue(np.allclose(nulls["mmd_rbf"], 0.03))

    def test_null_rows_that_actually_disagree_are_rejected(self):
        rows = draw_rows(n_resamples=5)
        for row in rows:
            if row["is_null"] == "True" and row["corruption"] == "flat":
                row["value"] = 0.9
        with self.assertRaises(ValueError) as caught:
            null_draws(wide_rows(rows), METRICS)
        self.assertIn("disagree across corruptions", str(caught.exception))

    def test_detection_uses_the_null_quantile_and_monotonicity_the_ladder(self):
        grouped = wide_rows(draw_rows())
        _, nulls = null_draws(grouped, METRICS)
        curves = curve_records(grouped, METRICS, [])
        verdicts, thresholds = verdict_records(curves, nulls, METRICS, 0.95)
        by_corruption = {row["corruption"]: row for row in verdicts}
        self.assertAlmostEqual(thresholds["mmd_rbf"], 0.03, places=9)
        # Reaches 1.0 at the top severity, far above the 0.03 null.
        self.assertTrue(by_corruption["strong"]["mmd_rbf__detected"])
        self.assertTrue(by_corruption["strong"]["mmd_rbf__monotone"])
        self.assertEqual(by_corruption["strong"]["mmd_rbf__null_draws_exceeded"], 50)
        # Below the null, so no detection. Still monotone: the verdict is
        # non-decreasing (b >= a), matching evaluate_bootstrap_null, so a flat
        # curve passes M even though it carries no signal.
        self.assertFalse(by_corruption["flat"]["mmd_rbf__detected"])
        self.assertTrue(by_corruption["flat"]["mmd_rbf__monotone"])
        self.assertEqual(by_corruption["flat"]["mmd_rbf__null_draws_exceeded"], 0)
        # Ends at 0.1, still above the null, but the ladder decreases.
        self.assertTrue(by_corruption["falling"]["mmd_rbf__detected"])
        self.assertFalse(by_corruption["falling"]["mmd_rbf__monotone"])

    def test_convert_writes_the_renderer_schema_with_its_scheme_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_draws(directory, draw_rows(null_jitter=1e-4))
            output = Path(directory) / "converted"
            summary = convert(path, output, quantile=0.95, scheme="temporal_resample")
            self.assertEqual(summary["replicates"], 50)
            self.assertEqual(summary["curve_replicates"], 50)
            self.assertEqual(summary["null_scheme"], "temporal_resample")
            for name in ("bootstrap_curves.csv", "bootstrap_null_draws.csv",
                         "bootstrap_null_verdicts.csv", "bootstrap_null_summary.json"):
                self.assertTrue((output / name).is_file(), name)
            with open(output / "bootstrap_null_draws.csv", newline="") as handle:
                header = next(csv.reader(handle))
            self.assertIn("temporal_resample__mmd_rbf", header)
            reloaded = json.loads((output / "bootstrap_null_summary.json").read_text())
            self.assertEqual(sorted(reloaded["metrics"]), sorted(METRICS))

    def test_too_few_resamples_refuse_to_produce_a_quantile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_draws(directory, draw_rows(n_resamples=5))
            with self.assertRaises(ValueError) as caught:
                convert(path, Path(directory) / "converted")
            self.assertIn("at least", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

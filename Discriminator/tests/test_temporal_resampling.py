import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from Discriminator.scripts.analyze_temporal_resamples import (
    compare_null_to_coordinate, reconstruct_scores_from_terms,
)
from Discriminator.scripts.temporal_resampling_pipeline import (
    _canonical_first, _equalize_discriminator_term_counts,
)
from Discriminator.scripts.temporal_resampling import (
    TemporalSchedule, aggregate_draws, fixed_schedules, learned_schedules, schedule_mask, write_csv_gz,
)


def config():
    return OmegaConf.create({
        "temporal_resampling": {
            "enabled": True, "learned_replicates": 5, "fixed_replicates": 10,
            "learned_test_windows": [[5, 11], [9, 15], [13, 19], [17, 23], [20, 26]],
            "buffer_days": 4, "random_test_start_range": [5, 20], "seed": 7,
            "active_schedule": None,
        }
    })


class TemporalResamplingTests(unittest.TestCase):
    def test_learned_windows_and_buffer_are_disjoint(self):
        schedule = learned_schedules(config())[0]
        values = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-02-01"))
        ranges = [(np.datetime64("2020-01-01"), np.datetime64("2020-01-31"))]
        test = schedule_mask(values, schedule, "test", ranges)
        train = schedule_mask(values, schedule, "train", ranges)
        self.assertEqual(values[test][0], np.datetime64("2020-01-05"))
        self.assertEqual(values[test][-1], np.datetime64("2020-01-11"))
        self.assertFalse(np.any(test & train))
        self.assertFalse(np.any(train[:15]))
        self.assertTrue(np.all(train[15:]))

    def test_fixed_schedules_are_deterministic_and_keep_seven_day_tests(self):
        ranges = [("2020-01-01", "2020-12-31")]
        first = fixed_schedules(config(), ranges)
        second = fixed_schedules(config(), ranges)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(first[4].test_days, (20, 26))
        for schedule in first[5:]:
            for month, _ in schedule.monthly_test_starts:
                lower, upper = schedule.test_window(month)
                self.assertEqual(upper - lower + 1, 7)
                self.assertGreaterEqual(lower, 5)
                self.assertLessEqual(upper, 26)

    def test_canonical_learned_fold_executes_first_without_renaming(self):
        schedules = learned_schedules(config())
        ordered = _canonical_first(schedules, "learned_04")
        self.assertEqual(ordered[0].resample_id, "learned_04")
        self.assertEqual(
            [schedule.resample_id for schedule in ordered[1:]],
            ["learned_00", "learned_01", "learned_02", "learned_03"],
        )

    def test_aggregates_preserve_draw_ids_and_requested_bounds(self):
        rows = [
            {"target": "GraphCast", "x": "12", "resample_id": f"fold_{i}", "score": value}
            for i, value in enumerate([1, 2, 3, 4, 10])
        ]
        result = aggregate_draws(rows, ["score"], ["target", "x"], "minmax")[0]
        self.assertEqual(result["score"], 4.0)
        self.assertEqual(result["score_lower"], 1.0)
        self.assertEqual(result["score_upper"], 10.0)
        self.assertEqual(result["n_resamples"], 5)
        self.assertEqual(result["draw_ids"], "fold_0,fold_1,fold_2,fold_3,fold_4")

    def test_query_counts_null_draws_above_lead_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draws.csv"
            fields = ["architecture", "kind", "target", "x", "is_era5_test_null",
                      "score", "resample_id"]
            records = []
            for index, score in enumerate([1, 2, 3, 4, 5]):
                records.append(dict(zip(fields, ["squeezenet", "forecast", "GraphCast", 0,
                                                 True, score, f"fold_{index}"])))
            for index, score in enumerate([2, 2, 2, 2, 2]):
                records.append(dict(zip(fields, ["squeezenet", "forecast", "GraphCast", 12,
                                                 False, score, f"fold_{index}"])))
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows(records)
            result = compare_null_to_coordinate(path, "squeezenet", "GraphCast", 12)
            self.assertEqual(result["candidate_mean"], 2.0)
            self.assertEqual(result["null_exceeding_count"], 3)
            self.assertEqual(result["null_count"], 5)

    def test_discriminator_terms_are_equalized_to_cross_fold_minimum(self):
        rows, terms = [], []
        for fold, ep_values, candidate_values in (
            ("learned_00", [1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0]),
            ("learned_01", [4.0, 6.0], [50.0, 60.0]),
        ):
            common = {
                "resample_id": fold, "architecture": "squeezenet",
                "kind": "forecast", "target": "GraphCast",
            }
            rows.append({
                **common, "source": "GraphCast", "x": "192.0",
                "n_samples": str(len(candidate_values)),
                "ep_n_samples": str(len(ep_values)), "score": "0",
            })
            terms.extend({
                **common, "role": "ep_reference", "source": "ERA5 test",
                "x": "", "sample_position": str(index),
                "transformed_term": str(value),
            } for index, value in enumerate(ep_values))
            terms.extend({
                **common, "role": "candidate", "source": "GraphCast",
                "x": "192.0", "sample_position": str(index),
                "transformed_term": str(value),
            } for index, value in enumerate(candidate_values))

        equalized_rows, equalized_terms = _equalize_discriminator_term_counts(rows, terms)
        self.assertEqual([row["n_samples"] for row in equalized_rows], [2, 2])
        self.assertEqual([row["ep_n_samples"] for row in equalized_rows], [2, 2])
        self.assertEqual(len(equalized_terms), 8)
        self.assertAlmostEqual(equalized_rows[0]["score"], -23.0)
        self.assertAlmostEqual(equalized_rows[1]["score"], -50.0)

    def test_raw_terms_exactly_reconstruct_reverse_kl_draw(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terms.csv.gz"
            common = {"resample_id": "learned_00", "architecture": "squeezenet",
                      "kind": "forecast", "target": "GraphCast"}
            terms = [
                {**common, "role": "ep_reference", "source": "ERA5 test", "x": "",
                 "transformed_term": value} for value in (-2.0, -4.0)
            ] + [
                {**common, "role": "candidate", "source": "GraphCast", "x": 12,
                 "transformed_term": value} for value in (-5.0, -7.0)
            ]
            write_csv_gz(path, terms)
            reconstructed = reconstruct_scores_from_terms(path)
            key = ("learned_00", "squeezenet", "forecast", "GraphCast", "GraphCast", 12.0)
            self.assertEqual(reconstructed[key], 3.0)


if __name__ == "__main__":
    unittest.main()

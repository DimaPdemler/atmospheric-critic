import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from Discriminator.scripts.histogram_matching import (
    HistogramMapStore, apply_quantile_table, map_id, quantile_table,
)
from Discriminator.scripts.histogram_matching_apply import match_raw


class HistogramMatchingTests(unittest.TestCase):
    def test_quantile_map_matches_shifted_empirical_distribution(self):
        source = np.linspace(-3.0, 3.0, 1001) ** 3
        target = np.linspace(250.0, 310.0, 1001)
        source_q, target_q = quantile_table(source, target, 1001)
        matched = apply_quantile_table(source, source_q, target_q)
        np.testing.assert_allclose(np.quantile(matched, [0, .25, .5, .75, 1]),
                                   np.quantile(target, [0, .25, .5, .75, 1]), atol=1e-5)
        self.assertTrue(np.all(np.diff(matched) >= 0.0))

    def test_repeated_source_quantiles_and_tails_are_safe(self):
        source_q, target_q = quantile_table([0, 0, 0, 1, 2], [10, 20, 30, 40, 50], 17)
        self.assertTrue(np.all(np.diff(source_q) > 0.0))
        matched = apply_quantile_table(np.asarray([-1.0, 0.0, 3.0, np.nan]), source_q, target_q)
        self.assertEqual(matched[0], target_q[0])
        self.assertEqual(matched[2], target_q[-1])
        self.assertTrue(np.isnan(matched[3]))

    def test_store_round_trip_and_runtime_channel_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            store = HistogramMapStore()
            key = map_id("standard", "forecast", "GraphCast", 6, "2m_temperature")
            store.add(key, np.arange(10.0), np.arange(10.0) + 100.0, knots=10)
            store.save(directory)
            loaded = HistogramMapStore.load(directory)
            np.testing.assert_allclose(loaded.maps[key].apply([0.0, 9.0]), [100.0, 109.0])
            cfg = OmegaConf.create({
                "histogram_matching": {"enabled": True},
                "pipeline": {"input_histogram_matching_dir": directory},
                "baseline": {"output_dir": str(Path(directory) / "unused")},
            })
            values = np.arange(4.0, dtype=np.float32).reshape(1, 2, 2)
            matched = match_raw(cfg, values, ["2m_temperature"], "standard",
                                "forecast", "GraphCast", 6)
            np.testing.assert_allclose(matched, values + 100.0)

    def test_zero_corruption_is_identity_without_map_lookup(self):
        cfg = OmegaConf.create({
            "histogram_matching": {"enabled": True},
            "pipeline": {"input_histogram_matching_dir": "/does/not/exist"},
            "baseline": {"output_dir": "/also/missing"},
        })
        values = np.arange(4.0, dtype=np.float32).reshape(1, 2, 2)
        self.assertIs(match_raw(cfg, values, ["x"], "standard", "corruption", "blur", 0), values)


if __name__ == "__main__":
    unittest.main()

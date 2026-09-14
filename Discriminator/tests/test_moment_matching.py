import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from Discriminator.scripts.fake_matching_apply import (
    match_raw, match_standardized, matching_mode,
)
from Discriminator.scripts.fake_matching_checkpoint import validate_binding, write_binding
from Discriminator.scripts.moment_matching import (
    MomentMapStore, RunningMoments, map_id,
)


def accumulator(values):
    result = RunningMoments(); result.update(values); return result


class MomentMatchingTests(unittest.TestCase):
    def config(self, directory, *, moments=True, histogram=False):
        return OmegaConf.create({
            "moment_matching": {"enabled": moments},
            "histogram_matching": {"enabled": histogram},
            "pipeline": {"input_moment_matching_dir": directory,
                         "input_histogram_matching_dir": None},
            "baseline": {"output_dir": str(Path(directory) / "unused")},
        })

    def test_fit_sample_matches_target_mean_and_std(self):
        source = np.linspace(-2.0, 5.0, 1001)
        target = np.linspace(250.0, 310.0, 1001) ** 1.01
        store = MomentMapStore()
        key = map_id("standard", "forecast", "GraphCast", 6, "x")
        store.add_moments(key, accumulator(source), accumulator(target))
        matched = store.maps[key].apply(source)
        self.assertAlmostEqual(float(matched.mean()), float(target.mean()), places=4)
        self.assertAlmostEqual(float(matched.std()), float(target.std()), places=4)

    def test_store_round_trip_and_standardized_application(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MomentMapStore()
            key = map_id("standard", "corruption", "noise", 0.2, "x")
            store.add_moments(key, accumulator([0.0, 2.0]), accumulator([10.0, 14.0]))
            store.save(directory)
            cfg = self.config(directory)
            standardized = np.asarray([[[0.0, 1.0]]], dtype=np.float32)
            matched = match_standardized(
                cfg, standardized, ["x"], {"x": 0.0}, {"x": 2.0},
                "standard", "corruption", "noise", 0.2,
            )
            np.testing.assert_allclose(matched, [[[5.0, 7.0]]])

    def test_zero_corruption_is_identity_without_artifact(self):
        cfg = self.config("/does/not/exist")
        values = np.arange(4.0, dtype=np.float32).reshape(1, 2, 2)
        self.assertIs(
            match_raw(cfg, values, ["x"], "standard", "corruption", "noise", 0.0),
            values,
        )

    def test_matching_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            matching_mode(self.config("/unused", moments=True, histogram=True))

    def test_checkpoint_binding_records_mode_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MomentMapStore()
            key = map_id("standard", "forecast", "GraphCast", 6, "x")
            store.add_moments(key, accumulator([0.0, 1.0]), accumulator([2.0, 4.0]))
            store.save(directory)
            cfg = self.config(directory)
            checkpoint = Path(directory) / "model.pth"; checkpoint.write_bytes(b"model")
            binding = write_binding(checkpoint, cfg)
            self.assertEqual(json.loads(binding.read_text())["mode"], "moments")
            validate_binding(checkpoint, cfg)
            with self.assertRaises(ValueError):
                validate_binding(checkpoint, self.config(directory, moments=False))


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf
import torch

from Discriminator.scripts.corruptions import (
    apply_gaussian_field_noise,
    apply_random_pixel_replace,
)

from Discriminator.scripts.plot_standard_metric_baselines import (
    apply_special_baseline_corruption,
    close_feature_memmaps,
    corruption_levels,
    corruption_max_severity,
    deranged_sample_positions,
    fieldwise_deranged_sample_positions,
    display_metric_value,
    displayed_metric_name,
    evaluate_corruption_metrics,
    finalize_global_mean_wasserstein,
    fit_vissio_ulam_grid,
    global_mean_wasserstein_diagnostic,
    representative_corruption_time_index,
    gallery_logit_bin_edges,
    repeated_diagnostic_labels,
    relative_corruption_coordinates,
    corruption_range_label,
    pairwise_sample_positions,
    mmd_rbf_bandwidths,
    mmd_rbf_distance,
    null_diamond_positions,
    reference_features_from_config,
    scwd_anchor_diagnostic,
    scwd_anchor_transport_costs,
    scwd_anchor_area_weights,
    scwd_area_weighted_from_responses,
    pointwise_moment_totals,
    plot_scwd_mean_response_differences,
    plot_scwd_anchor_diagnostics,
    plot_scwd_top_wasserstein_distributions,
    plot_discriminator_baselines,
    plot_normalized_corruption_metrics_by_type,
    plot_normalized_lead_metrics_by_model,
    plot_corruption_disturbances,
    plotted_metric_names,
    read_scwd_anchor_diagnostics,
    read_global_mean_wasserstein_diagnostics,
    standalone_metric_names,
    metric_normalization_scales,
    selected_distribution_metric_values,
    streaming_joint_features,
    streaming_reference_stats,
    structured_near_null_pattern,
    vissio_global_mean_wasserstein,
    write_discriminator_baselines,
    write_scwd_anchor_diagnostics,
    write_global_mean_wasserstein_diagnostics,
)
from Discriminator.scripts.temporal_holdout_utils import reconcile_checkpoint_variables


def baseline_test_config():
    return OmegaConf.create(
        {
            "baseline": {
                "pairwise_eval_samples": 6,
                "feature_chunk_size": 4,
                "filter_invalid_fields": True,
                "invalid_zero_atol": 1e-12,
                "invalid_min_std": 1e-12,
                "swd_pixels": 40,
                "swd_projections": 8,
                "swd_seed": 3,
                "backend": "numpy",
                "dtype": "float32",
                "mmd_standardize": True,
                "mmd_bandwidth": 1.0,
                "field_energy_chunk_size": 2,
                "spectrum_log_eps_factor": 1e-12,
                "scwd_anchor_lat_points": 4,
                "scwd_anchor_lon_points": 8,
                "scwd_domain_lat_points": 9,
                "scwd_domain_lon_points": 16,
                "scwd_radius_km": 4000.0,
                "scwd_order": 2.0,
                "scwd_quantiles": 10,
                "scwd_ot_samples": 6,
                "scwd_ot_progress": False,
                "scwd_top_wasserstein_anchors": 6,
                "scwd_anchor_chunk_size": 4,
                "corruption_seed": 11,
                "corruption_steps": 3,
                "corruption_severity_max": 0.05,
            }
        }
    )


def synthetic_temperature_dataset():
    values = np.random.default_rng(4).normal(size=(12, 5, 8)).astype(np.float32)
    return xr.Dataset(
        {"2m_temperature": (("time", "latitude", "longitude"), values)},
        coords={
            "time": pd.date_range("2020-01-01", periods=12, freq="6h"),
            "latitude": np.linspace(-72, 72, 5),
            "longitude": np.arange(8) * 45.0,
        },
    )


class RepeatedPlotSelectionTest(unittest.TestCase):
    def test_representative_mode_keeps_one_label_per_comparison_kind(self):
        diagnostics = [
            {"comparison_kind": "forecast", "label": "Pangu-Weather"},
            {"comparison_kind": "forecast", "label": "GraphCast"},
            {"comparison_kind": "corruption", "label": "grf"},
            {"comparison_kind": "corruption", "label": "gaussian_blur"},
            {"comparison_kind": "null", "label": "ERA5 null"},
        ]
        self.assertEqual(
            repeated_diagnostic_labels(diagnostics, representative_only=True),
            [
                ("corruption", "gaussian_blur"),
                ("forecast", "GraphCast"),
                ("null", "ERA5 null"),
            ],
        )
        self.assertEqual(len(repeated_diagnostic_labels(diagnostics)), 5)


class FullStatisticsBaselineTest(unittest.TestCase):
    def test_pointwise_moment_totals_do_not_apply_latitude_weights(self):
        values = np.asarray([[[0.0, 0.0], [10.0, 10.0]]])
        total, total_sq, count = pointwise_moment_totals(values)
        self.assertEqual(count, 4.0)
        self.assertEqual(total, 20.0)
        self.assertEqual(total_sq, 200.0)

    def test_signed_moment_metrics_are_absolute_only_for_plotting(self):
        row = {"mean_bias": -3.5, "std_ratio_error": -0.25, "scwd": -1.5}
        self.assertEqual(display_metric_value(row, "mean_bias"), 3.5)
        self.assertEqual(display_metric_value(row, "std_ratio_error"), 0.25)
        self.assertEqual(display_metric_value(row, "scwd"), -1.5)
        self.assertEqual(displayed_metric_name("mean_bias"), "|Mean bias|")
        self.assertEqual(displayed_metric_name("std_ratio_error"), "|Std. ratio error|")
        self.assertEqual(displayed_metric_name("mmd_rbf"), "RBF MMD")
        self.assertEqual(displayed_metric_name("scwd"), "SCWD")

    def test_corruption_coordinates_span_one_per_native_range(self):
        blur = [{"severity": 0.0}, {"severity": 0.5}, {"severity": 1.0}]
        pixels = [{"severity": 0.0}, {"severity": 0.025}, {"severity": 0.05}]
        self.assertEqual(relative_corruption_coordinates(blur, "severity"), [0.0, 0.5, 1.0])
        self.assertEqual(relative_corruption_coordinates(pixels, "severity"), [0.0, 0.5, 1.0])
        self.assertEqual(corruption_range_label("pixel_replace", pixels, "severity"), "Pixel replace (0.05)")

    def test_evaluation_only_metrics_are_not_selected_for_standard_plots(self):
        metrics = ["mean_bias", "crps_like_field_energy", "sliced_wasserstein",
                   "sliced_wasserstein_lon_corrected", "zonal_energy_spectrum_l2",
                   "scwd_area_weighted", "scwd"]
        self.assertEqual(plotted_metric_names(metrics), ["scwd"])
        self.assertEqual(standalone_metric_names(metrics), ["mean_bias", "scwd"])

    def test_mmd_fits_a_reference_bandwidth_per_field(self):
        reference = np.asarray(
            [[0.0, 0.0], [1.0, 100.0], [2.0, 200.0]], dtype=np.float64
        )
        bandwidths = mmd_rbf_bandwidths(reference, [(0, 1), (1, 2)])
        np.testing.assert_allclose(bandwidths, [1.0, 100.0])

    def test_per_field_mmd_remains_joint_across_fields(self):
        reference = np.asarray(
            [[-1.0, -1.0], [-0.5, -0.5], [0.5, 0.5], [1.0, 1.0]],
            dtype=np.float64,
        )
        candidate = reference.copy()
        candidate[:, 1] = candidate[::-1, 1]
        cfg = OmegaConf.create({"baseline": {
            "backend": "numpy",
            "mmd_standardize": False,
            "mmd_bandwidth_mode": "per_field",
            "mmd_bandwidth": 0.5,
        }})
        score = mmd_rbf_distance(candidate, reference, cfg, [(0, 1), (1, 2)])
        self.assertGreater(score, 0.1)
    def test_disturbance_gallery_renders_two_severity_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            levels = np.linspace(0.0, 0.2, 5, dtype=np.float32)[None, :]
            disturbance = np.arange(5 * 1 * 3 * 4, dtype=np.float32).reshape(1, 5, 1, 3, 4)
            xr.Dataset(
                {
                    "disturbance": (("corruption", "severity_index", "variable", "latitude", "longitude"), disturbance),
                    "corrupted_field": (("corruption", "severity_index", "variable", "latitude", "longitude"), disturbance + 273.15),
                },
                coords={
                    "corruption": ["fixture"], "severity": (("corruption", "severity_index"), levels),
                    "variable": ["2m_temperature"], "latitude": [-45.0, 0.0, 45.0],
                    "longitude": [0.0, 90.0, 180.0, 270.0],
                },
            ).to_netcdf(data_dir / "corruption_disturbances.nc")
            with patch("cartopy.mpl.geoaxes.GeoAxes.coastlines"), patch("cartopy.mpl.geoaxes.GeoAxes.add_feature"):
                plot_corruption_disturbances(root)
            self.assertTrue((root / "plots" / "corruption" / "disturbances" / "fixture.png").is_file())
            self.assertTrue((root / "plots" / "corruption" / "disturbances" / "fixture_corrupted.png").is_file())




    def test_null_diamond_positions_dodge_left_of_zero(self):
        positions = null_diamond_positions(5, 6.0)
        np.testing.assert_allclose(positions, [-6.0, -4.5, -3.0, -1.5, 0.0])
        self.assertEqual(len(np.unique(positions)), 5)

    def test_era5_forecast_gallery_uses_original_logits_on_independent_axis(self):
        panels = [
            ("GraphCast", np.asarray([-4.0, 4.0]), [(6, np.asarray([-6.0, 6.0]))]),
            ("Pangu-Weather", np.asarray([-2.0, 2.0]), [(6, np.asarray([-3.0, 3.0]))]),
            ("ERA5 Forecast", np.asarray([-0.2, 0.2]), [(6, np.asarray([-0.3, 0.3]))]),
        ]
        common, by_model = gallery_logit_bin_edges(panels)
        np.testing.assert_array_equal(by_model["GraphCast"], common)
        np.testing.assert_array_equal(by_model["Pangu-Weather"], common)
        self.assertAlmostEqual(float(by_model["ERA5 Forecast"][0]), -0.3)
        self.assertAlmostEqual(float(by_model["ERA5 Forecast"][-1]), 0.3)
        self.assertGreater(float(common[-1]), 5.0)

    def test_normalized_converse_plots_use_global_signed_metric_scales(self):
        metrics = ["positive", "mixed", "negative_only", "zero_only"]
        scale_rows = [
            {"positive": 2.0, "mixed": -3.0, "negative_only": -4.0, "zero_only": 0.0},
            {"positive": 8.0, "mixed": 2.0, "negative_only": -1.0, "zero_only": np.nan},
        ]
        scales = metric_normalization_scales(scale_rows, metrics)
        self.assertEqual(scales, {
            "positive": 8.0, "mixed": 2.0, "negative_only": 4.0, "zero_only": 1.0,
        })
        lead_rows = [
            {"label": "ERA5 Test vs Train", "variable": "T2M", "lead_hour": 0, **scale_rows[0]},
            {"label": "Model A", "variable": "T2M", "lead_hour": 6, **scale_rows[1]},
            {"label": "Model B", "variable": "T2M", "lead_hour": 6, **scale_rows[0]},
        ]
        corruption_rows = [
            {"corruption": "gaussian_blur", "variable": "T2M", "severity": 0.0, **scale_rows[0]},
            {"corruption": "gaussian_blur", "variable": "T2M", "severity": 1.0, **scale_rows[1]},
            {"corruption": "pixel_replace", "variable": "T2M", "severity": 0.0, **scale_rows[0]},
            {"corruption": "pixel_replace", "variable": "T2M", "severity": 0.05, **scale_rows[1]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            discriminator_rows = [
                {"architecture": "squeezenet", "kind": "forecast", "target": target,
                 "x": 6.0, "score": score, "score_lower": score - 0.1,
                 "score_upper": score + 0.1, "is_era5_test_null": False}
                for target, score in (("Model A", 2.0), ("Model B", 1.0))
            ]
            plot_normalized_lead_metrics_by_model(
                lead_rows, metrics, ["T2M"], root, discriminator_rows,
            )
            plot_normalized_corruption_metrics_by_type(corruption_rows, metrics, ["T2M"], root)
            self.assertTrue((root / "plots" / "lead_time_by_model_normalized.png").is_file())
            with np.load(root / "plots" / "lead_time_by_model_normalized.npz", allow_pickle=False) as bundle:
                metadata = json.loads(str(bundle["metadata_json"]))
            line_labels = {
                line["label"] for axis in metadata["axes"] for line in axis["lines"]
            }
            self.assertIn("Learned critic", line_labels)
            self.assertTrue((root / "plots" / "corruption_by_type_normalized.png").is_file())

    def test_discriminator_outputs_are_separated_by_architecture(self):
        cfg = OmegaConf.create({
            "baseline": {"discriminator": {"plot_yscale": "linear", "plot_linthresh": 1e-2}}
        })
        rows = []
        for architecture, variables in (
            ("squeezenet", "2m_temperature"),
            ("sfno_linear", ",".join([
                "2m_temperature", "10m_u_component_of_wind",
                "10m_v_component_of_wind", "mean_sea_level_pressure",
            ])),
        ):
            for kind, target, candidate_x in (
                ("forecast", "GraphCast", 6.0),
                ("corruption", "hf_noise", 0.1),
            ):
                for x, source, is_null in ((0.0, "ERA5 test", True), (candidate_x, target, False)):
                    rows.append({
                        "architecture": architecture,
                        "input_variables": variables,
                        "encoder_pretraining": "overlap" if architecture.startswith("sfno") else "",
                        "kind": kind, "target": target, "x": x, "source": source,
                        "is_era5_test_null": is_null, "score": x + 0.1,
                        "stderr": 0.01, "n_samples": 4, "ep_train": -1.0,
                    })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_discriminator_baselines(rows, cfg, root)
            plot_discriminator_baselines(rows, cfg, root)
            table = pd.read_csv(root / "data" / "discriminator_reverse_kl.csv")
            self.assertEqual(set(table["architecture"]), {"squeezenet", "sfno_linear"})
            for architecture in ("squeezenet", "sfno_linear"):
                self.assertTrue((root / "plots" / "discriminator" / architecture / "lead_time_reverse_kl.png").exists())
                self.assertTrue((root / "plots" / "discriminator" / architecture / "corruption_strength_reverse_kl.png").exists())

    def test_structured_near_null_patterns_have_expected_mean_and_unit_area_rms(self):
        latitudes = np.linspace(-90.0, 90.0, 9)
        weights = np.maximum(np.cos(np.deg2rad(latitudes)), 0.0)[:, None]
        denominator = np.sum(weights) * 16
        for name in (
            "equatorial_checker_texture",
            "meridional_scanlines",
            "checkerboard_2px",
        ):
            pattern = structured_near_null_pattern(name, latitudes, 16)
            self.assertAlmostEqual(float(np.sum(pattern * weights) / denominator), 0.0, places=6)
            rms = np.sqrt(np.sum(pattern**2 * weights) / denominator)
            self.assertAlmostEqual(float(rms), 1.0, places=6)
        zonal = structured_near_null_pattern("zonal_scanlines", latitudes, 16)
        self.assertAlmostEqual(float(np.mean(zonal)), 0.0, places=6)
        zonal_rms = np.sqrt(np.sum(zonal**2 * weights) / denominator)
        self.assertAlmostEqual(float(zonal_rms), 1.0, places=6)

    def test_hemisphere_splice_uses_severity_as_replacement_probability(self):
        cfg = baseline_test_config()
        cfg.baseline.corruption_severity_max = 0.05
        cfg.baseline.hemisphere_splice_latitude = 0.0
        latitudes = np.linspace(-75.0, 75.0, 7)
        clean = np.zeros((1, 7, 8), dtype=np.float32)
        donor = np.ones_like(clean)
        full = apply_special_baseline_corruption(
            clean, "hemisphere_splice", 0.05, latitudes, cfg, donor, random_seed=4,
        )
        self.assertTrue(np.allclose(full[:, latitudes < 0], 1.0))
        self.assertTrue(np.all(full[:, latitudes >= 0] == 0.0))
        outcomes = []
        for seed in range(32):
            half = apply_special_baseline_corruption(
                clean, "hemisphere_splice", 0.025, latitudes, cfg, donor, random_seed=seed,
            )
            southern = half[:, latitudes < 0]
            self.assertTrue(np.all(southern == 0.0) or np.all(southern == 1.0))
            self.assertTrue(np.all(half[:, latitudes >= 0] == 0.0))
            outcomes.append(bool(np.all(southern == 1.0)))
        self.assertIn(True, outcomes)
        self.assertIn(False, outcomes)
        permutation = deranged_sample_positions(8, seed=3)
        self.assertEqual(sorted(permutation.tolist()), list(range(8)))
        self.assertTrue(np.all(permutation != np.arange(8)))

    def test_field_splice_replaces_whole_fields_from_independent_donors(self):
        cfg = baseline_test_config()
        cfg.baseline.corruption_severity_max = 0.05
        latitudes = np.linspace(-75.0, 75.0, 7)
        clean = np.zeros((3, 7, 8), dtype=np.float32)
        donor = np.stack([np.full((7, 8), value, dtype=np.float32) for value in (1, 2, 3)])
        full = apply_special_baseline_corruption(
            clean, "field_splice", 0.05, latitudes, cfg, donor, random_seed=4,
        )
        np.testing.assert_array_equal(full, donor)
        for seed in range(16):
            partial = apply_special_baseline_corruption(
                clean, "field_splice", 0.025, latitudes, cfg, donor, random_seed=seed,
            )
            for channel, value in enumerate((1, 2, 3)):
                self.assertTrue(
                    np.all(partial[channel] == 0.0) or np.all(partial[channel] == value)
                )
        permutations = fieldwise_deranged_sample_positions(8, 3, seed=3)
        self.assertEqual(permutations.shape, (3, 8))
        for permutation in permutations:
            self.assertEqual(sorted(permutation.tolist()), list(range(8)))
            self.assertTrue(np.all(permutation != np.arange(8)))
        self.assertFalse(np.array_equal(permutations[0], permutations[1]))

    def test_streamed_corruption_metrics_include_all_new_families(self):
        cfg = baseline_test_config()
        cfg.monthly_split = {
            "strategy": "monthly_valid_time",
            "train_days": [1, 15], "test_days": [20, 26],
            "null_comparison_days": [16, 31],
            "corruption_time_range": ["2020-01-01", "2020-01-31"],
            "model_valid_time_ranges": [["2020-01-01", "2020-01-31"]],
            "require_exact_era5_match": True,
        }
        cfg.baseline.corruption_eval_samples = 4
        cfg.baseline.corruption_steps = 2
        cfg.baseline.corruption_severity_max = 0.05
        cfg.baseline.hemisphere_splice_latitude = 0.0
        cfg.baseline.corruptions = [
            "equatorial_checker_texture",
            "meridional_scanlines",
            "checkerboard_2px",
            "zonal_scanlines",
            "hemisphere_splice",
            "field_splice",
        ]
        dataset = synthetic_temperature_dataset().assign_coords(time=pd.to_datetime([
            "2020-01-01T00", "2020-01-05T00", "2020-01-10T00", "2020-01-15T00",
            "2020-01-16T00", "2020-01-19T00", "2020-01-20T00", "2020-01-22T00",
            "2020-01-26T00", "2020-01-27T00", "2020-01-30T00", "2020-01-31T00",
        ]))
        variables = ["2m_temperature"]
        metrics = ["mean_bias", "scwd"]
        with tempfile.TemporaryDirectory() as temporary_dir:
            rows, diagnostics, global_mean_diagnostics = evaluate_corruption_metrics(
                cfg, dataset, {}, variables, metrics, temporary_dir, return_scwd_diagnostics=True
            )
        self.assertEqual(len(rows), 3 * len(cfg.baseline.corruptions))
        self.assertEqual({row["corruption"] for row in rows}, set(cfg.baseline.corruptions))
        for name in cfg.baseline.corruptions:
            series = [row for row in rows if row["corruption"] == name]
            self.assertEqual([row["severity"] for row in series], [0.0, 0.0, 0.05])
            self.assertEqual(sum(bool(row["is_null"]) for row in series), 1)
            zero = next(row for row in series if not row["is_null"] and row["severity"] == 0.0)
            self.assertAlmostEqual(zero["scwd"], 0.0)
            self.assertTrue(all(np.isfinite(row["scwd"]) for row in series))
        self.assertEqual(len(diagnostics), len(cfg.baseline.corruptions))
        self.assertEqual({item["comparison_kind"] for item in diagnostics}, {"corruption"})
        self.assertEqual({item["severity"] for item in diagnostics}, {0.05})

    def test_area_weighted_scwd_downweights_polar_anchor_costs(self):
        cfg = baseline_test_config()
        n_anchors = cfg.baseline.scwd_anchor_lat_points * cfg.baseline.scwd_anchor_lon_points
        candidate = {"scwd_responses": np.zeros((2, 1, n_anchors), dtype=np.float32)}
        reference = {"scwd_responses": np.zeros((2, 1, n_anchors), dtype=np.float32)}
        candidate["scwd_responses"][:, :, :cfg.baseline.scwd_anchor_lon_points] = 1.0
        costs = scwd_anchor_transport_costs(candidate, reference, cfg)
        unweighted = float(np.mean(costs) ** (1.0 / cfg.baseline.scwd_order))
        area_weighted = scwd_area_weighted_from_responses(candidate, reference, cfg)
        self.assertAlmostEqual(float(np.sum(scwd_anchor_area_weights(cfg))), 1.0)
        self.assertLess(area_weighted, unweighted)

    def test_grf_preserves_input_dtype(self):
        values = torch.randn(2, 1, 9, 16, dtype=torch.float32)
        corrupted = apply_gaussian_field_noise(values, severity=0.05)
        self.assertEqual(corrupted.dtype, values.dtype)

    def test_grf_and_pixel_replace_preserve_channel_means(self):
        values = torch.randn(3, 2, 9, 16, dtype=torch.float32)
        for corruption in (apply_gaussian_field_noise, apply_random_pixel_replace):
            corrupted = corruption(values, severity=0.05)
            self.assertTrue(torch.allclose(
                corrupted.mean(dim=(-2, -1)), values.mean(dim=(-2, -1)), atol=2e-6
            ))

    def test_pixel_replace_leaves_unselected_pixels_untouched(self):
        values = torch.randn(2, 3, 11, 17, dtype=torch.float32)
        severity = 0.5
        torch.manual_seed(23)
        mask = torch.rand_like(values) < severity * 0.3
        torch.manual_seed(23)
        corrupted = apply_random_pixel_replace(values, severity=severity)
        self.assertTrue(torch.equal(corrupted[~mask], values[~mask]))
        self.assertTrue(torch.allclose(
            corrupted.mean(dim=(-2, -1)), values.mean(dim=(-2, -1)), atol=2e-6
        ))

    def test_pairwise_sample_positions_are_even_and_bounded(self):
        self.assertEqual(pairwise_sample_positions(4, 6), [0, 1, 2, 3])
        self.assertEqual(pairwise_sample_positions(12, 6), [0, 2, 4, 6, 8, 11])

    def test_baseline_corruption_levels_use_small_configured_range(self):
        levels = corruption_levels("grf", baseline_test_config())
        self.assertEqual(levels, [0.0, 0.025, 0.05])

    def test_per_corruption_severity_overrides_use_native_ranges(self):
        cfg = baseline_test_config()
        cfg.baseline.corruption_severity_max = 0.2
        cfg.baseline.corruption_steps = 3
        cfg.baseline.corruption_severity_max_overrides = {
            "gaussian_blur": 1.0,
            "pixel_replace": 0.05,
        }
        self.assertEqual(corruption_max_severity("grf", cfg), 0.2)
        self.assertEqual(corruption_max_severity("gaussian_blur", cfg), 1.0)
        self.assertEqual(corruption_max_severity("pixel_replace", cfg), 0.05)
        self.assertEqual(corruption_levels("grf", cfg), [0.0, 0.1, 0.2])
        self.assertEqual(corruption_levels("gaussian_blur", cfg), [0.0, 0.5, 1.0])
        self.assertEqual(corruption_levels("pixel_replace", cfg), [0.0, 0.025, 0.05])

    def test_representative_corruption_time_index_uses_middle_sweep_sample(self):
        indices = [0, 5, 10, 15, 20]
        self.assertEqual(representative_corruption_time_index(indices), 10)
        self.assertEqual(representative_corruption_time_index(indices, 15), 15)
        with self.assertRaises(ValueError):
            representative_corruption_time_index(indices, 9)

    def test_streaming_full_statistics_severity_zero_matches_clean_reference(self):
        cfg = baseline_test_config()
        ds = synthetic_temperature_dataset()
        variables = ["2m_temperature"]
        metrics = [
            "mean_bias",
            "std_ratio_error",
            "crps_like_field_energy",
            "zonal_energy_spectrum_l2",
            "sliced_wasserstein",
            "sliced_wasserstein_lon_corrected",
            "global_mean_wasserstein",
            "mmd_rbf",
            "scwd_area_weighted",
            "scwd",
        ]
        time_indices = list(range(ds.sizes["time"]))
        reference_stats = streaming_reference_stats(cfg, ds, variables, time_indices)

        with tempfile.TemporaryDirectory() as temporary_dir:
            clean = streaming_joint_features(
                cfg,
                ds,
                variables,
                reference_stats,
                time_indices,
                metrics,
                Path(temporary_dir) / "clean.dat",
                description="clean test fixture",
            )
            corrupted = streaming_joint_features(
                cfg,
                ds,
                variables,
                reference_stats,
                time_indices,
                metrics,
                Path(temporary_dir) / "corrupted.dat",
                corruption_type="hf_noise",
                severity=0.0,
                description="severity-zero test fixture",
            )
            result = selected_distribution_metric_values(corrupted, clean, metrics, cfg)

            self.assertEqual(result["n_samples"], 12)
            self.assertEqual(result["pairwise_n_samples"], 6)
            for metric in metrics:
                self.assertLess(abs(result[metric]), 2e-6, metric)

            costs = scwd_anchor_transport_costs(corrupted, clean, cfg)
            self.assertEqual(costs.shape, (32,))
            reconstructed = float(np.mean(costs) ** 0.5)
            self.assertAlmostEqual(reconstructed, result["scwd"], places=6)
            diagnostic = scwd_anchor_diagnostic(
                corrupted, clean, cfg, "Synthetic", 6, result["scwd"]
            )
            corruption_diagnostic = scwd_anchor_diagnostic(
                corrupted, clean, cfg, "fixture", 0, result["scwd"],
                comparison_kind="corruption", severity=0.05,
            )
            null_diagnostic = scwd_anchor_diagnostic(
                corrupted, clean, cfg, "ERA5 second-half null", 0, result["scwd"],
                comparison_kind="null",
            )
            self.assertEqual(diagnostic["anchor_local_wasserstein"].shape, (4, 8))
            self.assertEqual(diagnostic["anchor_mean_response_difference"].shape, (1, 4, 8))
            self.assertLess(np.max(np.abs(diagnostic["anchor_mean_response_difference"])), 2e-6)
            top_wasserstein = diagnostic["top_wasserstein_distributions"]
            self.assertEqual(len(top_wasserstein), 6)
            self.assertTrue(np.all(np.diff([item["wasserstein"] for item in top_wasserstein]) <= 0.0))
            with tempfile.TemporaryDirectory() as output_dir:
                output_root = Path(output_dir)
                diagnostics = [diagnostic, corruption_diagnostic, null_diagnostic]
                write_scwd_anchor_diagnostics(diagnostics, output_root)
                loaded = read_scwd_anchor_diagnostics(output_root)
                with patch("cartopy.mpl.geoaxes.GeoAxes.coastlines"), patch("cartopy.mpl.geoaxes.GeoAxes.add_feature"):
                    plot_scwd_anchor_diagnostics(diagnostics, output_root)
                    plot_scwd_mean_response_differences(diagnostics, output_root)
                plot_scwd_top_wasserstein_distributions(diagnostics, cfg, output_root)
                with xr.open_dataset(output_root / "data" / "scwd_anchor_contributions.nc") as saved:
                    self.assertEqual(saved.sizes["comparison"], 3)
                    self.assertEqual(saved.sizes["anchor_latitude"], 4)
                    self.assertEqual(saved.sizes["anchor_longitude"], 8)
                    self.assertEqual(str(saved.label.values[0]), "Synthetic")
                    self.assertEqual(str(saved.comparison_kind.values[1]), "corruption")
                    self.assertAlmostEqual(float(saved.scwd.values[0]), result["scwd"], places=6)
                    self.assertIn("top_local_wasserstein", saved)
                    self.assertEqual(int(saved.attrs["schema_version"]), 2)
                    self.assertIn("anchor_mean_response_difference", saved)
                    self.assertIn("candidate_response", saved)
                self.assertEqual(len(loaded), 3)
                self.assertEqual(loaded[1]["comparison_kind"], "corruption")
                self.assertEqual(loaded[2]["comparison_kind"], "null")
                self.assertEqual(len(loaded[0]["top_wasserstein_distributions"]), 6)
                np.testing.assert_allclose(
                    loaded[0]["top_wasserstein_distributions"][0]["candidate"],
                    diagnostic["top_wasserstein_distributions"][0]["candidate"],
                )
                self.assertTrue((output_root / "plots" / "scwd" / "Synthetic.png").is_file())
                self.assertTrue((output_root / "plots" / "scwd" / "top_wasserstein_distributions" / "Synthetic_006h.png").is_file())
                self.assertTrue((output_root / "plots" / "scwd" / "Synthetic_mean_response_difference.png").is_file())
                corruption_root = output_root / "plots" / "scwd" / "corruptions"
                self.assertTrue((corruption_root / "fixture_severity_0.05.png").is_file())
                self.assertTrue((corruption_root / "fixture_severity_0.05_mean_response_difference.png").is_file())
                self.assertTrue((corruption_root / "top_wasserstein_distributions" / "fixture_severity_0.05.png").is_file())
                null_root = output_root / "plots" / "scwd" / "null"
                self.assertTrue((null_root / "ERA5_second-half_null.png").is_file())
                self.assertTrue((null_root / "ERA5_second-half_null_mean_response_difference.png").is_file())
                self.assertTrue((null_root / "top_wasserstein_distributions" / "ERA5_second-half_null.png").is_file())

            close_feature_memmaps(corrupted)
            close_feature_memmaps(clean)

    def test_joint_gwd_detects_changed_cross_field_dependence(self):
        reference = np.repeat([[-1.0, -1.0], [1.0, 1.0]], 20, axis=0)
        candidate = np.repeat([[-1.0, 1.0], [1.0, -1.0]], 20, axis=0)
        grid = fit_vissio_ulam_grid([reference, candidate], n_bins=20)
        self.assertGreater(
            vissio_global_mean_wasserstein(candidate, reference, n_bins=20, grid=grid),
            0.0,
        )
        for field in range(2):
            np.testing.assert_array_equal(
                np.sort(candidate[:, field]), np.sort(reference[:, field])
            )

    def test_joint_scwd_uses_four_dimensional_euclidean_ground_cost(self):
        cfg = baseline_test_config()
        reference = {"scwd_responses": np.zeros((6, 4, 3), dtype=np.float32)}
        shift = np.asarray([1.0, 2.0, -0.5, 0.25], dtype=np.float32)
        candidate = {"scwd_responses": reference["scwd_responses"] + shift[None, :, None]}
        costs = scwd_anchor_transport_costs(candidate, reference, cfg)
        np.testing.assert_allclose(costs, np.sum(shift.astype(np.float64) ** 2))

    def test_joint_scwd_unequal_inputs_sample_both_full_ranges(self):
        cfg = baseline_test_config()
        candidate_values = np.zeros((10, 2, 1), dtype=np.float32)
        candidate_values[-1, :, 0] = 10.0
        candidate = {"scwd_responses": candidate_values}
        reference = {"scwd_responses": np.zeros((3, 2, 1), dtype=np.float32)}
        costs = scwd_anchor_transport_costs(candidate, reference, cfg)
        self.assertAlmostEqual(float(costs[0]), 200.0 / 3.0)

    def test_joint_gwd_finalization_uses_one_shared_grid(self):
        first_row, second_row = {}, {}
        diagnostics = [
            {"candidate": np.asarray([[0.0, 0.0], [1.0, 1.0]]),
             "reference": np.asarray([[0.0, 0.0], [0.5, 0.5]]),
             "row": first_row, "n_bins": 20},
            {"candidate": np.asarray([[-3.0, 2.0], [4.0, 5.0]]),
             "reference": np.asarray([[-1.0, 1.0], [2.0, 3.0]]),
             "row": second_row, "n_bins": 20},
        ]
        grid = finalize_global_mean_wasserstein(diagnostics, baseline_test_config())
        np.testing.assert_allclose(grid["lower"], [-3.0, 0.0])
        np.testing.assert_allclose(grid["upper"], [4.0, 5.0])
        self.assertIs(diagnostics[0]["grid"], diagnostics[1]["grid"])
        self.assertIn("global_mean_wasserstein", first_row)
        self.assertIn("global_mean_wasserstein", second_row)

    def test_joint_gwd_artifact_round_trip_persists_grid_and_ulam_measures(self):
        cfg = baseline_test_config()
        row = {}
        candidate = {"global_means": np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)}
        reference = {"global_means": np.asarray([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)}
        diagnostic = global_mean_wasserstein_diagnostic(
            candidate, reference, cfg, "fixture", 6, row=row,
        )
        finalize_global_mean_wasserstein([diagnostic], cfg)
        with tempfile.TemporaryDirectory() as output_dir:
            output_root = Path(output_dir)
            write_global_mean_wasserstein_diagnostics(
                [diagnostic], ["temperature", "pressure"], output_root,
            )
            loaded = read_global_mean_wasserstein_diagnostics(output_root)
            with xr.open_dataset(output_root / "data" / "global_mean_wasserstein_distributions.nc") as saved:
                self.assertEqual(int(saved.attrs["schema_version"]), 2)
                self.assertEqual(saved.attrs["estimator"], "joint_ulam_w2")
                self.assertIn("candidate_ulam_mass", saved)
                self.assertIn("reference_ulam_support", saved)
                np.testing.assert_allclose(saved.ulam_lower.values, [-1.0, 0.0])
                np.testing.assert_allclose(saved.ulam_upper.values, [2.0, 3.0])
            self.assertEqual(len(loaded), 1)
            np.testing.assert_allclose(loaded[0]["grid"]["lower"], [-1.0, 0.0])
            self.assertAlmostEqual(loaded[0]["distance"], row["global_mean_wasserstein"])

    def test_vissio_global_mean_wasserstein_is_bounded_and_detects_shift(self):
        reference = np.linspace(-2.0, 2.0, 101)[:, None]
        identical = vissio_global_mean_wasserstein(reference, reference, n_bins=20)
        shifted = vissio_global_mean_wasserstein(reference + 0.8, reference, n_bins=20)
        self.assertAlmostEqual(identical, 0.0)
        self.assertGreater(shifted, 0.0)
        self.assertLessEqual(shifted, 1.0)

    def test_reference_parameters_are_fitted_on_the_test_time_range(self):
        cfg = baseline_test_config()
        cfg.baseline.eval_samples = 0
        cfg.baseline.evaluation_time_range = ["2020-01-03", "2020-01-03 18:00"]
        cfg.baseline.reference_real_ranges = [["2020-01-01", "2020-01-02 18:00"]]
        ds = synthetic_temperature_dataset()
        variables = ["2m_temperature"]

        with tempfile.TemporaryDirectory() as temporary_dir:
            evaluation, train, stats = reference_features_from_config(
                cfg, ds, variables, ["mean_bias"], temporary_dir
            )
            expected = ds["2m_temperature"].isel(time=slice(8, 12)).values
            weights = np.cos(np.deg2rad(ds.latitude.values))[None, :, None]
            mass = float(weights.sum() * expected.shape[0] * expected.shape[2])
            expected_mean = float(np.sum(expected * weights) / mass)
            expected_std = float(np.sqrt(np.sum((expected - expected_mean) ** 2 * weights) / mass))
            self.assertAlmostEqual(stats["2m_temperature"]["mean"], expected_mean, places=6)
            self.assertAlmostEqual(stats["2m_temperature"]["std"], expected_std, places=6)
            self.assertEqual(evaluation["n_valid_samples"], 4)
            self.assertEqual(train["n_valid_samples"], 8)
            close_feature_memmaps(evaluation)
            close_feature_memmaps(train)

    def test_legacy_checkpoint_channel_count_recovers_configured_fields(self):
        cfg = OmegaConf.create(
            {
                "model_name": "squeezenet",
                "selected_variable": "mean_sea_level_pressure",
                "variables": [
                    "2m_temperature",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "mean_sea_level_pressure",
                ],
            }
        )
        state_dict = {"features.0.weight": torch.empty(64, 4, 3, 3)}
        recovered = reconcile_checkpoint_variables(cfg, ["mean_sea_level_pressure"], state_dict)
        self.assertEqual(recovered, list(cfg.variables))


if __name__ == "__main__":
    unittest.main()

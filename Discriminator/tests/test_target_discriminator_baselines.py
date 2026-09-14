import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import TensorDataset
from omegaconf import OmegaConf

from Discriminator.scripts.corruptions import gaussian_blur_effective_severity
from Discriminator.scripts.train_discriminator import WeatherDiscriminator, sample_power_law_severity
from Discriminator.scripts.train_target_discriminator_baselines import (
    BalancedTargetDataset,
    FrozenSFNOProbe,
    LinearProbe,
    ResidualMLPProbe,
    SFNOTargetDataset,
    SFNO_VARIABLES,
    apply_sfno_corruption,
    binary_classification_metrics,
    create_interpretability_gallery,
    create_sfno_representation_magnitude_gallery,
    integrated_gradients,
    resolve_attribution_baseline,
    sfno_representation_ratio_rows,
    build_sfno_probe,
    load_sfno_probe_checkpoint,
    save_sfno_probe_checkpoint,
    sample_target_corruption_severity,
    target_fake_severity_levels,
    training_corruption_severity,
    target_corruption_min,
    train_sfno_target,
)


def target_config():
    return OmegaConf.create({
        "lead_times": [6],
        "baseline": {
            "corruption_severity_max": 0.05,
            "hemisphere_splice_latitude": 0.0,
        },
        "target_discriminator": {
            "seed": 3,
            "epochs": 1,
            "batch_size": 4,
            "num_workers": 0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "max_train_samples": 0,
            "corruption_power": 2.0,
            "corruption_severity_max": 0.05,
            "sfno": {
                "embedding_channels": 2,
                "embedding_resolution": [2, 2],
                "pooling": "grid",
                "pool_grid": [2, 2],
                "mlp_hidden_multiplier": 2.0,
                "mlp_dropout": 0.1,
                "finetune_probe_architecture": None,
            },
        },
    })


def target_dataset():
    latitudes = np.linspace(-75.0, 75.0, 7)
    longitudes = np.arange(8) * 45.0
    values = np.stack([
        np.full((7, 8), float(index), dtype=np.float32) for index in range(6)
    ])
    return xr.Dataset(
        {"2m_temperature": (("time", "latitude", "longitude"), values)},
        coords={
            "time": pd.date_range("2020-01-01", periods=6, freq="6h"),
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )


def four_field_target_dataset():
    base = target_dataset()
    temperature = base["2m_temperature"].values
    return xr.Dataset(
        {
            "2m_temperature": (("time", "latitude", "longitude"), 280.0 + temperature),
            "10m_u_component_of_wind": (("time", "latitude", "longitude"), temperature + 1.0),
            "10m_v_component_of_wind": (("time", "latitude", "longitude"), temperature - 1.0),
            "mean_sea_level_pressure": (("time", "latitude", "longitude"), 100000.0 + temperature),
        },
        coords=base.coords,
    )


class MockSFNOEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.embedding_channels = 2
        self.embedding_resolution = (2, 2)
        self.pooling = "grid"
        self.pool_grid = (2, 2)
        self.feature_dim = 8
        self.register_buffer("norm_mean", torch.tensor([280.0, 0.0, 0.0, 100000.0]).view(1, 4, 1, 1))
        self.register_buffer("norm_std", torch.tensor([10.0, 5.0, 5.0, 1000.0]).view(1, 4, 1, 1))

    def extract_features(self, inputs, enable_input_grad=False):
        return self.extract_representation_maps(inputs)["pooled_embedding"]

    def extract_representation_maps(self, inputs):
        normalized = (inputs - self.norm_mean) / self.norm_std
        block6 = normalized * self.anchor
        block7 = torch.nn.functional.adaptive_avg_pool2d(block6, (2, 2))
        return {
            "block6_post_residual": block6,
            "block7_pre_projection": block7,
            "pooled_embedding": block7[:, :2].flatten(1),
        }


class FinetunableMockSFNOEncoder(MockSFNOEncoder):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.sfno_model = torch.nn.Module()
        self.model.sfno_model.last_encoder_block = torch.nn.Conv2d(2, 2, 1, bias=False)
        self.model.sfno_model.channel_down_scaling = torch.nn.Conv2d(2, 2, 1, bias=False)

    def extract_features(self, inputs, enable_input_grad=False):
        normalized = (inputs - self.norm_mean) / self.norm_std
        block7 = self.model.sfno_model.last_encoder_block(normalized[:, :2])
        embedding = self.model.sfno_model.channel_down_scaling(block7)
        return torch.nn.functional.adaptive_avg_pool2d(embedding, (2, 2)).flatten(1) * self.anchor


class TargetDiscriminatorBaselineTest(unittest.TestCase):
    def test_hemisphere_splice_training_always_uses_full_strength(self):
        for power in (0.0, 1.0, 2.0):
            np.random.seed(7)
            self.assertEqual(
                training_corruption_severity("hemisphere_splice", 0.2, power), 0.2
            )
        np.random.seed(7)
        self.assertLess(training_corruption_severity("grf", 0.2, 2.0), 0.2)

    def test_default_corruption_sampler_is_uniform_over_nonzero_test_levels(self):
        cfg = target_config()
        cfg.target_discriminator.corruption_steps = 4
        expected = target_fake_severity_levels(cfg, "grf", 0.2)
        self.assertTrue(np.array_equal(expected, [0.2 / 3.0, 2.0 * 0.2 / 3.0, 0.2]))
        np.random.seed(13)
        draws = np.asarray([
            sample_target_corruption_severity(cfg, "grf", 0.2, power=2.0)
            for _ in range(300)
        ])
        self.assertTrue(np.all(np.isin(draws, expected)))
        counts = np.asarray([(draws == level).sum() for level in expected])
        self.assertTrue(np.all(counts > 60))
        self.assertTrue(np.all(counts < 140))

    def test_discrete_test_severities_match_between_cnn_and_sfno(self):
        cfg = target_config()
        cfg.target_discriminator.corruption_steps = 5
        source = four_field_target_dataset()
        cnn = BalancedTargetDataset(
            source, source, ["2m_temperature"], {"2m_temperature": 0.0},
            {"2m_temperature": 1.0}, [6], corruption="pixel_replace", cfg=cfg,
            severity_max=0.05, deterministic_seed=17,
        )
        sfno = SFNOTargetDataset(
            source, source, MockSFNOEncoder(), [6], corruption="pixel_replace", cfg=cfg,
            severity_max=0.05, deterministic_seed=17,
        )
        for position in range(min(len(cnn.fake_i), 4)):
            self.assertEqual(cnn._severity(position), sfno._severity(position))
        levels = target_fake_severity_levels(cfg, "pixel_replace", 0.05)
        self.assertTrue(np.isin(cnn._severity(0), levels))

    def test_power_law_mode_remains_available(self):
        cfg = target_config()
        cfg.target_discriminator.corruption_severity_sampling = "power_law"
        value = sample_target_corruption_severity(cfg, "gaussian_blur", 1.0, 2.0, seed=3)
        self.assertGreaterEqual(value, 0.0)

    def test_gaussian_blur_zero_preserving_lerp_promotes_first_grid_point(self):
        self.assertEqual(gaussian_blur_effective_severity(0.0), 0.0)
        self.assertAlmostEqual(gaussian_blur_effective_severity(1.0 / 6.0), 1.0 / 3.0)
        self.assertAlmostEqual(gaussian_blur_effective_severity(0.5), 0.6)
        self.assertEqual(gaussian_blur_effective_severity(1.0), 1.0)

    def test_gaussian_blur_training_floor_avoids_identity_samples(self):
        np.random.seed(7)
        values = [sample_power_law_severity(1.0, 2.0, 0.2) for _ in range(20)]
        self.assertGreaterEqual(min(values), 0.2)
        cfg = target_config()
        cfg.target_discriminator.corruption_severity_max_overrides = {"gaussian_blur": 1.0}
        cfg.target_discriminator.corruption_severity_min_overrides = {"gaussian_blur": 0.2}
        self.assertEqual(target_corruption_min(cfg, "gaussian_blur"), 0.2)
        np.random.seed(7)
        self.assertGreaterEqual(
            training_corruption_severity("gaussian_blur", 1.0, 2.0, 0.2), 0.2
        )

    def test_attention_squeezenet_has_small_global_head(self):
        baseline = WeatherDiscriminator(1, "squeezenet", pretrained_backbone=False)
        attention = WeatherDiscriminator(
            1, "squeezenet_attention", pretrained_backbone=False
        ).eval()
        added_parameters = (
            sum(parameter.numel() for parameter in attention.parameters())
            - sum(parameter.numel() for parameter in baseline.parameters())
        )
        self.assertGreater(added_parameters, 60_000)
        self.assertLess(added_parameters, 70_000)
        self.assertEqual(attention(torch.zeros(2, 1, 121, 240)).shape, (2, 1))
        self.assertEqual(attention(torch.zeros(2, 1, 128, 256)).shape, (2, 1))
        with self.assertRaises(NotImplementedError):
            attention.pre_pool_logit_map(torch.zeros(2, 1, 121, 240))

    def build(self, corruption):
        dataset = target_dataset()
        return BalancedTargetDataset(
            dataset,
            dataset,
            ["2m_temperature"],
            {"2m_temperature": 0.0},
            {"2m_temperature": 1.0},
            [6],
            corruption=corruption,
            max_samples=0,
            power=0.0,
            severity_max=0.05,
            cfg=target_config(),
        )

    def test_structured_corruption_produces_full_strength_fake(self):
        dataset = self.build("equatorial_checker_texture")
        clean, _ = dataset[0]
        fake, label = dataset[dataset.n]
        self.assertEqual(fake.dtype, clean.dtype)
        self.assertEqual(fake.shape, clean.shape)
        self.assertEqual(float(label), 0.0)
        self.assertFalse(np.allclose(fake.numpy(), clean.numpy()))

    def test_hemisphere_fake_uses_a_distinct_deranged_donor(self):
        dataset = self.build("hemisphere_splice")
        fake, label = dataset[dataset.n]
        base_index = int(dataset.fake_i[0])
        donor_index = int(dataset.fake_i[int(dataset.donor_positions[0])])
        latitudes = dataset.latitudes
        self.assertNotEqual(base_index, donor_index)
        self.assertTrue(np.allclose(fake.numpy()[:, latitudes >= 0], float(base_index)))
        self.assertTrue(np.allclose(fake.numpy()[:, latitudes < 0], float(donor_index)))
        self.assertEqual(float(label), 0.0)

    def test_sfno_dataset_uses_all_four_raw_fields(self):
        source = four_field_target_dataset()
        dataset = SFNOTargetDataset(
            source, source, MockSFNOEncoder(), [6], max_samples=0,
            power=2.0, severity_max=0.05, cfg=target_config(),
        )
        sample, label = dataset[0]
        self.assertEqual(sample.shape, (4, 7, 8))
        self.assertEqual(float(label), 1.0)
        self.assertEqual(float(sample[0, 0, 0]), 280.0)
        self.assertEqual(float(sample[3, 0, 0]), 100000.0)
        self.assertEqual(dataset.sample_metadata(0)["true_class"], "real")
        self.assertEqual(dataset.sample_metadata(dataset.n)["true_class"], "fake")
        self.assertIn("time", dataset.sample_metadata(dataset.n))

    def test_sfno_corruption_operates_in_encoder_standardized_space(self):
        encoder = MockSFNOEncoder()
        raw = torch.stack([
            torch.full((7, 8), 280.0), torch.zeros((7, 8)),
            torch.zeros((7, 8)), torch.full((7, 8), 100000.0),
        ])
        torch.manual_seed(4)
        corrupted = apply_sfno_corruption(
            raw, encoder, "hf_noise", 0.05,
            np.linspace(-75.0, 75.0, 7), target_config(),
        )
        standardized = (corrupted - encoder.norm_mean.squeeze(0)) / encoder.norm_std.squeeze(0)
        self.assertEqual(corrupted.dtype, torch.float32)
        self.assertEqual(corrupted.shape, raw.shape)
        self.assertGreater(float(standardized.std()), 0.0)

    def test_sfno_probe_freezes_encoder_and_trains_head(self):
        encoder = MockSFNOEncoder()
        model = FrozenSFNOProbe(encoder, LinearProbe(encoder.feature_dim), "sfno_linear")
        inputs = torch.randn(3, 4, 7, 8)
        loss = model(inputs).sum()
        loss.backward()
        self.assertIsNone(encoder.anchor.grad)
        self.assertIsNotNone(model.head.output.weight.grad)
        self.assertFalse(encoder.training)

    def test_sfno_final_encoder_stage_finetuning_is_checkpointed(self):
        cfg = target_config()
        cfg.target_discriminator.sfno.unfreeze_final_encoder_stage = True
        cfg.target_discriminator.sfno.finetune_probe_architecture = "linear"
        encoder = FinetunableMockSFNOEncoder()
        model = build_sfno_probe(encoder, "sfno_linear", cfg)
        block = encoder.model.sfno_model.last_encoder_block
        projection = encoder.model.sfno_model.channel_down_scaling
        self.assertTrue(all(parameter.requires_grad for parameter in block.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in projection.parameters()))
        self.assertFalse(encoder.anchor.requires_grad)
        source = four_field_target_dataset()
        initial_block = block.weight.detach().clone()
        initial_projection = projection.weight.detach().clone()
        models = train_sfno_target(source, source, encoder, cfg, torch.device("cpu"), label="finetune-fixture")
        self.assertEqual(set(models), {"sfno_linear"})
        self.assertFalse(torch.equal(block.weight, initial_block))
        self.assertFalse(torch.equal(projection.weight, initial_projection))
        with torch.no_grad():
            block.weight.fill_(0.25)
            projection.weight.fill_(0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "finetuned.pth"
            save_sfno_probe_checkpoint(model, path)
            loaded, metadata = load_sfno_probe_checkpoint(
                path, cfg, torch.device("cpu"), encoder=FinetunableMockSFNOEncoder(),
            )
        self.assertTrue(metadata["unfreeze_final_encoder_stage"])
        self.assertTrue(torch.equal(
            loaded.encoder.model.sfno_model.last_encoder_block.weight,
            torch.full_like(loaded.encoder.model.sfno_model.last_encoder_block.weight, 0.25),
        ))
        self.assertTrue(torch.equal(
            loaded.encoder.model.sfno_model.channel_down_scaling.weight,
            torch.full_like(loaded.encoder.model.sfno_model.channel_down_scaling.weight, 0.5),
        ))

    def test_sfno_finetuning_requires_one_explicit_probe(self):
        cfg = target_config()
        cfg.target_discriminator.sfno.unfreeze_final_encoder_stage = True
        source = four_field_target_dataset()
        with self.assertRaisesRegex(ValueError, "finetune_probe_architecture"):
            train_sfno_target(
                source, source, FinetunableMockSFNOEncoder(), cfg, torch.device("cpu"), label="fixture",
            )

    def test_sfno_mlp_finetuning_trains_only_mlp(self):
        cfg = target_config()
        cfg.target_discriminator.sfno.unfreeze_final_encoder_stage = True
        cfg.target_discriminator.sfno.finetune_probe_architecture = "mlp"
        source = four_field_target_dataset()
        models = train_sfno_target(
            source, source, FinetunableMockSFNOEncoder(), cfg, torch.device("cpu"), label="fixture",
        )
        self.assertEqual(set(models), {"sfno_mlp"})

    def test_residual_mlp_and_checkpoint_round_trip(self):
        encoder = MockSFNOEncoder()
        head = ResidualMLPProbe(encoder.feature_dim, hidden_multiplier=2.0, dropout=0.1)
        model = FrozenSFNOProbe(encoder, head, "sfno_mlp").eval()
        expected = model(torch.randn(3, 4, 7, 8))
        self.assertEqual(expected.shape, (3, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            save_sfno_probe_checkpoint(model, path)
            loaded, metadata = load_sfno_probe_checkpoint(
                path, target_config(), torch.device("cpu"), encoder=MockSFNOEncoder(),
            )
            actual = loaded(torch.randn(2, 4, 7, 8))
        self.assertEqual(actual.shape, (2, 1))
        self.assertEqual(metadata["input_variables"], SFNO_VARIABLES)
        self.assertIn("overlaps", metadata["encoder_pretraining"])

    def test_joint_sfno_training_returns_both_probe_heads(self):
        source = four_field_target_dataset()
        encoder = MockSFNOEncoder()
        models = train_sfno_target(
            source, source, encoder, target_config(), torch.device("cpu"), label="fixture",
        )
        self.assertEqual(set(models), {"sfno_linear", "sfno_mlp"})
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))
        sample = torch.stack([SFNOTargetDataset(
            source, source, encoder, [6], cfg=target_config()
        )[0][0]] * 2)
        self.assertEqual(models["sfno_linear"](sample).shape, (2, 1))
        self.assertEqual(models["sfno_mlp"](sample).shape, (2, 1))


    def test_sfno_integrated_gradients_use_checkpoint_mean_and_render_four_fields(self):
        class Coordinates:
            latitudes = np.linspace(-75.0, 75.0, 7)
            longitudes = np.arange(8) * 45.0

        encoder = MockSFNOEncoder()
        head = LinearProbe(encoder.feature_dim)
        with torch.no_grad():
            head.output.weight.fill_(1.0)
            head.output.bias.zero_()
        model = FrozenSFNOProbe(encoder, head, "sfno_linear").eval()
        sample = encoder.norm_mean.squeeze(0).expand(-1, 7, 8).clone() + 1.0
        baseline = resolve_attribution_baseline(
            sample, {"baseline": {"kind": "global_training_mean"}}, model=model,
        )
        self.assertTrue(torch.equal(baseline, encoder.norm_mean.squeeze(0).expand_as(sample)))
        differentiable = sample.unsqueeze(0).detach().requires_grad_(True)
        model(differentiable).sum().backward()
        self.assertGreater(float(differentiable.grad.abs().sum()), 0.0)
        self.assertIsNone(encoder.anchor.grad)
        with torch.no_grad():
            logit = float(model(sample.unsqueeze(0)).item())
        cases = [
            {"input": sample, "logit": logit, "true_class": "real", "selection": "highest", "dataset_index": 0, "time": "2020-01-01"},
            {"input": sample + 0.5, "logit": logit, "true_class": "fake", "selection": "lowest", "dataset_index": 1, "time": "2020-01-02"},
        ]
        with tempfile.TemporaryDirectory() as directory, patch(
            "cartopy.mpl.geoaxes.GeoAxes.coastlines"
        ):
            output = Path(directory) / "sfno_gallery.png"
            rows = create_interpretability_gallery(
                model, cases, Coordinates(), SFNO_VARIABLES, {}, {},
                {"method": "integrated_gradients", "baseline": {"kind": "global_training_mean"},
                 "steps": 2, "internal_batch_size": 2},
                torch.device("cpu"), output, "sfno_linear", "forecast", "fixture",
            )
            self.assertTrue(output.is_file())
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["baseline_kind"] == "sfno_checkpoint_mean" for row in rows))
            self.assertTrue(all("attribution_sum_mean_sea_level_pressure" in row for row in rows))
            self.assertTrue(all(abs(row["completeness_residual"]) < 1e-4 for row in rows))

            representation_output = Path(directory) / "sfno_representation_magnitudes.png"
            create_sfno_representation_magnitude_gallery(
                model, cases, Coordinates(), representation_output,
                "sfno_linear", "forecast", "fixture",
            )
            self.assertTrue(representation_output.is_file())

    def test_sfno_representation_ratio_is_zero_for_identity_and_normalized_by_era5_pairs(self):
        cfg = target_config()
        cfg.target_discriminator.corruption_steps = 3
        cfg.target_discriminator.corruption_severity_max_overrides = {"pixel_replace": 1.0}
        encoder = MockSFNOEncoder()
        model = FrozenSFNOProbe(encoder, LinearProbe(encoder.feature_dim), "sfno_linear").eval()
        data = four_field_target_dataset()
        rows = sfno_representation_ratio_rows(
            model, data, data, None, "pixel_replace", cfg, torch.device("cpu"),
            maximum=0, batch_size=3,
        )
        self.assertEqual(len(rows), 9)
        self.assertEqual(
            {row["representation_layer"] for row in rows},
            {"block6_post_residual", "block7_pre_projection", "pooled_embedding"},
        )
        zero_rows = [row for row in rows if row["severity"] == 0.0]
        self.assertEqual(len(zero_rows), 3)
        self.assertTrue(all(row["reference_distance"] > 0.0 for row in zero_rows))
        self.assertTrue(all(abs(row["candidate_distance"]) < 1e-6 for row in zero_rows))
        self.assertTrue(all(abs(row["r_corr"]) < 1e-6 for row in zero_rows))
        strongest_rows = [row for row in rows if row["severity"] == 1.0]
        self.assertTrue(all(row["r_corr"] > 0.0 for row in strongest_rows))

    def test_integrated_gradients_is_complete_for_linear_logit(self):
        class LinearLogit(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
                self.bias = torch.nn.Parameter(torch.tensor(0.4))

            def forward(self, inputs):
                return (inputs.flatten(1) * self.weight).sum(dim=1, keepdim=True) + self.bias

        model = LinearLogit()
        sample = torch.tensor([[[1.5, -2.0]]])
        baseline = resolve_attribution_baseline(
            sample, {"baseline": {"kind": "global_training_mean"}}
        )
        attribution, diagnostics = integrated_gradients(
            model, sample, baseline, torch.device("cpu"), steps=8, internal_batch_size=3,
        )
        self.assertTrue(torch.allclose(attribution, sample * model.weight.reshape(1, 1, 2)))
        self.assertAlmostEqual(diagnostics["baseline_logit"], 0.4, places=6)
        self.assertAlmostEqual(diagnostics["completeness_residual"], 0.0, places=6)

    def test_case_selection_uses_raw_logit_within_each_class(self):
        class PixelLogit(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, inputs):
                return inputs.flatten(1)[:, :1] * self.scale

        values = torch.tensor([-2.0, 4.0, 1.0, 3.0, -5.0, 2.0, -1.0, 6.0]).reshape(-1, 1, 1, 1)
        labels = torch.tensor([1.0] * 4 + [0.0] * 4).reshape(-1, 1)
        metrics = binary_classification_metrics(
            PixelLogit(), TensorDataset(values, labels), torch.device("cpu"), 3,
            "selection fixture", random_samples_per_class=2, seed=9,
        )
        by_key = {(case["label"], case["selection"]): case for case in metrics["cases"]}
        self.assertEqual(by_key[(1, "highest")]["logit"], 4.0)
        self.assertEqual(by_key[(1, "lowest")]["logit"], -2.0)
        self.assertEqual(by_key[(0, "highest")]["logit"], 6.0)
        self.assertEqual(by_key[(0, "lowest")]["logit"], -5.0)
        for label in (0, 1):
            extrema = {by_key[(label, "highest")]["dataset_index"],
                       by_key[(label, "lowest")]["dataset_index"]}
            random_indices = {by_key[(label, "random_1")]["dataset_index"],
                              by_key[(label, "random_2")]["dataset_index"]}
            self.assertTrue(extrema.isdisjoint(random_indices))

    def test_held_out_corruption_is_reproducible(self):
        source = target_dataset()
        cfg = target_config()
        first = BalancedTargetDataset(
            source, source, ["2m_temperature"], {"2m_temperature": 0.0},
            {"2m_temperature": 1.0}, [6], corruption="hf_noise", cfg=cfg,
            deterministic_seed=17,
        )
        second = BalancedTargetDataset(
            source, source, ["2m_temperature"], {"2m_temperature": 0.0},
            {"2m_temperature": 1.0}, [6], corruption="hf_noise", cfg=cfg,
            deterministic_seed=17,
        )
        first_sample = first[first.n][0]
        second_sample = second[second.n][0]
        self.assertTrue(torch.equal(first_sample, second_sample))
        self.assertEqual(first.sample_metadata(first.n)["severity"],
                         second.sample_metadata(second.n)["severity"])


    def test_interpretability_gallery_smoke(self):
        class FlatLogit(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(24))

            def forward(self, inputs):
                return (inputs.flatten(1) * self.weight).sum(dim=1, keepdim=True)

        class Coordinates:
            latitudes = np.linspace(-60.0, 60.0, 4)
            longitudes = np.linspace(0.0, 300.0, 6)

        cases = []
        for index, (label, value) in enumerate((("real", 1.0), ("fake", -1.0))):
            cases.append({
                "input": torch.full((1, 4, 6), value), "logit": 24.0 * value,
                "true_class": label, "selection": "highest", "dataset_index": index,
                "time": f"2020-01-0{index + 1}",
            })
        with tempfile.TemporaryDirectory() as directory, patch(
            "cartopy.mpl.geoaxes.GeoAxes.coastlines"
        ):
            output = Path(directory) / "gallery.png"
            rows = create_interpretability_gallery(
                FlatLogit(), cases, Coordinates(), ["2m_temperature"],
                {"2m_temperature": 280.0}, {"2m_temperature": 2.0},
                {"method": "integrated_gradients", "baseline": {"kind": "global_training_mean"},
                 "steps": 2, "internal_batch_size": 2},
                torch.device("cpu"), output, "squeezenet", "model", "fixture",
            )
            self.assertTrue(output.is_file())
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(abs(row["completeness_residual"]) < 1e-5 for row in rows))


if __name__ == "__main__":
    unittest.main()

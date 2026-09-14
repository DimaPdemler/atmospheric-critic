"""Train target-only ERA5 discriminators and plot reverse-KL baseline curves."""
import csv
import os
from contextlib import nullcontext
from pathlib import Path
import sys
import hydra
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .plot_bundles import (
        CANDIDATE_COLOR, REFERENCE_COLOR, categorical_colors, lead_time_colors, all_plot_bundle_paths,
        configure_plot_bundle_saving_from_cfg, profiled_plot_path, save_figure_bundle,
    )
    from .fake_matching_apply import match_raw, match_standardized
    from .fake_matching_checkpoint import binding_path, validate_binding, write_binding
    from .temporal_resampling import active_schedule, write_split_membership
    from .train_discriminator import WeatherDiscriminator, apply_configured_corruption, safe_open_dataset, select_time_ranges, normalize_prediction_timedelta
    from .plot_standard_metric_baselines import (
        DATA_DEPENDENT_CORRUPTIONS,
        STRUCTURED_NEAR_NULL_CORRUPTIONS,
        apply_special_baseline_corruption,
        corruption_sample_seed,
        deranged_sample_positions,
        fieldwise_deranged_sample_positions,
    )
    from .monthly_split import (
        concatenate_forecasts,
        evenly_spaced_pairs,
        forecast_pairs,
        select_era5_split,
    )
except ImportError:
    from plot_bundles import (
        CANDIDATE_COLOR, REFERENCE_COLOR, categorical_colors, lead_time_colors, all_plot_bundle_paths,
        configure_plot_bundle_saving_from_cfg, profiled_plot_path, save_figure_bundle,
    )
    from fake_matching_apply import match_raw, match_standardized
    from fake_matching_checkpoint import binding_path, validate_binding, write_binding
    from temporal_resampling import active_schedule, write_split_membership
    from train_discriminator import WeatherDiscriminator, apply_configured_corruption, safe_open_dataset, select_time_ranges, normalize_prediction_timedelta
    from plot_standard_metric_baselines import (
        DATA_DEPENDENT_CORRUPTIONS,
        STRUCTURED_NEAR_NULL_CORRUPTIONS,
        apply_special_baseline_corruption,
        corruption_sample_seed,
        deranged_sample_positions,
        fieldwise_deranged_sample_positions,
    )
    from monthly_split import (
        concatenate_forecasts,
        evenly_spaced_pairs,
        forecast_pairs,
        select_era5_split,
    )


SPECIAL_CORRUPTIONS = STRUCTURED_NEAR_NULL_CORRUPTIONS | DATA_DEPENDENT_CORRUPTIONS
SFNO_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]


class LinearProbe(torch.nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.output = torch.nn.Linear(int(feature_dim), 1)

    def forward(self, features):
        return self.output(features)


class ResidualMLPProbe(torch.nn.Module):
    """Pre-normalized two-layer residual MLP followed by a scalar readout."""

    def __init__(self, feature_dim, hidden_multiplier=2.0, dropout=0.1):
        super().__init__()
        feature_dim = int(feature_dim)
        hidden_dim = int(round(feature_dim * float(hidden_multiplier)))
        self.normalization = torch.nn.BatchNorm1d(feature_dim)
        self.first = torch.nn.Linear(feature_dim, hidden_dim)
        self.second = torch.nn.Linear(hidden_dim, feature_dim)
        self.dropout = torch.nn.Dropout(float(dropout))
        self.output = torch.nn.Linear(feature_dim, 1)

    def forward(self, features):
        residual = self.normalization(features)
        residual = self.dropout(F.gelu(self.first(residual)))
        residual = self.dropout(self.second(residual))
        return self.output(F.gelu(features + residual))


def sfno_last_encoder_block(encoder):
    """Return the seventh SFNO block exposed by the supported adapter."""
    core = getattr(getattr(encoder, "model", None), "sfno_model", None)
    block = getattr(core, "last_encoder_block", None)
    if block is None:
        raise RuntimeError("The configured SFNO encoder does not expose last_encoder_block.")
    return block


def sfno_channel_projection(encoder):
    """Return the final 34-to-8 encoder channel projection."""
    core = getattr(getattr(encoder, "model", None), "sfno_model", None)
    projection = getattr(core, "channel_down_scaling", None)
    if projection is None:
        raise RuntimeError("The configured SFNO encoder does not expose channel_down_scaling.")
    return projection


def configure_sfno_trainability(encoder, unfreeze_final_encoder_stage=False):
    """Freeze the encoder except, optionally, its final block and projection."""
    encoder.requires_grad_(False)
    if unfreeze_final_encoder_stage:
        sfno_last_encoder_block(encoder).requires_grad_(True)
        sfno_channel_projection(encoder).requires_grad_(True)


class FrozenSFNOProbe(torch.nn.Module):
    """SFNO encoder with a scalar probe and optional final-stage finetuning."""

    expects_raw_fields = True

    def __init__(self, encoder, head, architecture, unfreeze_final_encoder_stage=False):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.architecture = str(architecture)
        self.input_variables = tuple(SFNO_VARIABLES)
        self.unfreeze_final_encoder_stage = bool(unfreeze_final_encoder_stage)
        configure_sfno_trainability(self.encoder, self.unfreeze_final_encoder_stage)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        if self.unfreeze_final_encoder_stage:
            sfno_last_encoder_block(self.encoder).train(mode)
            sfno_channel_projection(self.encoder).train(mode)
        return self

    def forward(self, inputs):
        # IG needs input gradients; finetuning needs a graph through block seven.
        features = self.encoder.extract_features(
            inputs,
            enable_input_grad=bool(torch.is_grad_enabled() and (
                inputs.requires_grad or self.unfreeze_final_encoder_stage
            )),
        )
        return self.head(features)


def sfno_settings(cfg):
    if cfg.get("target_discriminator") is not None:
        return cfg.target_discriminator.get("sfno", {}) or {}
    baseline = cfg.get("baseline", {}) or {}
    discriminator = baseline.get("discriminator", {}) or {}
    return discriminator.get("sfno", {}) or {}


def sfno_finetune_probe_architecture(cfg):
    """Return the one probe architecture allowed to update unfrozen SFNO weights."""
    selected = sfno_settings(cfg).get("finetune_probe_architecture")
    aliases = {"linear": "sfno_linear", "mlp": "sfno_mlp"}
    if selected is None or str(selected).strip() == "":
        raise ValueError(
            "SFNO final-stage finetuning requires "
            "target_discriminator.sfno.finetune_probe_architecture=linear or mlp."
        )
    try:
        return aliases[str(selected).strip().lower()]
    except KeyError as error:
        raise ValueError(
            "Unknown SFNO finetune probe architecture "
            f"{selected!r}; expected 'linear' or 'mlp'."
        ) from error


def sfno_context_settings(cfg):
    settings = sfno_settings(cfg)
    targets = tuple(settings.get("target_variables", ["2m_temperature"]))
    return bool(settings.get("use_era5_context_for_non_target_fields", False)), targets


def load_sfno_encoder(cfg, device):
    """Load the shared FeatureMetric SFNO adapter without requiring installation."""
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from FeatureMetric.utils.sfno_embedding import SFNOEmbedding

    settings = sfno_settings(cfg)
    repo_root = settings.get("repo_root") or os.environ.get("SFNO_REPO")
    encoder = SFNOEmbedding(
        embedding_channels=int(settings.get("embedding_channels", 8)),
        embedding_resolution=tuple(settings.get("embedding_resolution", [31, 60])),
        repo_root=repo_root,
        pooling=str(settings.get("pooling", "grid")),
        pool_grid=tuple(settings.get("pool_grid", [7, 8])),
        weights_subdir=str(settings.get("weights_subdir", "weights_4fields")),
    ).to(device)
    encoder.requires_grad_(False)
    return encoder.eval()


def build_sfno_head(encoder, architecture, cfg):
    settings = sfno_settings(cfg)
    if architecture == "sfno_linear":
        return LinearProbe(encoder.feature_dim)
    elif architecture == "sfno_mlp":
        return ResidualMLPProbe(
            encoder.feature_dim,
            hidden_multiplier=float(settings.get("mlp_hidden_multiplier", 2.0)),
            dropout=float(settings.get("mlp_dropout", 0.1)),
        )
    raise ValueError(f"Unknown SFNO probe architecture: {architecture}")


def build_sfno_probe(encoder, architecture, cfg, unfreeze_final_encoder_stage=None):
    if unfreeze_final_encoder_stage is None:
        unfreeze_final_encoder_stage = bool(sfno_settings(cfg).get("unfreeze_final_encoder_stage", False))
    return FrozenSFNOProbe(
        encoder, build_sfno_head(encoder, architecture, cfg).to(next(encoder.parameters()).device), architecture,
        unfreeze_final_encoder_stage=unfreeze_final_encoder_stage,
    )


def target_corruption_max(cfg, corruption):
    section = target_settings(cfg)
    maximum = section.get('corruption_severity_max', 0.2)
    overrides = section.get('corruption_severity_max_overrides', {}) or {}
    return float(overrides.get(str(corruption), maximum))


def target_corruption_min(cfg, corruption):
    """Return the training-only lower severity bound for one corruption."""
    section = target_settings(cfg)
    minimum = section.get('corruption_severity_min', 0.0)
    overrides = section.get('corruption_severity_min_overrides', {}) or {}
    return float(overrides.get(str(corruption), minimum))


def target_corruption_sampling_mode(cfg):
    mode = str(get(cfg, "corruption_severity_sampling", "discrete_uniform"))
    if mode not in {"discrete_uniform", "power_law"}:
        raise ValueError(
            "target_discriminator.corruption_severity_sampling must be "
            "'discrete_uniform' or 'power_law'."
        )
    return mode


def target_fake_severity_levels(cfg, corruption, severity_max=None):
    """Fake-class strengths aligned with the target corruption plot grid."""
    maximum = target_corruption_max(cfg, corruption) if severity_max is None else float(severity_max)
    if corruption in DATA_DEPENDENT_CORRUPTIONS:
        return np.asarray([maximum], dtype=np.float64)
    steps = int(get(cfg, "corruption_steps", 7))
    if steps < 2:
        raise ValueError("target_discriminator.corruption_steps must be at least 2.")
    # Zero is the identity transform and belongs only to the real class.
    return np.linspace(0.0, maximum, steps, dtype=np.float64)[1:]


def sample_target_corruption_severity(cfg, corruption, severity_max, power, *, seed=None):
    """Draw one train/test fake severity under the configured target protocol."""
    if corruption in DATA_DEPENDENT_CORRUPTIONS:
        return float(severity_max)
    mode = target_corruption_sampling_mode(cfg)
    if mode == "power_law":
        minimum = target_corruption_min(cfg, corruption)
        if seed is None:
            return training_corruption_severity(corruption, severity_max, power, minimum)
        return deterministic_corruption_severity(corruption, severity_max, power, minimum, seed)
    levels = target_fake_severity_levels(cfg, corruption, severity_max)
    if seed is None:
        return float(levels[np.random.randint(len(levels))])
    return float(levels[np.random.default_rng(int(seed)).integers(len(levels))])


def training_corruption_severity(corruption, severity_max, power, severity_min=0.0):
    """Sample fake strength, reserving an uncorrupted splice for the real class."""
    if corruption in DATA_DEPENDENT_CORRUPTIONS:
        return float(severity_max)
    severity_max, severity_min = float(severity_max), float(severity_min)
    if severity_min < 0.0 or severity_min > severity_max:
        raise ValueError(
            f"Expected 0 <= severity_min <= severity_max, got "
            f"{severity_min} and {severity_max}."
        )
    return severity_min + (severity_max - severity_min) * (
        np.random.random() ** float(power)
    )


def deterministic_corruption_severity(corruption, severity_max, power, severity_min, seed):
    """Deterministic counterpart used by held-out evaluation and attribution."""
    if corruption in DATA_DEPENDENT_CORRUPTIONS:
        return float(severity_max)
    severity_max, severity_min = float(severity_max), float(severity_min)
    if severity_min < 0.0 or severity_min > severity_max:
        raise ValueError(f"Expected 0 <= severity_min <= severity_max, got {severity_min} and {severity_max}.")
    draw = np.random.default_rng(int(seed)).random()
    return severity_min + (severity_max - severity_min) * (draw ** float(power))


def epoch_deranged_donor_positions(size, seed, epoch):
    """Return a reproducible full donor derangement for one training epoch."""
    size = int(size)
    if size < 2:
        raise ValueError("Data-dependent splice requires at least two samples.")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    source_order = rng.permutation(size)
    donor_positions = np.empty(size, dtype=int)
    donor_positions[source_order] = np.roll(source_order, -1)
    return donor_positions


def data_dependent_donor_positions(corruption, size, n_fields, seed, epoch=0):
    """Return epoch-specific donor permutations for whole-field corruptions."""
    if corruption == "hemisphere_splice":
        return epoch_deranged_donor_positions(size, seed, epoch)
    if corruption == "field_splice":
        epoch_seed = int(np.random.SeedSequence([int(seed), int(epoch)]).generate_state(1)[0])
        return fieldwise_deranged_sample_positions(size, n_fields, epoch_seed)
    return None


def standardized_donor_fields(ds, variables, means, stds, sample_indices, donor_positions, position, lead=None):
    """Load one complete standardized donor, allowing a distinct time per field."""
    if donor_positions.ndim == 1:
        return fields(ds, variables, means, stds, int(sample_indices[int(donor_positions[position])]), lead).numpy()
    return np.stack([
        fields(ds, [variable], means, stds, int(sample_indices[int(donor_positions[channel, position])]), lead)[0].numpy()
        for channel, variable in enumerate(variables)
    ])


def raw_donor_fields(ds, variables, sample_indices, donor_positions, position, lead=None):
    """Load one complete raw donor, allowing a distinct time per field."""
    if donor_positions.ndim == 1:
        return raw_fields(ds, variables, int(sample_indices[int(donor_positions[position])]), lead)
    return torch.stack([
        raw_fields(ds, [variable], int(sample_indices[int(donor_positions[channel, position])]), lead)[0]
        for channel, variable in enumerate(variables)
    ])


def target_settings(cfg):
    section = cfg.get("target_discriminator")
    if section is not None:
        return section
    baseline = cfg.get("baseline", {}) or {}
    return baseline.get("discriminator", {}) or {}


def get(cfg, key, default=None):
    value = target_settings(cfg).get(key)
    return default if value is None else value


def target_device(cfg):
    """Resolve `device: auto` to CUDA when available, otherwise CPU."""
    requested = get(cfg, "device", "auto")
    if requested is None or str(requested).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(requested))


def mask_equatorial_band(sample, latitudes, half_width_degrees):
    """Replace an equatorial latitude band by the standardized mean (zero)."""
    if float(half_width_degrees) <= 0.0:
        return sample
    masked = sample.clone()
    equator = torch.from_numpy(np.abs(np.asarray(latitudes)) <= float(half_width_degrees))
    masked[:, equator.to(device=masked.device), :] = 0.0
    return masked


def fields(ds, variables, means, stds, time_index, lead_index=None):
    sample = ds.isel(time=time_index)
    if lead_index is not None: sample = sample.isel(prediction_timedelta=lead_index)
    values = [
        np.nan_to_num(
            (
                sample[v].transpose("latitude", "longitude").values.astype(np.float32)
                - means[v]
            )
            / stds[v]
        ).astype(np.float32)
        for v in variables
    ]
    return torch.from_numpy(np.stack(values))


def raw_fields(ds, variables, time_index, lead_index=None):
    sample = ds.isel(time=time_index)
    if lead_index is not None:
        sample = sample.isel(prediction_timedelta=lead_index)
    values = [
        np.nan_to_num(
            sample[v].transpose("latitude", "longitude").values.astype(np.float32)
        ).astype(np.float32)
        for v in variables
    ]
    return torch.from_numpy(np.stack(values))


def sfno_normalization(encoder):
    cached = getattr(encoder, "_target_norm_cpu", None)
    if cached is None:
        cached = (
            encoder.norm_mean.detach().cpu().reshape(-1, 1, 1),
            encoder.norm_std.detach().cpu().reshape(-1, 1, 1),
        )
        encoder._target_norm_cpu = cached
    return cached


def apply_sfno_corruption(
    raw, encoder, corruption, severity, latitudes, cfg, donor=None,
    maximum_severity=None, random_seed=None, target_variables=None,
):
    """Corrupt a raw field in the standardized space expected by SFNO."""
    mean, std = sfno_normalization(encoder)
    standardized = (raw.to(torch.float32) - mean) / std
    channel_indices = ([SFNO_VARIABLES.index(variable) for variable in target_variables]
                       if target_variables is not None else list(range(len(SFNO_VARIABLES))))
    selected = standardized[channel_indices]
    if corruption in SPECIAL_CORRUPTIONS:
        standardized_donor = None
        if donor is not None:
            standardized_donor = ((donor.to(torch.float32) - mean) / std).numpy()[channel_indices]
        corrupted = torch.from_numpy(
            apply_special_baseline_corruption(
                selected.numpy(), corruption, severity, latitudes, cfg,
                standardized_donor,
                maximum_severity=(
                    target_corruption_max(cfg, corruption)
                    if maximum_severity is None else maximum_severity
                ),
                random_seed=random_seed,
            )
        )
    else:
        corrupted = apply_configured_corruption(selected, corruption, severity)
    result = standardized.clone()
    result[channel_indices] = corrupted.to(torch.float32)
    return (result * std + mean).to(torch.float32)


def indices(ds, maximum):
    n = ds.sizes.get("time", 0)
    return np.arange(n) if maximum <= 0 or n <= maximum else np.linspace(0, n-1, maximum, dtype=int)


def binary_classification_metrics(
    model, dataset, device, batch_size, description, random_samples_per_class=0, seed=0,
):
    """Compute held-out metrics and retain bounded logit-selected case tensors."""
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    model.eval()
    model_dtype = next(model.parameters()).dtype
    total_loss = total_correct = total_count = 0
    extrema = {0: {"highest": None, "lowest": None}, 1: {"highest": None, "lowest": None}}
    random_candidates = {0: [], 1: []}
    keep_random = max(0, int(random_samples_per_class)) + 4
    rng = np.random.default_rng(int(seed))

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=description, leave=False):
            start = total_count
            device_inputs = inputs.to(device=device, dtype=model_dtype)
            device_labels = labels.to(device=device, dtype=model_dtype)
            logits = model(device_inputs)
            losses = F.binary_cross_entropy_with_logits(logits, device_labels, reduction="none")
            total_loss += float(losses.sum().item())
            total_correct += int(((logits >= 0.0) == (device_labels >= 0.5)).sum().item())
            batch_count = int(labels.numel())
            if random_samples_per_class:
                for offset in range(batch_count):
                    dataset_index = start + offset
                    label = int(labels[offset].item() >= 0.5)
                    case = {
                        "dataset_index": dataset_index,
                        "label": label,
                        "logit": float(logits[offset].item()),
                        "loss": float(losses[offset].item()),
                        "input": inputs[offset].detach().cpu().clone(),
                    }
                    if hasattr(dataset, "sample_metadata"):
                        case.update(dataset.sample_metadata(dataset_index))
                    if extrema[label]["highest"] is None or case["logit"] > extrema[label]["highest"]["logit"]:
                        extrema[label]["highest"] = case
                    if extrema[label]["lowest"] is None or case["logit"] < extrema[label]["lowest"]["logit"]:
                        extrema[label]["lowest"] = case
                    random_candidates[label].append((float(rng.random()), case))
                    random_candidates[label].sort(key=lambda item: item[0])
                    del random_candidates[label][keep_random:]
            total_count += batch_count
    if not total_count:
        raise ValueError(f"No samples available for {description}.")

    selected = []
    for label in (1, 0):
        highest, lowest = extrema[label]["highest"], extrema[label]["lowest"]
        for selection, case in (("highest", highest), ("lowest", lowest)):
            if case is not None:
                selected.append({**case, "selection": selection})
        excluded = {case["dataset_index"] for case in (highest, lowest) if case is not None}
        random_cases = [case for _, case in random_candidates[label]
                        if case["dataset_index"] not in excluded][:int(random_samples_per_class)]
        selected.extend({**case, "selection": f"random_{number + 1}"}
                        for number, case in enumerate(random_cases))
    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "n_samples": total_count,
        "cases": selected,
    }


def resolve_attribution_baseline(inputs, settings, metadata=None, model=None):
    """Resolve an IG baseline for normalized CNN or raw SFNO inputs."""
    baseline = settings.get("baseline", {}) or {}
    kind = str(baseline.get("kind", "global_training_mean"))
    if kind != "global_training_mean":
        raise ValueError(f"Unsupported interpretability baseline kind: {kind}")
    if bool(getattr(model, "expects_raw_fields", False)):
        # SFNO receives physical units and normalizes internally. Its checkpoint
        # mean is therefore the raw-field baseline corresponding to zero input.
        mean = model.encoder.norm_mean.detach().cpu().reshape(-1, 1, 1)
        if inputs.shape[0] != mean.shape[0]:
            raise ValueError("SFNO attribution baseline has incompatible channel count.")
        return mean.expand_as(inputs).clone()
    # SqueezeNet inputs are standardized with the global training mean/std.
    return torch.zeros_like(inputs)


def integrated_gradients(model, inputs, baseline, device, steps=32, internal_batch_size=8):
    """Integrated gradients of the raw real-vs-fake logit using trapezoidal integration."""
    if int(steps) < 1:
        raise ValueError("Integrated gradients requires steps >= 1.")
    model.eval()
    dtype = next(model.parameters()).dtype
    inputs = inputs.detach().to(device=device, dtype=dtype)
    baseline = baseline.detach().to(device=device, dtype=dtype)
    delta = inputs - baseline
    alphas = torch.linspace(0.0, 1.0, int(steps) + 1, device=device, dtype=dtype)
    gradient_sum = torch.zeros_like(inputs)
    for start in range(0, len(alphas), int(internal_batch_size)):
        alpha = alphas[start:start + int(internal_batch_size)]
        scaled = baseline.unsqueeze(0) + alpha.reshape(-1, 1, 1, 1) * delta.unsqueeze(0)
        scaled.requires_grad_(True)
        logits = model(scaled).reshape(-1)
        gradients = torch.autograd.grad(logits.sum(), scaled, retain_graph=False)[0]
        weights = torch.ones_like(alpha)
        weights[alpha == 0.0] = 0.5
        weights[alpha == 1.0] = 0.5
        gradient_sum += (gradients * weights.reshape(-1, 1, 1, 1)).sum(dim=0)
    attribution = delta * gradient_sum / float(steps)
    with torch.no_grad():
        input_logit = float(model(inputs.unsqueeze(0)).item())
        baseline_logit = float(model(baseline.unsqueeze(0)).item())
    attribution_sum = float(attribution.sum().item())
    residual = (input_logit - baseline_logit) - attribution_sum
    return attribution.detach().cpu(), {
        "input_logit": input_logit,
        "baseline_logit": baseline_logit,
        "attribution_sum": attribution_sum,
        "completeness_residual": float(residual),
    }



def _logit_histogram_edges(values, bins=40):
    finite = [np.asarray(value)[np.isfinite(value)] for value in values]
    combined = np.concatenate([value for value in finite if value.size])
    return np.histogram_bin_edges(combined, bins=max(2, int(bins)))


def _plot_logit_histogram(reference, candidate, title, candidate_label, output_path, color=CANDIDATE_COLOR):
    edges = _logit_histogram_edges([reference, candidate])
    figure, axis = plt.subplots(figsize=(6.4, 5.0), layout="constrained")
    axis.hist(reference, bins=edges, density=True, histtype="stepfilled", color=REFERENCE_COLOR, alpha=0.35, label="ERA5 test")
    axis.hist(candidate, bins=edges, density=True, histtype="step", linewidth=2.0, color=color, label=candidate_label)
    axis.axvline(0.0, color="black", linewidth=0.7, alpha=0.45)
    axis.set(title=title, xlabel="Real-vs-fake logit", ylabel="Density")
    axis.grid(alpha=0.25)
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.20),
        ncol=2, frameon=False, fontsize=7,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_bundle(
        figure, output_path, plot_type="target_logit_histogram",
        payload={"reference_logits": reference, "candidate_logits": candidate, "bin_edges": edges}, dpi=200,
        paper_width_kind="half",
    ); plt.close(figure)


def _plot_logit_histogram_overlay(groups, title, output_path, colors=None):
    edges = _logit_histogram_edges([value for group in groups for value in (group["reference"], group["candidate"])])
    legend_columns = 2
    legend_rows = int(np.ceil((len(groups) + 1) / legend_columns))
    figure = plt.figure(
        figsize=(6.4, 5.2 + 0.65 * legend_rows), layout="constrained",
    )
    grid = figure.add_gridspec(
        2, 1, height_ratios=[4.2, max(1.15, 0.55 * legend_rows)],
    )
    axis = figure.add_subplot(grid[0])
    legend_axis = figure.add_subplot(grid[1])
    legend_axis.set_axis_off()
    pooled_reference = np.concatenate([group["reference"] for group in groups])
    axis.hist(pooled_reference, bins=edges, density=True, histtype="stepfilled", color=REFERENCE_COLOR, alpha=0.30, label="ERA5 test (pooled)")
    colors = colors or categorical_colors(len(groups), offset=1)
    for color, group in zip(colors, groups):
        axis.hist(group["candidate"], bins=edges, density=True, histtype="step", linewidth=1.7, color=color, label=group["label"])
    axis.axvline(0.0, color="black", linewidth=0.7, alpha=0.45)
    axis.set(xlabel="Real-vs-fake logit", ylabel="Density")
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    legend_axis.legend(
        handles, labels, loc="center", ncol=legend_columns,
        frameon=False, fontsize=7,
    )
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"bin_edges": edges, "pooled_reference_logits": pooled_reference}
    for index, group in enumerate(groups):
        payload[f"reference_logits_{index}"] = group["reference"]
        payload[f"candidate_logits_{index}"] = group["candidate"]
        payload[f"label_{index}"] = np.asarray(group["label"])
    save_figure_bundle(
        figure, output_path, plot_type="target_logit_histogram_overlay", payload=payload,
        dpi=200, paper_width_kind="half",
    ); plt.close(figure)


def _histogram_display_name(value):
    names = {
        "squeezenet": "SqueezeNet",
        "squeezenet_equator_mask": "Equator-masked SqueezeNet",
        "grf": "GRF noise",
        "hf_noise": "High-frequency noise",
        "checkerboard_2px": "2-pixel checkerboard",
        "equatorial_checker_texture": "Equatorial checkerboard",
        "hemisphere_splice": "Hemisphere splice",
        "pixel_replace": "Pixel replacement",
        "zonal_scanlines": "Zonal scanlines",
        "meridional_scanlines": "Meridional scanlines",
        "GraphCast": "GraphCast",
        "Pangu-Weather": "Pangu-Weather",
        "FuXi": "FuXi",
        "IFS HRES": "IFS HRES",
        "ERA5 Forecast": "ERA5 Forecast",
        "UCast member 0": "UCast member 0",
        "SWIFT": "SWIFT",
    }
    return names.get(str(value), str(value).replace("_", " ").title())


def plot_target_test_logit_histograms(model, architecture, kind, label, test_real, test_fake, records,
                                      corruption, variables, means, stds, cfg, device, maximum, batch_size,
                                      output_root=None):
    """Plot held-out real/fake logit densities for every target test point."""
    root = Path(str(output_root or get(cfg, "output_dir"))) / "plots" / "target_logit_distributions" / architecture / kind / safe_target_name(label)
    groups = []
    if corruption:
        selected = indices(test_fake, maximum)
        reference = logits_for(model, test_real, variables, means, stds, device, maximum, batch_size, cfg=cfg, selected_indices=selected)
        maximum_severity = target_corruption_max(cfg, corruption)
        levels = np.linspace(0.0, maximum_severity, int(get(cfg, "corruption_steps", 7)))
        for severity in levels:
            candidate = logits_for(model, test_fake, variables, means, stds, device, maximum, batch_size,
                                   corruption=corruption, severity=float(severity), cfg=cfg,
                                   maximum_severity=maximum_severity, selected_indices=selected)
            name = f"severity={severity:.3g}"
            path = root / f"severity_{severity:.3g}.png"
            _plot_logit_histogram(reference, candidate, f"{_histogram_display_name(label)}: {name} ({_histogram_display_name(architecture)})", label, path)
            groups.append({"label": name, "reference": reference, "candidate": candidate,
                           "path": profiled_plot_path(path)})
        overlay_path = root / "all_corruption_strengths.png"
        _plot_logit_histogram_overlay(groups, f"{_histogram_display_name(label)}: all corruption strengths ({_histogram_display_name(architecture)})", overlay_path)
        overlay_path = profiled_plot_path(overlay_path)
    else:
        for lead_index, lead in enumerate(np.asarray(test_fake.prediction_timedelta.values).astype("timedelta64[h]").astype(int)):
            selected_records = [record for record in records if record.lead_index == lead_index]
            selected_records = evenly_spaced_pairs(selected_records, maximum)
            if not selected_records:
                continue
            forecast_indices = [record.forecast_index for record in selected_records]
            reference_indices = [record.era5_index for record in selected_records]
            reference = logits_for(model, test_real, variables, means, stds, device, maximum, batch_size,
                                   cfg=cfg, selected_indices=reference_indices)
            candidate = logits_for(
                model, test_fake, variables, means, stds, device, maximum, batch_size,
                lead=lead_index, cfg=cfg, selected_indices=forecast_indices,
                context_ds=(test_real if bool(getattr(model, "sfno_use_era5_context", False)) else None),
                context_indices=(reference_indices if bool(getattr(model, "sfno_use_era5_context", False)) else None),
            )
            name = f"+{int(lead)}h"
            path = root / f"lead_{int(lead):03d}h.png"
            _plot_logit_histogram(reference, candidate, f"{_histogram_display_name(label)}: {name} ({_histogram_display_name(architecture)})", label, path)
            groups.append({"label": name, "reference": reference, "candidate": candidate,
                           "path": profiled_plot_path(path)})
        overlay_path = root / "all_lead_times.png"
        if groups:
            _plot_logit_histogram_overlay(
                groups,
                f"{_histogram_display_name(label)}: all lead times ({_histogram_display_name(architecture)})",
                overlay_path,
                colors=lead_time_colors(len(groups)),
            )
            overlay_path = profiled_plot_path(overlay_path)
    return [group["path"] for group in groups] + ([overlay_path] if groups else [])

def _sfno_layer_pair_distances(encoder, first_samples, second_samples, device, batch_size):
    """Mean L2 distances for named SFNO maps without retaining full test tensors."""
    if len(first_samples) != len(second_samples):
        raise ValueError("SFNO sample counts must match for representation distances.")
    totals = {}
    count = 0
    batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(first_samples), batch_size):
            stop = min(start + batch_size, len(first_samples))
            first = torch.stack(first_samples[start:stop]).to(device=device, dtype=torch.float32)
            second = torch.stack(second_samples[start:stop]).to(device=device, dtype=torch.float32)
            first_maps = encoder.extract_representation_maps(first)
            second_maps = encoder.extract_representation_maps(second)
            for layer, first_map in first_maps.items():
                norms = torch.linalg.vector_norm((first_map - second_maps[layer]).flatten(1), dim=1)
                totals[layer] = totals.get(layer, 0.0) + float(norms.sum().item())
            count += stop - start
    return {layer: total / count for layer, total in totals.items()}


def _sfno_reference_distances(encoder, reference_samples, donor_positions, device, batch_size):
    donors = [reference_samples[int(position)] for position in donor_positions]
    return _sfno_layer_pair_distances(encoder, reference_samples, donors, device, batch_size)


def _sfno_forecast_input(model, real, fake, record):
    """Construct precisely the four-field forecast input seen by the SFNO probe."""
    if not bool(getattr(model, "sfno_use_era5_context", False)):
        return raw_fields(fake, SFNO_VARIABLES, record.forecast_index, record.lead_index)
    target_variables = list(getattr(model, "sfno_target_variables", ["2m_temperature"]))
    context_variables = [variable for variable in SFNO_VARIABLES if variable not in target_variables]
    forecast = raw_fields(fake, target_variables, record.forecast_index, record.lead_index)
    context = raw_fields(real, context_variables, record.era5_index)
    return torch.stack([
        forecast[target_variables.index(variable)] if variable in target_variables
        else context[context_variables.index(variable)]
        for variable in SFNO_VARIABLES
    ])


def sfno_representation_ratio_rows(model, test_real, test_fake, records, corruption, cfg, device,
                                   maximum, batch_size):
    """Return held-out R_corr rows for a frozen SFNO representation.

    The denominator is the mean latent distance between a deterministic
    derangement of ERA5 test samples. The numerator is the mean distance between
    each reference and its matched forecast/corruption counterpart. Consequently
    R_corr is zero for an identity transform and about one when its average
    displacement matches a typical held-out ERA5 latent displacement.
    """
    if not bool(getattr(model, "expects_raw_fields", False)):
        return []
    encoder = model.encoder
    seed = int(get(cfg, "seed", 0))

    def summarize(reference_samples, candidate_samples, coordinate, reference_distances=None):
        if len(reference_samples) < 2:
            return None
        if reference_distances is None:
            donor_positions = deranged_sample_positions(len(reference_samples), seed)
            reference_distances = _sfno_reference_distances(
                encoder, reference_samples, donor_positions, device, batch_size,
            )
        candidate_distances = _sfno_layer_pair_distances(
            encoder, reference_samples, candidate_samples, device, batch_size,
        )
        return [{
            **coordinate,
            "representation_layer": layer,
            "reference_distance": reference_distance,
            "candidate_distance": candidate_distances[layer],
            "r_corr": (float("nan") if reference_distance <= np.finfo(np.float64).eps
                       else candidate_distances[layer] / reference_distance),
            "n_samples": int(len(reference_samples)),
        } for layer, reference_distance in reference_distances.items()]

    rows = []
    if corruption:
        selected = indices(test_real, maximum)
        reference_samples = [raw_fields(test_real, SFNO_VARIABLES, int(index)) for index in selected]
        donor_positions = (
            data_dependent_donor_positions(corruption, len(selected), len(SFNO_VARIABLES), seed)
            if corruption in DATA_DEPENDENT_CORRUPTIONS else None
        )
        maximum_severity = target_corruption_max(cfg, corruption)
        reference_distances = _sfno_reference_distances(
            encoder, reference_samples, deranged_sample_positions(len(reference_samples), seed), device, batch_size,
        ) if len(reference_samples) >= 2 else None
        for severity in np.linspace(0.0, maximum_severity, int(get(cfg, "corruption_steps", 7))):
            candidates = []
            for position, index in enumerate(selected):
                donor = (raw_donor_fields(
                    test_fake, SFNO_VARIABLES, selected, donor_positions, position
                ) if donor_positions is not None else None)
                candidates.append(apply_sfno_corruption(
                    reference_samples[position], model.encoder, corruption, float(severity),
                    np.asarray(test_real.latitude.values), cfg, donor,
                    maximum_severity=maximum_severity,
                    random_seed=corruption_sample_seed(seed, corruption, int(index)),
                    target_variables=(getattr(model, "sfno_target_variables", None)
                                      if getattr(model, "sfno_use_era5_context", False) else None),
                ))
            mapped_variables = (list(getattr(model, "sfno_target_variables", []))
                                if getattr(model, "sfno_use_era5_context", False) else SFNO_VARIABLES)
            mapped_indices = [SFNO_VARIABLES.index(variable) for variable in mapped_variables]
            for candidate_index, candidate in enumerate(candidates):
                matched = match_raw(cfg, candidate[mapped_indices], mapped_variables, "sfno",
                                    "corruption", getattr(model, "histogram_target", corruption), severity)
                candidate = candidate.clone(); candidate[mapped_indices] = matched
                candidates[candidate_index] = candidate
            row = summarize(
                reference_samples, candidates, {"severity": float(severity), "lead_hour": None},
                reference_distances=reference_distances,
            )
            if row is not None:
                rows.extend(row)
    else:
        lead_hours = np.asarray(test_fake.prediction_timedelta.values).astype("timedelta64[h]").astype(int)
        for lead_index, lead_hour in enumerate(lead_hours):
            selected_records = evenly_spaced_pairs(
                [record for record in records if record.lead_index == lead_index], maximum,
            )
            if len(selected_records) < 2:
                continue
            reference_samples = [raw_fields(test_real, SFNO_VARIABLES, record.era5_index) for record in selected_records]
            candidates = [_sfno_forecast_input(model, test_real, test_fake, record) for record in selected_records]
            mapped_variables = (list(getattr(model, "sfno_target_variables", []))
                                if getattr(model, "sfno_use_era5_context", False) else SFNO_VARIABLES)
            mapped_indices = [SFNO_VARIABLES.index(variable) for variable in mapped_variables]
            for candidate_index, candidate in enumerate(candidates):
                matched = match_raw(cfg, candidate[mapped_indices], mapped_variables, "sfno",
                                    "forecast", getattr(model, "histogram_target", "forecast"), lead_hour)
                candidate = candidate.clone(); candidate[mapped_indices] = matched
                candidates[candidate_index] = candidate
            row = summarize(reference_samples, candidates, {"severity": None, "lead_hour": int(lead_hour)})
            if row is not None:
                rows.extend(row)
    return rows


def plot_sfno_representation_ratio(rows, architecture, kind, label, output_path):
    """Plot R_corr across a target's lead times or corruption strengths."""
    if not rows:
        return None
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    corruptions = rows[0]["severity"] is not None
    layer_order = ["block6_post_residual", "block7_pre_projection", "pooled_embedding"]
    labels = {
        "block6_post_residual": "Block 6 post-residual (34×121×240)",
        "block7_pre_projection": "Block 7 pre-projection (34×31×60)",
        "pooled_embedding": "Pooled 8-channel embedding",
    }
    colors = {layer: color for layer, color in zip(layer_order, categorical_colors(len(layer_order)))}
    markers = ("o", "s", "^", "v", "D")
    for layer_index, layer in enumerate(layer_order):
        layer_rows = [row for row in rows if row.get("representation_layer", "pooled_embedding") == layer]
        if not layer_rows:
            continue
        x = np.asarray([row["severity"] if corruptions else row["lead_hour"] for row in layer_rows], dtype=float)
        y = np.asarray([row["r_corr"] for row in layer_rows], dtype=float)
        finite = np.isfinite(y)
        axis.plot(x[finite], y[finite], marker=markers[layer_index], color=colors[layer], label=labels[layer])
        if np.any(~finite):
            axis.scatter(x[~finite], np.zeros(np.count_nonzero(~finite)), marker="x", color=colors[layer])
    axis.legend(fontsize=8)
    axis.set(
        title=f"{architecture}: {label} — SFNO representation ratio",
        xlabel="Corruption strength" if corruptions else "Lead time (hours)",
        ylabel=r"$R_{corr}=E||\phi(x)-\phi(T(x))||_2 / E||\phi(x_i)-\phi(x_j)||_2$",
    )
    axis.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    save_figure_bundle(figure, output_path, plot_type="sfno_representation_ratio", dpi=200)
    plt.close(figure)
    return output_path


def write_sfno_representation_ratios(root, records):
    path = Path(root) / "data" / "sfno_representation_ratios.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["architecture", "kind", "target", "representation_layer", "severity", "lead_hour",
              "reference_distance", "candidate_distance", "r_corr", "n_samples"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
    print(f"Saved SFNO representation ratios to: {path}")
    return path


def safe_target_name(value):
    return "".join(character if character.isalnum() or character in "-_" else "_"
                   for character in str(value)).strip("_")


def create_interpretability_gallery(
    model, cases, dataset, variables, means, stds, settings, device,
    output_path, architecture, kind, target,
):
    """Create held-out physical-field/IG galleries for CNN or four-field SFNO inputs."""
    method = str(settings.get("method", "integrated_gradients"))
    if method != "integrated_gradients":
        raise ValueError(f"Unsupported interpretability method: {method}")
    raw_sfno = bool(getattr(model, "expects_raw_fields", False))
    rows = []
    for case in cases:
        baseline = resolve_attribution_baseline(case["input"], settings, case, model=model)
        attribution, diagnostics = integrated_gradients(
            model, case["input"], baseline, device,
            steps=int(settings.get("steps", 32)),
            internal_batch_size=int(settings.get("internal_batch_size", 8)),
        )
        if raw_sfno:
            physical = case["input"].numpy()
        else:
            physical = np.stack([
                case["input"][channel].numpy() * float(stds[variable]) + float(means[variable])
                for channel, variable in enumerate(variables)
            ])
        rows.append((case, physical, attribution.numpy(), diagnostics))
    if not rows:
        raise ValueError("No held-out cases were selected for interpretability.")

    projection = ccrs.PlateCarree()
    metadata_rows = []
    if not raw_sfno:
        physical_min = min(float(np.nanmin(row[1][0])) for row in rows)
        physical_max = max(float(np.nanmax(row[1][0])) for row in rows)
        relevance_limit = max(float(np.nanpercentile(
            np.concatenate([np.abs(row[2].sum(axis=0)).ravel() for row in rows]), 99.0
        )), np.finfo(np.float32).eps)
        figure, axes = plt.subplots(
            len(rows), 2, figsize=(10.5, max(2.65 * len(rows), 5.0)),
            subplot_kw={"projection": projection}, squeeze=False,
            layout="constrained",
        )
        physical_artist = relevance_artist = None
        for row_index, (case, physical, attribution, diagnostics) in enumerate(rows):
            relevance = attribution.sum(axis=0)
            physical_artist = axes[row_index, 0].pcolormesh(
                dataset.longitudes, dataset.latitudes, physical[0], shading="auto",
                cmap="coolwarm", vmin=physical_min, vmax=physical_max, transform=projection, rasterized=True,
            )
            relevance_artist = axes[row_index, 1].pcolormesh(
                dataset.longitudes, dataset.latitudes, relevance, shading="auto",
                cmap="RdBu_r", vmin=-relevance_limit, vmax=relevance_limit, transform=projection, rasterized=True,
            )
            _title_interpretability_case(axes[row_index, 0], case, diagnostics)
            for axis in axes[row_index]:
                axis.coastlines(linewidth=0.45); axis.set_global()
            metadata_rows.append(_interpretability_metadata(
                case, diagnostics, architecture, kind, target, output_path, "standardized_zero", variables, attribution,
            ))
        figure.suptitle(
            f"{architecture.replace('_', ' ').title()}: {kind} / "
            f"{str(target).replace('_', ' ')} held-out logit cases\n"
            "Signed IG: positive values support ERA5",
            fontsize=12,
        )
        figure.colorbar(
            physical_artist, ax=list(axes[:, 0]), orientation="horizontal",
            fraction=0.025, pad=0.025, aspect=45,
            label=f"{variables[0]} (physical units)",
        )
        figure.colorbar(
            relevance_artist, ax=list(axes[:, 1]), orientation="horizontal",
            fraction=0.025, pad=0.025, aspect=45,
            label="Integrated-gradient attribution",
        )
    else:
        if list(variables) != SFNO_VARIABLES:
            raise ValueError("SFNO interpretability requires the fixed four-field channel order.")
        field_limits = {
            variable: (min(float(np.nanmin(row[1][channel])) for row in rows),
                       max(float(np.nanmax(row[1][channel])) for row in rows))
            for channel, variable in enumerate(variables)
        }
        relevance_values = np.concatenate([
            np.concatenate([row[2].ravel(), row[2].sum(axis=0).ravel()]) for row in rows
        ])
        relevance_limit = max(float(np.nanpercentile(np.abs(relevance_values), 99.0)), np.finfo(np.float32).eps)
        figure, axes = plt.subplots(
            len(rows) * 2, 5, figsize=(18.0, max(3.6 * len(rows), 7.2)),
            subplot_kw={"projection": projection}, squeeze=False,
        )
        field_artists, relevance_artist = [None] * len(variables), None
        for row_index, (case, physical, attribution, diagnostics) in enumerate(rows):
            top, bottom = axes[2 * row_index], axes[2 * row_index + 1]
            _title_interpretability_case(top[0], case, diagnostics)
            for channel, variable in enumerate(variables):
                vmin, vmax = field_limits[variable]
                field_artists[channel] = top[channel].pcolormesh(
                    dataset.longitudes, dataset.latitudes, physical[channel], shading="auto", cmap="coolwarm",
                    vmin=vmin, vmax=vmax, transform=projection, rasterized=True,
                )
                relevance_artist = bottom[channel].pcolormesh(
                    dataset.longitudes, dataset.latitudes, attribution[channel], shading="auto", cmap="RdBu_r",
                    vmin=-relevance_limit, vmax=relevance_limit, transform=projection, rasterized=True,
                )
                # Keep the case title on the first panel's left side; using a
                # centered field label avoids overwriting it.
                top[channel].set_title(variable, fontsize=8, loc="center")
                bottom[channel].set_title(f"IG: {variable}", fontsize=8, loc="center")
            aggregate = attribution.sum(axis=0)
            relevance_artist = bottom[4].pcolormesh(
                dataset.longitudes, dataset.latitudes, aggregate, shading="auto", cmap="RdBu_r",
                vmin=-relevance_limit, vmax=relevance_limit, transform=projection, rasterized=True,
            )
            bottom[4].set_title("IG: all-channel sum", fontsize=8)
            top[4].set_visible(False)
            for axis in (*top[:4], *bottom):
                axis.coastlines(linewidth=0.4); axis.set_global()
            metadata_rows.append(_interpretability_metadata(
                case, diagnostics, architecture, kind, target, output_path, "sfno_checkpoint_mean", variables, attribution,
            ))
        figure.suptitle(f"{architecture}: {kind}/{target} — held-out SFNO integrated gradients", fontsize=12)
        figure.subplots_adjust(top=0.96, bottom=0.12, left=0.025, right=0.985, hspace=0.28, wspace=0.08)
        for channel, variable in enumerate(variables):
            figure.colorbar(field_artists[channel], cax=figure.add_axes([0.03 + channel * 0.23, 0.045, 0.17, 0.012]), orientation="horizontal", label=variable)
        figure.colorbar(relevance_artist, cax=figure.add_axes([0.83, 0.045, 0.14, 0.012]), orientation="horizontal", label="IG relevance")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_bundle(
        figure, output_path, plot_type="target_integrated_gradients",
        payload={
            "latitudes": np.asarray(dataset.latitudes), "longitudes": np.asarray(dataset.longitudes),
            "physical_inputs": np.stack([row[1] for row in rows]),
            "integrated_gradients": np.stack([row[2] for row in rows]),
            "logits": np.asarray([row[0]["logit"] for row in rows]),
            "true_classes": np.asarray([row[0]["true_class"] for row in rows]),
            "selections": np.asarray([row[0]["selection"] for row in rows]),
            "times": np.asarray([str(row[0].get("time", "")) for row in rows]),
            "lead_hours": np.asarray([
                np.nan if row[0].get("lead_hour") is None else float(row[0]["lead_hour"])
                for row in rows
            ]),
            "severities": np.asarray([
                np.nan if row[0].get("severity") is None else float(row[0]["severity"])
                for row in rows
            ]),
            "completeness_residuals": np.asarray([
                float(row[3]["completeness_residual"]) for row in rows
            ]),
        }, dpi=int(settings.get("figure_dpi", 140)), bbox_inches="tight",
        capture_artists=False,
    )
    plt.close(figure)
    return metadata_rows


def create_sfno_representation_magnitude_gallery(model, cases, dataset, output_path,
                                                 architecture, kind, target):
    """Render held-out SFNO block-six/seven per-position channel magnitudes."""
    if not bool(getattr(model, "expects_raw_fields", False)):
        raise ValueError("SFNO representation maps require an SFNO probe.")
    layers = ("block6_post_residual", "block7_pre_projection")
    layer_titles = {
        "block6_post_residual": "Block 6 post-residual | channel L2 magnitude",
        "block7_pre_projection": "Block 7 pre-projection | channel L2 magnitude",
    }
    maps_by_case = []
    model.eval()
    with torch.no_grad():
        for case in cases:
            inputs = case["input"].unsqueeze(0).to(
                device=next(model.parameters()).device, dtype=torch.float32,
            )
            representations = model.encoder.extract_representation_maps(inputs)
            maps_by_case.append({
                layer: torch.linalg.vector_norm(representations[layer][0], dim=0).cpu().numpy()
                for layer in layers
            })
    if not maps_by_case:
        raise ValueError("No held-out cases were selected for SFNO representation maps.")

    limits = {
        layer: max(
            float(np.nanpercentile(np.concatenate([maps[layer].ravel() for maps in maps_by_case]), 99.0)),
            np.finfo(np.float32).eps,
        )
        for layer in layers
    }
    projection = ccrs.PlateCarree()
    figure, axes = plt.subplots(
        len(cases), len(layers), figsize=(12, max(3.3 * len(cases), 5.0)),
        subplot_kw={"projection": projection}, squeeze=False,
    )
    artists = {}
    latitude = np.asarray(dataset.latitudes)
    longitude = np.asarray(dataset.longitudes)
    for row_index, (case, magnitude_maps) in enumerate(zip(cases, maps_by_case)):
        for column, layer in enumerate(layers):
            magnitude = magnitude_maps[layer]
            # The seventh block is lower resolution; retain the shared global extent.
            map_latitude = np.linspace(float(latitude.min()), float(latitude.max()), magnitude.shape[0])
            map_longitude = np.linspace(float(longitude.min()), float(longitude.max()), magnitude.shape[1], endpoint=False)
            axis = axes[row_index, column]
            artists[layer] = axis.pcolormesh(
                map_longitude, map_latitude, magnitude, shading="auto", cmap="magma",
                vmin=0.0, vmax=limits[layer], transform=projection, rasterized=True,
            )
            if column == 0:
                details = [case.get("time"),
                           f"+{case['lead_hour']}h" if case.get("lead_hour") is not None else None,
                           f"severity={case['severity']:.4g}" if case.get("severity") is not None else None]
                axis.set_title(
                    f"{case['true_class']} {case['selection']} | "
                    f"logit={case['logit']:.3f} | {', '.join(str(value) for value in details if value)}\n"
                    f"{layer_titles[layer]}", fontsize=8, loc="left",
                )
            else:
                axis.set_title(layer_titles[layer], fontsize=8)
            axis.coastlines(linewidth=0.4)
            axis.set_global()
    figure.suptitle(f"{architecture}: {kind}/{target} — held-out SFNO representation magnitudes", fontsize=12)
    figure.subplots_adjust(top=0.94, bottom=0.12, left=0.035, right=0.965, hspace=0.28, wspace=0.12)
    figure.colorbar(artists[layers[0]], cax=figure.add_axes([0.10, 0.045, 0.32, 0.014]),
                    orientation="horizontal", label="Block 6 channel L2 magnitude")
    figure.colorbar(artists[layers[1]], cax=figure.add_axes([0.58, 0.045, 0.32, 0.014]),
                    orientation="horizontal", label="Block 7 channel L2 magnitude")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_bundle(
        figure, output_path, plot_type="sfno_representation_magnitudes",
        payload={
            "latitudes": latitude, "longitudes": longitude,
            "block6_magnitudes": np.stack([maps[layers[0]] for maps in maps_by_case]),
            "block7_magnitudes": np.stack([maps[layers[1]] for maps in maps_by_case]),
            "logits": np.asarray([case["logit"] for case in cases]),
            "true_classes": np.asarray([case["true_class"] for case in cases]),
            "selections": np.asarray([case["selection"] for case in cases]),
        }, dpi=180, bbox_inches="tight",
    )
    plt.close(figure)
    return output_path


def _title_interpretability_case(axis, case, diagnostics):
    predicted = "real" if case["logit"] >= 0.0 else "fake"
    details = [
        case.get("time"),
        "+{} h".format(case.get("lead_hour"))
        if case.get("lead_hour") is not None else None,
        "severity {:.4g}".format(case.get("severity"))
        if case.get("severity") is not None else None,
    ]
    context = " | ".join(str(value) for value in details if value)
    axis.set_title(
        "{} / {} | logit {:+.3f} | predicted {}\n{}".format(
            case["true_class"].upper(),
            str(case["selection"]).replace("_", " "),
            case["logit"],
            predicted.upper(),
            context or "No temporal/corruption context",
        ),
        fontsize=7,
        loc="left",
    )


def _interpretability_metadata(case, diagnostics, architecture, kind, target, output_path, baseline_kind, variables, attribution):
    predicted = "real" if case["logit"] >= 0.0 else "fake"
    metadata = {
        "architecture": architecture, "kind": kind, "target": target,
        "true_class": case["true_class"], "selection": case["selection"],
        "dataset_index": case["dataset_index"], "time": case.get("time", ""),
        "initialization_time": case.get("initialization_time", ""),
        "valid_time": case.get("valid_time", ""), "lead_hour": case.get("lead_hour", ""),
        "severity": case.get("severity", ""), "logit": case["logit"],
        "predicted_class": predicted, "correct": predicted == case["true_class"],
        "baseline_kind": baseline_kind, **diagnostics, "gallery_path": str(output_path),
    }
    metadata.update({f"attribution_sum_{variable}": float(attribution[channel].sum()) for channel, variable in enumerate(variables)})
    return metadata


def write_interpretability_cases(root, records):
    path = Path(root) / "data" / "target_interpretability_cases.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "architecture", "kind", "target", "true_class", "selection", "dataset_index",
        "time", "initialization_time", "valid_time", "lead_hour", "severity", "logit",
        "predicted_class", "correct", "baseline_kind", "input_logit", "baseline_logit", "attribution_sum",
        "attribution_sum_2m_temperature", "attribution_sum_10m_u_component_of_wind",
        "attribution_sum_10m_v_component_of_wind", "attribution_sum_mean_sea_level_pressure",
        "completeness_residual", "gallery_path",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved target interpretability cases to: {path}")
    return path


def load_target_squeezenet_checkpoint(path, cfg, device, architecture, variables):
    validate_binding(path, cfg)
    """Recreate one saved SqueezeNet target critic for post-training plotting."""
    model = WeatherDiscriminator(
        len(variables), ("squeezenet" if architecture == "squeezenet_equator_mask" else architecture), learning_rate=get(cfg, "learning_rate"),
        pretrained_backbone=bool(get(cfg, "pretrained_backbone", True)),
    ).to(device)
    model.model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def plot_target_discriminator_interpretability(cfg):
    """Regenerate CNN and SFNO IG galleries from saved target checkpoints."""
    configure_plot_bundle_saving_from_cfg(cfg)
    settings = get(cfg, "interpretability", {}) or {}
    if not bool(settings.get("enabled", True)):
        return [], None
    variables = list(get(cfg, "variables"))
    root = Path(str(get(cfg, "output_dir")))
    checkpoint_root = Path(str(get(cfg, "checkpoint_dir") or (root / "models" / "target_discriminators")))
    seed = int(settings.get("seed", get(cfg, "seed", 0)))
    device = target_device(cfg)
    tasks = []
    for architecture in ("squeezenet", "squeezenet_attention", "squeezenet_equator_mask"):
        for kind, label, paths, corruption in target_specs(cfg, variables):
            if architecture == "squeezenet_equator_mask" and corruption != "hemisphere_splice":
                continue
            checkpoint = checkpoint_root / architecture / kind / label.replace(" ", "_") / "model.pth"
            if checkpoint.is_file():
                tasks.append((architecture, kind, label, paths, corruption, checkpoint, variables, None))

    sfno_tasks = []
    for architecture in ("sfno_linear", "sfno_mlp"):
        for kind, label, paths, corruption in target_specs(cfg, SFNO_VARIABLES):
            checkpoint = checkpoint_root / architecture / kind / label.replace(" ", "_") / "model.pth"
            if checkpoint.is_file():
                sfno_tasks.append((architecture, kind, label, paths, corruption, checkpoint, SFNO_VARIABLES, None))
    if sfno_tasks:
        try:
            load_sfno_encoder(cfg, device)
            tasks.extend(sfno_tasks)
        except FileNotFoundError as error:
            print(f"Skipping SFNO target interpretability: {error}")
    if not tasks:
        print(f"Skipping target interpretability: no compatible checkpoints under {checkpoint_root}.")
        return [], None
    if not Path(str(cfg.real_nc_file)).is_file():
        print(f"Skipping target interpretability: missing ERA5 input {cfg.real_nc_file}.")
        return [], None

    torch.manual_seed(seed)
    np.random.seed(seed)
    real = safe_open_dataset(cfg.real_nc_file)
    corruption_train = select_era5_split(real, cfg, "train", coverage="corruption")
    corruption_means = {variable: float(corruption_train[variable].mean()) for variable in variables}
    corruption_stds = {variable: max(float(corruption_train[variable].std()), 1e-8) for variable in variables}
    maximum = int(get(cfg, "max_eval_samples", 0))
    random_count = int(settings.get("random_samples_per_class", 2))
    galleries, rows = [], []
    try:
        for architecture, kind, label, paths, corruption, checkpoint, input_variables, _ in tqdm(
            tasks, desc="Plotting target interpretability"
        ):
            raw_sfno = architecture in {"sfno_linear", "sfno_mlp"}
            fake = corruption_train if corruption else open_model_forecasts(paths)
            try:
                test_real, test_fake, test_records = target_test_inputs(real, fake, cfg, corruption)
                if raw_sfno:
                    # Reload the base encoder for each checkpoint so a finetuned
                    # seventh block cannot leak into a subsequently frozen model.
                    model, _ = load_sfno_probe_checkpoint(checkpoint, cfg, device)
                    model.sfno_use_era5_context, model.sfno_target_variables = sfno_context_settings(cfg)
                    dataset = SFNOTargetDataset(
                        test_real, test_fake, model.encoder, cfg.lead_times, corruption, maximum,
                        float(get(cfg, "corruption_power")),
                        target_corruption_max(cfg, corruption) if corruption else float(get(cfg, "corruption_severity_max")),
                        cfg, test_records, deterministic_seed=seed, target_label=label,
                    )
                    gallery_means, gallery_stds = {}, {}
                else:
                    train_records = None if corruption else forecast_pairs(fake, real, cfg, "train", cfg.lead_times)
                    means, stds = ((corruption_means, corruption_stds) if corruption
                                   else matched_statistics(real, train_records, variables))
                    dataset = BalancedTargetDataset(
                        test_real, test_fake, variables, means, stds, cfg.lead_times, corruption, maximum,
                        float(get(cfg, "corruption_power")),
                        target_corruption_max(cfg, corruption) if corruption else float(get(cfg, "corruption_severity_max")),
                        cfg, test_records, deterministic_seed=seed,
                        equator_mask_degrees=(float((get(cfg, "equator_masked_hemisphere_splice", {}) or {}).get("half_width_degrees", 10.0))
                                              if architecture == "squeezenet_equator_mask" else 0.0),
                        target_label=label,
                    )
                    model = load_target_squeezenet_checkpoint(checkpoint, cfg, device, architecture, variables)
                    if architecture == "squeezenet_equator_mask":
                        model.equator_mask_degrees = float((get(cfg, "equator_masked_hemisphere_splice", {}) or {}).get("half_width_degrees", 10.0))
                    gallery_means, gallery_stds = means, stds
                cases = binary_classification_metrics(
                    model, dataset, device, int(get(cfg, "batch_size")),
                    f"Selecting IG cases {architecture} {kind}/{label}",
                    random_samples_per_class=random_count, seed=seed,
                ).get("cases", [])
                gallery_path = (root / "plots" / "target_interpretability" / architecture / kind /
                                f"{safe_target_name(label)}_integrated_gradients.png")
                gallery_rows = create_interpretability_gallery(
                    model, cases, dataset, input_variables, gallery_means, gallery_stds, settings, device,
                    gallery_path, architecture, kind, label,
                )
                galleries.append(gallery_path); rows.extend(gallery_rows)
                print(f"Saved target interpretability gallery to: {gallery_path}")
                if raw_sfno and bool(settings.get("sfno_representation_maps", True)):
                    representation_path = gallery_path.with_name(
                        gallery_path.name.replace("_integrated_gradients.png", "_representation_magnitudes.png")
                    )
                    create_sfno_representation_magnitude_gallery(
                        model, cases, dataset, representation_path, architecture, kind, label,
                    )
                    galleries.append(representation_path)
                    print(f"Saved SFNO representation magnitude gallery to: {representation_path}")
            finally:
                if not corruption:
                    fake.close()
    finally:
        real.close()
    return galleries, write_interpretability_cases(root, rows)


def target_test_inputs(real, fake, cfg, corruption):
    """Build the held-out counterpart of one target training problem."""
    if corruption:
        test = select_era5_split(real, cfg, "test", coverage="corruption")
        return test, test, None
    pairs = forecast_pairs(fake, real, cfg, "test", cfg.lead_times)
    if not pairs:
        raise ValueError("No exact monthly test forecast/ERA5 pairs remain for target evaluation.")
    return real, fake, pairs


class BalancedTargetDataset(Dataset):
    """Balanced real/fake data with reproducible held-out corruptions."""

    def __init__(self, real, fake, variables, means, stds, leads, corruption=None,
                 max_samples=0, power=2.0, severity_max=0.05, cfg=None,
                 paired_records=None, deterministic_seed=None, equator_mask_degrees=0.0, target_label=None):
        self.real, self.fake = real, fake
        self.variables, self.means, self.stds = variables, means, stds
        self.real_i, self.fake_i = indices(real, max_samples), indices(fake, max_samples)
        self.leads = ([i for i, hours in enumerate(
            np.asarray(fake.prediction_timedelta.values).astype("timedelta64[h]").astype(int)
        ) if int(hours) in leads] if "prediction_timedelta" in fake.dims else [None])
        self.corruption, self.power, self.severity_max = corruption, float(power), float(severity_max)
        self.paired_records = evenly_spaced_pairs(paired_records or [], max_samples)
        self.cfg = cfg
        self.deterministic_seed = None if deterministic_seed is None else int(deterministic_seed)
        self.latitudes = np.asarray(fake.latitude.values, dtype=np.float64)
        self.equator_mask_degrees = float(equator_mask_degrees)
        self.target_label = str(target_label or corruption or "forecast")
        self.longitudes = np.asarray(fake.longitude.values, dtype=np.float64)
        self.donor_positions = None
        self.donor_seed = int(get(cfg, "seed", 0))
        if corruption in DATA_DEPENDENT_CORRUPTIONS:
            self.set_epoch(0)
        self.n = len(self.paired_records) if self.paired_records else max(
            len(self.real_i), len(self.fake_i) * len(self.leads)
        )

    def __len__(self):
        return 2 * self.n

    def set_epoch(self, epoch):
        """Refresh whole-field splice donors once per training epoch."""
        if self.corruption in DATA_DEPENDENT_CORRUPTIONS:
            self.donor_positions = data_dependent_donor_positions(
                self.corruption, len(self.fake_i), len(self.variables), self.donor_seed, int(epoch)
            )

    def _sample_seed(self, position):
        if self.deterministic_seed is None:
            return None
        return corruption_sample_seed(self.deterministic_seed, str(self.corruption), int(position))

    def _severity(self, position):
        seed = self._sample_seed(position)
        return sample_target_corruption_severity(
            self.cfg, self.corruption, self.severity_max, self.power, seed=seed,
        )

    @staticmethod
    def _time(dataset, index):
        return str(np.asarray(dataset.time.values)[int(index)])

    def sample_metadata(self, index):
        is_fake, position = index >= self.n, index % self.n
        metadata = {"dataset_index": int(index), "true_class": "fake" if is_fake else "real"}
        if not is_fake:
            if self.paired_records:
                record = self.paired_records[position % len(self.paired_records)]
                metadata.update(time=str(record.valid_time), valid_time=str(record.valid_time))
            else:
                real_index = int(self.real_i[position % len(self.real_i)])
                metadata.update(time=self._time(self.real, real_index))
            return metadata
        if self.corruption:
            fake_position = position % len(self.fake_i)
            metadata.update(
                time=self._time(self.fake, int(self.fake_i[fake_position])),
                corruption=str(self.corruption), severity=float(self._severity(fake_position)),
            )
            return metadata
        if self.paired_records:
            record = self.paired_records[position % len(self.paired_records)]
            metadata.update(
                time=str(record.valid_time), initialization_time=str(record.initialization_time),
                valid_time=str(record.valid_time), lead_hour=int(record.lead_hour),
            )
            return metadata
        time_position, lead_position = divmod(position, len(self.leads))
        sample_index = int(self.fake_i[time_position % len(self.fake_i)])
        lead_index = self.leads[lead_position]
        lead_hour = 0 if lead_index is None else int(
            np.asarray(self.fake.prediction_timedelta.values)[lead_index]
            .astype("timedelta64[h]").astype(int)
        )
        metadata.update(time=self._time(self.fake, sample_index), lead_hour=lead_hour)
        return metadata

    def __getitem__(self, index):
        is_fake, position = index >= self.n, index % self.n
        if not is_fake:
            real_index = (self.paired_records[position % len(self.paired_records)].era5_index
                          if self.paired_records else int(self.real_i[position % len(self.real_i)]))
            return mask_equatorial_band(fields(self.real, self.variables, self.means, self.stds, int(real_index)), self.latitudes, self.equator_mask_degrees), torch.tensor([1.0])
        if self.corruption:
            fake_position = position % len(self.fake_i)
            sample = fields(self.fake, self.variables, self.means, self.stds,
                            int(self.fake_i[fake_position]))
            severity, seed = self._severity(fake_position), self._sample_seed(fake_position)
            if self.corruption in SPECIAL_CORRUPTIONS:
                donor = None
                if self.donor_positions is not None:
                    donor = standardized_donor_fields(
                        self.fake, self.variables, self.means, self.stds, self.fake_i,
                        self.donor_positions, fake_position,
                    )
                corrupted = torch.from_numpy(apply_special_baseline_corruption(
                    sample.numpy(), self.corruption, severity, self.latitudes, self.cfg,
                    donor, maximum_severity=self.severity_max, random_seed=seed,
                ))
            elif seed is None:
                corrupted = apply_configured_corruption(sample, self.corruption, severity)
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(seed))
                    corrupted = apply_configured_corruption(sample, self.corruption, severity)
            corrupted = match_standardized(self.cfg, corrupted, self.variables, self.means, self.stds,
                                             "standard", "corruption", self.target_label, severity)
            return mask_equatorial_band(corrupted.to(dtype=torch.float32), self.latitudes, self.equator_mask_degrees), torch.tensor([0.0])
        if self.paired_records:
            record = self.paired_records[position % len(self.paired_records)]
            candidate = fields(self.fake, self.variables, self.means, self.stds,
                               record.forecast_index, record.lead_index)
            candidate = match_standardized(self.cfg, candidate, self.variables, self.means, self.stds,
                                           "standard", "forecast", self.target_label, record.lead_hour)
            return mask_equatorial_band(candidate, self.latitudes, self.equator_mask_degrees), torch.tensor([0.0])
        time_position, lead_position = divmod(position, len(self.leads))
        return mask_equatorial_band(fields(self.fake, self.variables, self.means, self.stds,
                      int(self.fake_i[time_position % len(self.fake_i)]),
                      self.leads[lead_position]), self.latitudes, self.equator_mask_degrees), torch.tensor([0.0])


class SFNOTargetDataset(Dataset):
    """Balanced raw four-field samples for a frozen SFNO encoder."""

    def __init__(self, real, fake, encoder, leads, corruption=None, max_samples=0,
                 power=2.0, severity_max=0.2, cfg=None, paired_records=None,
                 deterministic_seed=None, target_label=None):
        self.real = real
        self.fake = fake
        self.encoder = encoder
        self.real_i = indices(real, max_samples)
        self.fake_i = indices(fake, max_samples)
        self.leads = (
            [
                i for i, hours in enumerate(
                    np.asarray(fake.prediction_timedelta.values).astype("timedelta64[h]").astype(int)
                ) if int(hours) in leads
            ]
            if "prediction_timedelta" in fake.dims else [None]
        )
        self.corruption = corruption
        self.power = float(power)
        self.severity_max = float(severity_max)
        self.cfg = cfg
        self.deterministic_seed = None if deterministic_seed is None else int(deterministic_seed)
        self.use_era5_context, self.target_variables = sfno_context_settings(cfg)
        self.target_label = str(target_label or corruption or "forecast")
        self.paired_records = evenly_spaced_pairs(paired_records or [], max_samples)
        self.latitudes = np.asarray(fake.latitude.values, dtype=np.float64)
        self.longitudes = np.asarray(fake.longitude.values, dtype=np.float64)
        self.donor_positions = None
        self.donor_seed = int(get(cfg, "seed", 0))
        if corruption in DATA_DEPENDENT_CORRUPTIONS:
            self.set_epoch(0)
        self.n = len(self.paired_records) if self.paired_records else max(len(self.real_i), len(self.fake_i) * len(self.leads))

    def __len__(self):
        return 2 * self.n

    def set_epoch(self, epoch):
        """Refresh whole-field splice donors once per training epoch."""
        if self.corruption in DATA_DEPENDENT_CORRUPTIONS:
            self.donor_positions = data_dependent_donor_positions(
                self.corruption, len(self.fake_i), len(SFNO_VARIABLES), self.donor_seed, int(epoch)
            )

    @staticmethod
    def _time(dataset, index):
        return str(np.asarray(dataset.time.values)[int(index)])

    def _sample_seed(self, position):
        if self.deterministic_seed is None:
            return None
        return corruption_sample_seed(self.deterministic_seed, str(self.corruption), int(position))

    def _severity(self, position):
        return sample_target_corruption_severity(
            self.cfg, self.corruption, self.severity_max, self.power,
            seed=self._sample_seed(position),
        )

    def sample_metadata(self, index):
        """Describe a held-out raw SFNO sample for logit-selected galleries."""
        is_fake, position = index >= self.n, index % self.n
        metadata = {"dataset_index": int(index), "true_class": "fake" if is_fake else "real"}
        if not is_fake:
            if self.paired_records:
                record = self.paired_records[position % len(self.paired_records)]
                metadata.update(time=str(record.valid_time), valid_time=str(record.valid_time))
            else:
                metadata.update(time=self._time(self.real, int(self.real_i[position % len(self.real_i)])))
            return metadata
        if self.corruption:
            fake_index = int(self.fake_i[position % len(self.fake_i)])
            metadata.update(
                time=self._time(self.fake, fake_index), corruption=str(self.corruption),
                severity=float(self._severity(position % len(self.fake_i))),
            )
            return metadata
        if self.paired_records:
            record = self.paired_records[position % len(self.paired_records)]
            metadata.update(
                time=str(record.valid_time), initialization_time=str(record.initialization_time),
                valid_time=str(record.valid_time), lead_hour=int(record.lead_hour),
            )
            return metadata
        time_position, lead_position = divmod(position, len(self.leads))
        sample_index = int(self.fake_i[time_position % len(self.fake_i)])
        lead_index = self.leads[lead_position]
        lead_hour = 0 if lead_index is None else int(
            np.asarray(self.fake.prediction_timedelta.values)[lead_index]
            .astype("timedelta64[h]").astype(int)
        )
        metadata.update(time=self._time(self.fake, sample_index), lead_hour=lead_hour)
        return metadata

    def __getitem__(self, index):
        is_fake = index >= self.n
        position = index % self.n
        if not is_fake:
            sample_index = (
                self.paired_records[position % len(self.paired_records)].era5_index
                if self.paired_records else int(self.real_i[position % len(self.real_i)])
            )
            return raw_fields(self.real, SFNO_VARIABLES, sample_index), torch.tensor([1.0])
        if self.corruption:
            fake_position = position % len(self.fake_i)
            sample_index = int(self.fake_i[fake_position])
            sample = raw_fields(self.fake, SFNO_VARIABLES, sample_index)
            donor = None
            if self.donor_positions is not None:
                donor = raw_donor_fields(
                    self.fake, SFNO_VARIABLES, self.fake_i, self.donor_positions, fake_position
                )
            severity = self._severity(fake_position)
            corrupted = apply_sfno_corruption(
                sample, self.encoder, self.corruption, severity, self.latitudes,
                self.cfg, donor, maximum_severity=self.severity_max,
                random_seed=self._sample_seed(fake_position),
                target_variables=self.target_variables if self.use_era5_context else None,
            )
            mapped_variables = self.target_variables if self.use_era5_context else SFNO_VARIABLES
            mapped_indices = [SFNO_VARIABLES.index(variable) for variable in mapped_variables]
            matched = match_raw(self.cfg, corrupted[mapped_indices], mapped_variables, "sfno",
                                "corruption", self.target_label, severity)
            corrupted = corrupted.clone(); corrupted[mapped_indices] = matched
            return corrupted, torch.tensor([0.0])
        if self.paired_records:
            record = self.paired_records[position % len(self.paired_records)]
            if self.use_era5_context:
                forecast = raw_fields(self.fake, self.target_variables, record.forecast_index, record.lead_index)
                forecast = match_raw(self.cfg, forecast, self.target_variables, "sfno",
                                     "forecast", self.target_label, record.lead_hour)
                context = raw_fields(self.real,
                                     [v for v in SFNO_VARIABLES if v not in self.target_variables],
                                     record.era5_index)
                values = []
                for variable in SFNO_VARIABLES:
                    values.append(forecast[self.target_variables.index(variable)] if variable in self.target_variables
                                  else context[[v for v in SFNO_VARIABLES if v not in self.target_variables].index(variable)])
                return torch.stack(values), torch.tensor([0.0])
            candidate = raw_fields(self.fake, SFNO_VARIABLES, record.forecast_index, record.lead_index)
            candidate = match_raw(self.cfg, candidate, SFNO_VARIABLES, "sfno",
                                  "forecast", self.target_label, record.lead_hour)
            return candidate, torch.tensor([0.0])
        time_position, lead_position = divmod(position, len(self.leads))
        sample_index = int(self.fake_i[time_position % len(self.fake_i)])
        return (
            raw_fields(self.fake, SFNO_VARIABLES, sample_index, self.leads[lead_position]),
            torch.tensor([0.0]),
        )


def logits_for(
    model, ds, variables, means, stds, device, maximum, batch_size, *,
    lead=None, corruption=None, severity=0., cfg=None, maximum_severity=None,
    progress_description=None, selected_indices=None, context_ds=None, context_indices=None,
):
    values=[]; batch=[]
    model_dtype = next(model.parameters()).dtype
    raw_sfno = bool(getattr(model, "expects_raw_fields", False))
    input_variables = list(getattr(model, "input_variables", variables))
    if selected_indices is None:
        selected_indices = indices(ds, maximum)
    else:
        selected_indices = np.asarray(selected_indices, dtype=int)
        if maximum > 0 and len(selected_indices) > maximum:
            keep = np.linspace(0, len(selected_indices) - 1, maximum, dtype=int)
            selected_indices = selected_indices[keep]
            if context_indices is not None:
                context_indices = np.asarray(context_indices, dtype=int)[keep]
    if context_indices is not None:
        context_indices = np.asarray(context_indices, dtype=int)
        if len(context_indices) != len(selected_indices):
            raise ValueError(
                "context_indices must align one-to-one with selected_indices "
                f"({len(context_indices)} != {len(selected_indices)})"
            )
    donor_positions = (
        data_dependent_donor_positions(
            corruption, len(selected_indices), len(input_variables), int(get(cfg, "seed", 0))
        ) if corruption in DATA_DEPENDENT_CORRUPTIONS else None
    )
    with torch.no_grad():
        iterator = selected_indices
        if progress_description is not None:
            iterator = tqdm(selected_indices, desc=progress_description, leave=False)
        for position, i in enumerate(iterator):
            if raw_sfno and context_ds is not None and bool(getattr(model, "sfno_use_era5_context", False)):
                target_variables = list(getattr(model, "sfno_target_variables", ["2m_temperature"]))
                context_variables = [v for v in input_variables if v not in target_variables]
                context_index = int(context_indices[position])
                candidate_fields = raw_fields(ds, target_variables, int(i), lead)
                context_fields = raw_fields(context_ds, context_variables, context_index)
                x = torch.stack([
                    candidate_fields[target_variables.index(variable)] if variable in target_variables
                    else context_fields[context_variables.index(variable)]
                    for variable in input_variables
                ])
            else:
                x = (
                    raw_fields(ds, input_variables, int(i), lead)
                    if raw_sfno else fields(ds,variables,means,stds,int(i),lead)
                )
            if raw_sfno and corruption:
                donor = None
                if donor_positions is not None:
                    donor = raw_donor_fields(
                        ds, input_variables, selected_indices, donor_positions, position, lead
                    )
                x = apply_sfno_corruption(
                    x, model.encoder, corruption, severity,
                    np.asarray(ds.latitude.values), cfg, donor,
                    maximum_severity=maximum_severity,
                    random_seed=corruption_sample_seed(get(cfg, "seed", 0), corruption, int(i)),
                    target_variables=(getattr(model, "sfno_target_variables", None)
                                      if getattr(model, "sfno_use_era5_context", False) else None),
                )
            elif corruption in SPECIAL_CORRUPTIONS:
                donor = None
                if donor_positions is not None:
                    donor = standardized_donor_fields(
                        ds, variables, means, stds, selected_indices, donor_positions, position, lead
                    )
                x = torch.from_numpy(apply_special_baseline_corruption(
                    x.numpy(), corruption, severity, np.asarray(ds.latitude.values), cfg,
                    donor, maximum_severity=(
                        target_corruption_max(cfg, corruption)
                        if maximum_severity is None else maximum_severity
                    ),
                    random_seed=corruption_sample_seed(get(cfg, "seed", 0), corruption, int(i)),
                ))
            elif corruption:
                sample_seed = corruption_sample_seed(get(cfg, "seed", 0), corruption, int(i))
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(sample_seed))
                    x = apply_configured_corruption(x, corruption, severity)
            histogram_target = getattr(model, "histogram_target", None)
            if histogram_target is not None and (corruption is not None or lead is not None):
                coordinate = severity if corruption is not None else int(
                    np.asarray(ds.prediction_timedelta.values)[lead].astype("timedelta64[h]").astype(int)
                )
                if raw_sfno:
                    mapped_variables = (list(getattr(model, "sfno_target_variables", []))
                                        if getattr(model, "sfno_use_era5_context", False) else input_variables)
                    mapped_indices = [input_variables.index(variable) for variable in mapped_variables]
                    matched = match_raw(cfg, x[mapped_indices], mapped_variables, "sfno",
                                        "corruption" if corruption is not None else "forecast",
                                        histogram_target, coordinate)
                    x = x.clone(); x[mapped_indices] = matched
                else:
                    x = match_standardized(
                        cfg, x, variables, means, stds, "standard",
                        "corruption" if corruption is not None else "forecast",
                        histogram_target, coordinate,
                    )
            x = mask_equatorial_band(
                x, ds.latitude.values, float(getattr(model, "equator_mask_degrees", 0.0))
            )
            batch.append(x)
            if len(batch) == batch_size:
                inputs = torch.stack(batch).to(device=device, dtype=model_dtype)
                values.extend(model(inputs).flatten().cpu().tolist()); batch=[]
        if batch:
            inputs = torch.stack(batch).to(device=device, dtype=model_dtype)
            values.extend(model(inputs).flatten().cpu().tolist())
    return np.asarray(values)


def score(model, ds, variables, means, stds, device, *, lead=None, corruption=None, severity=0., maximum=0, batch_size=32, cfg=None, maximum_severity=None, progress_description=None, selected_indices=None, context_ds=None, context_indices=None):
    z=logits_for(model,ds,variables,means,stds,device,maximum,batch_size,lead=lead,corruption=corruption,severity=severity,cfg=cfg,maximum_severity=maximum_severity,progress_description=progress_description,selected_indices=selected_indices,context_ds=context_ds,context_indices=context_indices); conj=z-1.; return float(conj.mean()), float(conj.std(ddof=1)/np.sqrt(len(z))) if len(z)>1 else 0., len(z)


def train_target(real, fake, variables, means, stds, cfg, device, *, corruption=None,
                 label=None, model_name=None, metric_logger=None, log_every_n_steps=20,
                 paired_records=None, equator_mask_degrees=0.0):
    selected_model_name = str(model_name or get(cfg, 'model_name'))
    model=WeatherDiscriminator(len(variables), selected_model_name, learning_rate=get(cfg,'learning_rate'), pretrained_backbone=bool(get(cfg,'pretrained_backbone',True))).to(device)
    dataset=BalancedTargetDataset(real,fake,variables,means,stds,cfg.lead_times,corruption,int(get(cfg,'max_train_samples',0)),float(get(cfg,'corruption_power')),target_corruption_max(cfg, corruption) if corruption else float(get(cfg,'corruption_severity_max')),cfg,paired_records,equator_mask_degrees=equator_mask_degrees,target_label=label)
    opt=torch.optim.AdamW(model.parameters(),lr=float(get(cfg,'learning_rate')),weight_decay=float(get(cfg,'weight_decay')))
    model.train()
    loader = DataLoader(
        dataset,
        batch_size=int(get(cfg,'batch_size')),
        shuffle=True,
        num_workers=int(get(cfg,'num_workers')),
    )
    target_label = label or corruption or "forecast"
    global_step = 0
    final_metrics = {}
    for epoch in range(int(get(cfg,'epochs'))):
        dataset.set_epoch(epoch)
        epoch_loss = epoch_correct = epoch_count = 0.0
        batches = tqdm(
            loader,
            desc=f"{target_label}: epoch {epoch + 1}/{int(get(cfg, 'epochs'))}",
            leave=False,
        )
        for x,y in batches:
            inputs, labels = x.to(device), y.to(device)
            opt.zero_grad(); logits = model(inputs)
            loss=F.binary_cross_entropy_with_logits(logits, labels); loss.backward(); opt.step()
            batch_count = labels.numel()
            batch_correct = ((logits.detach() >= 0.0) == (labels >= 0.5)).sum().item()
            epoch_loss += loss.detach().item() * batch_count
            epoch_correct += batch_correct
            epoch_count += batch_count
            global_step += 1
            if metric_logger is not None and global_step % max(int(log_every_n_steps), 1) == 0:
                metric_logger({
                    "train/batch_loss": float(loss.detach().item()),
                    "train/batch_accuracy": float(batch_correct / batch_count),
                    "train/global_step": global_step,
                    "train/epoch": epoch + 1,
                })
            batches.set_postfix(loss=f"{loss.detach().item():.4f}")
        if epoch_count:
            final_metrics = {
                "loss": float(epoch_loss / epoch_count),
                "accuracy": float(epoch_correct / epoch_count),
                "n_samples": int(epoch_count),
            }
            if metric_logger is not None:
                metric_logger({
                    "train/epoch_loss": final_metrics["loss"],
                    "train/epoch_accuracy": final_metrics["accuracy"],
                    "train/epoch": epoch + 1,
                })
    model.target_train_metrics = final_metrics
    return model.eval()


def train_sfno_target(real, fake, encoder, cfg, device, *, corruption=None, label=None,
                      metric_logger=None, log_every_n_steps=20, paired_records=None):
    """Train both frozen heads, or exactly one head when SFNO layers are unfrozen."""
    if int(get(cfg, "num_workers", 0)) != 0:
        raise ValueError("SFNO target training requires target_discriminator.num_workers=0.")
    severity_max = (
        target_corruption_max(cfg, corruption)
        if corruption else float(get(cfg, "corruption_severity_max"))
    )
    dataset = SFNOTargetDataset(
        real, fake, encoder, cfg.lead_times, corruption,
        int(get(cfg, "max_train_samples", 0)),
        float(get(cfg, "corruption_power")), severity_max, cfg, paired_records,
        target_label=label,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(get(cfg, "batch_size")),
        shuffle=True,
        num_workers=0,
    )
    unfreeze_final_encoder_stage = bool(sfno_settings(cfg).get("unfreeze_final_encoder_stage", False))
    selected_architectures = (
        (sfno_finetune_probe_architecture(cfg),)
        if unfreeze_final_encoder_stage else ("sfno_linear", "sfno_mlp")
    )
    heads = {
        architecture: build_sfno_head(encoder, architecture, cfg).to(device)
        for architecture in selected_architectures
    }
    configure_sfno_trainability(encoder, unfreeze_final_encoder_stage)
    parameters = [parameter for head in heads.values() for parameter in head.parameters()]
    parameters.extend(parameter for parameter in encoder.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(get(cfg, "learning_rate")),
        weight_decay=float(get(cfg, "weight_decay")),
    )
    epochs = int(sfno_settings(cfg).get("epochs", get(cfg, "epochs")))
    target_label = label or corruption or "forecast"
    encoder.eval()
    if unfreeze_final_encoder_stage:
        sfno_last_encoder_block(encoder).train()
        sfno_channel_projection(encoder).train()
    for head in heads.values():
        head.train()
    global_step = 0
    final_metrics = {}
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        epoch_loss = {name: 0.0 for name in heads}
        epoch_correct = {name: 0.0 for name in heads}
        epoch_count = 0
        batches = tqdm(loader, desc=f"SFNO {target_label}: epoch {epoch + 1}/{epochs}", leave=False)
        for inputs, labels in batches:
            # BatchNorm1d cannot estimate variance from a one-sample training batch.
            if inputs.shape[0] < 2:
                continue
            inputs = inputs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            if unfreeze_final_encoder_stage:
                features = encoder.extract_features(inputs, enable_input_grad=True)
            else:
                with torch.no_grad():
                    features = encoder.extract_features(inputs)
            optimizer.zero_grad()
            logits_by_head = {name: head(features) for name, head in heads.items()}
            losses = {
                name: F.binary_cross_entropy_with_logits(logits, labels)
                for name, logits in logits_by_head.items()
            }
            loss = torch.stack(list(losses.values())).sum()
            loss.backward()
            optimizer.step()
            count = labels.numel(); epoch_count += count; global_step += 1
            for name, logits in logits_by_head.items():
                logits = logits.detach()
                epoch_loss[name] += losses[name].detach().item() * count
                epoch_correct[name] += ((logits >= 0.0) == (labels >= 0.5)).sum().item()
            if metric_logger is not None and global_step % max(int(log_every_n_steps), 1) == 0:
                payload = {"train/global_step": global_step, "train/epoch": epoch + 1}
                for name in heads:
                    payload[f"{name}/batch_loss"] = float(losses[name].detach().item())
                    payload[f"{name}/batch_accuracy"] = float(
                        ((logits_by_head[name].detach() >= 0.0) == (labels >= 0.5)).float().mean().item()
                    )
                metric_logger(payload)
            batches.set_postfix(**{
                name.removeprefix("sfno_"): f"{loss.detach().item():.4f}"
                for name, loss in losses.items()
            })
        if epoch_count:
            final_metrics = {
                name: {
                    "loss": float(epoch_loss[name] / epoch_count),
                    "accuracy": float(epoch_correct[name] / epoch_count),
                    "n_samples": int(epoch_count),
                }
                for name in heads
            }
            if metric_logger is not None:
                payload = {"train/epoch": epoch + 1}
                for name, metrics in final_metrics.items():
                    payload[f"{name}/epoch_loss"] = metrics["loss"]
                    payload[f"{name}/epoch_accuracy"] = metrics["accuracy"]
                metric_logger(payload)
    probes = {}
    for name, head in heads.items():
        probe = FrozenSFNOProbe(
            encoder, head.eval(), name,
            unfreeze_final_encoder_stage=unfreeze_final_encoder_stage,
        ).eval()
        probe.target_train_metrics = final_metrics.get(name, {})
        probe.sfno_use_era5_context, probe.sfno_target_variables = sfno_context_settings(cfg)
        probes[name] = probe
    return probes


def sfno_checkpoint_metadata(encoder, architecture, unfreeze_final_encoder_stage=False):
    return {
        "architecture": str(architecture),
        "input_variables": list(SFNO_VARIABLES),
        "embedding_channels": int(encoder.embedding_channels),
        "embedding_resolution": list(encoder.embedding_resolution),
        "pooling": str(encoder.pooling),
        "pool_grid": list(encoder.pool_grid),
        "feature_dim": int(encoder.feature_dim),
        "encoder_frozen": not bool(unfreeze_final_encoder_stage),
        "unfreeze_final_encoder_stage": bool(unfreeze_final_encoder_stage),
        "encoder_pretraining": "ERA5 1975-2019; overlaps the temporal test partition",
    }


def save_sfno_probe_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "head_state_dict": model.head.state_dict(),
        "metadata": sfno_checkpoint_metadata(
            model.encoder, model.architecture, model.unfreeze_final_encoder_stage,
        ),
    }
    if model.unfreeze_final_encoder_stage:
        payload["last_encoder_block_state_dict"] = sfno_last_encoder_block(model.encoder).state_dict()
        payload["channel_down_scaling_state_dict"] = sfno_channel_projection(model.encoder).state_dict()
    torch.save(payload, path)


def load_sfno_probe_checkpoint(path, cfg, device, encoder=None):
    validate_binding(path, cfg)
    payload = torch.load(path, map_location=device, weights_only=True)
    metadata = payload.get("metadata", {})
    architecture = metadata.get("architecture")
    if architecture not in {"sfno_linear", "sfno_mlp"}:
        raise ValueError(f"Invalid SFNO probe architecture in {path}: {architecture!r}")
    if list(metadata.get("input_variables", [])) != SFNO_VARIABLES:
        raise ValueError(f"SFNO probe checkpoint {path} has incompatible input variables.")
    encoder = load_sfno_encoder(cfg, device) if encoder is None else encoder
    expected = sfno_checkpoint_metadata(encoder, architecture)
    for key in ("embedding_channels", "embedding_resolution", "pooling", "pool_grid", "feature_dim"):
        if metadata.get(key) != expected[key]:
            raise ValueError(
                f"SFNO probe checkpoint {path} has {key}={metadata.get(key)!r}; "
                f"configured encoder expects {expected[key]!r}."
            )
    unfreeze_final_encoder_stage = bool(metadata.get("unfreeze_final_encoder_stage", False))
    model = build_sfno_probe(
        encoder, architecture, cfg, unfreeze_final_encoder_stage=unfreeze_final_encoder_stage,
    ).to(device)
    if unfreeze_final_encoder_stage:
        block_state = payload.get("last_encoder_block_state_dict")
        projection_state = payload.get("channel_down_scaling_state_dict")
        if block_state is None or projection_state is None:
            raise ValueError(f"Finetuned SFNO checkpoint {path} is missing final encoder-stage weights.")
        sfno_last_encoder_block(model.encoder).load_state_dict(block_state)
        sfno_channel_projection(model.encoder).load_state_dict(projection_state)
    model.head.load_state_dict(payload["head_state_dict"])
    return model.eval(), metadata


def real_term(model, train, variables, means, stds, device, maximum, batch_size, *, progress_description=None, selected_indices=None):
    logits=logits_for(model,train,variables,means,stds,device,maximum,batch_size,progress_description=progress_description,selected_indices=selected_indices)
    v=-np.exp(np.minimum(-logits, 80)); return float(v.mean()), float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else 0.


def compatible(c, variables):
    return not (c in {'wind_patch_shuffle','wind_rotation'} and not {'10m_u_component_of_wind','10m_v_component_of_wind'}.issubset(variables))


def target_specs(cfg, variables):
    forecasts = []
    for label, configured_paths in get(cfg, "forecast_files").items():
        paths = [str(path) for path in configured_paths]
        missing = [path for path in paths if not Path(path).exists()]
        if missing:
            print(f"Skipping {label}: unavailable forecast files: {missing}")
            continue
        forecasts.append(("forecast", label, paths, None))
    corruptions = [
        ("corruption", str(name), None, str(name))
        for name in cfg.baseline.corruptions if compatible(str(name), variables)
    ]
    return forecasts + corruptions


def open_model_forecasts(paths):
    datasets = [normalize_prediction_timedelta(safe_open_dataset(path)) for path in paths]
    return concatenate_forecasts(datasets)


def matched_statistics(real, records, variables):
    if not records:
        raise ValueError("No forecast/ERA5 training pairs remain after monthly valid-time selection.")
    matched = real.isel(time=[record.era5_index for record in records])
    means = {variable: float(matched[variable].mean()) for variable in variables}
    stds = {variable: max(float(matched[variable].std()), 1e-8) for variable in variables}
    return means, stds


def write_target_train_test_metrics(root, records):
    """Persist one row per trained target/probe for train-versus-test comparison."""
    path = Path(root) / "data" / "target_train_test_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "architecture", "kind", "target", "train_loss", "train_accuracy",
        "train_n_samples", "test_loss", "test_accuracy", "test_n_samples",
        "path", "run_url",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved target train/test metrics to: {path}")
    return path


def train_target_discriminator_baselines(cfg, tracker=None):
    """Train configured target critics, optionally using one tracked run per target."""
    configure_plot_bundle_saving_from_cfg(cfg)
    torch.manual_seed(int(get(cfg,'seed'))); np.random.seed(int(get(cfg,'seed')))
    device = target_device(cfg)
    variables=list(get(cfg,'variables')); root=Path(str(get(cfg,'output_dir'))); root.mkdir(parents=True,exist_ok=True)
    resolved_path = root/'resolved_config.yaml'; OmegaConf.save(cfg,resolved_path,resolve=True)
    real=safe_open_dataset(cfg.real_nc_file)
    corruption_train=select_era5_split(real,cfg,"train",coverage="corruption")
    corruption_means={v:float(corruption_train[v].mean()) for v in variables}
    corruption_stds={v:max(float(corruption_train[v].std()),1e-8) for v in variables}
    checkpoint_root = Path(str(get(cfg, "checkpoint_dir") or (root / "models" / "target_discriminators")))
    outputs = []
    interpretability_rows = []
    representation_ratio_rows = []
    pipeline = cfg.get("pipeline", {}) or {}
    wandb_settings = pipeline.get("wandb", {}) or {}
    log_every = int(wandb_settings.get("log_every_n_steps", 20))
    upload = bool(wandb_settings.get("upload_checkpoints", True))
    render_diagnostics = bool(get(cfg, "render_diagnostics", True))
    diagnostics_root = Path(str(get(cfg, "diagnostics_output_dir", root)))

    def tracked_context(architecture, kind, label):
        if tracker is None:
            return nullcontext(None)
        schedule = active_schedule(cfg)
        resample_id = None if schedule is None else schedule.resample_id
        metadata = {
            "architecture": architecture, "target_kind": kind, "target": label,
            "variables": list(SFNO_VARIABLES if architecture == "sfno" else variables),
        }
        if resample_id is not None:
            metadata["temporal_resample_id"] = resample_id
        run_name = f"train/{architecture}/{kind}/{label}"
        if resample_id is not None:
            run_name = f"train/{resample_id}/{architecture}/{kind}/{label}"
        return tracker.run(
            run_name, "discriminator-training", cfg,
            metadata=metadata,
            tags=[architecture, kind, str(label), *([] if resample_id is None else [resample_id])],
        )

    def evaluate_target(model, architecture, kind, label, fake, corruption, means, stds, run, equator_mask_degrees=0.0):
        model.histogram_target = label
        test_real, test_fake, test_records = target_test_inputs(real, fake, cfg, corruption)
        maximum = int(get(cfg, "max_eval_samples", 0))
        interpretability = get(cfg, "interpretability", {}) or {}
        supports_attribution = architecture in {"squeezenet", "squeezenet_attention", "squeezenet_equator_mask", "sfno_linear", "sfno_mlp"}
        attribution_enabled = (render_diagnostics and bool(interpretability.get("enabled", True))
                               and supports_attribution)
        attribution_seed = int(interpretability.get("seed", get(cfg, "seed", 0)))
        random_count = int(interpretability.get("random_samples_per_class", 2)) if attribution_enabled else 0
        if bool(getattr(model, "expects_raw_fields", False)):
            dataset = SFNOTargetDataset(
                test_real, test_fake, model.encoder, cfg.lead_times, corruption, maximum,
                float(get(cfg, "corruption_power")),
                target_corruption_max(cfg, corruption) if corruption else float(get(cfg, "corruption_severity_max")),
                cfg, test_records, deterministic_seed=attribution_seed, target_label=label,
            )
        else:
            dataset = BalancedTargetDataset(
                test_real, test_fake, variables, means, stds, cfg.lead_times, corruption, maximum,
                float(get(cfg, "corruption_power")),
                target_corruption_max(cfg, corruption) if corruption else float(get(cfg, "corruption_severity_max")),
                cfg, test_records, deterministic_seed=attribution_seed,
                equator_mask_degrees=equator_mask_degrees, target_label=label,
            )
        test_metrics = binary_classification_metrics(
            model, dataset, device, int(get(cfg, "batch_size")),
            f"Testing {architecture} {kind}/{label}",
            random_samples_per_class=random_count, seed=attribution_seed,
        )
        cases = test_metrics.pop("cases", [])
        train_metrics = dict(getattr(model, "target_train_metrics", {}))
        record = {
            "architecture": architecture, "kind": kind, "target": label,
            "train_loss": train_metrics.get("loss", float("nan")),
            "train_accuracy": train_metrics.get("accuracy", float("nan")),
            "train_n_samples": train_metrics.get("n_samples", 0),
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_n_samples": test_metrics["n_samples"],
        }
        if run is not None:
            prefix = "" if architecture.startswith("squeezenet") else f"{architecture}/"
            run.log({
                f"{prefix}train/final_loss": record["train_loss"],
                f"{prefix}train/final_accuracy": record["train_accuracy"],
                f"{prefix}test/loss": record["test_loss"],
                f"{prefix}test/accuracy": record["test_accuracy"],
                f"{prefix}test/n_samples": record["test_n_samples"],
            })
            run.summary[f"{prefix}test/loss"] = record["test_loss"]
            run.summary[f"{prefix}test/accuracy"] = record["test_accuracy"]
            if corruption:
                run.summary["corruption/severity_sampling"] = target_corruption_sampling_mode(cfg)
                run.summary["corruption/fake_severity_levels"] = target_fake_severity_levels(
                    cfg, corruption,
                ).tolist()
        logit_histogram_paths = []
        if render_diagnostics:
            logit_histogram_paths = plot_target_test_logit_histograms(
                model, architecture, kind, label, test_real, test_fake, test_records, corruption,
                variables, means, stds, cfg, device, maximum, int(get(cfg, "batch_size")),
                output_root=diagnostics_root,
            )
        plot_paths = list(logit_histogram_paths)
        record["test_logit_histograms"] = [str(path) for path in logit_histogram_paths]
        if run is not None:
            tracker.log_images(run, logit_histogram_paths, diagnostics_root / "plots")
            run.summary["test/logit_histograms"] = len(logit_histogram_paths)
        ratio_settings = (sfno_settings(cfg).get("representation_ratio", {}) or {})
        if bool(getattr(model, "expects_raw_fields", False)) and bool(ratio_settings.get("enabled", True)):
            ratios = sfno_representation_ratio_rows(
                model, test_real, test_fake, test_records, corruption, cfg, device,
                int(ratio_settings.get("evaluation_samples", maximum)),
                min(int(get(cfg, "batch_size")), int(ratio_settings.get("batch_size", 8))),
            )
            for ratio in ratios:
                ratio.update({"architecture": architecture, "kind": kind, "target": label})
            representation_ratio_rows.extend(ratios)
            ratio_path = None
            if render_diagnostics:
                ratio_path = plot_sfno_representation_ratio(
                    ratios, architecture, kind, label,
                    diagnostics_root / "plots" / "sfno_representation_ratio" / architecture / kind /
                    f"{safe_target_name(label)}.png",
                )
            record["sfno_representation_ratios"] = ratios
            record["sfno_representation_ratio_plot"] = str(ratio_path) if ratio_path else ""
            if run is not None:
                for ratio in ratios:
                    coordinate = ratio["severity"] if ratio["severity"] is not None else ratio["lead_hour"]
                    run.log({
                        "sfno/representation_layer": ratio["representation_layer"],
                        "sfno/representation_ratio": ratio["r_corr"],
                        "sfno/representation_candidate_distance": ratio["candidate_distance"],
                        "sfno/representation_reference_distance": ratio["reference_distance"],
                        "sfno/representation_coordinate": coordinate,
                    })
                run.summary["sfno/representation_ratio_points"] = len(ratios)
                if ratio_path is not None:
                    tracker.log_images(run, [ratio_path], diagnostics_root / "plots")
                    plot_paths.append(ratio_path)
        if attribution_enabled:
            gallery_path = (diagnostics_root / "plots" / "target_interpretability" / architecture / kind /
                            f"{safe_target_name(label)}_integrated_gradients.png")
            try:
                input_variables = list(getattr(model, "input_variables", variables))
                rows = create_interpretability_gallery(
                    model, cases, dataset, input_variables, means, stds, interpretability,
                    device, gallery_path, architecture, kind, label,
                )
                interpretability_rows.extend(rows)
                record["interpretability_gallery"] = str(gallery_path)
                plot_paths.append(gallery_path)
                residuals = [abs(float(row["completeness_residual"])) for row in rows]
                if run is not None:
                    tracker.log_images(run, [gallery_path], diagnostics_root / "plots")
                    tracker.log_records_table(run, "interpretability/cases", rows)
                    run.summary["interpretability/status"] = "completed"
                    run.summary["interpretability/n_cases"] = len(rows)
                    run.summary["interpretability/max_abs_completeness_residual"] = max(residuals)
                print(f"Saved target interpretability gallery to: {gallery_path}")
                if (bool(getattr(model, "expects_raw_fields", False)) and
                        bool(interpretability.get("sfno_representation_maps", True))):
                    representation_path = gallery_path.with_name(
                        gallery_path.name.replace("_integrated_gradients.png", "_representation_magnitudes.png")
                    )
                    create_sfno_representation_magnitude_gallery(
                        model, cases, dataset, representation_path, architecture, kind, label,
                    )
                    record["sfno_representation_magnitude_gallery"] = str(representation_path)
                    plot_paths.append(representation_path)
                    if run is not None:
                        tracker.log_images(run, [representation_path], diagnostics_root / "plots")
                    print(f"Saved SFNO representation magnitude gallery to: {representation_path}")
            except Exception as error:
                print(f"Interpretability failed for {architecture} {kind}/{label}: {error}")
                if run is not None:
                    run.summary["interpretability/status"] = "failed"
                    run.summary["interpretability/error"] = str(error)
        elif run is not None:
            run.summary["interpretability/status"] = "not_applicable"
        if run is not None and tracker is not None and upload:
            bundle_paths = [
                member for path in plot_paths for member in all_plot_bundle_paths(path)
                if member.is_file()
            ]
            tracker.log_artifact(
                run, f"target-test-plots-{architecture}-{kind}-{label}", "plots", bundle_paths,
            )
        return record

    if bool(get(cfg, "train_squeezenet", True)):
        targets = target_specs(cfg, variables)
        for kind,label,paths,corruption in tqdm(targets, desc="Training SqueezeNet targets"):
            with tracked_context("squeezenet", kind, label) as run:
                fake = corruption_train if corruption else open_model_forecasts(paths)
                records = None if corruption else forecast_pairs(fake, real, cfg, "train", cfg.lead_times)
                means, stds = (
                    (corruption_means, corruption_stds)
                    if corruption else matched_statistics(real, records, variables)
                )
                model=train_target(
                    real if not corruption else corruption_train,fake,variables,means,stds,cfg,device,corruption=corruption,label=label,
                    metric_logger=None if run is None else run.log, log_every_n_steps=log_every,
                    paired_records=records,
                )
                out=checkpoint_root/kind/label.replace(' ','_'); out.mkdir(parents=True,exist_ok=True)
                output_path=out/'model.pth'; torch.save(model.model.state_dict(),output_path); write_binding(output_path, cfg)
                print(f"Saved SqueezeNet {kind} target discriminator: {output_path}")
                if tracker is not None and upload:
                    tracker.log_artifact(run, f"target-discriminator-squeezenet-{kind}-{label}", "model", [output_path, binding_path(output_path), resolved_path])
                summary = evaluate_target(model, "squeezenet", kind, label, fake, corruption, means, stds, run)
                summary.update(path=str(output_path), run_url=getattr(run, "url", None))
                outputs.append(summary)
                if not corruption: fake.close()
    masked = get(cfg, "equator_masked_hemisphere_splice", {}) or {}
    if bool(masked.get("enabled", True)):
        label, corruption = "hemisphere_splice", "hemisphere_splice"
        half_width = float(masked.get("half_width_degrees", 10.0))
        with tracked_context("squeezenet_equator_mask", "corruption", label) as run:
            model = train_target(
                corruption_train, corruption_train, variables, corruption_means, corruption_stds,
                cfg, device, corruption=corruption, label=label, model_name="squeezenet",
                metric_logger=None if run is None else run.log, log_every_n_steps=log_every,
                equator_mask_degrees=half_width,
            )
            model.equator_mask_degrees = half_width
            out = checkpoint_root / "squeezenet_equator_mask" / "corruption" / label
            out.mkdir(parents=True, exist_ok=True)
            output_path = out / "model.pth"; torch.save(model.model.state_dict(), output_path); write_binding(output_path, cfg)
            summary = evaluate_target(model, "squeezenet_equator_mask", "corruption", label,
                                      corruption_train, corruption, corruption_means, corruption_stds,
                                      run, equator_mask_degrees=half_width)
            summary.update(path=str(output_path), run_url=getattr(run, "url", None)); outputs.append(summary)

    if bool(get(cfg, "train_attention_squeezenet", False)):
        targets = target_specs(cfg, variables)
        for kind,label,paths,corruption in tqdm(
            targets, desc="Training attention-SqueezeNet targets"
        ):
            with tracked_context("squeezenet_attention", kind, label) as run:
                fake = corruption_train if corruption else open_model_forecasts(paths)
                records = None if corruption else forecast_pairs(fake, real, cfg, "train", cfg.lead_times)
                means, stds = (
                    (corruption_means, corruption_stds)
                    if corruption else matched_statistics(real, records, variables)
                )
                model=train_target(
                    real if not corruption else corruption_train, fake, variables, means, stds, cfg, device,
                    corruption=corruption, label=label, model_name="squeezenet_attention",
                    metric_logger=None if run is None else run.log, log_every_n_steps=log_every,
                    paired_records=records,
                )
                out=checkpoint_root/'squeezenet_attention'/kind/label.replace(' ','_'); out.mkdir(parents=True,exist_ok=True)
                output_path=out/'model.pth'; torch.save(model.model.state_dict(),output_path); write_binding(output_path, cfg)
                print(f"Saved attention-SqueezeNet {kind} target discriminator: {output_path}")
                if tracker is not None and upload:
                    tracker.log_artifact(run, f"target-discriminator-squeezenet-attention-{kind}-{label}", "model", [output_path, binding_path(output_path), resolved_path])
                summary = evaluate_target(model, "squeezenet_attention", kind, label, fake, corruption, means, stds, run)
                summary.update(path=str(output_path), run_url=getattr(run, "url", None))
                outputs.append(summary)
                if not corruption: fake.close()
    if bool(sfno_settings(cfg).get("enabled", True)):
        missing = [variable for variable in SFNO_VARIABLES if variable not in corruption_train.data_vars]
        if missing:
            raise ValueError(f"ERA5 is missing required SFNO variables: {missing}")
        try:
            encoder = load_sfno_encoder(cfg, device)
        except FileNotFoundError as error:
            print(f"Skipping optional SFNO target training: {error}")
            write_target_train_test_metrics(root, outputs)
            write_split_membership(cfg, real, root)
            write_interpretability_cases(root, interpretability_rows)
            write_sfno_representation_ratios(root, representation_ratio_rows)
            real.close()
            return outputs
        targets = target_specs(cfg, SFNO_VARIABLES)
        finetune_final_encoder_stage = bool(sfno_settings(cfg).get("unfreeze_final_encoder_stage", False))
        for kind,label,paths,corruption in tqdm(targets, desc="Training SFNO probe targets"):
            target_encoder = load_sfno_encoder(cfg, device) if finetune_final_encoder_stage else encoder
            with tracked_context("sfno", kind, label) as run:
                fake = corruption_train if corruption else open_model_forecasts(paths)
                records = None if corruption else forecast_pairs(fake, real, cfg, "train", cfg.lead_times)
                probes = train_sfno_target(
                    real if not corruption else corruption_train, fake, target_encoder, cfg, device, corruption=corruption, label=label,
                    metric_logger=None if run is None else run.log, log_every_n_steps=log_every,
                    paired_records=records,
                )
                paths = []
                for architecture, model in probes.items():
                    output_path = checkpoint_root/architecture/kind/label.replace(' ','_')/'model.pth'
                    save_sfno_probe_checkpoint(model, output_path); write_binding(output_path, cfg); paths.append(output_path)
                    print(f"Saved {architecture} {kind} target discriminator: {output_path}")
                    summary = evaluate_target(model, architecture, kind, label, fake, corruption, {}, {}, run)
                    summary.update(path=str(output_path), run_url=getattr(run, "url", None))
                    outputs.append(summary)
                if tracker is not None and upload:
                    tracker.log_artifact(run, f"target-discriminator-sfno-{kind}-{label}", "model", [*paths, *[binding_path(path) for path in paths], resolved_path])
                if not corruption: fake.close()
    write_target_train_test_metrics(root, outputs)
    write_split_membership(cfg, real, root)
    write_interpretability_cases(root, interpretability_rows)
    write_sfno_representation_ratios(root, representation_ratio_rows)
    real.close()
    return outputs


@hydra.main(version_base=None, config_path='../conf', config_name='target_discriminator_baselines')
def main(cfg: DictConfig):
    train_target_discriminator_baselines(cfg)

if __name__=='__main__': main()

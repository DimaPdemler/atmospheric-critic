"""Fit leakage-free per-target affine fake-distribution corrections."""

import csv

import numpy as np
import torch

try:
    from .moment_matching import MomentMapStore, RunningMoments, artifact_dir, enabled, map_id
    from .monthly_split import evenly_spaced_pairs, forecast_pairs, select_era5_split
    from .plot_standard_metric_baselines import (
        apply_special_baseline_corruption, corruption_levels, corruption_sample_seed,
    )
    from .train_discriminator import apply_configured_corruption, safe_open_dataset
    from .train_target_discriminator_baselines import (
        SPECIAL_CORRUPTIONS, SFNO_VARIABLES, DATA_DEPENDENT_CORRUPTIONS,
        apply_sfno_corruption, compatible, data_dependent_donor_positions, fields, indices,
        load_sfno_encoder, open_model_forecasts, raw_donor_fields, raw_fields,
        standardized_donor_fields, target_corruption_max, target_specs, target_device,
    )
except ImportError:
    from moment_matching import MomentMapStore, RunningMoments, artifact_dir, enabled, map_id
    from monthly_split import evenly_spaced_pairs, forecast_pairs, select_era5_split
    from plot_standard_metric_baselines import (
        apply_special_baseline_corruption, corruption_levels, corruption_sample_seed,
    )
    from train_discriminator import apply_configured_corruption, safe_open_dataset
    from train_target_discriminator_baselines import (
        SPECIAL_CORRUPTIONS, SFNO_VARIABLES, DATA_DEPENDENT_CORRUPTIONS,
        apply_sfno_corruption, compatible, data_dependent_donor_positions, fields, indices,
        load_sfno_encoder, open_model_forecasts, raw_donor_fields, raw_fields,
        standardized_donor_fields, target_corruption_max, target_specs, target_device,
    )


def _accumulate(iterable):
    accumulator = RunningMoments()
    for values in iterable:
        accumulator.update(values)
    return accumulator


def _add(store, family, kind, target_name, coordinate, variable, source, target):
    key = map_id(family, kind, target_name, coordinate, variable)
    store.add_moments(key, source, target, metadata={
        "family": family, "kind": kind, "target": str(target_name),
        "coordinate": float(coordinate), "variable": variable,
    })


def fit_moment_matching_maps(cfg):
    """Fit ordinary pooled-pixel moments using only the active training split."""
    if not enabled(cfg):
        return []
    if bool((cfg.get("histogram_matching", {}) or {}).get("enabled", False)):
        raise ValueError("Histogram matching and moment matching are mutually exclusive.")
    settings = cfg.moment_matching
    max_fields = int(settings.get("max_fit_fields", 1000))
    seed = int(settings.get("seed", 0))
    store = MomentMapStore()
    real = safe_open_dataset(cfg.real_nc_file)
    train = select_era5_split(real, cfg, "train", coverage="corruption")
    standard_variables = list(cfg.baseline.variables)
    all_variables = [variable for variable in SFNO_VARIABLES if variable in train]
    normalization_variables = list(dict.fromkeys([*standard_variables, *all_variables]))
    means = {variable: float(train[variable].mean()) for variable in normalization_variables}
    stds = {variable: max(float(train[variable].std()), 1e-8) for variable in normalization_variables}

    # Forecast maps use exact valid-time paired ERA5 training fields.
    for _, label, paths, corruption in target_specs(cfg, standard_variables):
        if corruption:
            continue
        fake = open_model_forecasts(paths)
        records = forecast_pairs(fake, real, cfg, "train", cfg.lead_times)
        for lead_hour in sorted({record.lead_hour for record in records}):
            selected = evenly_spaced_pairs(
                [record for record in records if record.lead_hour == lead_hour], max_fields,
            )
            for variable in [v for v in normalization_variables if v in fake and v in real]:
                source = _accumulate(
                    raw_fields(fake, [variable], record.forecast_index, record.lead_index)[0]
                    for record in selected
                )
                target = _accumulate(
                    raw_fields(real, [variable], record.era5_index)[0] for record in selected
                )
                if variable in standard_variables:
                    _add(store, "standard", "forecast", label, lead_hour, variable, source, target)
                _add(store, "sfno", "forecast", label, lead_hour, variable, source, target)
        fake.close()

    selected = indices(train, max_fields)
    for corruption in [str(value) for value in cfg.baseline.corruptions
                       if compatible(str(value), standard_variables)]:
        maximum = target_corruption_max(cfg, corruption)
        donor_positions = (
            data_dependent_donor_positions(corruption, len(selected), len(standard_variables), seed)
            if corruption in DATA_DEPENDENT_CORRUPTIONS else None
        )
        for severity in [value for value in corruption_levels(corruption, cfg) if float(value) != 0.0]:
            source = {variable: RunningMoments() for variable in standard_variables}
            target = {variable: RunningMoments() for variable in standard_variables}
            for position, index in enumerate(selected):
                sample = fields(train, standard_variables, means, stds, int(index))
                donor = None
                if donor_positions is not None:
                    donor = standardized_donor_fields(
                        train, standard_variables, means, stds, selected, donor_positions, position
                    )
                if corruption in SPECIAL_CORRUPTIONS:
                    corrupted = apply_special_baseline_corruption(
                        sample.numpy(), corruption, float(severity), train.latitude.values, cfg,
                        donor, maximum_severity=maximum,
                        random_seed=corruption_sample_seed(seed, corruption, int(index)),
                    )
                else:
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(corruption_sample_seed(seed, corruption, int(index)))
                        corrupted = apply_configured_corruption(sample, corruption, float(severity)).numpy()
                for channel, variable in enumerate(standard_variables):
                    source[variable].update(corrupted[channel] * stds[variable] + means[variable])
                    target[variable].update(raw_fields(train, [variable], int(index))[0])
            for variable in standard_variables:
                _add(store, "standard", "corruption", corruption, severity, variable,
                     source[variable], target[variable])

    # Fit the equivalent physical-input maps for optional SFNO probes.
    sfno = cfg.target_discriminator.get("sfno", {}) or {}
    if bool(sfno.get("enabled", True)):
        try:
            encoder = load_sfno_encoder(cfg, target_device(cfg))
        except FileNotFoundError as error:
            print(f"Skipping optional SFNO moment maps: {error}")
            encoder = None
        if encoder is not None:
            use_context = bool(sfno.get("use_era5_context_for_non_target_fields", False))
            target_variables = list(sfno.get("target_variables", ["2m_temperature"]))
            mapped_variables = target_variables if use_context else all_variables
            for corruption in [str(value) for value in cfg.baseline.corruptions
                               if compatible(str(value), all_variables)]:
                maximum = target_corruption_max(cfg, corruption)
                donor_positions = (
                    data_dependent_donor_positions(corruption, len(selected), len(SFNO_VARIABLES), seed)
                    if corruption in DATA_DEPENDENT_CORRUPTIONS else None
                )
                for severity in [value for value in corruption_levels(corruption, cfg)
                                 if float(value) != 0.0]:
                    source = {variable: RunningMoments() for variable in mapped_variables}
                    target = {variable: RunningMoments() for variable in mapped_variables}
                    for position, index in enumerate(selected):
                        sample = raw_fields(train, SFNO_VARIABLES, int(index))
                        donor = None
                        if donor_positions is not None:
                            donor = raw_donor_fields(
                                train, SFNO_VARIABLES, selected, donor_positions, position
                            )
                        corrupted = apply_sfno_corruption(
                            sample, encoder, corruption, float(severity), train.latitude.values,
                            cfg, donor, maximum_severity=maximum,
                            random_seed=corruption_sample_seed(seed, corruption, int(index)),
                            target_variables=target_variables if use_context else None,
                        )
                        for variable in mapped_variables:
                            channel = SFNO_VARIABLES.index(variable)
                            source[variable].update(corrupted[channel])
                            target[variable].update(sample[channel])
                    for variable in mapped_variables:
                        _add(store, "sfno", "corruption", corruption, severity, variable,
                             source[variable], target[variable])

    output = artifact_dir(cfg, writing=True)
    paths = list(store.save(output, config={
        "max_fit_fields": max_fields, "seed": seed,
        "train_days": list(cfg.monthly_split.train_days),
        "weighting": "ordinary pooled grid cells",
    }))
    summary = output / "fit_summary.csv"
    if store.metadata:
        with open(summary, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(store.metadata[0]))
            writer.writeheader(); writer.writerows(store.metadata)
        paths.append(summary)
    real.close()
    print(f"Saved moment-matching maps to: {output}")
    return paths

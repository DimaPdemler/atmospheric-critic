"""Fit pipeline-wide leakage-free histogram-matching artifacts."""

import csv

import numpy as np
import torch

try:
    from .histogram_matching import HistogramMapStore, artifact_dir, deterministic_subsample, enabled
    from .monthly_split import evenly_spaced_pairs, forecast_pairs, select_era5_split
    from .plot_standard_metric_baselines import (
        apply_special_baseline_corruption, corruption_levels, corruption_sample_seed, deranged_sample_positions,
    )
    from .train_discriminator import apply_configured_corruption, safe_open_dataset
    from .train_target_discriminator_baselines import (
        SPECIAL_CORRUPTIONS, SFNO_VARIABLES, DATA_DEPENDENT_CORRUPTIONS, data_dependent_donor_positions, raw_donor_fields, standardized_donor_fields, apply_sfno_corruption, compatible, fields, indices,
        load_sfno_encoder, open_model_forecasts, raw_fields, target_corruption_max,
        target_corruption_sampling_mode, target_fake_severity_levels, target_specs, target_device,
    )
except ImportError:
    from histogram_matching import HistogramMapStore, artifact_dir, deterministic_subsample, enabled
    from monthly_split import evenly_spaced_pairs, forecast_pairs, select_era5_split
    from plot_standard_metric_baselines import (
        apply_special_baseline_corruption, corruption_levels, corruption_sample_seed, deranged_sample_positions,
    )
    from train_discriminator import apply_configured_corruption, safe_open_dataset
    from train_target_discriminator_baselines import (
        SPECIAL_CORRUPTIONS, SFNO_VARIABLES, DATA_DEPENDENT_CORRUPTIONS, data_dependent_donor_positions, raw_donor_fields, standardized_donor_fields, apply_sfno_corruption, compatible, fields, indices,
        load_sfno_encoder, open_model_forecasts, raw_fields, target_corruption_max,
        target_corruption_sampling_mode, target_fake_severity_levels, target_specs, target_device,
    )


def pooled(fields_iterable, maximum, seed):
    fields_list = list(fields_iterable)
    if not fields_list:
        return np.asarray([], dtype=np.float64)
    per_field = 0 if maximum <= 0 else max(1, int(np.ceil(maximum / len(fields_list))))
    pieces = [deterministic_subsample(field, per_field, seed + index)
              for index, field in enumerate(fields_list)]
    return deterministic_subsample(np.concatenate(pieces), maximum, seed)


def fit_histogram_matching_maps(cfg):
    if not enabled(cfg):
        return []
    if target_corruption_sampling_mode(cfg) != "discrete_uniform":
        raise ValueError(
            "Per-severity histogram matching requires "
            "target_discriminator.corruption_severity_sampling=discrete_uniform."
        )
    settings = cfg.histogram_matching
    max_values = int(settings.get("max_fit_values_per_variable", 1_000_000))
    max_fields = int(settings.get("max_fit_fields", 256))
    knots = int(settings.get("quantile_knots", 2049))
    seed = int(settings.get("seed", 0))
    store = HistogramMapStore()
    real = safe_open_dataset(cfg.real_nc_file)
    train = select_era5_split(real, cfg, "train", coverage="corruption")
    standard_variables = list(cfg.baseline.variables)
    all_variables = [variable for variable in SFNO_VARIABLES if variable in train]
    normalization_variables = list(dict.fromkeys([*standard_variables, *all_variables]))
    means = {variable: float(train[variable].mean()) for variable in normalization_variables}
    stds = {variable: max(float(train[variable].std()), 1e-8) for variable in all_variables}

    def add(family, kind, target_name, coordinate, variable, source_fields, target_fields):
        source = pooled(source_fields, max_values, seed)
        target = pooled(target_fields, max_values, seed + 104729)
        key = "|".join((family, kind, str(target_name), format(float(coordinate), ".12g"), variable))
        store.add(key, source, target, knots=knots, metadata={
            "family": family, "kind": kind, "target": str(target_name),
            "coordinate": float(coordinate), "variable": variable,
        })

    # Physical forecast maps are fitted against exact paired ERA5 valid times.
    for _, label, paths, corruption in target_specs(cfg, standard_variables):
        if corruption:
            continue
        fake = open_model_forecasts(paths)
        records = forecast_pairs(fake, real, cfg, "train", cfg.lead_times)
        for lead_hour in sorted({record.lead_hour for record in records}):
            selected = evenly_spaced_pairs(
                [record for record in records if record.lead_hour == lead_hour], max_fields,
            )
            forecast_variables = [variable for variable in normalization_variables
                                  if variable in fake and variable in real]
            for variable in forecast_variables:
                source = (raw_fields(fake, [variable], record.forecast_index, record.lead_index)[0]
                          for record in selected)
                target = (raw_fields(real, [variable], record.era5_index)[0] for record in selected)
                values_source, values_target = list(source), list(target)
                if variable in standard_variables:
                    add("standard", "forecast", label, lead_hour, variable,
                        values_source, values_target)
                add("sfno", "forecast", label, lead_hour, variable, values_source, values_target)
        fake.close()

    selected = indices(train, max_fields)
    for corruption in [str(value) for value in cfg.baseline.corruptions
                       if compatible(str(value), standard_variables)]:
        maximum = target_corruption_max(cfg, corruption)
        donor_positions = (
            data_dependent_donor_positions(
                corruption, len(selected), len(standard_variables), seed
            )
            if corruption in DATA_DEPENDENT_CORRUPTIONS else None
        )
        for severity in [value for value in corruption_levels(corruption, cfg) if float(value) != 0.0]:
            sources = {variable: [] for variable in standard_variables}
            targets = {variable: [] for variable in standard_variables}
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
                    sources[variable].append(corrupted[channel] * stds[variable] + means[variable])
                    targets[variable].append(raw_fields(train, [variable], int(index))[0])
            for variable in standard_variables:
                add("standard", "corruption", corruption, severity, variable,
                    sources[variable], targets[variable])

    # SFNO keeps its existing encoder-normalized corruption construction.
    sfno = cfg.target_discriminator.get("sfno", {}) or {}
    if bool(sfno.get("enabled", True)):
        try:
            encoder = load_sfno_encoder(cfg, target_device(cfg))
        except FileNotFoundError as error:
            print(f"Skipping optional SFNO histogram maps: {error}")
            encoder = None
        if encoder is not None:
            use_context = bool(sfno.get("use_era5_context_for_non_target_fields", False))
            target_variables = list(sfno.get("target_variables", ["2m_temperature"]))
            mapped_variables = target_variables if use_context else all_variables
            for corruption in [str(value) for value in cfg.baseline.corruptions
                               if compatible(str(value), all_variables)]:
                maximum = target_corruption_max(cfg, corruption)
                donor_positions = (
                    data_dependent_donor_positions(
                        corruption, len(selected), len(SFNO_VARIABLES), seed
                    )
                    if corruption in DATA_DEPENDENT_CORRUPTIONS else None
                )
                for severity in [value for value in corruption_levels(corruption, cfg) if float(value) != 0.0]:
                    sources = {variable: [] for variable in mapped_variables}
                    targets = {variable: [] for variable in mapped_variables}
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
                            sources[variable].append(corrupted[channel])
                            targets[variable].append(sample[channel])
                    for variable in mapped_variables:
                        add("sfno", "corruption", corruption, severity, variable,
                            sources[variable], targets[variable])

    output = artifact_dir(cfg, writing=True)
    paths = list(store.save(output, config={
        "quantile_knots": knots, "max_fit_values_per_variable": max_values,
        "max_fit_fields": max_fields, "seed": seed,
        "train_days": list(cfg.monthly_split.train_days),
    }))
    summary = output / "fit_summary.csv"
    if store.metadata:
        with open(summary, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(store.metadata[0]))
            writer.writeheader(); writer.writerows(store.metadata)
        paths.append(summary)
    real.close()
    print(f"Saved histogram matching maps to: {output}")
    return paths

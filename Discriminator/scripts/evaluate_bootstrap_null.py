"""Resample the ERA5 null distribution behind the corruption blind-spot table.

The shipped sweep computes one null per metric -- ERA5 days 1-15 against days
16-31 -- and calls a corruption detected when its maximum-severity score clears
that single number.  That null is itself a random quantity: it depends on which
days land in which split, on the evenly spaced subsample, and (for MMD) on the
bandwidth fitted to those samples.  This script replaces it with a resampled
distribution.  Real ERA5 days outside the corruption test window are pooled,
two disjoint sample sets of the sizes the sweep actually uses are drawn many
times, and every metric is scored on each draw.  The upper quantile of that
distribution becomes the detection threshold.

Run it exactly like the other baseline entry points, for example:

    python scripts/evaluate_bootstrap_null.py \
        bootstrap_null.replicates=200 bootstrap_null.pool_samples=3000
"""

import csv
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

try:
    from . import plot_standard_metric_baselines as P
    from .monthly_split import datetime_mask, time_ranges
    from .fake_matching_apply import matching_mode
except ImportError:
    import plot_standard_metric_baselines as P
    from monthly_split import datetime_mask, time_ranges
    from fake_matching_apply import matching_mode


# Everything the blind-spot figure plots (DEFAULT_METRICS minus
# PLOTTING_DISABLED_METRICS), plus the l2 spectrum variant so both spectral and
# both SCWD variants can be compared against the published table. Sliced
# Wasserstein is excluded: it needs the random-projection buffers, which cannot
# be resampled from stored per-sample features.
TABLE_METRICS = [
    "mean_bias",
    "std_ratio_error",
    "zonal_energy_spectrum_l2",
    "zonal_energy_spectrum_log_l2",
    "global_mean_wasserstein",
    "mmd_rbf",
    "scwd",
    "scwd_area_weighted",
]


def bootstrap_get(cfg, key, default):
    settings = cfg.get("bootstrap_null") or {}
    value = settings.get(key)
    return default if value is None else value


def pool_time_indices(cfg, ds, pool_samples):
    """Evenly spaced real-ERA5 times outside the corruption test-day window.

    The sweep's null split (days 16-31) contains its test split (days 20-26), so
    a pool built from the shipped splits would share samples with the corruption
    reference.  Excluding the test days keeps the null independent of it.
    """
    test_lower, test_upper = (int(value) for value in cfg["monthly_split"]["test_days"])
    times = np.asarray(ds.time.values)
    ranges = time_ranges(cfg, "corruption")
    in_range = datetime_mask(times, ranges, (1, 31))
    is_test_day = datetime_mask(times, ranges, (test_lower, test_upper))
    positions = np.flatnonzero(in_range & ~is_test_day)
    if positions.size == 0:
        raise ValueError("No ERA5 samples available outside the corruption test days.")
    if 0 < pool_samples < positions.size:
        positions = positions[np.linspace(0, positions.size - 1, pool_samples, dtype=int)]
    return positions.tolist()


def per_sample_features(
    cfg, ds, variables, reference_stats, time_indices, metric_names,
    corruption_type=None, severity=0.0, donor_positions=None, description="features",
):
    """Extract per-sample metric inputs so any subset can be scored later.

    Mirrors `streaming_joint_features` sample for sample, but keeps every
    quantity unaggregated: the sweep folds spectra and weighted moments into
    running totals, which cannot be resampled afterwards.
    """
    chunk_size = max(1, int(P.baseline_get(cfg, "feature_chunk_size", 32)))
    latitudes = np.asarray(ds[variables[0]].latitude.values, dtype=np.float64)
    longitudes = np.asarray(ds[variables[0]].longitude.values, dtype=np.float64)
    per_variable_pixels = max(1, int(P.baseline_get(cfg, "swd_pixels", 4096)) // len(variables))
    need_scwd = bool({"scwd", "scwd_area_weighted"} & set(metric_names))

    numpy_weights = torch_weights = None
    device = None
    if need_scwd:
        weights = P.scwd_weight_vectors({"latitudes": latitudes, "longitudes": longitudes}, cfg)
        n_pixels = len(latitudes) * len(longitudes)
        device = P.torch_metric_device(cfg)
        if device is None:
            numpy_weights = P.scwd_sparse_weight_matrix_numpy(weights, n_pixels)
        else:
            torch_weights = P.scwd_sparse_weight_matrix_torch(weights, n_pixels, cfg, device)

    base_seed = int(P.baseline_get(cfg, "corruption_seed", 0))
    special = corruption_type in P.STRUCTURED_NEAR_NULL_CORRUPTIONS | P.DATA_DEPENDENT_CORRUPTIONS

    moments, spectra, vectors, global_means, scwd_responses, positions = [], [], [], [], [], []
    for start in tqdm(range(0, len(time_indices), chunk_size), desc=description, leave=False):
        chunk_indices = time_indices[start:start + chunk_size]
        chunk_ds = ds.isel(time=chunk_indices)
        raw_by_variable = {
            variable: np.asarray(
                chunk_ds[variable].transpose("time", "latitude", "longitude").values, dtype=np.float32
            )
            for variable in variables
        }
        donor_raw_by_variable = None
        if donor_positions is not None:
            donor_raw_by_variable = {}
            for channel, variable in enumerate(variables):
                positions_for_field = (
                    donor_positions if donor_positions.ndim == 1 else donor_positions[channel]
                )
                donor_indices = [
                    time_indices[int(positions_for_field[position])]
                    for position in range(start, start + len(chunk_indices))
                ]
                donor_chunk = ds.isel(time=donor_indices)
                donor_raw_by_variable[variable] = np.asarray(
                    donor_chunk[variable].transpose("time", "latitude", "longitude").values,
                    dtype=np.float32,
                )

        valid_fields, valid_positions = [], []
        for local_idx, time_idx in enumerate(chunk_indices):
            raw_fields = [raw_by_variable[variable][local_idx] for variable in variables]
            if any(P.should_skip_field(cfg, values) for values in raw_fields):
                continue
            donor_fields = None
            if donor_raw_by_variable is not None:
                donor_fields = [donor_raw_by_variable[variable][local_idx] for variable in variables]
                if any(P.should_skip_field(cfg, values) for values in donor_fields):
                    continue
            standardized = np.stack([
                (P.canonical_latlon(values, latitudes) - reference_stats[variable]["mean"])
                / reference_stats[variable]["std"]
                for variable, values in zip(variables, raw_fields)
            ]).astype(np.float32)
            if corruption_type is not None and severity > 0.0:
                if special:
                    donor = None
                    if donor_fields is not None:
                        donor = np.stack([
                            (P.canonical_latlon(values, latitudes) - reference_stats[variable]["mean"])
                            / reference_stats[variable]["std"]
                            for variable, values in zip(variables, donor_fields)
                        ]).astype(np.float32)
                    standardized = P.apply_special_baseline_corruption(
                        standardized, corruption_type, severity, latitudes, cfg, donor,
                        random_seed=P.corruption_sample_seed(base_seed, corruption_type, time_idx),
                    )
                else:
                    seed = P.corruption_sample_seed(base_seed, corruption_type, time_idx)
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(seed)
                        standardized = P.apply_configured_corruption(
                            torch.from_numpy(standardized), corruption_type, float(severity)
                        ).detach().cpu().numpy().astype(np.float32)
            valid_fields.append(standardized)
            valid_positions.append(start + local_idx)

        if not valid_fields:
            continue
        field_batch = np.stack(valid_fields)
        global_means.append(P.area_weighted_global_means(field_batch, latitudes))
        for sample_fields in field_batch:
            moments.append(P.pointwise_moment_totals(sample_fields[None]))
            spectra.append(np.concatenate([P.zonal_energy_spectrum(f, latitudes) for f in sample_fields]))
            vectors.append(np.concatenate([P.field_vector(f, latitudes, per_variable_pixels) for f in sample_fields]))
        if need_scwd:
            scwd_responses.append(P.scwd_response_batch(field_batch, numpy_weights, torch_weights, cfg, device))
        positions.extend(valid_positions)

    if not positions:
        raise ValueError(f"No valid fields available for {description}.")
    return {
        "moments": np.asarray(moments, dtype=np.float64),
        "spectra": np.asarray(spectra, dtype=np.float64),
        "vectors": np.asarray(vectors, dtype=np.float32),
        "global_means": np.concatenate(global_means, axis=0).astype(np.float32),
        "scwd_responses": (
            np.concatenate(scwd_responses, axis=0).astype(np.float32)
            if scwd_responses else np.empty((0, 0, 0), dtype=np.float32)
        ),
        "latitudes": latitudes,
        "longitudes": longitudes,
        "mmd_field_slices": [
            (field * per_variable_pixels, (field + 1) * per_variable_pixels)
            for field in range(len(variables))
        ],
        "positions": np.asarray(positions, dtype=int),
    }


def subset_features(pool, rows, pairwise_cap):
    """Assemble a pipeline-shaped feature dict from selected pool rows."""
    rows = np.asarray(rows, dtype=int)
    moments = pool["moments"][rows]
    total, total_sq, mass = moments[:, 0].sum(), moments[:, 1].sum(), moments[:, 2].sum()
    mean = total / mass
    variance = max(total_sq / mass - mean * mean, 0.0)
    pair_rows = rows[P.pairwise_sample_positions(rows.size, pairwise_cap)]
    features = {
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
        "spectrum": pool["spectra"][rows].mean(axis=0),
        "vectors": pool["vectors"][pair_rows],
        "global_means": pool["global_means"][rows],
        "fields": np.empty((0, 0, 0), dtype=np.float32),
        "latitudes": pool["latitudes"],
        "longitudes": pool["longitudes"],
        "mmd_field_slices": pool["mmd_field_slices"],
        "n_valid_samples": int(rows.size),
        "pairwise_n_samples": int(pair_rows.size),
    }
    if pool["scwd_responses"].size:
        features["scwd_responses"] = pool["scwd_responses"][rows]
    return features


def displayed(name, values):
    """Apply the sweep's presentation convention before any comparison.

    `mean_bias` and `std_ratio_error` are signed, and their null straddles zero,
    so an upper quantile of the raw values would be a small positive number that
    a large negative bias would never exceed. `display_metric_value` already
    takes the magnitude for exactly these two; do the same here so thresholds,
    verdicts and ribbons are all on the quantity the figure shows. The CSVs keep
    the raw signed values.
    """
    values = np.asarray(values, dtype=np.float64)
    return np.abs(values) if name in P.ABSOLUTE_DISPLAY_METRICS else values


def score(candidate, reference, metric_names, cfg, dedup_scwd=True):
    """Score one candidate/reference pair, optionally sharing the SCWD costs.

    `scwd` and `scwd_area_weighted` differ only in how they aggregate
    `scwd_anchor_transport_costs`, which is the expensive part and by far the
    dominant cost of a resampling loop. Computing it once and aggregating twice
    halves the scoring time. The cross-check in `main` scores the reference side
    through the pipeline's own `distribution_metric_values`, so a mistake here
    fails the run rather than passing silently.
    """
    requested = list(metric_names)
    pair = [name for name in ("scwd", "scwd_area_weighted") if name in requested]
    shareable = (
        dedup_scwd and len(pair) == 2
        and "scwd_responses" in candidate and "scwd_responses" in reference
    )
    if not shareable:
        return P.distribution_metric_values(candidate, reference, cfg, requested)

    rest = [name for name in requested if name not in pair]
    metrics = P.distribution_metric_values(candidate, reference, cfg, rest) if rest else {}
    costs = P.scwd_anchor_transport_costs(candidate, reference, cfg)
    order = float(P.baseline_get(cfg, "scwd_order", 2.0))
    if costs.size == 0:
        metrics["scwd"] = np.nan
        metrics["scwd_area_weighted"] = np.nan
        return metrics
    weights = P.scwd_anchor_area_weights(cfg)
    if costs.size != weights.size:
        raise ValueError(f"SCWD anchor-weight mismatch: {costs.size} costs, {weights.size} weights.")
    metrics["scwd"] = float(np.mean(costs) ** (1.0 / order))
    metrics["scwd_area_weighted"] = float(np.sum(weights * costs) ** (1.0 / order))
    return metrics


def day_of_month(times):
    """Calendar day of month for each pooled sample."""
    stamps = np.asarray(times).astype("datetime64[ns]")
    days = stamps.astype("datetime64[D]")
    return (days - days.astype("datetime64[M]")).astype(int) + 1


def day_partition_split(days, available, rng, side_size):
    """Split by a random complementary pair of day-of-month sets.

    This is the shipped null's own construction with the arbitrary choice
    randomized.  That null contrasts ERA5 days 1-15 with days 16-31, evenly
    spaced within each side; here the 1-15/16-31 cut is replaced by a random
    half/half cut of the available days, and both sides are still evenly spaced.
    Every replicate is therefore a null the sweep could equally well have
    reported, which is exactly what makes their spread the right yardstick.

    Splitting by day of month also keeps the two sides disjoint while leaving
    their month, year, and hour composition matched -- the same reason the
    original cut works.
    """
    chosen = set(rng.permutation(available)[: available.size // 2].tolist())
    mask = np.array([day in chosen for day in days])
    left = np.flatnonzero(mask)
    right = np.flatnonzero(~mask)
    return trim_evenly(left, side_size), trim_evenly(right, side_size)


def unstratified_split(n_pool, rng, side_size):
    """Plain random split, kept to show what unmatched composition costs."""
    draw = rng.permutation(n_pool)
    return np.sort(draw[:side_size]), np.sort(draw[side_size:2 * side_size])


def trim_evenly(rows, size):
    """Reduce a time-ordered row set to an exact size, preserving its spread.

    Mirrors `sample_time_indices`, which is how the sweep thins every split.
    """
    if rows.size < size:
        raise ValueError(f"Split side has {rows.size} samples, need {size}.")
    if rows.size == size:
        return rows
    return rows[np.linspace(0, rows.size - 1, size, dtype=int)]


def bootstrap_null_output_dir(cfg, variables):
    """Namespaced output directory for the resampled-null experiment."""
    return P.baseline_output_dir(cfg, variables) / "bootstrap_null"


def evaluate_bootstrap_null(cfg):
    """Resample the null, sweep the severity ladder; returns written paths."""
    variables = P.variables_from_config(cfg)
    if matching_mode(cfg) != "none":
        raise ValueError("Bootstrap null evaluation does not yet support fake-distribution matching.")
    metric_names = [m for m in TABLE_METRICS if m in set(P.metric_names_from_config(cfg))]
    replicates = int(bootstrap_get(cfg, "replicates", 200))
    pool_samples = int(bootstrap_get(cfg, "pool_samples", 3000))
    quantile = float(bootstrap_get(cfg, "threshold_quantile", 0.95))
    seed = int(bootstrap_get(cfg, "seed", 0))
    curve_replicates = int(bootstrap_get(cfg, "curve_replicates", 50))
    curve_pool_samples = int(bootstrap_get(cfg, "curve_pool_samples", 2000))
    dedup_scwd = bool(bootstrap_get(cfg, "scwd_dedup", True))
    eval_samples = int(P.baseline_get(cfg, "corruption_eval_samples", 1000))
    pairwise_cap = int(P.baseline_get(cfg, "pairwise_eval_samples", 256))

    output_root = bootstrap_null_output_dir(cfg, variables)
    (output_root / "data").mkdir(parents=True, exist_ok=True)

    real_ds = P.select_level(P.safe_open_dataset(cfg.real_nc_file), cfg.get("level"))
    print(f"metrics={metric_names}\n  null: {replicates} replicates, pool {pool_samples}, "
          f"per-side n={eval_samples} (pairwise {pairwise_cap})\n  curves: {curve_replicates} "
          f"replicates, pool {curve_pool_samples}, scwd_dedup={dedup_scwd}")

    # --- corruption side: exactly the sweep's test split and its statistics ---
    test_ds = P.select_level(P.select_era5_split(real_ds, cfg, "test", coverage="corruption"), cfg.get("level"))
    test_indices = P.sample_time_indices(test_ds, eval_samples)
    test_stats = P.streaming_reference_stats(cfg, test_ds, variables, test_indices)
    test_pool = per_sample_features(cfg, test_ds, variables, test_stats, test_indices,
                                    metric_names, description="clean ERA5 test")
    test_features = subset_features(test_pool, np.arange(test_pool["positions"].size), pairwise_cap)

    # --- cross-check: the pipeline's own feature builder must agree ---
    import tempfile
    with tempfile.TemporaryDirectory(prefix="bootstrap-null-check-") as tmp:
        pipeline_test = P.streaming_joint_features(
            cfg, test_ds, variables, test_stats, test_indices, metric_names,
            Path(tmp) / "check.dat", description="pipeline cross-check",
        )
        blur_max = P.corruption_max_severity("gaussian_blur", cfg)
        pipeline_blur = P.streaming_joint_features(
            cfg, test_ds, variables, test_stats, test_indices, metric_names,
            Path(tmp) / "check-blur.dat", corruption_type="gaussian_blur", severity=blur_max,
            description="pipeline cross-check (blur)",
        )
        # Deliberately the pipeline's own scorer, so this validates the SCWD
        # dedup as well as the feature extraction.
        reference_scores = P.distribution_metric_values(
            pipeline_blur, pipeline_test, cfg, metric_names
        )
        mine_pool = per_sample_features(cfg, test_ds, variables, test_stats, test_indices,
                                        metric_names, corruption_type="gaussian_blur",
                                        severity=blur_max, description="cross-check (blur)")
        mine = score(subset_features(mine_pool, np.arange(mine_pool["positions"].size), pairwise_cap),
                     test_features, metric_names, cfg, dedup_scwd=dedup_scwd)
        print("\ncross-check against the shipped feature builder (gaussian_blur @ max severity):")
        worst = 0.0
        for name in metric_names:
            a, b = float(reference_scores[name]), float(mine[name])
            rel = abs(a - b) / max(abs(a), 1e-30)
            worst = max(worst, rel)
            print(f"  {name:32s} pipeline={a:.6g}  resampler={b:.6g}  rel.diff={rel:.2e}")
        P.close_feature_memmaps(pipeline_test)
        P.close_feature_memmaps(pipeline_blur)
        if worst > 1e-5:
            raise SystemExit(f"Feature extraction diverges from the pipeline (max rel.diff {worst:.2e}).")

    # --- null side: pooled real ERA5 outside the test-day window ---
    pool_indices = pool_time_indices(cfg, real_ds, pool_samples)
    pool_ds = P.select_level(real_ds, cfg.get("level"))
    # Standardize the null pool with the test-split statistics the corruption
    # scores use.  SCWD and the global-mean Wasserstein both scale with the
    # normalizing standard deviation, so a pool-fitted one would put the null on
    # a different scale from the scores it is meant to threshold.
    null_pool = per_sample_features(cfg, pool_ds, variables, test_stats, pool_indices,
                                    metric_names, description="ERA5 null pool")
    n_pool = null_pool["positions"].size
    if n_pool < 2 * eval_samples:
        raise SystemExit(f"Pool has {n_pool} samples, need at least {2 * eval_samples}.")

    pool_times = np.asarray(real_ds.time.values).astype("datetime64[ns]")[
        np.asarray(pool_indices)][null_pool["positions"]]
    pool_days = day_of_month(pool_times)
    available_days = np.unique(pool_days)
    hours = (pool_times.astype("datetime64[h]").astype(np.int64)) % 24
    hour_counts = dict(zip(*(x.tolist() for x in np.unique(hours, return_counts=True))))
    print(f"pool: {n_pool} samples, day-of-month values {available_days.min()}-{available_days.max()} "
          f"({available_days.size} distinct), hour histogram {hour_counts}")

    probe_left, probe_right = day_partition_split(pool_days, available_days,
                                                  np.random.default_rng(seed), eval_samples)
    for name, rows in [("side A", probe_left), ("side B", probe_right)]:
        months = pool_times[rows].astype("datetime64[M]").astype(np.int64) % 12 + 1
        hrs = (pool_times[rows].astype("datetime64[h]").astype(np.int64)) % 24
        print(f"  example {name}: n={rows.size} "
              f"months={np.bincount(months, minlength=13)[1:].tolist()} "
              f"hours={np.bincount(hrs, minlength=24)[::6].tolist()}")

    rng = np.random.default_rng(seed)
    null_rows = []
    for replicate in tqdm(range(replicates), desc="bootstrap null"):
        row = {"replicate": replicate}
        for scheme in ("day_partition", "unstratified"):
            if scheme == "day_partition":
                left, right = day_partition_split(pool_days, available_days, rng, eval_samples)
            else:
                left, right = unstratified_split(n_pool, rng, eval_samples)
            values = score(subset_features(null_pool, left, pairwise_cap),
                           subset_features(null_pool, right, pairwise_cap), metric_names, cfg,
                           dedup_scwd=dedup_scwd)
            row.update({f"{scheme}__{k}": float(values[k]) for k in metric_names})
        null_rows.append(row)

    null_array = {
        scheme: {
            name: displayed(name, [r[f"{scheme}__{name}"] for r in null_rows])
            for name in metric_names
        }
        for scheme in ("day_partition", "unstratified")
    }

    # --- the sweep's own single null, for reference ---
    train_ds = P.select_level(P.select_era5_split(real_ds, cfg, "train", coverage="corruption"), cfg.get("level"))
    shipped_null_ds = P.select_level(P.select_era5_split(real_ds, cfg, "null", coverage="corruption"), cfg.get("level"))
    train_indices = P.sample_time_indices(train_ds, eval_samples)
    shipped_indices = P.sample_time_indices(shipped_null_ds, eval_samples)
    # The sweep fits these on its train split; keep that, so this row reproduces
    # the number the sweep actually reports rather than a rescaled version of it.
    shipped_stats = P.streaming_reference_stats(cfg, train_ds, variables, train_indices)
    train_pool = per_sample_features(cfg, train_ds, variables, shipped_stats, train_indices,
                                     metric_names, description="ERA5 train (shipped null ref)")
    shipped_pool = per_sample_features(cfg, shipped_null_ds, variables, shipped_stats, shipped_indices,
                                       metric_names, description="ERA5 days 16-31 (shipped null)")
    print("normalization check: test std=%.6f  train std=%.6f  (ratio %.5f)" % (
        test_stats[variables[0]]["std"], shipped_stats[variables[0]]["std"],
        test_stats[variables[0]]["std"] / shipped_stats[variables[0]]["std"]))
    shipped_null = score(
        subset_features(shipped_pool, np.arange(shipped_pool["positions"].size), pairwise_cap),
        subset_features(train_pool, np.arange(train_pool["positions"].size), pairwise_cap),
        metric_names, cfg, dedup_scwd=dedup_scwd,
    )

    # --- corruption severity ladder with paired resampling ---
    corruption_types = P.compatible_corruption_types(P.corruption_types_from_config(cfg), variables)
    curve_indices = P.sample_time_indices(test_ds, curve_pool_samples)
    curve_clean = per_sample_features(cfg, test_ds, variables, test_stats, curve_indices,
                                      metric_names, description="clean ERA5 curve pool")
    n_curve = curve_clean["positions"].size
    if n_curve < eval_samples:
        raise SystemExit(f"Curve pool has {n_curve} samples, need at least {eval_samples}.")

    # One set of row subsets, shared by every corruption and severity. A
    # replicate therefore means the same 1000 days everywhere, which keeps the
    # ribbons paired and makes monotonicity evaluable within a replicate.
    curve_rng = np.random.default_rng(seed + 1)
    subsets = [
        np.arange(n_curve) if n_curve == eval_samples
        else np.sort(curve_rng.choice(n_curve, eval_samples, replace=False))
        for _ in range(curve_replicates)
    ]

    zero_reference = subset_features(curve_clean, subsets[0], pairwise_cap)
    zero_values = score(zero_reference, zero_reference, metric_names, cfg, dedup_scwd=dedup_scwd)
    worst_zero = max(abs(float(zero_values[name])) for name in metric_names)
    print(f"severity-0 self-comparison: max |value| = {worst_zero:.3g} (expected exactly 0)")

    curve_rows = []
    ladder_work = [
        (corruption_type, severity)
        for corruption_type in corruption_types
        for severity in P.corruption_levels(corruption_type, cfg)
        if float(severity) > 0.0
    ]
    ladder_progress = tqdm(total=len(ladder_work) * curve_replicates, desc="curve resamples")
    for corruption_type in corruption_types:
        levels = P.corruption_levels(corruption_type, cfg)
        donor_seed = int(P.baseline_get(cfg, "corruption_seed", 0))
        if corruption_type == "hemisphere_splice":
            donor_positions = P.deranged_sample_positions(len(curve_indices), donor_seed)
        elif corruption_type == "field_splice":
            donor_positions = P.fieldwise_deranged_sample_positions(
                len(curve_indices), len(variables), donor_seed
            )
        else:
            donor_positions = None
        for severity in levels:
            if float(severity) == 0.0:
                # Self-comparison; identically zero, verified above.
                curve_rows.extend(
                    {"corruption": corruption_type, "severity": 0.0, "replicate": replicate,
                     **{name: 0.0 for name in metric_names}}
                    for replicate in range(curve_replicates)
                )
                continue
            corrupt = per_sample_features(
                cfg, test_ds, variables, test_stats, curve_indices, metric_names,
                corruption_type=corruption_type, severity=severity,
                donor_positions=donor_positions,
                description=f"{corruption_type} @ {severity:g}",
            )
            if not np.array_equal(corrupt["positions"], curve_clean["positions"]):
                raise SystemExit(
                    f"{corruption_type} @ {severity:g} retained a different sample set than the "
                    "clean pool, so the paired comparison would be misaligned."
                )
            for replicate, rows in enumerate(subsets):
                values = score(subset_features(corrupt, rows, pairwise_cap),
                               subset_features(curve_clean, rows, pairwise_cap),
                               metric_names, cfg, dedup_scwd=dedup_scwd)
                curve_rows.append({"corruption": corruption_type, "severity": float(severity),
                                   "replicate": replicate,
                                   **{name: float(values[name]) for name in metric_names}})
                ladder_progress.update(1)
            del corrupt
    ladder_progress.close()

    # --- verdicts, taken from the mean of the resampled curves ---
    matched = null_array["day_partition"]
    unstratified = null_array["unstratified"]
    thresholds = {name: float(np.quantile(matched[name], quantile)) for name in metric_names}

    by_point = {}
    for row in curve_rows:
        by_point.setdefault((row["corruption"], row["severity"]), []).append(row)

    verdict_rows = []
    for corruption_type in corruption_types:
        ladder = sorted({s for (c, s) in by_point if c == corruption_type and s > 0.0})
        record = {"corruption": corruption_type, "severity": ladder[-1]}
        for name in metric_names:
            columns = [displayed(name, [r[name] for r in by_point[(corruption_type, s)]]) for s in ladder]
            curve = [float(column.mean()) for column in columns]
            value, threshold = curve[-1], thresholds[name]
            monotone_per_replicate = [
                all(b >= a for a, b in zip(sequence, sequence[1:]))
                for sequence in np.stack(columns, axis=1)
            ]
            record.update({
                f"{name}__value": value,
                f"{name}__sd": float(columns[-1].std(ddof=1)) if columns[-1].size > 1 else 0.0,
                f"{name}__threshold": threshold,
                f"{name}__margin": value / threshold if threshold > 0 else float("inf"),
                f"{name}__detected": bool(value > threshold),
                f"{name}__null_draws_exceeded": int((matched[name] < value).sum()),
                f"{name}__p_value": float((matched[name] >= value).mean()),
                f"{name}__monotone": bool(all(b >= a for a, b in zip(curve, curve[1:]))),
                f"{name}__monotone_fraction": float(np.mean(monotone_per_replicate)),
            })
        verdict_rows.append(record)

    print(f"\n{'':34s} " + "".join(f"{n[:14]:>16s}" for n in metric_names))
    for label, getter in [
        ("day-partition null mean", lambda n: matched[n].mean()),
        ("day-partition null sd", lambda n: matched[n].std(ddof=1)),
        ("day-partition null min", lambda n: matched[n].min()),
        (f"day-partition p{quantile*100:g} (THRESHOLD)", lambda n: thresholds[n]),
        ("day-partition null max", lambda n: matched[n].max()),
        ("spread (max/min)", lambda n: matched[n].max() / max(matched[n].min(), 1e-30)),
        ("unstratified null mean", lambda n: unstratified[n].mean()),
        ("shipped single null", lambda n: float(displayed(n, shipped_null[n]))),
        ("  pctile vs day-partition",
         lambda n: 100.0 * float((matched[n] <= float(displayed(n, shipped_null[n]))).mean())),
        ("  shipped/threshold",
         lambda n: float(displayed(n, shipped_null[n])) / max(thresholds[n], 1e-30)),
    ]:
        print(f"{label:34s} " + "".join(f"{getter(n):16.6g}" for n in metric_names))

    print(f"\nN / M at maximum severity (N threshold = day-partition null p{quantile*100:g}):")
    print(f"{'corruption':30s} " + "".join(f"{n[:16]:>18s}" for n in metric_names))
    for record in verdict_rows:
        cells = []
        for name in metric_names:
            n_mark = "N" if record[f"{name}__detected"] else "-"
            m_mark = "M" if record[f"{name}__monotone"] else "-"
            cells.append(f"{n_mark}{m_mark} {record[f'{name}__null_draws_exceeded']:3d}/{replicates}".rjust(18))
        print(f"{record['corruption']:30s} " + "".join(cells))
    print("\nN = above threshold, M = monotone; the fraction is null draws exceeded.")

    with open(output_root / "data" / "bootstrap_null_draws.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(null_rows[0].keys()))
        writer.writeheader()
        writer.writerows(null_rows)
    with open(output_root / "data" / "bootstrap_curves.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)
    with open(output_root / "data" / "bootstrap_null_verdicts.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(verdict_rows[0].keys()))
        writer.writeheader()
        writer.writerows(verdict_rows)
    with open(output_root / "data" / "bootstrap_null_summary.json", "w") as handle:
        json.dump({
            "metrics": metric_names, "replicates": replicates, "pool_samples": int(n_pool),
            "curve_replicates": curve_replicates, "curve_pool_samples": int(n_curve),
            "scwd_dedup": dedup_scwd,
            "eval_samples": eval_samples, "pairwise_eval_samples": pairwise_cap,
            "threshold_quantile": quantile, "thresholds": thresholds,
            "shipped_single_null": {n: float(shipped_null[n]) for n in metric_names},
            "day_partition_mean": {n: float(matched[n].mean()) for n in metric_names},
            "day_partition_sd": {n: float(matched[n].std(ddof=1)) for n in metric_names},
            "unstratified_mean": {n: float(unstratified[n].mean()) for n in metric_names},
        }, handle, indent=2)
    written = [
        output_root / "data" / name for name in (
            "bootstrap_null_draws.csv", "bootstrap_curves.csv",
            "bootstrap_null_verdicts.csv", "bootstrap_null_summary.json",
        )
    ]
    print(f"\nWrote {output_root / 'data'}")
    return written


@hydra.main(version_base=None, config_path="../conf", config_name="baseline_config")
def main(cfg: DictConfig):
    evaluate_bootstrap_null(cfg)


if __name__ == "__main__":
    main()

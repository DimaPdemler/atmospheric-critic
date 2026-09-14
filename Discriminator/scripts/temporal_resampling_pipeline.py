"""Fold-aware orchestration and aggregation for the baseline pipeline."""

from __future__ import annotations

import csv
import gzip
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm

try:
    from .temporal_resampling import (
        active_schedule, aggregate_draws, fixed_schedules, learned_schedules,
        settings, write_csv_gz, write_manifest,
    )
except ImportError:
    from temporal_resampling import (
        active_schedule, aggregate_draws, fixed_schedules, learned_schedules,
        settings, write_csv_gz, write_manifest,
    )


STANDARD_KEYS = {
    "lead_time": ["label", "variable", "lead_hour", "is_null"],
    "corruption_strength": ["label", "variable", "corruption", "severity", "is_null"],
}
DISCRIMINATOR_KEYS = [
    "architecture", "input_variables", "encoder_pretraining", "kind", "target",
    "x", "source", "is_era5_test_null",
]


def _read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with open(temporary, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
            handle.flush(); os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _write_draw_npz(path, rows):
    """Persist the long-form draw table as arrays for direct numerical reuse."""
    rows = list(rows)
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            **{field: np.asarray([str(row.get(field, "")) for row in rows]) for field in fields},
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _ranges(cfg):
    monthly = cfg.monthly_split
    return [monthly.corruption_time_range, *list(monthly.model_valid_time_ranges)]


def _fold_cfg(cfg, schedule, work_dir, checkpoint_dir=None, training_output_dir=None,
              render_diagnostics=False, diagnostics_output_dir=None):
    """Make an in-process fold configuration without a child pipeline run."""
    fold = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    fold.temporal_resampling.active_schedule = schedule.to_dict()
    fold.pipeline.resume = False
    fold.baseline.output_dir = str(work_dir)
    target_root = (Path(training_output_dir) if training_output_dir is not None
                   else Path(work_dir) / "target_discriminator")
    fold.target_discriminator.output_dir = str(target_root)
    fold.target_discriminator.render_diagnostics = bool(render_diagnostics)
    fold.target_discriminator.diagnostics_output_dir = str(diagnostics_output_dir or target_root)
    fold.pipeline.output_root = str(target_root)
    if checkpoint_dir is not None:
        fold.target_discriminator.checkpoint_dir = str(checkpoint_dir)
        if fold.baseline.get("discriminator") is not None:
            fold.baseline.discriminator.checkpoint_dir = str(checkpoint_dir)
    return fold


def _prior_checkpoint(cfg, schedule):
    configured = settings(cfg).get("input_run_dir") or cfg.pipeline.get("input_checkpoint_dir")
    if configured is None:
        return None
    root = Path(str(configured))
    if root.name == "target_discriminators":
        return root / schedule.resample_id if (root / schedule.resample_id).is_dir() else root
    candidates = [
        root / "models" / "target_discriminators" / schedule.resample_id,
        root / _parent_output_root(cfg).name / "models" / "target_discriminators" / schedule.resample_id,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Expected checkpoint directory for {schedule.resample_id}; checked {candidates}"
    )


def _local_checkpoint(cfg, schedule):
    candidate = _parent_output_root(cfg) / "models" / "target_discriminators" / schedule.resample_id
    return candidate if candidate.is_dir() else _prior_checkpoint(cfg, schedule)

def _validate_counts(rows, group_fields, count_fields):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(field) for field in group_fields), []).append(row)
    for key, group in grouped.items():
        for field in count_fields:
            values = {int(float(row[field])) for row in group if row.get(field, "") != ""}
            if len(values) > 1:
                raise ValueError(f"Inconsistent {field} across temporal resamples for {dict(zip(group_fields, key))}: {sorted(values)}")


def _metric_names(cfg):
    try:
        from .plot_standard_metric_baselines import metric_names_from_config
    except ImportError:
        from plot_standard_metric_baselines import metric_names_from_config
    return metric_names_from_config(cfg)


def aggregate_standard(cfg, schedules, fold_data):
    """Aggregate persisted per-fold files into the parent data contract."""
    metric_names = _metric_names(cfg)
    output_root = _parent_output_root(cfg)
    draw_rows, outputs = [], []
    for experiment, keys in STANDARD_KEYS.items():
        rows = []
        for schedule in schedules:
            path = fold_data[schedule.resample_id] / f"{experiment}.csv"
            for source in _read_csv(path):
                source["resample_id"] = schedule.resample_id
                rows.append(source)
                for metric in metric_names:
                    draw_rows.append({
                        "family": "fixed", "resample_id": schedule.resample_id,
                        "experiment": experiment, "metric": metric,
                        **{key: source.get(key, "") for key in keys},
                        "value": source[metric], "n_samples": source.get("n_samples", ""),
                        "pairwise_n_samples": source.get("pairwise_n_samples", ""),
                    })
        _validate_counts(rows, keys, ("n_samples", "pairwise_n_samples"))
        aggregate = aggregate_draws(rows, metric_names, keys, "p05_p95")
        representatives = {tuple(row.get(key) for key in keys): row for row in rows}
        for row in aggregate:
            representative = representatives[tuple(row.get(key) for key in keys)]
            for field in ("n_samples", "pairwise_n_samples", "n_pairs", "initialization_start",
                          "initialization_end", "valid_start", "valid_end"):
                if field in representative:
                    row[field] = representative[field]
        outputs.append(_write_csv(output_root / "data" / f"{experiment}.csv", aggregate))
    outputs.append(_write_csv(output_root / "data" / "fixed_metric_draws.csv", draw_rows))
    outputs.append(_write_draw_npz(output_root / "data" / "fixed_metric_draws.npz", draw_rows))
    split_path = aggregate_split_manifests(cfg, schedules, "fixed", fold_data)
    if split_path is not None:
        outputs.append(split_path)
    canonical = schedules[min(4, len(schedules) - 1)]
    canonical_data = fold_data[canonical.resample_id]
    for name in ("scwd_anchor_contributions.nc", "global_mean_wasserstein_distributions.nc",
                 "corruption_disturbances.nc"):
        source = canonical_data / name
        if source.is_file():
            target = output_root / "data" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            outputs.append(target)
    return outputs


def _evenly_spaced(items, count):
    """Select ``count`` ordered records while retaining the sampled span."""
    if len(items) <= count:
        return list(items)
    selected = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[int(index)] for index in selected]


def _equalize_discriminator_term_counts(rows, terms):
    """Equalize persisted held-out term counts across temporal folds.

    A forecast source can occasionally lack a valid time in one otherwise
    equal-width calendar window. Use the minimum available count at each
    evaluated coordinate and deterministically retain evenly spaced raw terms.
    Scores and standard errors are then reconstructed without repeating model
    inference, while every temporal fold retains equal statistical weight.
    """
    ep_groups, candidate_groups = {}, {}
    for term in terms:
        base = (
            term["resample_id"], term["architecture"], term["kind"],
            term["target"],
        )
        if term["role"] in {"ep_reference", "ep_train"}:
            ep_groups.setdefault(base, []).append(term)
        else:
            key = (*base, term["source"], float(term["x"]))
            candidate_groups.setdefault(key, []).append(term)

    for group in list(ep_groups.values()) + list(candidate_groups.values()):
        group.sort(key=lambda item: int(float(item.get("sample_position", 0))))

    ep_minimum = {}
    for key, group in ep_groups.items():
        cross_fold = key[1:]
        ep_minimum[cross_fold] = min(ep_minimum.get(cross_fold, len(group)), len(group))
    candidate_minimum = {}
    for key, group in candidate_groups.items():
        cross_fold = key[1:]
        candidate_minimum[cross_fold] = min(
            candidate_minimum.get(cross_fold, len(group)), len(group)
        )

    selected_ep = {
        key: _evenly_spaced(group, ep_minimum[key[1:]])
        for key, group in ep_groups.items()
    }
    selected_candidates = {
        key: _evenly_spaced(group, candidate_minimum[key[1:]])
        for key, group in candidate_groups.items()
    }
    equalized_terms = [
        term
        for group in list(selected_ep.values()) + list(selected_candidates.values())
        for term in group
    ]

    changed = set()
    for row in rows:
        base = (
            row["resample_id"], row["architecture"], row["kind"], row["target"],
        )
        candidate_key = (*base, row["source"], float(row["x"]))
        ep_values = np.asarray([
            float(term["transformed_term"]) for term in selected_ep[base]
        ])
        candidate_values = np.asarray([
            float(term["transformed_term"])
            for term in selected_candidates[candidate_key]
        ])
        original_counts = (int(float(row["n_samples"])), int(float(row["ep_n_samples"])))
        new_counts = (len(candidate_values), len(ep_values))
        if original_counts != new_counts:
            changed.add((row["architecture"], row["kind"], row["target"], row["source"], row["x"], *new_counts))
        ep = float(ep_values.mean())
        candidate = float(candidate_values.mean())
        ep_stderr = (
            float(ep_values.std(ddof=1) / np.sqrt(len(ep_values)))
            if len(ep_values) > 1 else 0.0
        )
        candidate_stderr = (
            float(candidate_values.std(ddof=1) / np.sqrt(len(candidate_values)))
            if len(candidate_values) > 1 else 0.0
        )
        row.update(
            score=ep - candidate,
            stderr=float(np.hypot(ep_stderr, candidate_stderr)),
            n_samples=len(candidate_values),
            ep_reference=ep,
            ep_n_samples=len(ep_values),
        )
    if changed:
        print(
            f"Equalized {len(changed)} discriminator coordinates to their "
            "minimum held-out counts across temporal folds."
        )
    return rows, equalized_terms


def aggregate_discriminator(cfg, schedules, fold_data):
    output_root = _parent_output_root(cfg)
    rows, terms = [], []
    for schedule in schedules:
        data = fold_data[schedule.resample_id]
        for row in _read_csv(data / "discriminator_reverse_kl.csv"):
            row["resample_id"] = schedule.resample_id
            rows.append(row)
        terms_path = data / "discriminator_terms.csv.gz"
        if terms_path.is_file():
            with gzip.open(terms_path, "rt", newline="") as handle:
                for row in csv.DictReader(handle):
                    row["resample_id"] = schedule.resample_id
                    terms.append(row)
    rows, terms = _equalize_discriminator_term_counts(rows, terms)
    _validate_counts(rows, DISCRIMINATOR_KEYS, ("n_samples", "ep_n_samples"))
    try:
        from .analyze_temporal_resamples import reconstruct_scores_from_terms
    except ImportError:
        from analyze_temporal_resamples import reconstruct_scores_from_terms
    terms_path = write_csv_gz(output_root / "data" / "discriminator_terms.csv.gz", terms)
    reconstructed = reconstruct_scores_from_terms(terms_path)
    for row in rows:
        key = (row["resample_id"], row["architecture"], row["kind"], row["target"],
               row["source"], float(row["x"]))
        if key not in reconstructed or not np.isclose(float(row["score"]), reconstructed[key], rtol=1e-10, atol=1e-10):
            raise ValueError(f"Saved KL terms do not reconstruct draw {key}")
    aggregate = aggregate_draws(rows, ["score"], DISCRIMINATOR_KEYS, "minmax")
    representative = {tuple(row.get(key) for key in DISCRIMINATOR_KEYS): row for row in rows}
    for row in aggregate:
        source = representative[tuple(row.get(key) for key in DISCRIMINATOR_KEYS)]
        for field in (
            "stderr", "n_samples", "ep_reference", "ep_reference_split",
            "ep_train", "ep_n_samples",
        ):
            if source.get(field, "") != "":
                row[field] = source[field]
        matching = [item for item in rows if tuple(item.get(key) for key in DISCRIMINATOR_KEYS) == tuple(row.get(key) for key in DISCRIMINATOR_KEYS)]
        row["checkpoint_paths"] = ",".join(item.get("checkpoint_path", "") for item in matching)
        row["checkpoint_sha256s"] = ",".join(item.get("checkpoint_sha256", "") for item in matching)
    score_path = _write_csv(output_root / "data" / "discriminator_reverse_kl.csv", aggregate)
    draw_path = _write_csv(output_root / "data" / "discriminator_metric_draws.csv", rows)
    draw_npz_path = _write_draw_npz(output_root / "data" / "discriminator_metric_draws.npz", rows)
    split_path = aggregate_split_manifests(cfg, schedules, "learned", fold_data)
    return [score_path, terms_path, draw_path, draw_npz_path] + ([split_path] if split_path is not None else [])

def _parent_output_root(cfg):
    try:
        from .plot_standard_metric_baselines import baseline_output_dir, variables_from_config
    except ImportError:
        from plot_standard_metric_baselines import baseline_output_dir, variables_from_config
    return baseline_output_dir(cfg, variables_from_config(cfg))


def aggregate_split_manifests(cfg, schedules, family, fold_data):
    output_root = _parent_output_root(cfg)
    rows = []
    for schedule in schedules:
        path = fold_data[schedule.resample_id] / "split_manifest.csv.gz"
        if not path.is_file():
            continue
        with gzip.open(path, "rt", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return write_csv_gz(output_root / "data" / f"{family}_split_manifest.csv.gz", rows) if rows else None


def aggregate_all_draws(cfg):
    output_root = _parent_output_root(cfg)
    sources = [output_root / "data" / "fixed_metric_draws.csv",
               output_root / "data" / "discriminator_metric_draws.csv"]
    rows = []
    for path in sources:
        if path.is_file():
            for row in _read_csv(path):
                row["source_table"] = path.name; rows.append(row)
    draw_path = _write_csv(output_root / "data" / "metric_draws.csv", rows) if rows else None
    split_rows = []
    for path in (output_root / "data" / "fixed_split_manifest.csv.gz",
                 output_root / "data" / "learned_split_manifest.csv.gz"):
        if path.is_file():
            with gzip.open(path, "rt", newline="") as handle:
                split_rows.extend(csv.DictReader(handle))
    if split_rows:
        write_csv_gz(output_root / "data" / "split_manifest.csv.gz", split_rows)
    return draw_path


def _fit_fold_fake_matching(cfg):
    paths = []
    if bool((cfg.get("histogram_matching", {}) or {}).get("enabled", False)):
        try:
            from .fit_histogram_matching import fit_histogram_matching_maps
        except ImportError:
            from fit_histogram_matching import fit_histogram_matching_maps
        paths.extend(fit_histogram_matching_maps(cfg))
    if bool((cfg.get("moment_matching", {}) or {}).get("enabled", False)):
        try:
            from .fit_moment_matching import fit_moment_matching_maps
        except ImportError:
            from fit_moment_matching import fit_moment_matching_maps
        paths.extend(fit_moment_matching_maps(cfg))
    return paths


def _publish_canonical_histogram_maps(cfg, schedule):
    """Expose the canonical learned-fold maps at the parent plot-data path."""
    if not bool((cfg.get("histogram_matching", {}) or {}).get("enabled", False)):
        return None
    root = _parent_output_root(cfg)
    source = root / "training" / schedule.resample_id / "data" / "preprocessing" / "histogram_matching"
    if not (source / "maps.npz").is_file() or not (source / "manifest.json").is_file():
        return None
    target = root / "data" / "preprocessing" / "histogram_matching"
    shutil.copytree(source, target, dirs_exist_ok=True)
    return target


def _publish_canonical_moment_maps(cfg, schedule):
    """Expose the canonical learned-fold moment maps at the parent data path."""
    if not bool((cfg.get("moment_matching", {}) or {}).get("enabled", False)):
        return None
    root = _parent_output_root(cfg)
    source = root / "training" / schedule.resample_id / "data" / "preprocessing" / "moment_matching"
    if not (source / "moments.npz").is_file() or not (source / "manifest.json").is_file():
        return None
    target = root / "data" / "preprocessing" / "moment_matching"
    shutil.copytree(source, target, dirs_exist_ok=True)
    return target


def _archive_fold_matching(cfg, fold, schedule, family):
    """Retain small preprocessing artifacts that would otherwise live in scratch."""
    archived = []
    for mode, archive_name in (("histogram_matching", "maps.npz"),
                               ("moment_matching", "moments.npz")):
        if not bool((fold.get(mode, {}) or {}).get("enabled", False)):
            continue
        source = Path(str(fold.pipeline.output_root)) / "data" / "preprocessing" / mode
        if not (source / archive_name).is_file():
            continue
        target = (_parent_output_root(cfg) / "data" / "preprocessing" / mode / "resamples" /
                  family / schedule.resample_id)
        shutil.copytree(source, target, dirs_exist_ok=True)
        archived.extend(path for path in target.iterdir() if path.is_file())
    return archived


def _fold_status_path(cfg):
    return _parent_output_root(cfg) / "data" / "resample_status.csv"


def _write_fold_status(cfg, rows):
    """Update one stage's fold status without discarding other stage records."""
    path = _fold_status_path(cfg)
    previous = _read_csv(path) if path.is_file() else []
    stages = {row.get("stage") for row in rows}
    return _write_csv(path, [row for row in previous if row.get("stage") not in stages] + list(rows))


def _upload_fold_checkpoint_bundle(cfg, fold, tracker, schedule, checkpoint_dir, training_root):
    """Upload one coherent checkpoint artifact per fold, not one artifact per target."""
    if not bool((cfg.pipeline.get("wandb", {}) or {}).get("upload_checkpoints", True)):
        return None
    paths = sorted(path for path in Path(checkpoint_dir).rglob("*") if path.is_file())
    paths.extend(path for path in (
        Path(training_root) / "data" / "target_train_test_metrics.csv",
        Path(training_root) / "resolved_config.yaml",
    ) if path.is_file())
    if not paths:
        return None
    metadata = {"temporal_resample_id": schedule.resample_id, "checkpoint_count": len(paths)}
    with tracker.run(
        f"train/{schedule.resample_id}/checkpoint-bundle", "checkpoint-bundle", fold,
        metadata=metadata, tags=["checkpoint-bundle", schedule.resample_id],
    ) as run:
        tracker.log_csv_table(
            run, "training/train_test_metrics",
            Path(training_root) / "data" / "target_train_test_metrics.csv",
        )
        return tracker.log_artifact(
            run, f"target-discriminators-{schedule.resample_id}", "model", paths,
            metadata=metadata,
        )


def _check_parent_storage_budget(cfg):
    try:
        from .run_baseline_pipeline import check_storage_budget
    except ImportError:
        from run_baseline_pipeline import check_storage_budget
    run_dir = cfg.pipeline.get("run_dir")
    return None if run_dir is None else check_storage_budget(cfg, Path(str(run_dir)))


def _canonical_first(schedules, canonical):
    """Run the diagnostic fold first without changing identifiers or aggregation order."""
    return sorted(schedules, key=lambda schedule: schedule.resample_id != canonical)


def _train_learned_folds(cfg, schedules, tracker):
    try:
        from .train_target_discriminator_baselines import train_target_discriminator_baselines
    except ImportError:
        from train_target_discriminator_baselines import train_target_discriminator_baselines
    root = _parent_output_root(cfg)
    previous = _read_csv(_fold_status_path(cfg)) if _fold_status_path(cfg).is_file() else []
    status = [row for row in previous if row.get("stage") == "train_discriminators"]
    records = []
    requested_canonical = str((cfg.pipeline.get("storage", {}) or {}).get(
        "canonical_diagnostic_fold", "learned_04"
    ))
    schedule_ids = {schedule.resample_id for schedule in schedules}
    canonical = requested_canonical if requested_canonical in schedule_ids else schedules[-1].resample_id
    execution_order = _canonical_first(schedules, canonical)
    for schedule in tqdm(execution_order, desc="Training learned temporal resamples"):
        checkpoint_dir = root / "models" / "target_discriminators" / schedule.resample_id
        training_root = root / "training" / schedule.resample_id
        metrics_path = training_root / "data" / "target_train_test_metrics.csv"
        completed = next((row for row in status
                          if row.get("resample_id") == schedule.resample_id
                          and row.get("status") == "completed"), None)
        if bool(cfg.pipeline.get("resume", False)) and completed and metrics_path.is_file():
            fold_records = _read_csv(metrics_path)
            if fold_records and all(Path(row.get("path", "")).is_file() for row in fold_records):
                for record in fold_records:
                    record["resample_id"] = schedule.resample_id
                    records.append(record)
                print(f"Skipping completed learned fold {schedule.resample_id}")
                continue
        status = [row for row in status if row.get("resample_id") != schedule.resample_id]
        render = schedule.resample_id == canonical
        fold = _fold_cfg(
            cfg, schedule, training_root, checkpoint_dir=checkpoint_dir,
            training_output_dir=training_root, render_diagnostics=render,
            diagnostics_output_dir=root,
        )
        # Scalar training runs remain individually inspectable, while their heavy
        # files are uploaded once in a fold-level artifact below.
        OmegaConf.update(fold, "pipeline.wandb.upload_checkpoints", False, merge=False)
        _fit_fold_fake_matching(fold)
        try:
            fold_records = train_target_discriminator_baselines(fold, tracker=tracker)
        except BaseException as error:
            status.append({"stage": "train_discriminators", "resample_id": schedule.resample_id,
                           "status": "failed", "error": f"{type(error).__name__}: {error}"})
            _write_fold_status(cfg, status)
            raise
        for record in fold_records:
            record["resample_id"] = schedule.resample_id
            records.append(record)
        _upload_fold_checkpoint_bundle(cfg, fold, tracker, schedule, checkpoint_dir, training_root)
        status.append({"stage": "train_discriminators", "resample_id": schedule.resample_id,
                       "status": "completed", "checkpoint_dir": str(checkpoint_dir),
                       "diagnostics_rendered": render})
        _write_fold_status(cfg, status)
        _check_parent_storage_budget(cfg)
    _write_csv(root / "data" / "learned_target_train_test_draws.csv", records)
    return records, status


def _cache_fold_data(source, target, names):
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = Path(source) / name
        if path.is_file():
            shutil.copy2(path, target / name)
    return target


def _fold_cache_complete(path, names):
    return all((Path(path) / name).is_file() for name in names)


def _evaluate_standard_folds(cfg, schedules):
    try:
        from .plot_standard_metric_baselines import evaluate_standard_metrics, baseline_output_dir, variables_from_config
    except ImportError:
        from plot_standard_metric_baselines import evaluate_standard_metrics, baseline_output_dir, variables_from_config
    scratch_dir = cfg.baseline.get("scratch_dir")
    with tempfile.TemporaryDirectory(prefix="temporal-resampling-standard-",
                                     dir=None if scratch_dir is None else str(scratch_dir)) as directory:
        scratch = Path(directory)
        cache_root = _parent_output_root(cfg) / "data" / "resume_cache" / "standard_metrics"
        previous = _read_csv(_fold_status_path(cfg)) if _fold_status_path(cfg).is_file() else []
        fold_data = {}
        status = [row for row in previous if row.get("stage") == "evaluate_standard_metrics"]
        matching_paths = []
        canonical_id = schedules[min(4, len(schedules) - 1)].resample_id
        required = ("lead_time.csv", "corruption_strength.csv", "split_manifest.csv.gz")
        ancillary = ("scwd_anchor_contributions.nc", "global_mean_wasserstein_distributions.nc",
                     "corruption_disturbances.nc")
        for schedule in tqdm(schedules, desc="Evaluating standard-metric temporal resamples"):
            cached = cache_root / schedule.resample_id
            completed = next((row for row in status if row.get("resample_id") == schedule.resample_id
                              and row.get("status") == "completed"), None)
            if (bool(cfg.pipeline.get("resume", False)) and completed
                    and _fold_cache_complete(cached, required)):
                fold_data[schedule.resample_id] = cached
                print(f"Skipping completed standard-metric fold {schedule.resample_id}")
                continue
            status = [row for row in status if row.get("resample_id") != schedule.resample_id]
            fold = _fold_cfg(cfg, schedule, scratch / schedule.resample_id)
            _fit_fold_fake_matching(fold)
            matching_paths.extend(_archive_fold_matching(cfg, fold, schedule, "fixed"))
            try:
                evaluate_standard_metrics(fold)
            except BaseException as error:
                status.append({"stage": "evaluate_standard_metrics", "resample_id": schedule.resample_id,
                               "status": "failed", "error": f"{type(error).__name__}: {error}"})
                _write_fold_status(cfg, status)
                raise
            source_data = baseline_output_dir(fold, variables_from_config(fold)) / "data"
            names = required + (ancillary if schedule.resample_id == canonical_id else ())
            fold_data[schedule.resample_id] = _cache_fold_data(source_data, cached, names)
            status.append({"stage": "evaluate_standard_metrics", "resample_id": schedule.resample_id,
                           "status": "completed"})
            _write_fold_status(cfg, status)
            _check_parent_storage_budget(cfg)
        outputs = aggregate_standard(cfg, schedules, fold_data) + matching_paths
        shutil.rmtree(cache_root, ignore_errors=True)
        return outputs, status


def _evaluate_discriminator_folds(cfg, schedules):
    try:
        from .plot_standard_metric_baselines import evaluate_discriminator_metrics, baseline_output_dir, variables_from_config
    except ImportError:
        from plot_standard_metric_baselines import evaluate_discriminator_metrics, baseline_output_dir, variables_from_config
    scratch_dir = cfg.baseline.get("scratch_dir")
    with tempfile.TemporaryDirectory(prefix="temporal-resampling-discriminator-",
                                     dir=None if scratch_dir is None else str(scratch_dir)) as directory:
        scratch = Path(directory)
        cache_root = _parent_output_root(cfg) / "data" / "resume_cache" / "discriminator_metrics"
        previous = _read_csv(_fold_status_path(cfg)) if _fold_status_path(cfg).is_file() else []
        fold_data = {}
        status = [row for row in previous if row.get("stage") == "evaluate_discriminator_metrics"]
        matching_paths = []
        required = ("discriminator_reverse_kl.csv", "discriminator_terms.csv.gz", "split_manifest.csv.gz")
        for schedule in tqdm(schedules, desc="Evaluating discriminator temporal resamples"):
            cached = cache_root / schedule.resample_id
            completed = next((row for row in status if row.get("resample_id") == schedule.resample_id
                              and row.get("status") == "completed"), None)
            if (bool(cfg.pipeline.get("resume", False)) and completed
                    and _fold_cache_complete(cached, required)):
                fold_data[schedule.resample_id] = cached
                print(f"Skipping completed discriminator-metric fold {schedule.resample_id}")
                continue
            status = [row for row in status if row.get("resample_id") != schedule.resample_id]
            checkpoint = _local_checkpoint(cfg, schedule)
            if checkpoint is None:
                raise FileNotFoundError(f"No discriminator checkpoints found for {schedule.resample_id}")
            fold = _fold_cfg(cfg, schedule, scratch / schedule.resample_id, checkpoint_dir=checkpoint)
            _fit_fold_fake_matching(fold)
            matching_paths.extend(_archive_fold_matching(cfg, fold, schedule, "learned"))
            try:
                evaluate_discriminator_metrics(fold)
            except BaseException as error:
                status.append({"stage": "evaluate_discriminator_metrics", "resample_id": schedule.resample_id,
                               "status": "failed", "error": f"{type(error).__name__}: {error}"})
                _write_fold_status(cfg, status)
                raise
            source_data = baseline_output_dir(fold, variables_from_config(fold)) / "data"
            fold_data[schedule.resample_id] = _cache_fold_data(source_data, cached, required)
            status.append({"stage": "evaluate_discriminator_metrics", "resample_id": schedule.resample_id,
                           "status": "completed", "checkpoint_dir": str(checkpoint)})
            _write_fold_status(cfg, status)
            _check_parent_storage_budget(cfg)
        outputs = aggregate_discriminator(cfg, schedules, fold_data) + matching_paths
        shutil.rmtree(cache_root, ignore_errors=True)
        return outputs, status


def run_resampled_stage(stage, cfg, tracker, output_root, resolved_path):
    """Run all temporal schedules in the parent pipeline and aggregate in place."""
    learned = learned_schedules(cfg)
    fixed = fixed_schedules(cfg, _ranges(cfg))
    if stage == "plot":
        canonical = learned[min(4, len(learned) - 1)]
        checkpoint = _local_checkpoint(cfg, canonical)
        if checkpoint is not None:
            OmegaConf.update(cfg, "target_discriminator.checkpoint_dir", str(checkpoint), merge=False)
            if cfg.baseline.get("discriminator") is not None:
                OmegaConf.update(cfg, "baseline.discriminator.checkpoint_dir", str(checkpoint), merge=False)
        maps = _publish_canonical_histogram_maps(cfg, canonical)
        if maps is not None:
            OmegaConf.update(cfg, "pipeline.input_histogram_matching_dir", str(maps), force_add=True)
        moment_maps = _publish_canonical_moment_maps(cfg, canonical)
        if moment_maps is not None:
            OmegaConf.update(cfg, "pipeline.input_moment_matching_dir", str(moment_maps), force_add=True)
        return None
    if stage in {"fit_histogram_matching", "fit_moment_matching"}:
        return [], []
    if stage == "train_discriminators":
        records, status = _train_learned_folds(cfg, learned, tracker)
        paths = [Path(record["path"]) for record in records if record.get("path")]
        paths.extend([_parent_output_root(cfg) / "data" / "learned_target_train_test_draws.csv",
                      _fold_status_path(cfg)])
        paths.extend((_parent_output_root(cfg) / "training").glob(
            "learned_*/data/preprocessing/moment_matching/*"
        ))
        return [str(path) for path in paths if path.is_file()], status
    if stage == "evaluate_standard_metrics":
        paths, status = _evaluate_standard_folds(cfg, fixed)
        draw_path = aggregate_all_draws(cfg)
        if draw_path is not None:
            paths.append(draw_path)
        with tracker.run("evaluation/standard-metrics-aggregate", "temporal-resampling-aggregate", cfg,
                         tags=["evaluation", "standard-metrics", "temporal-resampling"]) as run:
            tracker.log_csv_table(run, "metrics/temporal_standard_draws",
                                  _parent_output_root(cfg) / "data" / "fixed_metric_draws.csv")
            if bool(cfg.pipeline.wandb.get("upload_evaluation_data", True)):
                tracker.log_artifact(run, "temporal-standard-metric-samples", "evaluation",
                                     [*paths, resolved_path], metadata={"pipeline_id": tracker.group})
        return [str(path) for path in paths], status
    if stage == "evaluate_discriminator_metrics":
        paths, status = _evaluate_discriminator_folds(cfg, learned)
        draw_path = aggregate_all_draws(cfg)
        if draw_path is not None:
            paths.append(draw_path)
        with tracker.run("evaluation/discriminator-metrics-aggregate", "temporal-resampling-aggregate", cfg,
                         tags=["evaluation", "discriminator-metrics", "temporal-resampling"]) as run:
            tracker.log_csv_table(run, "metrics/temporal_discriminator_draws",
                                  _parent_output_root(cfg) / "data" / "discriminator_metric_draws.csv")
            if bool(cfg.pipeline.wandb.get("upload_evaluation_data", True)):
                tracker.log_artifact(run, "temporal-discriminator-metric-samples", "evaluation",
                                     [*paths, resolved_path], metadata={"pipeline_id": tracker.group})
        return [str(path) for path in paths], status
    return None

def finalize_resampling_manifest(cfg):
    output_root = _parent_output_root(cfg)
    data = output_root / "data"
    paths = list(data.glob("*")) if data.exists() else []
    payload = {
        "pipeline_id": str(cfg.pipeline.id),
        "learned_schedules": [item.to_dict() for item in learned_schedules(cfg)],
        "fixed_schedules": [item.to_dict() for item in fixed_schedules(cfg, _ranges(cfg))],
        "learned_bounds": "minmax", "fixed_bounds": "p05_p95",
        "within_test_bootstrap": False,
    }
    return write_manifest(data / "resampling", payload, paths)

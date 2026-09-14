#!/usr/bin/env python3
"""Chunked U-Cast inference over a full year, keeping only the analysis fields.

Upstream's ``run_inference_standalone.py`` accumulates every initial condition's full
83-variable output in memory and writes once at the end. Over 732 ICs at horizon 20 that
is ~340 GB, so it cannot be used directly for a year-long run. This driver reuses
upstream's functions unmodified and adds the three things a production run needs:

  1. Opens the monthly ERA5 zarr stores directly, so no single concatenated store has to
     be built. Upstream's ``open_era5_zarr`` insists on one directory named exactly as
     the checkpoint config's ``dataset`` field; we bypass it.
  2. Keeps only the nine fields we actually analyse. ``build_xarray_dataset`` iterates
     whatever ``var_names`` it is handed, so passing a subset is enough -- the model
     still runs all 83 channels internally, we simply store less.
  3. Writes one file per chunk of ICs, so peak memory stays flat and a crash costs one
     chunk instead of the whole run.

Output here is "raw" (init_time/lead_time, ``*_predicted`` names). Run
``convert_to_baseline_schema.py`` afterwards to get baseline-shaped files.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import logging
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

log = logging.getLogger("ucast2020")


# The nine fields we keep. Upstream flattens pressure levels into the variable name,
# so `temperature_850` is temperature at 850 hPa.
SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
LEVEL850_VARS = [
    "temperature_850",
    "u_component_of_wind_850",
    "v_component_of_wind_850",
    "specific_humidity_850",
]
Z500_VARS = ["geopotential_500"]
KEEP_VARS = SURFACE_VARS + LEVEL850_VARS + Z500_VARS

# Union of the lead times our baselines use (12/24/48/96/192) and the ones the U-Cast
# paper reports in Table 1 (1d/3d/10d = 24/72/240). 6 h is deliberately absent: U-Cast
# steps 12-hourly and cannot produce it.
DEFAULT_LEAD_HOURS = [12, 24, 48, 72, 96, 192, 240]


def load_upstream(repo: Path):
    """Import upstream's standalone script as a module.

    It guards its entry point with ``if __name__ == "__main__"`` and has no import-time
    side effects, so importing it is safe.
    """
    script = repo / "run_inference_standalone.py"
    if not script.is_file():
        raise SystemExit(f"Cannot find {script}. Pass --ucast-repo pointing at the u-cast clone.")
    # Upstream resolves --stats-dir relative to the working directory.
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("ucast_standalone", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def open_era5_stores(data_glob: str):
    """Open one or many ERA5 zarr stores as a single dataset.

    ``data_vars="minimal"`` is load-bearing: the default broadcasts the two static
    fields (geopotential_at_surface, land_sea_mask) along every timestep, which turns
    two 113 KiB maps into gigabytes and makes them look time-varying.
    """
    paths = sorted(glob.glob(data_glob))
    if not paths:
        raise SystemExit(f"No zarr stores matched {data_glob!r}")
    log.info(f"Opening {len(paths)} store(s) matching {data_glob}")
    if len(paths) == 1:
        return xr.open_zarr(paths[0])
    return xr.open_mfdataset(
        paths, engine="zarr", combine="by_coords", data_vars="minimal", coords="minimal", compat="override"
    )


def build_ic_dates(ds, start: str, end: str, hours, limit=None):
    """Every timestamp in [start, end] whose hour is in `hours` and which the data covers."""
    times = ds.sel(time=slice(start, end)).time.values
    wanted = [t for t in times if int((t - t.astype("datetime64[D]")) / np.timedelta64(1, "h")) in hours]
    if limit:
        wanted = wanted[:limit]
    return wanted


def drop_targets(ds):
    """Keep forecasts only. Targets are ERA5, which we already have on disk, and the
    baseline files we are matching contain forecasts only."""
    return ds[[v for v in ds.data_vars if v.endswith("_predicted")]]


class ScoreAccumulator:
    """Running CRPS/RMSE over ICs, without holding any forecast in memory.

    Upstream's ``score_forecasts`` sums over the list of ICs it is given and divides by
    the count, so scoring one IC at a time and averaging the results here is exactly
    equivalent to scoring them all at once -- but costs nothing, which matters because
    the whole point of this driver is not to retain per-IC results.
    """

    def __init__(self, upstream, latitude, var_names, hourly_resolution):
        self._up = upstream
        self._lat = latitude
        self._vars = var_names
        self._hres = hourly_resolution
        self._sums = None
        self._n = 0

    def add(self, result):
        s = self._up.score_forecasts([result], self._lat, self._vars, self._hres)
        if self._sums is None:
            self._sums = {k: {v: np.asarray(a, dtype=float).copy() for v, a in d.items()} for k, d in s.items()}
        else:
            for k, d in s.items():
                for v, a in d.items():
                    self._sums[k][v] += np.asarray(a, dtype=float)
        self._n += 1

    def result(self):
        if not self._n:
            return None
        return {k: {v: a / self._n for v, a in d.items()} for k, d in self._sums.items()}


def run(args):
    up = load_upstream(Path(args.ucast_repo).expanduser())
    import torch
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = yaml.safe_load(open(args.config_path))
    dm_cfg = cfg["datamodule"]
    var_names = list(dm_cfg["input_vars"])
    window = dm_cfg["window"]
    hourly_resolution = dm_cfg["hourly_resolution"]

    missing = [v for v in KEEP_VARS if v not in var_names]
    if missing:
        raise SystemExit(f"Requested fields absent from the model's variables: {missing}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Must happen before the model is built.
    up.set_circular_height_padding()

    ds = open_era5_stores(args.data_glob)
    ds = up.ensure_latitude_is_ascending(ds)
    log.info(f"Dataset: {len(ds.time)} timesteps, {str(ds.time.values[0])[:16]} -> {str(ds.time.values[-1])[:16]}")

    normalizer = up.setup_normalizer(args.stats_dir, var_names, ds, device=args.device)
    static_cond = up.load_static_conditions(ds, list(dm_cfg["static_fields"]))

    ckpt = up._resolve_path(args.ckpt_path)
    model = up.load_model_from_checkpoint(ckpt, cfg, device=args.device)
    log.info(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ic_dates = build_ic_dates(ds, args.start, args.end, args.ic_hours, args.limit_ics)
    log.info(f"{len(ic_dates)} initial conditions, {args.start}..{args.end} at hours {args.ic_hours}")
    if not ic_dates:
        raise SystemExit("No initial conditions selected.")

    keep_leads = [np.timedelta64(int(h), "h") for h in args.lead_hours]
    max_lead = max(args.lead_hours)
    if max_lead > args.prediction_horizon * hourly_resolution:
        raise SystemExit(
            f"--lead-hours asks for {max_lead}h but horizon {args.prediction_horizon} "
            f"only reaches {args.prediction_horizon * hourly_resolution}h."
        )

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = (len(ic_dates) + args.chunk_size - 1) // args.chunk_size
    written = []
    peak_mem = 0.0
    scorer = ScoreAccumulator(up, ds.latitude.values, KEEP_VARS, hourly_resolution) if args.score else None

    for chunk_idx in range(n_chunks):
        chunk = ic_dates[chunk_idx * args.chunk_size : (chunk_idx + 1) * args.chunk_size]
        chunk_path = out_dir / f"{args.prefix}_chunk{chunk_idx:03d}.nc"
        if chunk_path.exists() and args.skip_existing:
            log.info(f"[chunk {chunk_idx + 1}/{n_chunks}] exists, skipping: {chunk_path.name}")
            written.append(chunk_path)
            continue

        per_ic = []
        for i, ic_dt in enumerate(chunk):
            batch = up.extract_batch_for_ic(
                ds=ds,
                ic_datetime=ic_dt,
                var_names=var_names,
                static_cond=static_cond,
                normalizer=normalizer,
                window=window,
                prediction_horizon=args.prediction_horizon,
                hourly_resolution=hourly_resolution,
                device=args.device,
            )
            with torch.no_grad():
                results = up.run_autoregressive_inference(
                    model=model,
                    batch=batch,
                    var_names=var_names,
                    normalizer=normalizer,
                    window=window,
                    prediction_horizon=args.prediction_horizon,
                    ensemble_size=args.ensemble_size,
                    device=args.device,
                )
            # Passing KEEP_VARS rather than var_names is what keeps this run in memory:
            # only these fields are converted to numpy and retained.
            ic_ds = up.build_xarray_dataset(results, batch, ic_dt, hourly_resolution, KEEP_VARS)
            ic_ds = drop_targets(ic_ds.sel(lead_time=keep_leads))
            per_ic.append(ic_ds.expand_dims("init_time"))

            if scorer is not None:
                # Scored at every step of the rollout, not just the saved lead times.
                scorer.add(results)

            del results, batch
            if args.device.startswith("cuda"):
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e9)

            if (i + 1) % 10 == 0 or i + 1 == len(chunk):
                log.info(f"[chunk {chunk_idx + 1}/{n_chunks}] {i + 1}/{len(chunk)} ICs")

        chunk_ds = xr.concat(per_ic, dim="init_time")
        chunk_ds.to_netcdf(chunk_path)
        size_gb = chunk_path.stat().st_size / 1e9
        log.info(f"[chunk {chunk_idx + 1}/{n_chunks}] wrote {chunk_path.name} ({size_gb:.2f} GB)")
        written.append(chunk_path)
        del per_ic, chunk_ds

    if peak_mem:
        log.info(f"Peak CUDA memory: {peak_mem:.2f} GB")

    if scorer is not None:
        scores = scorer.result()
        lead_hours = np.arange(1, args.prediction_horizon + 1) * hourly_resolution
        log.info("=" * 62)
        log.info(f"SCORES (area-weighted, averaged over {len(ic_dates)} ICs)")
        log.info("=" * 62)
        for var in KEEP_VARS:
            log.info(f"  {var}:")
            for h_idx, h in enumerate(lead_hours):
                if int(h) in args.report_lead_hours:
                    log.info(
                        f"    Lead {int(h):4d}h:  CRPS={scores['crps'][var][h_idx]:.4f}  "
                        f"RMSE={scores['rmse'][var][h_idx]:.4f}"
                    )
        if args.wandb_project:
            up.log_scores_to_wandb(
                scores=scores,
                var_names=KEEP_VARS,
                hourly_resolution=hourly_resolution,
                wandb_project=args.wandb_project,
                # Never leave this to the default: upstream falls back to the generic
                # "era5_inference", which collides with anyone else's runs.
                wandb_run_name=args.wandb_run_name or f"{args.prefix}-h{args.prediction_horizon}-e{args.ensemble_size}",
                # Must be explicit: the account's default entity is a personal one.
                wandb_entity=args.wandb_entity,
                extra_config={
                    "ensemble_size": args.ensemble_size,
                    "prediction_horizon": args.prediction_horizon,
                    "n_ics": len(ic_dates),
                    "ic_hours": args.ic_hours,
                    "saved_vars": KEEP_VARS,
                    "saved_lead_hours": args.lead_hours,
                    "ckpt": str(args.ckpt_path),
                },
            )

    log.info(f"Done. {len(written)} chunk file(s) in {out_dir}")
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ucast-repo", default="~/u-cast", help="Path to the u-cast clone.")
    p.add_argument("--ckpt-path", required=True, help="Checkpoint path, or hf:<repo>/<file>.")
    p.add_argument(
        "--config-path",
        default="configs/config_inference.yaml",
        help="Relative to --ucast-repo. Upstream's own default (configs/config.yaml) does not exist.",
    )
    p.add_argument("--stats-dir", default="data/stats", help="Relative to --ucast-repo.")
    p.add_argument("--data-glob", required=True, help="Glob for the ERA5 zarr store(s).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", default="ucast2020")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2020-12-31T23:59:59")
    p.add_argument("--ic-hours", type=int, nargs="+", default=[0, 12], help="UTC hours to use as ICs.")
    p.add_argument("--prediction-horizon", type=int, default=20, help="Autoregressive steps (20 x 12h = 240h).")
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--lead-hours", type=int, nargs="+", default=DEFAULT_LEAD_HOURS)
    p.add_argument("--chunk-size", type=int, default=50, help="ICs per output file.")
    p.add_argument("--limit-ics", type=int, default=None, help="Cap IC count (testing).")
    p.add_argument("--skip-existing", action="store_true", help="Resume: skip chunks already written.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--score", action="store_true", help="Area-weighted CRPS/RMSE, accumulated per IC.")
    p.add_argument(
        "--report-lead-hours",
        type=int,
        nargs="+",
        default=[24, 72, 240],
        help="Lead times to print (default 1d/3d/10d, matching the paper's Table 1).",
    )
    p.add_argument("--wandb-project", default=None, help="e.g. ucast-inference. Requires --score.")
    p.add_argument(
        "--wandb-entity",
        default=None,
        help="Pass weather-realism-pmlr explicitly; the account default is a personal entity.",
    )
    p.add_argument("--wandb-run-name", default=None)
    args = p.parse_args()

    repo = Path(args.ucast_repo).expanduser()
    if not Path(args.config_path).is_absolute():
        args.config_path = str(repo / args.config_path)
    args.output_dir = str(Path(args.output_dir).expanduser())
    args.data_glob = str(Path(args.data_glob).expanduser())
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Turn raw U-Cast chunk files into baseline-shaped forecast files.

``run_ucast_2020.py`` writes chunks in upstream's own layout (``init_time`` /
``lead_time``, ``*_predicted`` names, pressure levels flattened into the variable name).
The comparison models in ``team07/data/`` use a different convention, established by the
existing files:

    surface  (*_surf_*.nc)   2m_temperature, 10m_u_component_of_wind,
                             10m_v_component_of_wind, mean_sea_level_pressure
                             dims (time, prediction_timedelta, longitude, latitude)

    nonsurf  (nonsurf/*.nc)  temperature, u_component_of_wind,
                             v_component_of_wind, specific_humidity
                             same dims, plus a scalar level=850 coord

This script converts to that layout. The one intentional difference is an extra
``ensemble_member`` dimension -- the baselines are deterministic, U-Cast is not -- so
comparison code selects a member with ``ds.isel(ensemble_member=i)``.

geopotential_500 goes to its own file. Folding it into nonsurf would need a ragged
level axis (z at 500, everything else at 850), which would break parity with the
baseline nonsurf files for no benefit; it exists only to check our run against the
paper's Table 1.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import xarray as xr

SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]
LEVEL_VARS = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]
# Baseline dim order, with ensemble_member inserted where it does least harm.
DIM_ORDER = ["time", "prediction_timedelta", "ensemble_member", "longitude", "latitude"]


def load_chunks(pattern: str) -> xr.Dataset:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No chunk files matched {pattern!r}")
    print(f"Concatenating {len(paths)} chunk file(s)")
    ds = xr.open_mfdataset(paths, combine="by_coords", data_vars="minimal")
    # Upstream names every forecast variable "<var>_predicted"; targets were already
    # dropped at write time.
    ds = ds.rename({v: v[: -len("_predicted")] for v in ds.data_vars if v.endswith("_predicted")})
    return ds.rename({"init_time": "time", "lead_time": "prediction_timedelta"})


def _finish(ds: xr.Dataset) -> xr.Dataset:
    order = [d for d in DIM_ORDER if d in ds.dims]
    return ds.transpose(*order)


def split_outputs(ds: xr.Dataset, level_hpa: int = 850):
    """Return (surface, level, z500) datasets in baseline layout."""
    surface = _finish(ds[[v for v in SURFACE_VARS if v in ds]])

    suffix = f"_{level_hpa}"
    level_map = {f"{v}{suffix}": v for v in LEVEL_VARS if f"{v}{suffix}" in ds}
    level = ds[list(level_map)].rename(level_map)
    level = _finish(level).assign_coords(level=level_hpa)

    z500 = None
    if "geopotential_500" in ds:
        z500 = ds[["geopotential_500"]].rename({"geopotential_500": "geopotential"})
        z500 = _finish(z500).assign_coords(level=500)

    return surface, level, z500


def write(ds: xr.Dataset, path: Path, compress: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoding = None
    if compress:
        encoding = {v: {"zlib": True, "complevel": 1} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=encoding)
    print(f"  wrote {path}  ({path.stat().st_size / 1e9:.2f} GB)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunks", required=True, help="Glob for the raw chunk files.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tag", default="ucast", help="Leading token of the output filenames.")
    p.add_argument("--span", default="2020-01-01_2020-12-31", help="Date span used in the filenames.")
    p.add_argument("--level", type=int, default=850)
    p.add_argument("--compress", action="store_true", help="zlib level 1; roughly halves size.")
    args = p.parse_args()

    out = Path(args.output_dir).expanduser()
    ds = load_chunks(str(Path(args.chunks).expanduser()))
    print(f"  {ds.sizes.get('time', 0)} init times, {ds.sizes.get('prediction_timedelta', 0)} lead times, "
          f"{ds.sizes.get('ensemble_member', 0)} members")

    surface, level, z500 = split_outputs(ds, args.level)

    # "6steps" in the baseline filenames is inherited from their download tooling; kept
    # so the names sort alongside the existing files.
    write(surface, out / f"{args.tag}_6steps_surf_1.5deg_{args.span}.nc", args.compress)
    write(level, out / "nonsurf" / f"{args.tag}_6steps_1.5deg_{args.span}.nc", args.compress)
    if z500 is not None:
        write(z500, out / f"{args.tag}_z500_1.5deg_{args.span}.nc", args.compress)
    print("Done.")


if __name__ == "__main__":
    main()

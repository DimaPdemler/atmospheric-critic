#!/usr/bin/env python3
"""Download WeatherBench2 forecast model data (GraphCast, Pangu, etc.) as NetCDF.

The forecast zarr stores have an extra `prediction_timedelta` dimension. This
script selects a single lead time and saves with only the `time` dimension so
the output matches the ERA5 format used by AtmosphereDataset.
"""
import argparse
from pathlib import Path
import numpy as np
import xarray as xr


SURFACE_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]

# 5-var order expected by the SFNO encoder (adds 6h-accumulated precipitation).
# Pangu does NOT forecast precipitation, so this only applies to ERA5 + GraphCast.
SFNO_SURFACE_VARS = SURFACE_VARS + ["total_precipitation_6hr"]

# Known WeatherBench2 GCS paths at 240x121 (1.5 deg, equiangular with poles)
SOURCES = {
    "pangu": "gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr",
    # GraphCast is split by year; override with --source if you need a different range
    "graphcast": "gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr",
    # ERA5 reanalysis (no prediction_timedelta — handled as analysis data). Has
    # total_precipitation_6hr, so it can supply the SFNO 5th channel.
    "era5": "gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr",
}


def download_forecast(
    source: str,
    output_path: Path,
    variables: list[str],
    time_start: str,
    time_end: str,
    lead_hours: int,
):
    """Download one lead time of a WeatherBench2 forecast zarr and save as NetCDF.

    Selects the requested variables and ``lead_hours`` slice, converts init time
    to valid time, slices the time range, and writes an ERA5-compatible file.
    """
    print(f"Opening {source}...")
    ds = xr.open_zarr(source, chunks="auto", decode_timedelta=True)

    print(f"Available variables: {list(ds.data_vars)}")
    print(f"Dimensions: {dict(ds.dims)}")

    # Select requested variables
    available = [v for v in variables if v in ds]
    missing = [v for v in variables if v not in ds]
    if missing:
        print(f"Warning: variables not found in dataset: {missing}")
    if not available:
        raise SystemExit("None of the requested variables are available.")
    ds = ds[available]

    # Select lead time
    target_delta = np.timedelta64(lead_hours, "h")
    if "prediction_timedelta" in ds.dims:
        available_deltas = ds.prediction_timedelta.values
        if target_delta not in available_deltas:
            hours = [int(d / np.timedelta64(1, "h")) for d in available_deltas]
            raise SystemExit(
                f"Lead time {lead_hours}h not found. Available: {hours}h"
            )
        ds = ds.sel(prediction_timedelta=target_delta)
        # Convert init time → valid time so the dimension matches ERA5
        ds = ds.assign_coords(time=ds.time.values + target_delta)
        print(f"Selected lead time {lead_hours}h; converted init time → valid time.")
    else:
        print("No prediction_timedelta dimension found; treating as analysis data.")

    # Time slice
    ds = ds.sel(time=slice(time_start, time_end))
    if ds.time.size == 0:
        raise SystemExit(f"No data in time range {time_start} – {time_end}.")

    print(f"Saving {ds.time.size} timesteps → {output_path}")
    ds.to_netcdf(output_path, format="NETCDF4")
    print("Done!")


def main():
    """Parse CLI arguments and download the selected forecast model's surface data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        choices=list(SOURCES.keys()),
        help="Forecast model to download",
    )
    parser.add_argument("output", type=Path, help="Output .nc file path")
    parser.add_argument(
        "--source",
        default=None,
        help="Override GCS zarr source URL (uses built-in default if omitted)",
    )
    parser.add_argument("-s", "--time-start", required=True, help="YYYY-MM-DD")
    parser.add_argument("-e", "--time-end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "-v",
        "--variables",
        nargs="+",
        default=SURFACE_VARS,
        help="Variables to download (default: 4 surface vars matching ERA5)",
    )
    parser.add_argument(
        "--sfno-vars",
        action="store_true",
        help="Download the 5 surface vars (incl. total_precipitation_6hr) in the "
             "order the SFNO encoder expects. Overrides -v. Not valid for Pangu.",
    )
    parser.add_argument(
        "--lead-hours",
        type=int,
        default=24,
        help="Forecast lead time in hours to extract (default: 24)",
    )
    args = parser.parse_args()

    if args.sfno_vars:
        if args.model == "pangu":
            raise SystemExit("Pangu has no precipitation variable; --sfno-vars is unsupported.")
        variables = SFNO_SURFACE_VARS
    else:
        variables = args.variables

    source = args.source or SOURCES[args.model]
    download_forecast(
        source,
        args.output,
        variables,
        args.time_start,
        args.time_end,
        args.lead_hours,
    )


if __name__ == "__main__":
    main()

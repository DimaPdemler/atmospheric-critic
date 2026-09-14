#!/usr/bin/env python3
"""Convert a SWIFT forecast Zarr store to the baseline pipeline's NetCDF layout.

SWIFT stores forecasts as ``init_time × member × prediction_timedelta ×
latitude × longitude``. The discriminator pipeline expects ``time`` as the
initialization coordinate and no singleton member dimension. Initialization
times are deliberately preserved: valid time is computed downstream as
``time + prediction_timedelta``.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


SURFACE_VARIABLES = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
)


def lead_hours(values):
    """Convert a prediction-timedelta coordinate to integer hours."""
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[h]").astype(int)
    return values.astype(int)


def prepare_swift_forecasts(dataset, variables=SURFACE_VARIABLES, lead_hour_values=None, member=0):
    """Select SWIFT surface fields and normalize its forecast dimensions."""
    missing = [variable for variable in variables if variable not in dataset.data_vars]
    if missing:
        raise ValueError(f"SWIFT Zarr is missing requested variables: {missing}")
    if "init_time" not in dataset.dims:
        raise ValueError("SWIFT Zarr must have an init_time dimension.")
    if "prediction_timedelta" not in dataset.dims:
        raise ValueError("SWIFT Zarr must have a prediction_timedelta dimension.")

    selected = dataset[list(variables)]
    if "member" in selected.dims:
        available_members = np.asarray(selected.member.values)
        if member not in available_members:
            raise ValueError(f"Requested member={member}; available members: {available_members.tolist()}")
        selected = selected.sel(member=member, drop=True)

    available_hours = lead_hours(selected.prediction_timedelta.values)
    if lead_hour_values is not None:
        requested_hours = np.asarray(lead_hour_values, dtype=int)
        missing_hours = sorted(set(requested_hours) - set(available_hours))
        if missing_hours:
            raise ValueError(
                f"Requested lead hours are unavailable: {missing_hours}; "
                f"available: {available_hours.tolist()}"
            )
        selected = selected.isel(
            prediction_timedelta=np.flatnonzero(np.isin(available_hours, requested_hours))
        )

    selected = selected.rename({"init_time": "time"})
    # Zarr's xarray metadata is an internal storage attribute, not a NetCDF
    # attribute. It still refers to ``init_time`` after the rename and netCDF4
    # rejects it while writing the variable attributes.
    for name in selected.variables:
        selected[name].attrs.pop("_ARRAY_DIMENSIONS", None)
    return selected.transpose("time", "prediction_timedelta", "latitude", "longitude")


def forecast_encoding(dataset, compression_level):
    """Storage layout tuned for how the pipeline actually reads these files.

    Every consumer reads one (time, lead) field at a time, in shuffled order.
    Compressed variables whose chunks span many timesteps make that pathological:
    the first SWIFT export used zlib with HDF5's auto-chosen [244, 2, 41, 80]
    chunks, so a single field read decompressed chunks covering 244 timesteps,
    and discriminator training ran at 7.8 s/step against 0.16 s/step for the
    uncompressed GraphCast file -- a ~49x penalty for a ~25% space saving.

    So default to uncompressed and contiguous, matching the other forecast files.
    When compression is explicitly requested, chunk one timestep and lead at a
    time so a read decompresses exactly the field it asked for.
    """
    encoding = {}
    for variable in dataset.data_vars:
        array = dataset[variable]
        if compression_level <= 0:
            encoding[variable] = {"zlib": False, "complevel": 0, "contiguous": True}
            continue
        sizes = [1 if dimension in ("time", "prediction_timedelta") else array.sizes[dimension]
                 for dimension in array.dims]
        encoding[variable] = {
            "zlib": True, "complevel": compression_level, "chunksizes": tuple(sizes),
        }
    return encoding


def sample_oriented_netcdf_encoding(dataset, compression_level=4):
    """Backward-compatible name for the sample-oriented forecast encoding."""
    return forecast_encoding(dataset, int(compression_level))


def convert_swift_zarr(input_path, output_path, variables=SURFACE_VARIABLES, lead_hour_values=None,
                       member=0, compression_level=4, overwrite=False):
    """Write a selected SWIFT forecast Zarr subset as a pipeline-compatible NetCDF."""
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"No such Zarr store: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Opening SWIFT Zarr: {input_path}")
    dataset = xr.open_zarr(input_path, chunks="auto", decode_timedelta=True)
    try:
        converted = prepare_swift_forecasts(dataset, variables, lead_hour_values, member)
        print(f"Writing {dict(converted.sizes)}; leads={lead_hours(converted.prediction_timedelta.values).tolist()} h")
        encoding = forecast_encoding(converted, int(compression_level))
        temporary_output = output_path.with_name(f".{output_path.name}.partial")
        try:
            temporary_output.unlink(missing_ok=True)
            converted.to_netcdf(temporary_output, format="NETCDF4", encoding=encoding)
            temporary_output.replace(output_path)
        except BaseException:
            temporary_output.unlink(missing_ok=True)
            raise
    finally:
        dataset.close()
    print(f"Wrote pipeline-compatible SWIFT forecast: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input SWIFT .zarr directory")
    parser.add_argument("output", type=Path, help="Output forecast .nc file")
    parser.add_argument("--lead-hours", type=int, nargs="+", default=None,
                        help="Forecast leads to retain. Defaults to every available lead.")
    parser.add_argument("--member", type=int, default=0, help="Ensemble member to export (default: 0)")
    parser.add_argument("--compression-level", type=int, default=0,
                        help="NetCDF zlib compression level, 0-9 (default: 0, uncompressed). "
                             "Non-zero chunks per timestep so reads stay cheap.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    arguments = parser.parse_args()
    if not 0 <= arguments.compression_level <= 9:
        parser.error("--compression-level must be between 0 and 9")
    convert_swift_zarr(
        arguments.input, arguments.output, lead_hour_values=arguments.lead_hours,
        member=arguments.member, compression_level=arguments.compression_level,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()

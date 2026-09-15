#!/bin/bash
# Download a small local test slice of Pangu and GraphCast surface data.
# Matches the variable set and format of data/test_data_local.nc (ERA5 surface).
#
# Requires: gcsfs, zarr, xarray in the pmlr conda env.
# Install if missing:
#   /opt/miniconda3/envs/pmlr/bin/pip install gcsfs

set -eo pipefail

PYTHON=/opt/miniconda3/envs/pmlr/bin/python
SCRIPT="$(dirname "$0")/download_forecast_netcdf.py"
OUTPUT_DIR="$(dirname "$0")/../data"

mkdir -p "$OUTPUT_DIR"

# 2020: full test year at 24h lead time
TIME_START="2020-01-01"
TIME_END="2020-12-31"
LEAD_HOURS=24

echo "=== Downloading Pangu surface data ==="
$PYTHON "$SCRIPT" pangu \
    "$OUTPUT_DIR/pangu_surface_2020_lead${LEAD_HOURS}h.nc" \
    -s "$TIME_START" -e "$TIME_END" \
    --lead-hours "$LEAD_HOURS"

echo ""
echo "=== Downloading GraphCast surface data ==="
$PYTHON "$SCRIPT" graphcast \
    "$OUTPUT_DIR/graphcast_surface_2020_lead${LEAD_HOURS}h.nc" \
    -s "$TIME_START" -e "$TIME_END" \
    --lead-hours "$LEAD_HOURS"

echo ""
echo "Done. Files written to $OUTPUT_DIR/"

#!/usr/bin/env python3
"""Split ensemble U-Cast files into per-member deterministic files.

The discriminator baseline pipeline consumes forecast files shaped like the other
comparison models: one field per (time, prediction_timedelta), with no ensemble axis.
Our U-Cast output carries an ``ensemble_member`` dimension, so this writes each member
out separately, byte-compatible with the deterministic baselines.

Register the members in ``Discriminator/conf/baseline_config.yaml`` under
``forecast_files``. Registering a single member gives the cleanest like-for-like
comparison against the deterministic models; registering several lets the realism
metric see the ensemble spread.

Do not average the members into an ensemble mean and score that. Averaging reintroduces
exactly the blurring that the deterministic baselines already suffer from, so it would
manufacture the very result the comparison is meant to test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr


def split(path: Path, out_dir: Path, members, tag: str, compress: bool):
    ds = xr.open_dataset(path)
    if "ensemble_member" not in ds.dims:
        raise SystemExit(f"{path.name} has no ensemble_member dimension.")
    available = list(range(ds.sizes["ensemble_member"]))
    chosen = members or available
    bad = [m for m in chosen if m not in available]
    if bad:
        raise SystemExit(f"Members {bad} absent; file has {available}.")

    written = []
    for m in chosen:
        sub = ds.isel(ensemble_member=m, drop=True)
        # "<tag>_..." -> "<tag>-m<N>_..." so the files sort next to the baselines and
        # each member is unambiguous in the pipeline's model list.
        stem = path.name.replace(f"{tag}_", f"{tag}-m{m}_", 1)
        dest = out_dir / stem
        dest.parent.mkdir(parents=True, exist_ok=True)
        encoding = {v: {"zlib": True, "complevel": 1} for v in sub.data_vars} if compress else None
        sub.to_netcdf(dest, encoding=encoding)
        written.append(dest)
        print(f"  member {m} -> {dest}  ({dest.stat().st_size / 1e9:.2f} GB)")
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="Baseline-shaped files carrying ensemble_member.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--members", type=int, nargs="+", default=None, help="Default: all.")
    p.add_argument("--tag", default="ucast", help="Leading filename token to rewrite.")
    p.add_argument("--compress", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir).expanduser()
    for raw in args.inputs:
        src = Path(raw).expanduser()
        if not src.is_file():
            raise SystemExit(f"Missing input: {src}")
        # Preserve the nonsurf/ subdirectory split that the baselines use.
        dest_dir = out_dir / "nonsurf" if src.parent.name == "nonsurf" else out_dir
        print(f"{src.name}:")
        split(src, dest_dir, args.members, args.tag, args.compress)
    print("Done.")


if __name__ == "__main__":
    main()

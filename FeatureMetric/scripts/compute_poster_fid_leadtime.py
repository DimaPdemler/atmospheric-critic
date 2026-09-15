"""Compute latent-FID(ERA5-GT, forecast_at_lead) for the poster's bottom panel.

For each forecast model (IFS HRES, GraphCast, Pangu-Weather) and each lead
time {6, 12, 24, 48, 96, 192} hours, this script computes FID in two modes:

Non-temporal (default):
  Uses the standard 4-channel I-JEPA from results/may_07_512_encoder/.
  Each snapshot is normalized with abs_mean/abs_std and fed directly to the
  encoder — no prior frame needed.

Temporal (--temporal flag):
  Uses the 8-channel temporal-exp3 I-JEPA from
  results/may_13_temporal_exp3_phase_d512_maxpool/.
  Builds a temporal pair (X_t = forecast, X_{t-24h} = ERA5-GT ground truth)
  and feeds the phase-composed 8-channel tensor to the target encoder.

Output: plots/poster_fid_leadtime_data_nontemporal.npz (non-temporal) or
        plots/poster_fid_leadtime_data_temporal.npz (--temporal).
Override with --output.

The plotting script (scripts/plot_poster_fid_combined.py) imports neither
torch nor any project models, so teammates can iterate on style freely.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import argparse
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from utils.model_io import build_model, load_model_checkpoint
from utils.features import extract_features_for_loader
from utils.temporal import compose_temporal_input
from eval.eval_distances import calculate_frechet_distance


VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]

LEAD_HOURS = (6, 12, 24, 48, 96, 192)
PRIOR_DELTA_HOURS = 24


class InMemoryRawSource:
    """Loads (T, 4, H, W) raw float32 into memory once and exposes
    ``read_raw(idx) -> (4, H, W)``. Matches the contract used by the
    eval-time TemporalPairDataset in ``eval/eval_real_vs_forecast.py``.

    Forecast files: pass ``lead_hours`` to slice a single
    ``prediction_timedelta`` and shift the time coord to valid time.
    """

    def __init__(self, nc_path, lead_hours=None):
        ds = xr.open_dataset(nc_path)
        if lead_hours is not None:
            if "prediction_timedelta" not in ds.dims:
                raise ValueError(f"{nc_path}: no prediction_timedelta dim")
            td = np.timedelta64(int(lead_hours), "h")
            ds = ds.sel(prediction_timedelta=td)
            ds = ds.assign_coords(time=ds.time + td)
        ds = ds.transpose("time", "latitude", "longitude")
        arrs = [ds[v].values.astype(np.float32) for v in VARIABLES]
        self.data = np.stack(arrs, axis=1)  # (T, 4, H, W)
        self.times = pd.DatetimeIndex(ds.time.values)
        ds.close()

    def read_raw(self, idx):
        """Return the raw ``(4, H, W)`` field at time index ``idx``."""
        return self.data[idx]

    def __len__(self):
        """Number of time steps held in memory."""
        return self.data.shape[0]


class SimpleNormalizedDataset(Dataset):
    """4-channel normalized dataset for the non-temporal model path."""

    def __init__(self, src, indices, abs_mean, abs_std):
        self.src = src
        self.indices = list(indices)
        self.abs_mean = abs_mean
        self.abs_std = abs_std

    def __len__(self):
        """Number of selected snapshots."""
        return len(self.indices)

    def __getitem__(self, i):
        """Return the normalized 4-channel ("none"-mode) input for snapshot ``i``."""
        raw = self.src.read_raw(self.indices[i])  # (4, H, W)
        sample = compose_temporal_input(raw, None, "none", self.abs_mean, self.abs_std)
        return torch.from_numpy(sample)


class TemporalPairDataset(Dataset):
    """Composes 8-channel temporal-phase input from (prior, present) pairs."""

    def __init__(self, prior_src, present_src, prior_idx, present_idx,
                 abs_mean, abs_std, diff_mean, diff_std):
        if len(prior_idx) != len(present_idx):
            raise ValueError("prior_idx and present_idx must have the same length")
        self.prior_src = prior_src
        self.present_src = present_src
        self.prior_idx = list(prior_idx)
        self.present_idx = list(present_idx)
        self.abs_mean = abs_mean
        self.abs_std = abs_std
        self.diff_mean = diff_mean
        self.diff_std = diff_std

    def __len__(self):
        """Number of (prior, present) pairs."""
        return len(self.prior_idx)

    def __getitem__(self, i):
        """Compose the 8-channel phase input from the prior/present pair at index ``i``."""
        prior = self.prior_src.read_raw(self.prior_idx[i])
        present = self.present_src.read_raw(self.present_idx[i])
        sample = compose_temporal_input(
            present, prior, "phase",
            self.abs_mean, self.abs_std, self.diff_mean, self.diff_std,
        )
        return torch.from_numpy(sample)


def build_time_lookup(times):
    """Map each timestamp to its integer index for O(1) time-based lookups."""
    return {pd.Timestamp(t): i for i, t in enumerate(pd.DatetimeIndex(times))}


def build_paired_indices(present_times, prior_lookup, prior_offset_hours=24):
    """For each present_time t, find prior_idx where prior_times == t - Δt exactly.

    Returns (present_idx, prior_idx, dropped). Drops samples whose prior is
    not present in the lookup (boundary effects at the start of the year).
    """
    offset = pd.Timedelta(hours=prior_offset_hours)
    present_out, prior_out = [], []
    dropped = 0
    for i, t in enumerate(pd.DatetimeIndex(present_times)):
        j = prior_lookup.get(pd.Timestamp(t) - offset)
        if j is None:
            dropped += 1
            continue
        present_out.append(i)
        prior_out.append(j)
    return present_out, prior_out, dropped


def cap_pairs(present_idx, prior_idx, n, rng):
    """Randomly subsample a matched (present, prior) index pair down to ``n`` entries."""
    total = len(present_idx)
    if n >= total:
        return present_idx, prior_idx
    perm = rng.permutation(total)[:n].tolist()
    return [present_idx[i] for i in perm], [prior_idx[i] for i in perm]


def features_for_pairs(model, prior_src, present_src, prior_idx, present_idx,
                       abs_mean, abs_std, diff_mean, diff_std, device,
                       batch_size, num_workers, label):
    """Temporal path: compose phase pairs and extract encoder features for them."""
    pair_ds = TemporalPairDataset(
        prior_src, present_src, prior_idx, present_idx,
        abs_mean, abs_std, diff_mean, diff_std,
    )
    loader = DataLoader(
        pair_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    print(f"  Extracting features: {label} ({len(pair_ds)} pairs)...")
    return extract_features_for_loader(model, loader, device)


def features_for_source(model, src, indices, abs_mean, abs_std, device,
                        batch_size, num_workers, label):
    """Non-temporal path: normalize 4-channel snapshots and extract features."""
    ds = SimpleNormalizedDataset(src, indices, abs_mean, abs_std)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    print(f"  Extracting features: {label} ({len(ds)} samples)...")
    return extract_features_for_loader(model, loader, device)


def parse_args():
    """Parse CLI arguments for the lead-time FID computation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ifs", type=Path,
        default=Path("data/forecasts_2020/ifs_hres_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc"),
    )
    parser.add_argument(
        "--graphcast", type=Path,
        default=Path("data/forecasts_2020/graphcast_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc"),
    )
    parser.add_argument(
        "--pangu", type=Path,
        default=Path("data/forecasts_2020/pangu_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc"),
    )
    parser.add_argument(
        "--era5-gt", type=Path,
        default=Path("data/forecasts_2020/era5-gt_6steps_surf_1.5deg_2020-01-01_2020-12-31.nc"),
    )
    # Non-temporal model (default path).
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("results/may_07_512_encoder"),
        help="Model directory for the non-temporal 4-channel I-JEPA (used unless --temporal).",
    )
    # Temporal model path (only loaded when --temporal is set).
    parser.add_argument(
        "--temporal-model-dir", type=Path,
        default=Path("results/may_13_temporal_exp3_phase_d512_maxpool"),
        help="Model directory for the temporal 8-channel I-JEPA (used with --temporal).",
    )
    parser.add_argument(
        "--temporal", action="store_true",
        help="Use the 8-channel temporal-exp3 model instead of the 4-channel non-temporal one.",
    )
    parser.add_argument("--n-samples", type=int, default=400,
                        help="Cap per (model, lead) FID point; same cap is used for ERA5 ref.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .npz path. Defaults to plots/poster_fid_leadtime_data_nontemporal.npz "
                             "(non-temporal) or plots/poster_fid_leadtime_data_temporal.npz "
                             "(--temporal).")
    return parser.parse_args()


def main():
    """Compute FID(ERA5-GT, forecast) for each forecast model × lead time and save to .npz."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Default output path depends on which model is used.
    if args.output is None:
        if args.temporal:
            args.output = Path("plots/poster_fid_leadtime_data_temporal.npz")
        else:
            args.output = Path("plots/poster_fid_leadtime_data_nontemporal.npz")

    # -----------------------------------------------------------------------
    # Load model and stats
    # -----------------------------------------------------------------------
    if args.temporal:
        model_dir = args.temporal_model_dir
        model_path = model_dir / "best_ijepa_model_twin_d512_tm-phase.pth"
        abs_mean  = np.load(model_dir / "data_mean.npy")
        abs_std   = np.load(model_dir / "data_std.npy")
        diff_mean = np.load(model_dir / "diff_mean_dt24h.npy")
        diff_std  = np.load(model_dir / "diff_std_dt24h.npy")
        print(f"\nMode: temporal (8-channel phase, {model_dir.name})")
        print(f"Loading model: {model_path}")
        model = build_model("ijepa", device=device, model_size="twin",
                            embed_dim=512, in_chans=8)
    else:
        model_dir = args.model_dir
        model_path = model_dir / "best_ijepa_model_twin_d512.pth"
        abs_mean = np.load(model_dir / "data_mean.npy")
        abs_std  = np.load(model_dir / "data_std.npy")
        diff_mean = diff_std = None
        print(f"\nMode: non-temporal (4-channel, {model_dir.name})")
        print(f"Loading model: {model_path}")
        model = build_model("ijepa", device=device, model_size="twin",
                            embed_dim=512, in_chans=4)

    model = load_model_checkpoint("ijepa", model, model_path, device)
    model.eval()

    # -----------------------------------------------------------------------
    # ERA5-GT source
    # -----------------------------------------------------------------------
    print(f"\nLoading ERA5-GT into memory: {args.era5_gt}")
    era5_src = InMemoryRawSource(args.era5_gt)
    print(f"  {len(era5_src)} samples, range {era5_src.times[0]}..{era5_src.times[-1]}")

    # -----------------------------------------------------------------------
    # ERA5 reference distribution
    # -----------------------------------------------------------------------
    if args.temporal:
        era5_lookup = build_time_lookup(era5_src.times)
        print("\nBuilding ERA5 reference temporal pairs...")
        ref_present_all, ref_prior_all, ref_dropped = build_paired_indices(
            era5_src.times, era5_lookup, PRIOR_DELTA_HOURS,
        )
        print(f"  {len(ref_present_all)} pairs available ({ref_dropped} dropped at boundary)")
        ref_present, ref_prior = cap_pairs(ref_present_all, ref_prior_all, args.n_samples, rng)
        print(f"  Capping to {len(ref_present)} pairs")
        print("\nExtracting ERA5 reference features (once)...")
        z_ref = features_for_pairs(
            model, era5_src, era5_src, ref_prior, ref_present,
            abs_mean, abs_std, diff_mean, diff_std,
            device, args.batch_size, args.num_workers, "ERA5 reference",
        )
        n_ref = len(ref_present)
    else:
        total = len(era5_src)
        ref_indices = rng.choice(total, size=min(args.n_samples, total), replace=False).tolist()
        print(f"\nExtracting ERA5 reference features ({len(ref_indices)} samples)...")
        z_ref = features_for_source(
            model, era5_src, ref_indices, abs_mean, abs_std,
            device, args.batch_size, args.num_workers, "ERA5 reference",
        )
        n_ref = len(ref_indices)
        era5_lookup = build_time_lookup(era5_src.times)  # still needed for time-matching

    z_ref_np = z_ref.numpy()
    mu_ref = np.mean(z_ref_np, axis=0)
    sigma_ref = np.cov(z_ref_np, rowvar=False)
    print(f"  z_ref shape: {z_ref_np.shape}")

    # -----------------------------------------------------------------------
    # Loop over (forecast model, lead time)
    # -----------------------------------------------------------------------
    forecast_paths = {
        "ifs_hres": args.ifs,
        "graphcast": args.graphcast,
        "pangu": args.pangu,
    }
    fid_results = {k: [] for k in forecast_paths}
    n_used_results = {k: [] for k in forecast_paths}

    for lead in LEAD_HOURS:
        print(f"\n{'=' * 60}\nLead time: {lead}h\n{'=' * 60}")
        for model_key, fc_path in forecast_paths.items():
            print(f"\n--- {model_key} @ {lead}h ---")
            fc_src = InMemoryRawSource(fc_path, lead_hours=lead)
            print(f"  {len(fc_src)} forecast snapshots, "
                  f"range {fc_src.times[0]}..{fc_src.times[-1]}")

            if args.temporal:
                fc_present, fc_prior, dropped = build_paired_indices(
                    fc_src.times, era5_lookup, PRIOR_DELTA_HOURS,
                )
                print(f"  {len(fc_present)} forecast→ERA5 pairs ({dropped} dropped)")
                fc_present, fc_prior = cap_pairs(fc_present, fc_prior, args.n_samples, rng)
                print(f"  Capping to {len(fc_present)} pairs")
                z_fc = features_for_pairs(
                    model, era5_src, fc_src, fc_prior, fc_present,
                    abs_mean, abs_std, diff_mean, diff_std,
                    device, args.batch_size, args.num_workers,
                    f"{model_key} @ {lead}h",
                )
                n_used = len(fc_present)
            else:
                total_fc = len(fc_src)
                fc_indices = rng.choice(total_fc, size=min(args.n_samples, total_fc),
                                        replace=False).tolist()
                print(f"  Using {len(fc_indices)} snapshots")
                z_fc = features_for_source(
                    model, fc_src, fc_indices, abs_mean, abs_std,
                    device, args.batch_size, args.num_workers,
                    f"{model_key} @ {lead}h",
                )
                n_used = len(fc_indices)

            z_fc_np = z_fc.numpy()
            mu_fc = np.mean(z_fc_np, axis=0)
            sigma_fc = np.cov(z_fc_np, rowvar=False)
            fid = calculate_frechet_distance(mu_ref, sigma_ref, mu_fc, sigma_fc)
            print(f"  FID: {fid:.4f}")
            fid_results[model_key].append(float(fid))
            n_used_results[model_key].append(int(n_used))
            del fc_src

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        lead_times_hours=np.array(LEAD_HOURS, dtype=np.int64),
        fid_ifs_hres=np.array(fid_results["ifs_hres"], dtype=np.float64),
        fid_graphcast=np.array(fid_results["graphcast"], dtype=np.float64),
        fid_pangu=np.array(fid_results["pangu"], dtype=np.float64),
        n_used_ifs_hres=np.array(n_used_results["ifs_hres"], dtype=np.int64),
        n_used_graphcast=np.array(n_used_results["graphcast"], dtype=np.int64),
        n_used_pangu=np.array(n_used_results["pangu"], dtype=np.int64),
        n_ref=np.int64(n_ref),
        n_samples_cap=np.int64(args.n_samples),
        seed=np.int64(args.seed),
        temporal=bool(args.temporal),
        model_path=str(model_path),
        era5_gt_path=str(args.era5_gt),
        ifs_path=str(args.ifs),
        graphcast_path=str(args.graphcast),
        pangu_path=str(args.pangu),
    )
    print(f"\nSaved: {args.output}")
    print("\nSummary (FID vs ERA5 reference):")
    print(f"  {'lead (h)':>8}  {'IFS HRES':>10}  {'GraphCast':>10}  {'Pangu':>10}")
    for i, lh in enumerate(LEAD_HOURS):
        print(f"  {lh:>8d}  {fid_results['ifs_hres'][i]:>10.4f}  "
              f"{fid_results['graphcast'][i]:>10.4f}  {fid_results['pangu'][i]:>10.4f}")


if __name__ == "__main__":
    main()

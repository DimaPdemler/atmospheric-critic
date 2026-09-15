"""Compute latent-FID(clean ERA5, corrupted ERA5) for the poster's top panel.

Mirrors scripts/compute_poster_fid_leadtime.py: runs in two modes.

Non-temporal (default):
  Uses the standard 4-channel I-JEPA from results/may_07_512_encoder/.
  Each (corrupted) snapshot is normalized with abs_mean/abs_std and fed
  directly to the encoder — no prior frame needed.

Temporal (--temporal flag):
  Uses the 8-channel temporal-exp3 I-JEPA from
  results/may_13_temporal_exp3_phase_d512_maxpool/.
  Corruption is applied to the RAW present frame, then the 8-channel temporal
  input is re-composed via utils/temporal.compose_temporal_input — so the
  absolute (X_t) and diff (X_t - X_{t-24h}) halves of the model input stay
  consistent. Applying the corruption to the already-composed 8-channel
  tensor would leave the diff channels reflecting the clean past, which
  pushes the model far out-of-distribution for reasons unrelated to the
  corruption itself.

Output: plots/poster_fid_severity_data_nontemporal.npz (non-temporal) or
        plots/poster_fid_severity_data_temporal.npz (--temporal).
Override with --output. The key schema (severities, fid_noise, fid_rotation)
is identical in both modes, so scripts/plot_poster_fid_combined.py consumes
either unchanged.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import argparse
import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from utils.corruptions import (
    apply_high_freq_noise,
    apply_wind_channel_rotation,
)
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

PRIOR_DELTA_HOURS = 24
DELTA_STEPS_6H = 4  # 24h / 6h = 4 indices on the 6-hourly local file


class InMemoryRawSource:
    """Loads (T, 4, H, W) raw float32 into memory; exposes read_raw(idx)."""

    def __init__(self, nc_path):
        ds = xr.open_dataset(nc_path)
        ds = ds.transpose("time", "latitude", "longitude")
        arrs = [ds[v].values.astype(np.float32) for v in VARIABLES]
        self.data = np.stack(arrs, axis=1)  # (T, 4, H, W)
        self.times = ds.time.values
        ds.close()

    def read_raw(self, idx):
        """Return the raw ``(4, H, W)`` field at time index ``idx``."""
        return self.data[idx]

    def __len__(self):
        """Number of time steps held in memory."""
        return self.data.shape[0]


def corrupt_raw_present(present_raw, severity, apply_fn, abs_mean, abs_std):
    """Apply a corruption (expecting normalized 4-channel input) to a raw present frame.

    The corruption functions in utils/corruptions are calibrated to normalized
    units (std ~ 1 per channel), so we normalize first, call the corruption,
    then denormalize back to raw units. Operating on the unpadded interior
    (4, 121, 240) keeps the corruption's spatial semantics intact.

    Returns: numpy (4, 121, 240) raw float32.
    """
    m = np.asarray(abs_mean, dtype=np.float32).reshape(-1, 1, 1)
    s = np.asarray(abs_std, dtype=np.float32).reshape(-1, 1, 1)
    present_norm = (present_raw - m) / s          # (4, H, W)
    x = torch.from_numpy(present_norm[None, ...])  # (1, 4, H, W)
    x_corrupted = apply_fn(x, severity)
    present_norm_corrupted = x_corrupted.detach().cpu().numpy()[0]
    return present_norm_corrupted * s + m


class CorruptedSeverityDataset(Dataset):
    """Yields model input where the present frame has been corrupted in raw
    space before composition.

    temporal=True  -> 8-channel temporal-phase input (needs prior_idx).
    temporal=False -> 4-channel "none" input (prior_idx ignored).

    severity=0 (and apply_fn=None) yields the clean reference distribution.
    """

    def __init__(self, source, present_idx, prior_idx,
                 severity, apply_fn,
                 abs_mean, abs_std, diff_mean, diff_std, temporal):
        if temporal and len(present_idx) != len(prior_idx):
            raise ValueError("present_idx and prior_idx must match in length")
        self.source = source
        self.present_idx = list(present_idx)
        self.prior_idx = list(prior_idx) if prior_idx is not None else None
        self.severity = severity
        self.apply_fn = apply_fn
        self.abs_mean = abs_mean
        self.abs_std = abs_std
        self.diff_mean = diff_mean
        self.diff_std = diff_std
        self.temporal = temporal

    def __len__(self):
        """Number of present-frame samples."""
        return len(self.present_idx)

    def __getitem__(self, i):
        """Corrupt the raw present frame (if severity>0), then compose the model input."""
        present_raw = self.source.read_raw(self.present_idx[i])
        if self.severity > 0 and self.apply_fn is not None:
            present_raw = corrupt_raw_present(
                present_raw, self.severity, self.apply_fn,
                self.abs_mean, self.abs_std,
            )
        if self.temporal:
            prior_raw = self.source.read_raw(self.prior_idx[i])
            sample = compose_temporal_input(
                present_raw, prior_raw, "phase",
                self.abs_mean, self.abs_std, self.diff_mean, self.diff_std,
            )
        else:
            sample = compose_temporal_input(
                present_raw, None, "none", self.abs_mean, self.abs_std,
            )
        return torch.from_numpy(sample)


def features_for_dataset(model, dataset, device, batch_size, num_workers, label):
    """Build a DataLoader over ``dataset`` and extract mean-pooled encoder features."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    print(f"  Extracting features: {label} ({len(dataset)} samples)...")
    return extract_features_for_loader(model, loader, device)


def fid_from_features(z_ref_np, mu_ref, sigma_ref, z_cor):
    """Compute FID between the reference Gaussian and the corrupted features (clamped ≥ 0)."""
    z_cor_np = z_cor.numpy()
    mu_c  = np.mean(z_cor_np, axis=0)
    sig_c = np.cov(z_cor_np, rowvar=False)
    fid = calculate_frechet_distance(mu_ref, sigma_ref, mu_c, sig_c)
    return float(max(fid, 0.0))


def parse_args():
    """Parse CLI arguments for the corruption-severity FID computation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/test_data_local.nc"),
        help="ERA5 NetCDF source (6-hourly).",
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("results/may_07_512_encoder"),
        help="Model directory for the non-temporal 4-channel I-JEPA (used unless --temporal).",
    )
    parser.add_argument(
        "--temporal-model-dir", type=Path,
        default=Path("results/may_13_temporal_exp3_phase_d512_maxpool"),
        help="Model directory for the temporal 8-channel I-JEPA (used with --temporal).",
    )
    parser.add_argument(
        "--temporal", action="store_true",
        help="Use the 8-channel temporal-exp3 model instead of the 4-channel non-temporal one.",
    )
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--n-severity-steps", type=int, default=5,
                        help="Number of severity levels including 0 (default 5 → [0, 0.25, 0.5, 0.75, 1.0]).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .npz path. Defaults to "
                             "plots/poster_fid_severity_data_nontemporal.npz (non-temporal) or "
                             "plots/poster_fid_severity_data_temporal.npz (--temporal).")
    return parser.parse_args()


def main():
    """Compute FID(clean ERA5, corrupted ERA5) across severity levels and save to .npz."""
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
            args.output = Path("plots/poster_fid_severity_data_temporal.npz")
        else:
            args.output = Path("plots/poster_fid_severity_data_nontemporal.npz")

    # Model + stats
    if args.temporal:
        model_dir = args.temporal_model_dir
        model_path = model_dir / "best_ijepa_model_twin_d512_tm-phase.pth"
        abs_mean  = np.load(model_dir / "data_mean.npy")
        abs_std   = np.load(model_dir / "data_std.npy")
        diff_mean = np.load(model_dir / "diff_mean_dt24h.npy")
        diff_std  = np.load(model_dir / "diff_std_dt24h.npy")
        in_chans = 8
        print(f"\nMode: temporal (8-channel phase, {model_dir.name})")
    else:
        model_dir = args.model_dir
        model_path = model_dir / "best_ijepa_model_twin_d512.pth"
        abs_mean = np.load(model_dir / "data_mean.npy")
        abs_std  = np.load(model_dir / "data_std.npy")
        diff_mean = diff_std = None
        in_chans = 4
        print(f"\nMode: non-temporal (4-channel, {model_dir.name})")

    print(f"Loading model: {model_path}")
    model = build_model("ijepa", device=device, model_size="twin",
                        embed_dim=512, in_chans=in_chans)
    model = load_model_checkpoint("ijepa", model, model_path, device)
    model.eval()

    # ERA5 source — clean present (+ clean 24h-prior in temporal mode)
    print(f"\nLoading ERA5 source into memory: {args.data}")
    src = InMemoryRawSource(args.data)
    print(f"  {len(src)} samples; first/last: {src.times[0]} / {src.times[-1]}")

    if args.temporal:
        # Build pair indices: present at i, prior at i - delta_steps (24h on 6h grid).
        valid_present = np.arange(DELTA_STEPS_6H, len(src))
        n_avail = len(valid_present)
        n_use = min(args.n_samples, n_avail)
        chosen = rng.permutation(n_avail)[:n_use]
        present_idx = valid_present[chosen].tolist()
        prior_idx   = (valid_present[chosen] - DELTA_STEPS_6H).tolist()
        print(f"  Using {n_use}/{n_avail} pair samples")
    else:
        # Non-temporal: no prior needed; sample present frames over all of src.
        n_avail = len(src)
        n_use = min(args.n_samples, n_avail)
        present_idx = rng.permutation(n_avail)[:n_use].tolist()
        prior_idx = None
        print(f"  Using {n_use}/{n_avail} samples")

    # Clean reference features (once).
    ref_ds = CorruptedSeverityDataset(
        src, present_idx, prior_idx, severity=0.0, apply_fn=None,
        abs_mean=abs_mean, abs_std=abs_std,
        diff_mean=diff_mean, diff_std=diff_std, temporal=args.temporal,
    )
    z_ref = features_for_dataset(
        model, ref_ds, device, args.batch_size, args.num_workers, "clean reference",
    )
    z_ref_np = z_ref.numpy()
    mu_ref  = np.mean(z_ref_np, axis=0)
    sigma_ref = np.cov(z_ref_np, rowvar=False)
    print(f"  z_ref shape: {z_ref_np.shape}")

    severities = np.linspace(0.0, 1.0, args.n_severity_steps)
    corruptions = {
        "noise":    apply_high_freq_noise,
        "rotation": apply_wind_channel_rotation,
    }
    fid_results = {k: [] for k in corruptions}

    for sev in severities:
        print(f"\n--- severity = {sev:.2f} ---")
        for key, apply_fn in corruptions.items():
            if sev == 0.0:
                fid = 0.0
                print(f"  {key:8s}: FID = 0.0000 (clean reference)")
            else:
                cor_ds = CorruptedSeverityDataset(
                    src, present_idx, prior_idx, severity=float(sev), apply_fn=apply_fn,
                    abs_mean=abs_mean, abs_std=abs_std,
                    diff_mean=diff_mean, diff_std=diff_std, temporal=args.temporal,
                )
                z_cor = features_for_dataset(
                    model, cor_ds, device, args.batch_size, args.num_workers,
                    f"sev={sev:.2f} / {key}",
                )
                fid = fid_from_features(z_ref_np, mu_ref, sigma_ref, z_cor)
                print(f"  {key:8s}: FID = {fid:.4f}")
            fid_results[key].append(fid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        severities=np.array(severities, dtype=np.float64),
        fid_noise=np.array(fid_results["noise"], dtype=np.float64),
        fid_rotation=np.array(fid_results["rotation"], dtype=np.float64),
        n_samples=np.int64(n_use),
        seed=np.int64(args.seed),
        temporal=bool(args.temporal),
        model_path=str(model_path),
        data_path=str(args.data),
    )
    print(f"\nSaved: {args.output}")
    print("\nSummary (clean ↔ corrupted FID):")
    print(f"  {'severity':>8}  {'noise':>10}  {'rotation':>10}")
    for i, sev in enumerate(severities):
        print(f"  {sev:>8.2f}  {fid_results['noise'][i]:>10.4f}  {fid_results['rotation'][i]:>10.4f}")


if __name__ == "__main__":
    main()

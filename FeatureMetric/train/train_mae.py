"""Train a Masked Autoencoder on ERA5 surface fields."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from utils.dataset import AtmosphereDataset
from utils.model_io import checkpoint_path, save_mae_checkpoint
from utils.models import build_mae
from utils.temporal import IN_CHANS_BY_MODE, derive_delta_steps

CLUSTER_DATA_PATH = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local.nc"
LARGE_LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local_5y.nc"

def train_epoch(model, loader, optimizer, device):
    """Run one MAE training epoch at 75% masking; return mean batch loss."""
    model.train()
    total_loss = 0

    for batch_idx, data in enumerate(loader):
        data = data.to(device)

        optimizer.zero_grad()
        loss, _, _ = model(data, mask_ratio=0.75)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate_epoch(model, loader, device):
    """Run one MAE validation epoch (no grad) at 75% masking; return mean loss."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            loss, _, _ = model(data, mask_ratio=0.75)
            total_loss += loss.item()

    return total_loss / len(loader)

def main():
    """CLI entry point: parse args, build data/model/scheduler, and run the MAE training loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run locally using subset data")
    parser.add_argument("--large-local", action="store_true", help="Run locally using the larger 5-year dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--stats-chunk-size", type=int, default=64)
    parser.add_argument("--model-size", choices=["default", "twin"], default="twin")
    parser.add_argument("--embed-dim", type=int, default=None,
                        help="Override encoder embed_dim (latent dim). Auto-derives num_heads as embed_dim//64.")
    parser.add_argument("--num-heads", type=int, default=None, help="Override encoder num_heads.")
    parser.add_argument("--depth", type=int, default=None, help="Override encoder depth.")
    parser.add_argument("--recompute-stats", action="store_true", help="Ignore saved normalization stats and recompute them")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory for checkpoint + normalization stats output (default: checkpoints).")
    parser.add_argument("--lazy", dest="lazy", action="store_true", help="Enable lazy dataloading")
    parser.add_argument("--eager", dest="lazy", action="store_false", help="Force eager dataloading")
    parser.set_defaults(lazy=None)
    parser.add_argument("--temporal-mode", choices=["none", "diff", "concat", "phase"], default="none",
                        help="Temporal-pair input mode. 'none' is the original single-timestep input. "
                             "'diff'=X_t-X_{t-Δt} (4ch); 'concat'=[X_{t-Δt},X_t] (8ch); 'phase'=[X_t, X_t-X_{t-Δt}] (8ch).")
    parser.add_argument("--delta-hours", type=int, default=24,
                        help="Δt in hours for temporal-pair construction (default 24, ignored when temporal-mode=none).")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # On macOS we can also try MPS
    if torch.backends.mps.is_available():
        device = torch.device('cpu')
        print("Using CPU for local robust execution.")
    else:
        print(f"Using device: {device}")

    # Paths and Hyperparameters
    if args.local and args.large_local:
        raise ValueError("Use only one of --local or --large-local.")

    if args.local:
        data_path = LOCAL_DATA_PATH
        batch_size = 16 # smaller batch size locally
        epochs = min(args.epochs, 2) # force small epochs for local testing
        num_workers = 0 if args.num_workers is None else args.num_workers
        print("Running in LOCAL mode.")
    elif args.large_local:
        data_path = LARGE_LOCAL_DATA_PATH
        batch_size = args.batch_size
        epochs = args.epochs
        num_workers = 0 if args.num_workers is None else args.num_workers
        print("Running in LARGE LOCAL mode.")
    else:
        data_path = CLUSTER_DATA_PATH
        batch_size = args.batch_size
        epochs = args.epochs
        num_workers = 0 if args.num_workers is None else args.num_workers
        print("Running in CLUSTER mode.")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"Loading data from {data_path}...")

    stats_dir = Path(args.output_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    mean_path = stats_dir / "data_mean.npy"
    std_path = stats_dir / "data_std.npy"

    stats = None
    if not args.recompute_stats and mean_path.exists() and std_path.exists():
        print(f"Loading cached normalization stats from {stats_dir}...")
        stats = (np.load(mean_path), np.load(std_path))
    else:
        print("Normalization stats not found in checkpoints or recomputation requested.")

    # Derive Δt index step from the file's time coordinate, or 0 for non-temporal.
    if args.temporal_mode == "none":
        delta_steps = 0
    else:
        import xarray as xr
        with xr.open_dataset(data_path, decode_times=True) as _ds:
            time_coord = _ds.time.values
        delta_steps = derive_delta_steps(time_coord, args.delta_hours)
        print(f"Temporal mode '{args.temporal_mode}': Δt={args.delta_hours}h → {delta_steps} index steps.")

    # Diff stats (only needed for 'diff' and 'phase'). Stored per-Δt so multiple
    # experiments at different Δt can coexist in the same output dir.
    diff_stats = None
    diff_mean_path = stats_dir / f"diff_mean_dt{args.delta_hours}h.npy"
    diff_std_path  = stats_dir / f"diff_std_dt{args.delta_hours}h.npy"

    # Load Datasets
    lazy_load = (not args.local) if args.lazy is None else args.lazy
    print(f"Lazy loading: {lazy_load} | num_workers: {num_workers}")

    if args.temporal_mode in ("diff", "phase"):
        if not args.recompute_stats and diff_mean_path.exists() and diff_std_path.exists():
            print(f"Loading cached diff stats from {stats_dir} (Δt={args.delta_hours}h)...")
            diff_stats = (np.load(diff_mean_path), np.load(diff_std_path))
        else:
            # Need absolute stats first if not cached.
            if stats is None:
                bootstrap = AtmosphereDataset(
                    data_path, split="train", lazy=True,
                    stats_chunk_size=args.stats_chunk_size,
                )
                stats = bootstrap.get_stats()
                np.save(mean_path, stats[0])
                np.save(std_path, stats[1])
            print(f"Computing diff stats (Δt={args.delta_hours}h = {delta_steps} steps)...")
            bootstrap = AtmosphereDataset(
                data_path, split="train", stats=stats, lazy=True,
                stats_chunk_size=args.stats_chunk_size,
            )
            diff_mean_arr, diff_std_arr = bootstrap.compute_diff_stats(delta_steps=delta_steps)
            np.save(diff_mean_path, diff_mean_arr)
            np.save(diff_std_path,  diff_std_arr)
            diff_stats = (diff_mean_arr, diff_std_arr)
            print(f"  saved {diff_mean_path.name}, {diff_std_path.name}")

    train_dataset = AtmosphereDataset(
        data_path,
        split="train",
        stats=stats,
        lazy=lazy_load,
        stats_chunk_size=args.stats_chunk_size,
        temporal_mode=args.temporal_mode,
        delta_steps=delta_steps,
        diff_stats=diff_stats,
    )
    stats = train_dataset.get_stats()
    val_dataset = AtmosphereDataset(
        data_path,
        split="val",
        stats=stats,
        lazy=lazy_load,
        stats_chunk_size=args.stats_chunk_size,
        temporal_mode=args.temporal_mode,
        delta_steps=delta_steps,
        diff_stats=diff_stats,
    )

    # Save stats for future training/evaluation runs
    np.save(mean_path, stats[0])
    np.save(std_path, stats[1])

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    # Initialize Model
    in_chans = IN_CHANS_BY_MODE[args.temporal_mode]
    model = build_mae(
        model_size=args.model_size,
        in_chans=in_chans,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
    ).to(device)
    print(
        f"MAE setup: in_chans={in_chans}, embed_dim={model.patch_embed.proj.out_channels}, "
        f"depth={len(model.encoder_blocks)}, num_heads={model.encoder_blocks[0].attn.num_heads}"
    )

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("\nStarting Training...")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, val_loader, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            variant = None if args.temporal_mode == "none" else f"tm-{args.temporal_mode}"
            ckpt_path = checkpoint_path(
                "mae", args.model_size, stats_dir,
                embed_dim=args.embed_dim, variant=variant,
            )
            save_mae_checkpoint(ckpt_path, model, optimizer, epoch + 1, val_loss, args)
            print(f"  -> Saved best model to {ckpt_path}")

    # End-of-training diagnostic: per-input-channel L2 norm of the patch-embed weight.
    # For temporal modes that include the prior in the input (concat/phase), a low
    # ratio (< ~0.2) means the network is ignoring half of its input.
    with torch.no_grad():
        w = model.patch_embed.proj.weight.detach().cpu()  # (embed_dim, in_chans, 16, 16)
        norms = w.pow(2).sum(dim=(0, 2, 3)).sqrt()
        print(f"\nPatchEmbed per-input-channel L2 norms: {norms.tolist()}")
        if args.temporal_mode == "concat":
            ratio = (norms[0:4].mean() / norms[4:8].mean()).item()
            print(f"  concat prior/present norm ratio = {ratio:.3f}  "
                  f"({'OK' if ratio > 0.2 else 'WARN: prior may be ignored'})")
        elif args.temporal_mode == "phase":
            # Phase: ch 0..3 are abs X_t, ch 4..7 are diff.
            ratio = (norms[4:8].mean() / norms[0:4].mean()).item()
            print(f"  phase diff/abs norm ratio = {ratio:.3f}  "
                  f"({'OK' if ratio > 0.2 else 'WARN: diff half may be ignored'})")

    print("\nTraining Complete.")

if __name__ == "__main__":
    main()

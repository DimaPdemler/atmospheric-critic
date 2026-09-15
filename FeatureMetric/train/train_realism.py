"""Train the contrastive realism metric on ERA5 (from scratch, ERA5 only).

The model learns the manifold of realistic atmospheric states from ERA5 plus a
subtle corruption suite, and at eval time scores forecasts as off-manifold. See
utils/realism.py for the model and loss.

Usage (local smoke):
    /opt/miniconda3/envs/pmlr/bin/python train/train_realism.py --local

Cluster:
    python train/train_realism.py --epochs 100 --batch-size 64
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import mannwhitneyu

from utils.dataset import AtmosphereDataset
from utils.model_io import checkpoint_path
from utils.realism import RealismMetric, compute_realism_losses


def realism_auc(clean, corrupt):
    """AUC = P(r(corrupted) > r(clean)); 0.5 = no discrimination, 1.0 = perfect."""
    if len(clean) == 0 or len(corrupt) == 0:
        return float("nan")
    u, _ = mannwhitneyu(corrupt, clean, alternative="greater")
    return float(u) / (len(clean) * len(corrupt))

CLUSTER_DATA_PATH = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local.nc"
LARGE_LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local_5y.nc"


def run_epoch(model, loader, optimizer, device, args, train=True):
    """Run one epoch; returns mean losses + clean-vs-corrupted discrimination AUC."""
    model.train(train)
    sums = {"total": 0.0, "loss_inv": 0.0, "loss_real": 0.0, "loss_comp": 0.0}
    r_clean_all, r_corr_all = [], []
    n = 0
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for data in loader:
            data = data.to(device)
            if train:
                optimizer.zero_grad()
            loss, logs = compute_realism_losses(
                model, data,
                w_inv=args.w_inv, w_real=args.w_real, w_comp=args.w_comp,
                margin=args.margin,
                update_center=train,  # never move the center on val
            )
            if train:
                loss.backward()
                optimizer.step()
            for k in sums:
                sums[k] += logs[k]
            r_clean_all.append(logs["r_clean"].cpu().numpy())
            r_corr_all.append(logs["r_corr"].cpu().numpy())
            n += 1
    means = {k: v / max(n, 1) for k, v in sums.items()}
    means["auc"] = realism_auc(
        np.concatenate(r_clean_all), np.concatenate(r_corr_all)
    ) if n else float("nan")
    return means


def main():
    """CLI entry point: build data/model, run the realism training loop, save checkpoint."""
    parser = argparse.ArgumentParser(description="Train the contrastive realism metric.")
    parser.add_argument("--local", action="store_true", help="Run locally on the 1-year subset.")
    parser.add_argument("--large-local", action="store_true", help="Run locally on the 5-year subset.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--stats-chunk-size", type=int, default=64)
    parser.add_argument("--model-size", choices=["default", "twin"], default="twin")
    parser.add_argument("--embed-dim", type=int, default=None,
                        help="Override encoder embed_dim (auto-derives num_heads).")
    parser.add_argument("--proj-dim", type=int, default=128, help="VICReg projector output dim.")
    parser.add_argument("--w-inv", type=float, default=0.5, help="Weight on the VICReg invariance loss.")
    parser.add_argument("--w-real", type=float, default=50.0, help="Weight on the severity-regression loss.")
    parser.add_argument("--w-comp", type=float, default=1.0, help="Weight on the compactness loss.")
    parser.add_argument("--margin", type=float, default=0.5, help="Severity margin for the compactness push.")
    parser.add_argument("--recompute-stats", action="store_true", help="Ignore cached normalization stats.")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory for checkpoint + normalization stats (default: checkpoints).")
    parser.add_argument("--lazy", dest="lazy", action="store_true")
    parser.add_argument("--eager", dest="lazy", action="store_false")
    parser.set_defaults(lazy=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.local and args.large_local:
        raise ValueError("Use only one of --local or --large-local.")
    if args.local:
        data_path, batch_size, epochs = LOCAL_DATA_PATH, 16, min(args.epochs, 2)
        print("Running in LOCAL mode.")
    elif args.large_local:
        data_path, batch_size, epochs = LARGE_LOCAL_DATA_PATH, args.batch_size, args.epochs
        print("Running in LARGE LOCAL mode.")
    else:
        data_path, batch_size, epochs = CLUSTER_DATA_PATH, args.batch_size, args.epochs
        print("Running in CLUSTER mode.")
    num_workers = 0 if args.num_workers is None else args.num_workers

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    stats_dir = Path(args.output_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    mean_path, std_path = stats_dir / "data_mean.npy", stats_dir / "data_std.npy"

    stats = None
    if not args.recompute_stats and mean_path.exists() and std_path.exists():
        print(f"Loading cached normalization stats from {stats_dir}...")
        stats = (np.load(mean_path), np.load(std_path))

    lazy_load = (not args.local) if args.lazy is None else args.lazy
    print(f"Loading data from {data_path} (lazy={lazy_load}, num_workers={num_workers})...")

    train_dataset = AtmosphereDataset(
        data_path, split="train", stats=stats, lazy=lazy_load,
        stats_chunk_size=args.stats_chunk_size,
    )
    stats = train_dataset.get_stats()
    val_dataset = AtmosphereDataset(
        data_path, split="val", stats=stats, lazy=lazy_load,
        stats_chunk_size=args.stats_chunk_size,
    )
    np.save(mean_path, stats[0])
    np.save(std_path, stats[1])

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                         pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_kwargs)

    model = RealismMetric(
        model_size=args.model_size, in_chans=4,
        embed_dim=args.embed_dim, proj_dim=args.proj_dim,
    ).to(device)
    print(f"Model built (embed_dim={model.embed_dim}). Params: "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_path = checkpoint_path("realism", args.model_size, stats_dir, embed_dim=args.embed_dim)
    # Select on the realism signal (clean-vs-corrupted AUC), NOT total loss — the
    # total is dominated by the VICReg invariance term, which is not what we care
    # about for the realism metric.
    best_auc = -1.0
    for epoch in range(epochs):
        tr = run_epoch(model, train_loader, optimizer, device, args, train=True)
        va = run_epoch(model, val_loader, optimizer, device, args, train=False)
        scheduler.step()
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train {tr['total']:.4f} (inv {tr['loss_inv']:.3f} real {tr['loss_real']:.3f} comp {tr['loss_comp']:.3f}) auc {tr['auc']:.3f} | "
            f"val {va['total']:.4f} (inv {va['loss_inv']:.3f} real {va['loss_real']:.3f} comp {va['loss_comp']:.3f}) auc {va['auc']:.3f}"
        )
        if np.isfinite(va["auc"]) and va["auc"] > best_auc:
            best_auc = va["auc"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": va["total"],
                    "val_auc": best_auc,
                    "config": vars(args),
                },
                ckpt_path,
            )
            print(f"  ↳ saved best checkpoint to {ckpt_path} (val auc {best_auc:.3f})")

    print(f"Done. Best val clean-vs-corrupted AUC: {best_auc:.3f}. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

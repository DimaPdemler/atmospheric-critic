"""Train an I-JEPA encoder on ERA5 surface fields."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import math

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from utils.dataset import AtmosphereDataset
from utils.ijepa_masking import MultiBlockMaskGenerator
from utils.model_io import build_model, checkpoint_path, save_ijepa_checkpoint
from utils.temporal import IN_CHANS_BY_MODE, derive_delta_steps


CLUSTER_DATA_PATH = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local.nc"
LARGE_LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local_5y.nc"


class WarmupCosineSchedule:
    """Learning-rate schedule: linear warmup to ``ref_lr`` then cosine decay to ``final_lr``."""
    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, final_lr, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.total_steps = max(1, total_steps)
        self.step_num = 0

    def step(self):
        """Advance one step, set the new LR on every param group, and return it."""
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            progress = self.step_num / self.warmup_steps
            lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.final_lr + 0.5 * (self.ref_lr - self.final_lr) * (1.0 + math.cos(math.pi * progress))

        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


class CosineWDSchedule:
    """Weight-decay schedule: cosine anneal from ``start_wd`` to ``final_wd`` over training."""
    def __init__(self, optimizer, start_wd, final_wd, total_steps):
        self.optimizer = optimizer
        self.start_wd = start_wd
        self.final_wd = final_wd
        self.total_steps = max(1, total_steps)
        self.step_num = 0

    def step(self):
        """Advance one step, set WD on non-excluded param groups, and return it."""
        self.step_num += 1
        progress = self.step_num / self.total_steps
        wd = self.final_wd + 0.5 * (self.start_wd - self.final_wd) * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            if group.get("WD_exclude", False):
                continue
            group["weight_decay"] = wd
        return wd


def make_optimizer(model, lr, weight_decay):
    """Build AdamW over the context encoder + predictor.

    Splits params into two groups so weight decay is excluded from biases and
    1-D params (norms), as in the original I-JEPA recipe.
    """
    param_groups = [
        {
            "params": [
                p
                for n, p in list(model.context_encoder.named_parameters()) + list(model.predictor.named_parameters())
                if p.requires_grad and ("bias" not in n) and (len(p.shape) != 1)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in list(model.context_encoder.named_parameters()) + list(model.predictor.named_parameters())
                if p.requires_grad and (("bias" in n) or (len(p.shape) == 1))
            ],
            "weight_decay": 0.0,
            "WD_exclude": True,
        },
    ]
    return AdamW(param_groups, lr=lr)


def resolve_data_path(args):
    """Pick the dataset path from the ``--local`` / ``--large-local`` flags (else cluster)."""
    if args.local and args.large_local:
        raise ValueError("Use only one of --local or --large-local.")
    if args.local:
        return LOCAL_DATA_PATH
    if args.large_local:
        return LARGE_LOCAL_DATA_PATH
    return CLUSTER_DATA_PATH


def load_stats(checkpoint_dir, recompute_stats):
    """Load cached ``(mean, std)`` normalization stats, or None to recompute them."""
    mean_path = checkpoint_dir / "data_mean.npy"
    std_path = checkpoint_dir / "data_std.npy"
    if not recompute_stats and mean_path.exists() and std_path.exists():
        return np.load(mean_path), np.load(std_path)
    return None


def build_datasets(data_path, stats, lazy_load, stats_chunk_size,
                   temporal_mode="none", delta_steps=0, diff_stats=None):
    """Build train/val :class:`AtmosphereDataset`s sharing the train-split stats.

    Returns ``(train_dataset, val_dataset, stats)`` where ``stats`` are the
    normalization stats actually used (computed on train if not supplied).
    """
    train_dataset = AtmosphereDataset(
        data_path,
        split="train",
        stats=stats,
        lazy=lazy_load,
        stats_chunk_size=stats_chunk_size,
        temporal_mode=temporal_mode,
        delta_steps=delta_steps,
        diff_stats=diff_stats,
    )
    stats = train_dataset.get_stats()
    val_dataset = AtmosphereDataset(
        data_path,
        split="val",
        stats=stats,
        lazy=lazy_load,
        stats_chunk_size=stats_chunk_size,
        temporal_mode=temporal_mode,
        delta_steps=delta_steps,
        diff_stats=diff_stats,
    )
    return train_dataset, val_dataset, stats


def truncate_for_smoke(dataset, limit):
    """Return a Subset of the first ``limit`` samples for quick smoke tests."""
    limit = min(limit, len(dataset))
    return Subset(dataset, list(range(limit)))


def train_epoch(model, loader, optimizer, device, mask_generator, lr_schedule, wd_schedule, momentum_schedule, grad_clip):
    """Run one training epoch and return averaged loss/diagnostic metrics.

    Per batch: sample context/target masks, forward+backward, optional grad
    clipping, optimizer step, EMA target-encoder update, and LR/WD schedule steps.
    """
    model.train()
    total_loss = 0.0
    pred_std_sum = 0.0
    target_std_sum = 0.0
    total_batches = 0
    current_lr = optimizer.param_groups[0]["lr"]
    current_wd = optimizer.param_groups[0].get("weight_decay", 0.0)
    pred_shape = None
    target_shape = None

    clip_params = [
        p for p in list(model.context_encoder.parameters()) + list(model.predictor.parameters())
        if p.requires_grad
    ]

    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        context_masks, target_masks = mask_generator.sample(batch.shape[0], device=device)

        optimizer.zero_grad()
        loss, pred_tokens, target_tokens = model(batch, context_masks, target_masks)
        if not torch.isfinite(loss):
            raise RuntimeError("Encountered non-finite JEPA loss")

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=grad_clip)
        optimizer.step()
        model.update_target_encoder(next(momentum_schedule))
        current_lr = lr_schedule.step()
        current_wd = wd_schedule.step()

        total_loss += loss.item()
        pred_std_sum += pred_tokens.detach().float().std().item()
        target_std_sum += target_tokens.detach().float().std().item()
        total_batches += 1
        pred_shape = pred_tokens.shape
        target_shape = target_tokens.shape

    n = max(1, total_batches)
    return {
        "loss": total_loss / n,
        "pred_std": pred_std_sum / n,
        "target_std": target_std_sum / n,
        "lr": current_lr,
        "wd": current_wd,
        "pred_shape": pred_shape,
        "target_shape": target_shape,
    }


@torch.no_grad()
def val_epoch(model, loader, device, mask_generator):
    """Run one validation epoch (no grad/updates); return averaged loss metrics."""
    model.eval()
    total_loss = 0.0
    pred_std_sum = 0.0
    target_std_sum = 0.0
    total_batches = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        context_masks, target_masks = mask_generator.sample(batch.shape[0], device=device)
        loss, pred_tokens, target_tokens = model(batch, context_masks, target_masks)
        if not torch.isfinite(loss):
            raise RuntimeError("Encountered non-finite JEPA loss")

        total_loss += loss.item()
        pred_std_sum += pred_tokens.float().std().item()
        target_std_sum += target_tokens.float().std().item()
        total_batches += 1

    n = max(1, total_batches)
    return {
        "loss": total_loss / n,
        "pred_std": pred_std_sum / n,
        "target_std": target_std_sum / n,
    }


def main():
    """CLI entry point: parse args, build data/model/schedules, and run the I-JEPA training loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--large-local", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--stats-chunk-size", type=int, default=64)
    parser.add_argument("--recompute-stats", action="store_true")
    parser.add_argument("--lazy", dest="lazy", action="store_true")
    parser.add_argument("--eager", dest="lazy", action="store_false")
    parser.set_defaults(lazy=None)
    parser.add_argument("--model-size", choices=["tiny", "small", "twin"], default="tiny")
    parser.add_argument("--embed-dim", type=int, default=None,
                        help="Override encoder embed_dim (latent dim). Auto-derives num_heads as embed_dim//64.")
    parser.add_argument("--num-heads", type=int, default=None, help="Override encoder num_heads.")
    parser.add_argument("--depth", type=int, default=None, help="Override encoder depth.")
    parser.add_argument("--start-lr", type=float, default=2e-4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--final-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--final-weight-decay", type=float, default=0.1)
    parser.add_argument("--ema-start", type=float, default=0.996)
    parser.add_argument("--ema-end", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max grad norm; set to 0 to disable clipping.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=64)
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop if val loss doesn't improve for this many epochs. 0 = disabled.")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory for checkpoint + normalization stats output (default: checkpoints).")
    parser.add_argument("--temporal-mode", choices=["none", "diff", "concat", "phase"], default="none",
                        help="Temporal-pair input mode. 'none' is original single-timestep. "
                             "'diff'=X_t-X_{t-Δt} (4ch); 'concat'=[X_{t-Δt},X_t] (8ch); 'phase'=[X_t, X_t-X_{t-Δt}] (8ch).")
    parser.add_argument("--delta-hours", type=int, default=24,
                        help="Δt in hours for temporal-pair construction (default 24, ignored when temporal-mode=none).")
    args = parser.parse_args()

    if args.model_size == "small":
        if parser.get_default("lr") == args.lr:
            args.lr = 1e-3
        if parser.get_default("final_weight_decay") == args.final_weight_decay:
            args.final_weight_decay = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and device.type != "cuda":
        device = torch.device("cpu")
    print(f"Using device: {device}")

    data_path = resolve_data_path(args)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats = load_stats(checkpoint_dir, args.recompute_stats)

    # Derive Δt index step from file's time coordinate.
    if args.temporal_mode == "none":
        delta_steps = 0
    else:
        import xarray as xr
        with xr.open_dataset(data_path, decode_times=True) as _ds:
            time_coord = _ds.time.values
        delta_steps = derive_delta_steps(time_coord, args.delta_hours)
        print(f"Temporal mode '{args.temporal_mode}': Δt={args.delta_hours}h → {delta_steps} index steps.")

    diff_stats = None
    diff_mean_path = checkpoint_dir / f"diff_mean_dt{args.delta_hours}h.npy"
    diff_std_path  = checkpoint_dir / f"diff_std_dt{args.delta_hours}h.npy"
    if args.temporal_mode in ("diff", "phase"):
        if not args.recompute_stats and diff_mean_path.exists() and diff_std_path.exists():
            print(f"Loading cached diff stats from {checkpoint_dir} (Δt={args.delta_hours}h)...")
            diff_stats = (np.load(diff_mean_path), np.load(diff_std_path))
        else:
            if stats is None:
                bootstrap = AtmosphereDataset(
                    data_path, split="train", lazy=True,
                    stats_chunk_size=args.stats_chunk_size,
                )
                stats = bootstrap.get_stats()
                np.save(checkpoint_dir / "data_mean.npy", stats[0])
                np.save(checkpoint_dir / "data_std.npy", stats[1])
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

    lazy_load = (not args.local) if args.lazy is None else args.lazy
    num_workers = 0 if args.num_workers is None else args.num_workers
    train_dataset, val_dataset, stats = build_datasets(
        data_path=data_path,
        stats=stats,
        lazy_load=lazy_load,
        stats_chunk_size=args.stats_chunk_size,
        temporal_mode=args.temporal_mode,
        delta_steps=delta_steps,
        diff_stats=diff_stats,
    )
    np.save(checkpoint_dir / "data_mean.npy", stats[0])
    np.save(checkpoint_dir / "data_std.npy", stats[1])

    if args.smoke_test:
        train_dataset = truncate_for_smoke(train_dataset, args.smoke_samples)
        val_dataset = truncate_for_smoke(val_dataset, max(8, args.smoke_samples // 2))
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 8)

    if args.local and not args.smoke_test:
        args.epochs = min(args.epochs, 5)
        args.batch_size = min(args.batch_size, 8)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    in_chans = IN_CHANS_BY_MODE[args.temporal_mode]
    model = build_model(
        "ijepa",
        device=device,
        model_size=args.model_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        in_chans=in_chans,
    )
    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    mask_generator = MultiBlockMaskGenerator(
        input_size=(128, 256),
        patch_size=model.patch_size,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=4,
        allow_overlap=False,
    )

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = min(total_steps, max(1, len(train_loader) * args.warmup_epochs))
    lr_schedule = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        start_lr=args.start_lr,
        ref_lr=args.lr,
        final_lr=args.final_lr,
        total_steps=total_steps,
    )
    wd_schedule = CosineWDSchedule(
        optimizer=optimizer,
        start_wd=args.weight_decay,
        final_wd=args.final_weight_decay,
        total_steps=total_steps,
    )
    momentum_schedule = iter(
        np.linspace(args.ema_start, args.ema_end, total_steps + 1, dtype=np.float64).tolist()
    )

    best_val_loss = float("inf")
    variant = None if args.temporal_mode == "none" else f"tm-{args.temporal_mode}"
    best_path = checkpoint_path(
        "ijepa", args.model_size, checkpoint_dir,
        embed_dim=args.embed_dim, variant=variant,
    )
    epochs_no_improve = 0

    print(
        f"JEPA setup: in_chans={in_chans}, grid={model.grid_size}, embed_dim={model.embed_dim}, "
        f"batch_size={args.batch_size}, epochs={args.epochs}, warmup_steps={warmup_steps}"
    )

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mask_generator=mask_generator,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            momentum_schedule=momentum_schedule,
            grad_clip=args.grad_clip,
        )
        val_metrics = val_epoch(
            model=model,
            loader=val_loader,
            device=device,
            mask_generator=mask_generator,
        )
        val_loss = val_metrics["loss"]
        msg = (
            f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {train_metrics['lr']:.6f} | WD: {train_metrics['wd']:.4f} | "
            f"Pred std: {train_metrics['pred_std']:.3f} (val {val_metrics['pred_std']:.3f}) | "
            f"Target std: {train_metrics['target_std']:.3f}"
        )
        if epoch == 0:
            msg += (
                f" | Pred shape: {tuple(train_metrics['pred_shape'])} | "
                f"Target shape: {tuple(train_metrics['target_shape'])}"
            )
        print(msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_ijepa_checkpoint(best_path, model, optimizer, epoch + 1, val_loss, args)
            print(f"  -> Saved best model to {best_path}")
        else:
            epochs_no_improve += 1

        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(f"\nEarly stopping: val loss did not improve for {args.early_stopping_patience} epochs.")
            break

    # End-of-training diagnostic: per-input-channel L2 norm of patch_embed.proj.weight
    # for BOTH the context encoder (gradient-trained) and the target encoder (EMA).
    # For concat/phase: if the prior half is being ignored the ratio will collapse.
    with torch.no_grad():
        for enc_name, enc in [
            ("context_encoder", model.context_encoder),
            ("target_encoder",  model.target_encoder),
        ]:
            w = enc.patch_embed.proj.weight.detach().cpu()  # (embed_dim, in_chans, 16, 16)
            norms = w.pow(2).sum(dim=(0, 2, 3)).sqrt()
            print(f"\n[{enc_name}] PatchEmbed per-input-channel L2 norms: {norms.tolist()}")
            if args.temporal_mode == "concat":
                ratio = (norms[0:4].mean() / norms[4:8].mean()).item()
                print(f"  concat prior/present ratio = {ratio:.3f}  "
                      f"({'OK' if ratio > 0.2 else 'WARN: prior may be ignored'})")
            elif args.temporal_mode == "phase":
                ratio = (norms[4:8].mean() / norms[0:4].mean()).item()
                print(f"  phase diff/abs ratio = {ratio:.3f}  "
                      f"({'OK' if ratio > 0.2 else 'WARN: diff half may be ignored'})")


if __name__ == "__main__":
    main()

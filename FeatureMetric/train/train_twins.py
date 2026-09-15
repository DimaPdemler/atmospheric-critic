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
from utils.ijepa_masking import MultiBlockMaskGenerator, target_masks_to_mae_masks
from utils.model_io import build_model, checkpoint_path, save_mae_checkpoint, save_ijepa_checkpoint

CLUSTER_DATA_PATH = Path("/cluster/courses/pmlr/teams/team07/data/era5_1.5deg_2004-01-01_2023-12-31.nc")
LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local.nc"
LARGE_LOCAL_DATA_PATH = Path(__file__).parent.parent / "data" / "test_data_local_5y.nc"


# ---------------------------------------------------------------------------
# Schedules (identical to train_ijepa)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Optimizer - generic, works for both models
# ---------------------------------------------------------------------------

def make_optimizer(model, lr, weight_decay):
    """Build AdamW over all trainable params, excluding biases/norms from weight decay.

    Generic version that works for both MAE and I-JEPA twin models.
    """
    # Collect all trainable parameters regardless of model type
    all_named_params = list(model.named_parameters())
    decay_params = [
        p for n, p in all_named_params
        if p.requires_grad and "bias" not in n and p.ndim != 1
    ]
    no_decay_params = [
        p for n, p in all_named_params
        if p.requires_grad and ("bias" in n or p.ndim == 1)
    ]
    return AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0, "WD_exclude": True},
        ],
        lr=lr,
    )


# ---------------------------------------------------------------------------
# Per-model forward pass - the only place the two models diverge
# ---------------------------------------------------------------------------

def forward_mae(model, batch, mask_generator, device, args):
    """MAE forward returning the loss; uses random or I-JEPA-style block masking."""
    if args.mae_mask_mode == "random":
        loss, _, _ = model(batch, mask_ratio=0.75)
        return loss

    _, target_masks = mask_generator.sample(batch.shape[0], device=device)
    visible_indices, loss_mask = target_masks_to_mae_masks(target_masks, model.patch_embed.num_patches)
    loss, _, _ = model(batch, visible_indices=visible_indices, loss_mask=loss_mask)
    return loss


def forward_ijepa(model, batch, mask_generator, device, args):
    """I-JEPA forward returning the loss; samples context + target masks per batch."""
    context_masks, target_masks = mask_generator.sample(batch.shape[0], device=device)
    loss, _, _ = model(batch, context_masks, target_masks)
    return loss


def get_forward_fn(model_name):
    """Return the per-model forward function (``forward_mae`` or ``forward_ijepa``)."""
    if model_name == "mae":
        return forward_mae
    if model_name == "ijepa":
        return forward_ijepa
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, forward_fn, mask_generator,
                lr_schedule, wd_schedule, grad_clip, args):
    """Run one training epoch for either model type; return averaged loss + LR/WD.

    Uses the model-specific ``forward_fn``, applies optional grad clipping, and
    EMA-updates the target encoder when the model is I-JEPA.
    """
    model.train()
    total_loss = 0.0
    total_batches = 0
    current_lr = optimizer.param_groups[0]["lr"]
    current_wd = optimizer.param_groups[0].get("weight_decay", 0.0)

    # Only IJEPA has an EMA target encoder to update
    is_ijepa = hasattr(model, "update_target_encoder")

    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad()
        loss = forward_fn(model, batch, mask_generator, device, args)

        if not torch.isfinite(loss):
            raise RuntimeError("Encountered non-finite loss")

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            )
        optimizer.step()

        if is_ijepa:
            model.update_target_encoder(next(momentum_schedule))  # see note below

        current_lr = lr_schedule.step()
        current_wd = wd_schedule.step()
        total_loss += loss.item()
        total_batches += 1

    n = max(1, total_batches)
    return {"loss": total_loss / n, "lr": current_lr, "wd": current_wd}


@torch.no_grad()
def val_epoch(model, loader, device, forward_fn, mask_generator, args):
    """Run one validation epoch (no grad/updates); return averaged loss."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        loss = forward_fn(model, batch, mask_generator, device, args)
        if not torch.isfinite(loss):
            raise RuntimeError("Encountered non-finite val loss")
        total_loss += loss.item()
        total_batches += 1
    return {"loss": total_loss / max(1, total_batches)}


# ---------------------------------------------------------------------------
# Checkpointing - unified
# ---------------------------------------------------------------------------

def save_checkpoint(model_name, model, optimizer, epoch, val_loss, args, path):
    """Dispatch to the MAE or I-JEPA checkpoint saver by model type."""
    if model_name == "mae":
        save_mae_checkpoint(path, model, optimizer, epoch, val_loss, args)
    elif model_name == "ijepa":
        save_ijepa_checkpoint(path, model, optimizer, epoch, val_loss, args)
    else:
        raise ValueError(f"model {model_name} not recognized!")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def resolve_data_path(args):
    """Pick the dataset path from the ``--local`` / ``--large-local`` flags (else cluster)."""
    if args.local and args.large_local:
        raise ValueError("Use only one of --local or --large-local.")
    if args.local:
        return LOCAL_DATA_PATH
    if args.large_local:
        return LARGE_LOCAL_DATA_PATH
    return CLUSTER_DATA_PATH


def build_datasets(data_path, stats, lazy_load, stats_chunk_size):
    """Build train/val datasets sharing the train-split normalization stats."""
    train_dataset = AtmosphereDataset(data_path, split="train", stats=stats,
                                       lazy=lazy_load, stats_chunk_size=stats_chunk_size)
    stats = train_dataset.get_stats()
    val_dataset = AtmosphereDataset(data_path, split="val", stats=stats,
                                     lazy=lazy_load, stats_chunk_size=stats_chunk_size)
    return train_dataset, val_dataset, stats


def build_mask_generator(args):
    """Construct the I-JEPA mask generator from the chosen context/target mask modes."""
    return MultiBlockMaskGenerator(
        input_size=(128, 256),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=4,
        allow_overlap=False,
        context_mode=args.ijepa_context_mode,
        disjoint_targets=args.target_mask_mode == "shared-blocks",
    )


def resolve_variant(model_name, args):
    """Derive the checkpoint variant tag from the masking-mode flags (or None)."""
    tags = []
    if args.target_mask_mode == "shared-blocks":
        tags.append("shared-targets")
    if model_name == "mae" and args.mae_mask_mode == "random":
        tags = []
    if model_name == "ijepa" and args.ijepa_context_mode == "world-band":
        tags.append("world-band")
    if args.variant_suffix:
        tags.append(args.variant_suffix)
    return "__".join(tags) or None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """CLI entry point: train an MAE or I-JEPA "twin" (matched-architecture) model.

    The twin models share an identical encoder so MAE and I-JEPA can be compared
    head-to-head as realism-metric backbones.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mae", "ijepa"], required=True)
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
    # Shared optimizer/schedule hyperparams
    parser.add_argument("--start-lr", type=float, default=2e-4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--final-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--final-weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-start", type=float, default=0.996)
    parser.add_argument("--ema-end", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-samples", type=int, default=64)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--mae-mask-mode", choices=["random", "target-blocks"], default="random")
    parser.add_argument("--target-mask-mode", choices=["baseline", "shared-blocks"], default="baseline")
    parser.add_argument("--ijepa-context-mode", choices=["multiblock", "world-band"], default="multiblock")
    parser.add_argument("--variant-suffix", type=str, default=None)
    args = parser.parse_args()

    if args.model == "mae" and args.mae_mask_mode == "target-blocks":
        args.target_mask_mode = "shared-blocks"
    if args.model == "mae" and args.ijepa_context_mode != "multiblock":
        raise ValueError("MAE training does not use ijepa context masks; keep --ijepa-context-mode=multiblock.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available() and device.type != "cuda":
        device = torch.device("cpu")
    print(f"Training {args.model.upper()} twin on {device}")

    data_path = resolve_data_path(args)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    mean_path = checkpoint_dir / "data_mean.npy"
    std_path = checkpoint_dir / "data_std.npy"
    stats = None
    if not args.recompute_stats and mean_path.exists() and std_path.exists():
        stats = (np.load(mean_path), np.load(std_path))

    lazy_load = (not args.local) if args.lazy is None else args.lazy
    num_workers = 0 if args.num_workers is None else args.num_workers

    train_dataset, val_dataset, stats = build_datasets(
        data_path, stats, lazy_load, args.stats_chunk_size)
    np.save(mean_path, stats[0])
    np.save(std_path, stats[1])

    if args.smoke_test:
        train_dataset = Subset(train_dataset, list(range(min(args.smoke_samples, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(max(8, args.smoke_samples // 2), len(val_dataset)))))
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 8)

    if args.local and not args.smoke_test:
        args.epochs = min(args.epochs, 5)
        args.batch_size = min(args.batch_size, 8)

    loader_kwargs = {"batch_size": args.batch_size, "num_workers": num_workers,
                     "pin_memory": device.type == "cuda"}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = build_model(args.model, device=device, model_size="twin")
    optimizer = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    forward_fn = get_forward_fn(args.model)
    mask_generator = build_mask_generator(args)

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = min(total_steps, max(1, len(train_loader) * args.warmup_epochs))
    lr_schedule = WarmupCosineSchedule(optimizer, warmup_steps, args.start_lr,
                                        args.lr, args.final_lr, total_steps)
    wd_schedule = CosineWDSchedule(optimizer, args.weight_decay,
                                    args.final_weight_decay, total_steps)

    global momentum_schedule
    momentum_schedule = iter(
        np.linspace(args.ema_start, args.ema_end, total_steps + 1, dtype=np.float64).tolist()
    )

    best_val_loss = float("inf")
    variant = resolve_variant(args.model, args)
    best_path = checkpoint_path(
        args.model,
        model_size="twin",
        checkpoint_dir=checkpoint_dir,
        variant=variant,
    )
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_metrics = train_epoch(model, train_loader, optimizer, device, forward_fn,
                                     mask_generator, lr_schedule, wd_schedule, args.grad_clip, args)
        val_metrics = val_epoch(model, val_loader, device, forward_fn, mask_generator, args)
        val_loss = val_metrics["loss"]

        print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {train_metrics['lr']:.6f} | WD: {train_metrics['wd']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(args.model, model, optimizer, epoch + 1, val_loss, args, best_path)
            print(f"  -> Saved best model to {best_path}")
        else:
            epochs_no_improve += 1

        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(f"\nEarly stopping after {args.early_stopping_patience} epochs without improvement.")
            break


if __name__ == "__main__":
    main()

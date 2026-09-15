import torch
from pathlib import Path
from .models import build_mae, build_ijepa


def build_model(model_name, device, model_size, embed_dim=None, num_heads=None, depth=None, in_chans=4):
    """Construct an MAE or I-JEPA model and move it to ``device``.

    Single entry point that dispatches to :func:`build_mae` / :func:`build_ijepa`
    by ``model_name`` ("mae" or "ijepa").
    """
    if model_name == "mae":
        model = build_mae(
            model_size=model_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
        )
    elif model_name == "ijepa":
        model = build_ijepa(
            model_size=model_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


def checkpoint_path(model_name, model_size, checkpoint_dir=Path("checkpoints"), variant=None, embed_dim=None):
    """Build the canonical checkpoint filename for a model configuration.

    Encodes size, optional ``embed_dim`` (``dN``), and optional ``variant`` into
    ``best_<model>_model_<suffix>.pth`` under ``checkpoint_dir``.
    """
    if model_name in ("mae", "ijepa", "realism"):
        parts = [model_size]
        if embed_dim is not None:
            parts.append(f"d{embed_dim}")
        if variant:
            parts.append(variant)
        suffix = "_".join(parts)
        return checkpoint_dir / f"best_{model_name}_model_{suffix}.pth"
    raise ValueError(f"Unknown model: {model_name}")

def save_mae_checkpoint(path, model, optimizer, epoch, val_loss, args):
    """Save MAE weights, optimizer state, epoch, val loss, and config to ``path``."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": vars(args),
        },
        path,
    )

def save_ijepa_checkpoint(path, model, optimizer, epoch, val_loss, args):
    """Save I-JEPA submodules (context/target encoders, predictor) + training state."""
    torch.save(
        {
            "context_encoder": model.context_encoder.state_dict(),
            "target_encoder": model.target_encoder.state_dict(),
            "predictor": model.predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": vars(args),
        },
        path,
    )


def load_model_checkpoint(model_name, model, path, device):
    """Load weights from ``path`` into ``model`` for the given model type.

    For MAE, accepts both the new dict format and a legacy raw state dict; for
    I-JEPA, restores the context encoder, target encoder, and predictor.
    """
    checkpoint = torch.load(path, map_location=device)
    print(f"loading path {path}")
    if model_name == "mae":
        # Support both old format (raw state dict) and new format (dict with "model" key)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model

    # IJEPA
    checkpoint = torch.load(path, map_location=device)
    model.context_encoder.load_state_dict(checkpoint["context_encoder"])
    model.target_encoder.load_state_dict(checkpoint["target_encoder"])
    model.predictor.load_state_dict(checkpoint["predictor"])
    return model

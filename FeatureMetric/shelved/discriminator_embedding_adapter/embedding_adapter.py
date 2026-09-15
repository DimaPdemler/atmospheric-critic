"""Frozen encoder front-end for the weather discriminator.

Maps a discriminator-normalized field batch ``(B, 4, 121, 240)`` through a
frozen FeatureMetric encoder (MAE / I-JEPA / SFNO) and returns a spatial
embedding upsampled back to ``(B, D, 121, 240)``, so the unchanged SqueezeNet
(with ``in_channels=D``) trains on embeddings instead of raw fields.

The adapter de-normalizes with the discriminator's own scalar ERA5 stats,
then applies each encoder's native preprocessing:
  - MAE / I-JEPA: FeatureMetric per-channel stats + (lat zero-pad 4/3, lon
    wrap 8/8) padding to 128x256 — exactly `utils/temporal.compose_temporal_input`
    with mode="none".
  - SFNO: raw physical units (it normalizes internally).

Corruptions therefore stay in discriminator-normalized field space, applied by
the dataset before this module, identical to the raw-field pipeline.

FeatureMetric is imported the same way `utils/sfno_embedding.py` resolves its
sibling repo: explicit config value -> $FEATUREMETRIC_DIR -> ../../FeatureMetric.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The 4-variable set (and order) every FeatureMetric encoder was trained on.
ENCODER_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
]

# FeatureMetric padding scheme (utils/temporal.py PAD_H / PAD_W).
PAD_TOP, PAD_BOTTOM = 4, 3
PAD_LEFT, PAD_RIGHT = 8, 8


def _resolve_featuremetric_dir(fm_dir=None):
    """Explicit arg -> $FEATUREMETRIC_DIR -> sibling FeatureMetric/ directory."""
    candidates = [
        fm_dir,
        os.environ.get("FEATUREMETRIC_DIR"),
        Path(__file__).resolve().parents[2] / "FeatureMetric",
    ]
    for cand in candidates:
        if cand and Path(cand).is_dir():
            return Path(cand).resolve()
    raise FileNotFoundError(
        "FeatureMetric directory not found. Set embedding_encoder.featuremetric_dir "
        "in the config or the FEATUREMETRIC_DIR environment variable."
    )


def _import_featuremetric(fm_dir=None):
    """Make FeatureMetric's `utils` package importable; return its root path.

    Appends (not prepends) to sys.path so Discriminator-local modules keep
    priority over any same-named FeatureMetric modules.
    """
    root = _resolve_featuremetric_dir(fm_dir)
    if str(root) not in sys.path:
        sys.path.append(str(root))
    return root


def embedding_cfg_from(cfg):
    """Extract the `embedding_encoder` block as a plain dict, or None."""
    enc_cfg = cfg.get("embedding_encoder") if cfg is not None else None
    if not enc_cfg:
        return None
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(enc_cfg):
            enc_cfg = OmegaConf.to_container(enc_cfg, resolve=True)
    except ImportError:
        pass
    return dict(enc_cfg)


def maybe_build_adapter(cfg, variables, device=None):
    """Build a FrozenEncoderAdapter when the config enables one, else None."""
    enc_cfg = embedding_cfg_from(cfg)
    if enc_cfg is None:
        return None
    adapter = FrozenEncoderAdapter(enc_cfg, list(variables))
    if device is not None:
        adapter = adapter.to(device)
    return adapter


def load_field_stats_sidecar(weights_path):
    """Load the `<weights>.stats.json` sidecar written at train time, or None."""
    sidecar = Path(f"{weights_path}.stats.json")
    if not sidecar.exists():
        return None
    with open(sidecar) as f:
        payload = json.load(f)
    return payload["means"], payload["stds"]


class FrozenEncoderAdapter(nn.Module):
    """Frozen encoder -> ``(B, D, 121, 240)`` embedding input adapter."""

    def __init__(self, enc_cfg, variables):
        super().__init__()
        if list(variables) != ENCODER_VARS:
            raise ValueError(
                f"embedding encoders were trained on {ENCODER_VARS}; "
                f"config variables are {list(variables)}"
            )
        self.name = enc_cfg["name"]
        self.variables = list(variables)
        fm_root = _import_featuremetric(enc_cfg.get("featuremetric_dir"))

        if self.name in ("mae", "ijepa"):
            from utils.model_io import build_model, load_model_checkpoint

            ckpt_path = enc_cfg["checkpoint"]
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model_cfg = ckpt.get("config", {})
            if model_cfg.get("temporal_mode", "none") != "none":
                raise ValueError(f"{ckpt_path} is a temporal checkpoint; need temporal_mode=none")
            self.encoder = build_model(
                self.name, torch.device("cpu"), model_cfg.get("model_size", "twin"),
                embed_dim=model_cfg.get("embed_dim"),
                num_heads=model_cfg.get("num_heads"),
                depth=model_cfg.get("depth"),
            )
            load_model_checkpoint(self.name, self.encoder, ckpt_path, torch.device("cpu"))
            stats_dir = Path(enc_cfg.get("stats_dir") or (fm_root / "checkpoints"))
            fm_mean = np.load(stats_dir / "data_mean.npy").reshape(1, -1, 1, 1)
            fm_std = np.load(stats_dir / "data_std.npy").reshape(1, -1, 1, 1)
            self.register_buffer("fm_mean", torch.tensor(fm_mean, dtype=torch.float32))
            self.register_buffer("fm_std", torch.tensor(fm_std, dtype=torch.float32))
            # patch_embed lives on the MAE itself / on I-JEPA's target encoder.
            patch_embed = (self.encoder.patch_embed if self.name == "mae"
                           else self.encoder.target_encoder.patch_embed)
            patch = patch_embed.patch_size
            patch_h = patch[0] if isinstance(patch, (tuple, list)) else patch
            patch_w = patch[1] if isinstance(patch, (tuple, list)) else patch
            self.token_grid = (
                (121 + PAD_TOP + PAD_BOTTOM) // patch_h,
                (240 + PAD_LEFT + PAD_RIGHT) // patch_w,
            )
            self._out_channels = int(patch_embed.proj.out_channels)
        elif self.name == "sfno":
            from utils.sfno_embedding import SFNOEmbedding

            sfno_cfg = enc_cfg.get("sfno") or {}
            self.encoder = SFNOEmbedding(
                embedding_channels=sfno_cfg.get("embedding_channels", 8),
                embedding_resolution=tuple(sfno_cfg.get("embedding_resolution", (31, 60))),
                repo_root=sfno_cfg.get("repo_root"),
            )
            self._out_channels = int(self.encoder.embedding_channels)
        else:
            raise ValueError(f"unknown embedding encoder {self.name!r}")

        self.encoder.eval()
        self.requires_grad_(False)

        # Discriminator-side scalar ERA5 stats; filled via set_field_stats().
        self.register_buffer("field_mean", torch.zeros(1, len(variables), 1, 1))
        self.register_buffer("field_std", torch.ones(1, len(variables), 1, 1))
        self._field_stats_set = False

    @property
    def out_channels(self):
        return self._out_channels

    def set_field_stats(self, means, stds):
        """Set the discriminator dataset's per-variable scalar z-score stats."""
        mean = torch.tensor([float(means[v]) for v in self.variables], dtype=torch.float32)
        std = torch.tensor([float(stds[v]) for v in self.variables], dtype=torch.float32)
        self.field_mean.copy_(mean.view(1, -1, 1, 1))
        self.field_std.copy_(std.view(1, -1, 1, 1))
        self._field_stats_set = True

    def train(self, mode=True):
        """Stay in eval mode regardless of the parent LightningModule's mode."""
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        if not self._field_stats_set:
            raise RuntimeError("FrozenEncoderAdapter: call set_field_stats() before forward()")
        # Encoders need exact preprocessing; disable any surrounding autocast
        # (16-mixed training) and run in fp32 (SFNO's SHT requires it).
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            # The discriminator dataset keeps the NetCDF (lon, lat) = (240, 121)
            # orientation; encoders expect (lat, lon) = (121, 240).
            if x.shape[-2:] == (240, 121):
                x = x.transpose(-2, -1)
            elif x.shape[-2:] != (121, 240):
                raise ValueError(f"expected (240, 121) or (121, 240) fields, got {tuple(x.shape)}")
            x_phys = x * self.field_std + self.field_mean

            if self.name == "sfno":
                emb = self.encoder.encode(x_phys)                       # (B, C, h, w)
                return F.interpolate(emb, size=(121, 240), mode="bilinear",
                                     align_corners=False)

            # MAE / I-JEPA: FeatureMetric normalization + padding (temporal._pad
            # order: lat zero-pad first, then lon wrap — corners wrap the zeros).
            x_fm = (x_phys - self.fm_mean) / self.fm_std
            x_fm = F.pad(x_fm, (0, 0, PAD_TOP, PAD_BOTTOM), mode="constant", value=0.0)
            x_fm = F.pad(x_fm, (PAD_LEFT, PAD_RIGHT, 0, 0), mode="circular")

            tokens = self.encoder.extract_patch_tokens(x_fm)            # (B, N, D)
            gh, gw = self.token_grid
            grid = tokens.view(-1, gh, gw, tokens.shape[-1]).permute(0, 3, 1, 2)
            up = F.interpolate(grid, size=(121 + PAD_TOP + PAD_BOTTOM,
                                           240 + PAD_LEFT + PAD_RIGHT),
                               mode="bilinear", align_corners=False)
            return up[:, :, PAD_TOP:PAD_TOP + 121, PAD_LEFT:PAD_LEFT + 240]

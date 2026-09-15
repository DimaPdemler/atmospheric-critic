"""Contrastive realism metric: a feature embedding trained (from scratch, on ERA5
only) to be sensitive to the *subtle* differences that separate real reanalysis
from ML forecasts.

The model learns the manifold of realistic atmospheric states from ERA5 plus a
suite of subtle, physically-motivated corruptions (see
``utils.corruptions.CONTRASTIVE_CORRUPTION_SPECS``). At test time it scores a
field's distance from that manifold *reference-free* via a realism head (and,
independently, distance to an EMA manifold center). The key claim: trained
without ever seeing forecasts, it still flags forecasts as off-manifold.

Three training losses per ERA5 batch:
  1. Invariance (VICReg) between two longitude-rolled clean views — tightens the
     manifold, removes nuisance variance, prevents collapse.
  2. Severity regression — realism head predicts 0 for clean views and the
     normalized corruption strength for a subtly-corrupted view.
  3. Compactness (deep-SVDD + severity margin) — pull clean embeddings toward an
     EMA center; push the corrupted embedding farther by a severity-scaled margin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.models import build_mae
from utils.corruptions import sample_contrastive_corruption


def random_longitude_roll(x):
    """Realism-preserving augmentation: cyclic shift along longitude (width).

    Longitude is periodic, so a roll yields another physically valid field. The
    dataset pads longitude by wrapping, so rolling the full padded tensor keeps
    the wrap structure consistent up to the pad width (a negligible edge effect
    for augmentation purposes).
    """
    shift = int(torch.randint(0, x.shape[-1], (1,)).item())
    return torch.roll(x, shifts=shift, dims=-1)


class MLPHead(nn.Module):
    """Two-layer MLP head (Linear → GELU → Linear)."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def vicreg_loss(p1, p2, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
    """Standard VICReg loss: invariance + variance (anti-collapse) + covariance.

    ``p1``, ``p2`` are projector outputs of two views, shape (B, D). Returns
    ``(loss, logs)``.
    """
    inv = F.mse_loss(p1, p2)

    def _std_term(p):
        p = p - p.mean(dim=0)
        std = torch.sqrt(p.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1.0 - std))

    std = _std_term(p1) + _std_term(p2)

    def _cov_term(p):
        n, d = p.shape
        p = p - p.mean(dim=0)
        cov = (p.T @ p) / (n - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).sum() / d

    cov = _cov_term(p1) + _cov_term(p2)
    loss = sim_coeff * inv + std_coeff * std + cov_coeff * cov
    logs = {"vic_inv": inv.item(), "vic_std": std.item(), "vic_cov": cov.item()}
    return loss, logs


class RealismMetric(nn.Module):
    """ViT encoder (from scratch) + projector + realism head + EMA manifold center."""

    def __init__(
        self,
        model_size="twin",
        in_chans=4,
        embed_dim=None,
        proj_dim=128,
        head_hidden=256,
        center_momentum=0.99,
    ):
        super().__init__()
        # Reuse the MAE ViT as the encoder backbone (the decoder is unused here).
        self.backbone = build_mae(model_size=model_size, in_chans=in_chans, embed_dim=embed_dim)
        D = self.backbone.pos_embed.shape[-1]
        self.embed_dim = D

        self.projector = MLPHead(D, head_hidden, proj_dim)
        self.realism_head = MLPHead(D, head_hidden, 1)

        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(D))
        self.register_buffer("center_initialized", torch.zeros(1, dtype=torch.bool))

    def embed(self, imgs):
        """Run the encoder (no masking) and mean-pool patch tokens (CLS excluded).

        Implemented inline so the embedding is always mean-pooled, independent of
        the ``EXTRACT_FEATURES_POOLING`` env var used by the SSL eval scripts.
        """
        bb = self.backbone
        x = bb.patch_embed(imgs)
        x = x + bb.pos_embed[:, 1:, :]
        cls = (bb.cls_token + bb.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        for blk in bb.encoder_blocks:
            x = blk(x)
        x = bb.encoder_norm(x)
        return x[:, 1:, :].mean(dim=1)

    def project(self, z):
        return self.projector(z)

    def predict_realism(self, z):
        """Reference-free scalar realism score (higher = less realistic)."""
        return self.realism_head(z).squeeze(-1)

    @torch.no_grad()
    def update_center(self, z):
        """EMA-update the manifold center using L2-normalized clean embeddings."""
        batch_center = F.normalize(z, dim=-1).mean(dim=0)
        if not bool(self.center_initialized):
            self.center.copy_(batch_center)
            self.center_initialized.fill_(True)
        else:
            self.center.mul_(self.center_momentum).add_(
                batch_center, alpha=1.0 - self.center_momentum
            )

    def center_distance(self, z):
        """Distance from L2-normalized embedding to the EMA manifold center."""
        return torch.norm(F.normalize(z, dim=-1) - self.center, dim=-1)


def compute_realism_losses(
    model,
    x_clean,
    w_inv=1.0,
    w_real=1.0,
    w_comp=1.0,
    margin=0.5,
    types=None,
    generator=None,
    update_center=True,
):
    """Compute the combined realism training loss for one ERA5 batch.

    ``x_clean``: (B, C, H, W) normalized, padded ERA5 fields from AtmosphereDataset.
    Returns ``(total_loss, logs)``.
    """
    # Two realism-preserving views + one subtly-corrupted view (paired to v1).
    v1 = random_longitude_roll(x_clean)
    v2 = random_longitude_roll(x_clean)
    x_corr, strength, name = sample_contrastive_corruption(v1, generator=generator, types=types)

    z1 = model.embed(v1)
    z2 = model.embed(v2)
    zc = model.embed(x_corr)

    # 1. Invariance (VICReg) over the two clean views.
    p1, p2 = model.project(z1), model.project(z2)
    l_inv, inv_logs = vicreg_loss(p1, p2)

    # 2. Severity regression: clean → 0, corrupted → normalized strength.
    r1 = model.predict_realism(z1)
    r2 = model.predict_realism(z2)
    rc = model.predict_realism(zc)
    t = torch.full_like(rc, float(strength))
    l_real = (
        F.smooth_l1_loss(r1, torch.zeros_like(r1))
        + F.smooth_l1_loss(r2, torch.zeros_like(r2))
        + F.smooth_l1_loss(rc, t)
    )

    # 3. Compactness: pull clean to EMA center; push corrupted out by severity margin.
    if update_center:
        model.update_center(z1.detach())
    c = model.center.detach()
    z1n = F.normalize(z1, dim=-1)
    zcn = F.normalize(zc, dim=-1)
    d_clean = ((z1n - c) ** 2).sum(dim=-1)
    d_corr = ((zcn - c) ** 2).sum(dim=-1)
    l_comp = d_clean.mean() + F.relu(margin * float(strength) + d_clean - d_corr).mean()

    total = w_inv * l_inv + w_real * l_real + w_comp * l_comp
    logs = {
        "total": total.item(),
        "loss_inv": l_inv.item(),
        "loss_real": l_real.item(),
        "loss_comp": l_comp.item(),
        "corruption": name,
        "strength": float(strength),
        # Per-sample realism scores (detached) for monitoring clean-vs-corrupted
        # discrimination AUC during training.
        "r_clean": r1.detach(),
        "r_corr": rc.detach(),
        **inv_logs,
    }
    return total, logs


if __name__ == "__main__":
    torch.manual_seed(0)
    model = RealismMetric(model_size="twin")
    dummy = torch.randn(4, 4, 128, 256)
    loss, logs = compute_realism_losses(model, dummy)
    loss.backward()
    print("Smoke test OK. embed_dim =", model.embed_dim)
    print("Loss:", round(loss.item(), 4))
    print("Logs:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in logs.items()})

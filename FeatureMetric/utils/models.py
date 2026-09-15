import math
import os
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def _pool_patch_tokens(x):
    """Reduce patch-token sequence (B, N, D) to image vector (B, D) or (B, 2D).

    Mode is read from the EXTRACT_FEATURES_POOLING env var per call so eval
    scripts can toggle without rebuilding the model. Defaults to "mean".
    'concat' concatenates mean and max → output dim is 2*D.
    """
    mode = os.environ.get("EXTRACT_FEATURES_POOLING", "mean").lower()
    if mode == "mean":
        return x.mean(dim=1)
    if mode == "max":
        return x.amax(dim=1)
    if mode == "concat":
        return torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1)
    raise ValueError(
        f"EXTRACT_FEATURES_POOLING={mode!r} not understood; expected 'mean', 'max', or 'concat'."
    )


# ---------------------------------------------------------------------------
# Positional embedding helpers (MAE only — used by MaskedAutoencoderViT)
# ---------------------------------------------------------------------------

def _mae_get_1d_sincos_pos_embed(embed_dim, positions):
    """Build 1-D sine/cosine positional embeddings for the given positions.

    Returns an ``(len(positions), embed_dim)`` array; half the dims encode sine,
    half cosine, at geometrically spaced frequencies.
    """
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)
    positions = positions.reshape(-1).astype(np.float32)
    out = np.einsum("m,d->md", positions, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _mae_get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """Build 2-D sin/cos positional embeddings for an H×W patch grid.

    Concatenates separate height and width 1-D embeddings per patch. If
    ``cls_token`` is set, a zero row is prepended for the CLS token.
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h, indexing="xy")
    grid = np.stack(grid, axis=0).reshape(2, -1)
    emb_h = _mae_get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    emb_w = _mae_get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), pos_embed], axis=0)
    return pos_embed


# ---------------------------------------------------------------------------
# MAE PatchEmbed
# ---------------------------------------------------------------------------

class PatchEmbedMAE(nn.Module):
    """Splits an image into non-overlapping patches and linearly embeds each.

    Implemented as a strided conv; output is a ``(B, num_patches, embed_dim)``
    token sequence.
    """

    def __init__(self, img_size=(128, 256), patch_size=16, in_chans=4, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """Embed ``(B, C, H, W)`` image into ``(B, num_patches, embed_dim)`` tokens."""
        return self.proj(x).flatten(2).transpose(1, 2)


# ---------------------------------------------------------------------------
# Masked Autoencoder ViT (MAE)
# ---------------------------------------------------------------------------

class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with a ViT encoder/decoder for ERA5 fields.

    A fraction of patches is masked; the encoder processes only visible patches
    (plus a CLS token) and the lightweight decoder reconstructs the raw pixel
    values of the masked patches. Loss is L1 on masked patches only.
    ``extract_features`` reuses the encoder (no masking) to produce a latent
    feature vector for the realism metric.
    """

    def __init__(
        self,
        img_size=(128, 256),
        patch_size=16,
        in_chans=4,
        embed_dim=256,
        depth=6,
        num_heads=8,
        decoder_embed_dim=128,
        decoder_depth=3,
        decoder_num_heads=4,
        mlp_ratio=4.0,
        init_std=0.02,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedMAE(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        dpr = [x.item() for x in torch.linspace(0, 0., depth)]  # no drop path by default
        self.encoder_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim, nhead=decoder_num_heads,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.init_std=init_std
        self.initialize_weights()

    def initialize_weights(self):
        """Fill fixed sin/cos positional embeddings and init all learnable weights."""
        enc_pe = _mae_get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.grid_size[0], self.grid_size[1], cls_token=True)
        dec_pe = _mae_get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.grid_size[0], self.grid_size[1], cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(enc_pe).float().unsqueeze(0))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dec_pe).float().unsqueeze(0))
        # nn.init.normal_(self.cls_token, std=self.init_std)
        # nn.init.normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        # apply same init scheme as in VisionTransformer
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        """Rescale per-layer weights by 1/sqrt(2*layer_id) for stable deep init."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.encoder_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        """Per-module weight init: truncated-normal linears/convs, unit LayerNorm."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def random_masking(self, x, mask_ratio):
        """Randomly keep a (1-mask_ratio) subset of patches per sample.

        Returns ``(x_masked, mask, ids_restore)``: the kept tokens, the binary
        mask (1 = removed) in original order, and indices to un-shuffle the
        sequence in the decoder.
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def structured_masking(self, x, visible_indices, loss_mask):
        """Apply an externally supplied mask (e.g. I-JEPA blocks) instead of random.

        Given explicit ``visible_indices`` and a ``loss_mask`` (1 = masked), gathers
        the visible tokens and reconstructs ``ids_restore``. Requires a fixed count
        of masked patches per sample so the batch stays rectangular.
        """
        _, L, D = x.shape
        ids_keep = visible_indices.long()
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = loss_mask.to(device=x.device, dtype=x.dtype)
        masked_counts = mask.sum(dim=1).to(dtype=torch.int64)
        if not torch.all(masked_counts == masked_counts[0]):
            raise ValueError("Structured masking requires a fixed number of masked patches per batch")

        ids_mask = torch.argsort(mask, dim=1, descending=True)[:, :masked_counts[0].item()]
        ids_shuffle = torch.cat([ids_keep, ids_mask], dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        if ids_restore.shape[1] != L:
            raise ValueError("Structured masking produced an invalid restore index shape")
        return x_visible, mask, ids_restore

    def forward_encoder(self, x, mask_ratio=None, visible_indices=None, loss_mask=None):
        """Patch-embed, mask, prepend CLS, and run the encoder transformer.

        Uses structured masking when ``visible_indices`` is given, otherwise random
        masking at ``mask_ratio``. Returns ``(encoded, mask, ids_restore)``.
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        if visible_indices is not None:
            if loss_mask is None:
                raise ValueError("loss_mask is required when visible_indices are provided")
            x, mask, ids_restore = self.structured_masking(x, visible_indices, loss_mask)
        else:
            if mask_ratio is None:
                raise ValueError("mask_ratio is required for random masking")
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """Re-insert mask tokens, restore patch order, and predict raw pixels.

        Returns per-patch pixel predictions of shape ``(B, num_patches, p^2 * C)``.
        """
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        x = self.decoder(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        """Mean L1 reconstruction error, averaged over masked patches only."""
        target = self.patchify(imgs)
        loss = (pred - target).abs().mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def patchify(self, imgs):
        """Reshape ``(B, C, H, W)`` images into ``(B, num_patches, p^2 * C)`` targets."""
        p = self.patch_embed.patch_size
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    def forward(self, imgs, mask_ratio=0.75, visible_indices=None, loss_mask=None):
        """Full MAE forward: encode → decode → loss. Returns ``(loss, pred, mask)``."""
        latent, mask, ids_restore = self.forward_encoder(
            imgs,
            mask_ratio=mask_ratio,
            visible_indices=visible_indices,
            loss_mask=loss_mask,
        )
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def extract_patch_tokens(self, imgs):
        """Encode all patches (no masking) and return per-patch tokens (B, N, D).

        CLS token is excluded; N = grid_h * grid_w in row-major patch order.
        """
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        return x[:, 1:, :]

    def extract_features(self, imgs):
        """Encode all patches (no masking) and pool to a latent feature vector.

        This is the metric backbone's representation: runs the full encoder at
        mask_ratio 0 and pools the patch tokens (CLS excluded).
        """
        return _pool_patch_tokens(self.extract_patch_tokens(imgs))


# ===========================================================================
# I-JEPA — copied verbatim from Meta's original implementation
# (https://github.com/facebookresearch/ijepa, src/models/vision_transformer.py)
# Only changes are:
#   • get_2d_sincos_pos_embed: accepts (grid_size_h, grid_size_w) instead of
#     a single square grid_size
#   • PatchEmbed: img_size is a (H, W) tuple; in_chans=4
#   • VisionTransformerPredictor / VisionTransformer: pass (grid_h, grid_w)
#     to get_2d_sincos_pos_embed instead of int(num_patches**.5)
#   • VisionTransformer.forward: drop interpolate_pos_encoding (fixed size)
# Everything else — Attention, Block, MLP, DropPath, init schemes,
# fix_init_weight, trunc_normal_, apply_masks, repeat_interleave_batch —
# is a direct copy.
# ===========================================================================

# ---------------------------------------------------------------------------
# Copied from src/utils/tensors.py
# ---------------------------------------------------------------------------

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """In-place fill ``tensor`` with a truncated normal via the inverse-CDF method."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Public wrapper: fill ``tensor`` with values from a truncated normal in [a, b]."""
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


def repeat_interleave_batch(x, B, repeat):
    """Repeat each per-batch block of ``x`` ``repeat`` times, keeping batches grouped.

    Used to align predictor mask tokens with the number of context masks.
    """
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x


# ---------------------------------------------------------------------------
# Copied from src/models/vision_transformer.py
# get_2d_sincos_pos_embed: adapted for rectangular (grid_size_h, grid_size_w)
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w=None, cls_token=False):
    """
    grid_size_h, grid_size_w: ints for height and width of the patch grid.
    If grid_size_w is None, falls back to square (grid_size_w = grid_size_h).
    return:
    pos_embed: [grid_size_h*grid_size_w, embed_dim] or [1+grid_size_h*grid_size_w, embed_dim]
    """
    if grid_size_w is None:
        grid_size_w = grid_size_h
    grid_h = np.arange(grid_size_h, dtype=float)
    grid_w = np.arange(grid_size_w, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Combine separate height/width 1-D sin/cos embeddings for a coordinate grid."""
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega   # (D/2,)

    pos = pos.reshape(-1)   # (M,)
    out = np.einsum('m,d->md', pos, omega)   # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# ---------------------------------------------------------------------------
# Copied verbatim: drop_path, DropPath, MLP, Attention, Block
# ---------------------------------------------------------------------------

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Stochastic depth: randomly zero whole residual paths per sample at train time."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    """Two-layer feed-forward block (Linear → activation → Linear) used in transformer blocks."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Standard multi-head self-attention; returns ``(output, attention_weights)``."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    """Pre-norm transformer block: attention + MLP, each with a residual connection."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# PatchEmbed — adapted for rectangular img_size tuple and in_chans=4
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=(128, 256), patch_size=16, in_chans=4, embed_dim=768):
        super().__init__()
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# VisionTransformerPredictor — copied verbatim, with one adaptation:
#   • __init__ accepts grid_size=(H, W) and passes (grid_h, grid_w) to
#     get_2d_sincos_pos_embed instead of int(num_patches**.5)
# ---------------------------------------------------------------------------

class VisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        num_patches,
        grid_size,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False)
        predictor_pos_embed = get_2d_sincos_pos_embed(
            self.predictor_pos_embed.shape[-1],
            grid_size[0], grid_size[1],  # rectangular adaptation
            cls_token=False)
        self.predictor_pos_embed.data.copy_(
            torch.from_numpy(predictor_pos_embed).float().unsqueeze(0))
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        """Rescale per-layer weights by 1/sqrt(2*layer_id) for stable deep init."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        """Per-module weight init: truncated-normal linears/convs, unit LayerNorm."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks_x, masks):
        """Predict target-block representations from context tokens.

        Takes context tokens ``x`` (selected by ``masks_x``) plus learned mask
        tokens placed at the ``masks`` target positions, runs the predictor
        transformer, and returns only the predictions for the target patches.
        """
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # -- Batch Size
        B = len(x) // len(masks_x)

        # -- map from encoder-dim to predictor-dim
        x = self.predictor_embed(x)

        # -- add positional embedding to x tokens
        x_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # -- concat mask tokens to x
        pos_embs = self.predictor_pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        # --
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        # --
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # -- return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x


# Alias used by IJEPA wrapper
IJEPAPredictor = VisionTransformerPredictor


# ---------------------------------------------------------------------------
# VisionTransformer — copied verbatim, with two adaptations:
#   • img_size is a (H, W) tuple; grid_size computed accordingly
#   • get_2d_sincos_pos_embed called with (grid_h, grid_w) not int(sqrt(N))
#   • interpolate_pos_encoding removed (fixed input size)
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=(128, 256),
        patch_size=16,
        in_chans=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        # --
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        # --
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            grid_size[0], grid_size[1],  # rectangular adaptation
            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        """Rescale per-layer weights by 1/sqrt(2*layer_id) for stable deep init."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        """Per-module weight init: truncated-normal linears/convs, unit LayerNorm."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, masks=None):
        """Patch-embed, add positions, optionally keep only ``masks`` patches, encode.

        Returns the normalized patch-token sequence. When ``masks`` is given (a
        list of index tensors), only those patches are processed.
        """
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # -- patchify x
        x = self.patch_embed(x)

        # -- add positional embedding to x
        x = x + self.pos_embed

        # -- mask x
        if masks is not None:
            x = apply_masks(x, masks)

        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        if self.norm is not None:
            x = self.norm(x)

        return x

    def extract_features(self, x):
        """Encode all patches (no masking) and pool to a latent feature vector."""
        return _pool_patch_tokens(self.forward(x))


# Alias used elsewhere
RectangularVisionTransformer = VisionTransformer


# ---------------------------------------------------------------------------
# IJEPA wrapper (unchanged)
# ---------------------------------------------------------------------------

class IJEPA(nn.Module):
    """I-JEPA model bundling the context encoder, EMA target encoder, and predictor.

    Training predicts target-block representations (produced by the frozen,
    EMA-updated target encoder) from context-block representations, with an MSE/
    smooth-L1 loss in latent space — no pixel reconstruction. ``extract_features``
    always uses the target encoder for the realism metric.
    """

    def __init__(
        self,
        img_size=(128, 256),
        patch_size=16,
        in_chans=4,
        embed_dim=192,
        depth=8,
        num_heads=3,
        predictor_embed_dim=192,
        predictor_depth=3,
        predictor_num_heads=3,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.context_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.target_encoder = deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        num_patches = self.context_encoder.patch_embed.num_patches
        self.predictor = VisionTransformerPredictor(
            num_patches=num_patches,
            grid_size=grid_size,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.embed_dim = embed_dim

    @torch.no_grad()
    def update_target_encoder(self, momentum):
        """EMA-update the target encoder toward the context encoder weights."""
        for context_param, target_param in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data.mul_(momentum).add_(context_param.data, alpha=1.0 - momentum)

    def forward(self, imgs, context_masks, target_masks):
        """Run target/context encoders + predictor; return ``(loss, preds, targets)``.

        Target representations come from the frozen target encoder (no grad);
        the loss is smooth-L1 between predicted and target patch embeddings.
        """
        self.target_encoder.eval()
        with torch.no_grad():
            target_tokens = self.target_encoder(imgs)
            target_tokens = F.layer_norm(target_tokens, (target_tokens.size(-1),))
            target_tokens = apply_masks(target_tokens, target_masks)

        context_tokens = self.context_encoder(imgs, masks=context_masks)
        pred_tokens = self.predictor(context_tokens, context_masks, target_masks)
        loss = F.smooth_l1_loss(pred_tokens, target_tokens)
        return loss, pred_tokens, target_tokens

    @torch.no_grad()
    def extract_patch_tokens(self, imgs):
        """Per-patch tokens (B, N, D) from the target encoder; no CLS token."""
        return self.target_encoder(imgs)

    @torch.no_grad()
    def extract_features(self, imgs):
        """Latent feature vector from the target encoder (the metric backbone)."""
        return self.target_encoder.extract_features(imgs)

    @property
    def encoder(self):
        """The trainable context encoder (alias used by external callers)."""
        return self.context_encoder


def _auto_num_heads(embed_dim, target_head_dim=64):
    """Pick num_heads so head_dim is close to `target_head_dim` and divides embed_dim."""
    n = max(1, embed_dim // target_head_dim)
    while n > 1 and embed_dim % n != 0:
        n -= 1
    return n


def build_ijepa(
    model_size="tiny",
    img_size=(128, 256),
    patch_size=16,
    in_chans=4,
    embed_dim=None,
    num_heads=None,
    depth=None,
    predictor_embed_dim=None,
):
    """Construct an :class:`IJEPA` from a named size preset ("tiny"/"small"/"twin").

    Any of ``embed_dim``/``num_heads``/``depth``/``predictor_embed_dim`` override
    the preset; ``num_heads`` is auto-derived from ``embed_dim`` when omitted.
    Validates that ``embed_dim`` is divisible by ``num_heads``.
    """
    configs = {
        "tiny": {
            "embed_dim": 192,
            "depth": 8,
            "num_heads": 3,
            "predictor_embed_dim": 192,
            "predictor_depth": 3,
            "predictor_num_heads": 3,
        },
        "small": {
            "embed_dim": 384,
            "depth": 10,
            "num_heads": 6,
            "predictor_embed_dim": 192,
            "predictor_depth": 4,
            "predictor_num_heads": 3,
        },
        "twin": {
            # Matches MAE 'twin' encoder exactly
            "embed_dim": 192,
            "depth": 8,
            "num_heads": 3,
            "predictor_embed_dim": 192,
            "predictor_depth": 3,
            "predictor_num_heads": 3,
        },
    }
    if model_size not in configs:
        raise ValueError(f"Unknown I-JEPA model size: {model_size}")
    cfg = dict(configs[model_size])
    if embed_dim is not None:
        cfg["embed_dim"] = embed_dim
        if num_heads is None:
            cfg["num_heads"] = _auto_num_heads(embed_dim)
    if num_heads is not None:
        cfg["num_heads"] = num_heads
    if depth is not None:
        cfg["depth"] = depth
    if predictor_embed_dim is not None:
        cfg["predictor_embed_dim"] = predictor_embed_dim
    if cfg["embed_dim"] % cfg["num_heads"] != 0:
        raise ValueError(
            f"embed_dim ({cfg['embed_dim']}) must be divisible by num_heads ({cfg['num_heads']})"
        )
    return IJEPA(img_size=img_size, patch_size=patch_size, in_chans=in_chans, **cfg)


def build_mae(
    model_size="twin",
    img_size=(128, 256),
    patch_size=16,
    in_chans=4,
    embed_dim=None,
    num_heads=None,
    depth=None,
    decoder_embed_dim=None,
):
    """Construct a :class:`MaskedAutoencoderViT` from a named size preset.

    Overrides mirror :func:`build_ijepa`; when ``embed_dim`` is given the decoder
    width and head counts are auto-derived. Validates head divisibility for both
    encoder and decoder.
    """
    configs = {
        "default": {
            "embed_dim": 256,
            "depth": 6,
            "num_heads": 8,
            "decoder_embed_dim": 128,
            "decoder_depth": 2,
            "decoder_num_heads": 4,
        },
        "twin": {
            # Encoder matches IJEPA tiny exactly for fair comparison
            "embed_dim": 192,
            "depth": 8,
            "num_heads": 3,
            "decoder_embed_dim": 96,
            "decoder_depth": 2,
            "decoder_num_heads": 3,
        },
    }
    if model_size not in configs:
        raise ValueError(f"Unknown MAE model size: {model_size}")
    cfg = dict(configs[model_size])
    if embed_dim is not None:
        cfg["embed_dim"] = embed_dim
        if num_heads is None:
            cfg["num_heads"] = _auto_num_heads(embed_dim)
        if decoder_embed_dim is None:
            cfg["decoder_embed_dim"] = max(64, embed_dim // 2)
            cfg["decoder_num_heads"] = _auto_num_heads(cfg["decoder_embed_dim"], target_head_dim=32)
    if num_heads is not None:
        cfg["num_heads"] = num_heads
    if depth is not None:
        cfg["depth"] = depth
    if decoder_embed_dim is not None:
        cfg["decoder_embed_dim"] = decoder_embed_dim
        cfg["decoder_num_heads"] = _auto_num_heads(decoder_embed_dim, target_head_dim=32)
    if cfg["embed_dim"] % cfg["num_heads"] != 0:
        raise ValueError(
            f"embed_dim ({cfg['embed_dim']}) must be divisible by num_heads ({cfg['num_heads']})"
        )
    if cfg["decoder_embed_dim"] % cfg["decoder_num_heads"] != 0:
        raise ValueError(
            f"decoder_embed_dim ({cfg['decoder_embed_dim']}) must be divisible by "
            f"decoder_num_heads ({cfg['decoder_num_heads']})"
        )
    return MaskedAutoencoderViT(img_size=img_size, patch_size=patch_size, in_chans=in_chans, **cfg)

if __name__ == "__main__":
    model = MaskedAutoencoderViT()
    dummy_input = torch.randn(2, 4, 128, 256)
    loss, pred, mask = model(dummy_input)
    print("Loss:", loss.item())

    z = model.extract_features(dummy_input)
    print("Latent shape:", z.shape)

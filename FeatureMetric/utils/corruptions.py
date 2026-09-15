"""Synthetic corruptions for self-supervised evaluation.

A parallel copy lives at Discriminator/scripts/corruptions.py with blur-floor
remapping and mean-preservation additions; this copy adds spectral-lowpass
filtering and contrastive corruption sampling.
"""

import torch
import scipy.ndimage as ndimage
import numpy as np


MAX_SEVERITY = 2.0
T2M_CHANNEL = 0
U10_CHANNEL = 1
V10_CHANNEL = 2
MSL_CHANNEL = 3
PAD_TOP = 4
PAD_BOTTOM = 3
PAD_LEFT = 8
PAD_RIGHT = 8
PADDED_SHAPE = (128, 256)


def get_corruption_ladder(corruption_type, n_steps=9):
    """
    Returns discrete severity levels in [0, MAX_SEVERITY] for distance evaluation.
    """
    if corruption_type in (
        "blur",
        "noise",
        "grf",
        "pixel_replace",
        "wind_patch_shuffle",
        "wind_rotation",
    ):
        return np.linspace(0.0, MAX_SEVERITY, n_steps).tolist()
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")


def _is_model_padded_shape(x):
    """True if ``x`` has the model's padded ``(128, 256)`` spatial shape."""
    return tuple(x.shape[-2:]) == PADDED_SHAPE


def _crop_interior(x):
    """Strip the dataset padding, returning the real ERA5 interior region."""
    return x[:, :, PAD_TOP:x.shape[-2] - PAD_BOTTOM, PAD_LEFT:x.shape[-1] - PAD_RIGHT]


def _repad_like_dataset(x):
    """Re-apply the dataset's padding (wrap in longitude, zero-pad in latitude)."""
    x_np = x.detach().cpu().numpy()
    x_np = np.pad(x_np, pad_width=((0, 0), (0, 0), (0, 0), (PAD_LEFT, PAD_RIGHT)), mode="wrap")
    x_np = np.pad(x_np, pad_width=((0, 0), (0, 0), (PAD_TOP, PAD_BOTTOM), (0, 0)), mode="constant", constant_values=0)
    return torch.from_numpy(x_np).to(device=x.device, dtype=x.dtype)


def _finalize_like_input(original_x, work_x):
    """Re-pad ``work_x`` if the original input was padded; otherwise return as-is."""
    if _is_model_padded_shape(original_x):
        return _repad_like_dataset(work_x)
    return work_x


def apply_gaussian_blur(x, severity):
    """
    x: torch tensor of shape (N, C, H, W)
    severity: float in [0, 1]. Maps to sigma in [0, 1.125].
    """
    if severity <= 0:
        return x
    sigma = severity * 1.125
    if _is_model_padded_shape(x):
        interior = _crop_interior(x)
        interior_np = interior.detach().cpu().numpy()
        blurred_np = ndimage.gaussian_filter(interior_np, sigma=(0, 0, sigma, sigma))
        return _repad_like_dataset(torch.from_numpy(blurred_np).to(device=x.device, dtype=x.dtype))

    x_np = x.detach().cpu().numpy()
    x_blurred_np = ndimage.gaussian_filter(x_np, sigma=(0, 0, sigma, sigma))
    return torch.from_numpy(x_blurred_np).to(device=x.device, dtype=x.dtype)


def apply_high_freq_noise(x, severity):
    """
    x: torch tensor of shape (N, C, H, W)
    severity: float in [0, 1]. Maps to std in [0, 0.25].
    """
    if severity <= 0:
        return x
    std = severity * 0.25
    if _is_model_padded_shape(x):
        interior = _crop_interior(x)
        noise = torch.randn_like(interior) * std
        return _repad_like_dataset(interior + noise)

    noise = torch.randn_like(x) * std
    return x + noise


def apply_gaussian_field_noise(x, severity, len_scale=10.0):
    """
    x: torch tensor of shape (N, C, H, W)
    severity: float in [0, 1]. Maps to std in [0, 0.375].
    Adds spatially correlated Gaussian Random Field noise using FFT.
    The power spectrum is the Fourier transform of a Gaussian covariance
    C(r) = std^2 * exp(-r^2 / len_scale^2), which is also a Gaussian in
    frequency space.
    """
    if severity <= 0:
        return x

    std = severity * 0.375
    work_x = _crop_interior(x) if _is_model_padded_shape(x) else x
    N, C, H, W = work_x.shape

    # Frequency grids
    freq_y = np.fft.fftfreq(H).astype(np.float32)
    freq_x = np.fft.fftfreq(W).astype(np.float32)
    fy, fx = np.meshgrid(freq_y, freq_x, indexing="ij")

    # Power spectrum of Gaussian covariance: S(f) = pi * l^2 * exp(-pi^2 * l^2 * |f|^2)
    # We scale so that the spatial-domain variance equals std^2.
    freq_sq = fy**2 + fx**2
    power = np.exp(-2.0 * (np.pi * len_scale) ** 2 * freq_sq)

    # For ifft2 normalization (1/(H*W)): Var(Re(x)) = sum(A^2) / (H*W)^2
    # We want Var = std^2, so sum(A^2) = std^2 * (H*W)^2
    # With A[k] = c * sqrt(power[k]): c = std * H * W / sqrt(sum(power))
    amplitude = np.sqrt(power) * std * (H * W) / np.sqrt(power.sum())
    amplitude = torch.from_numpy(amplitude)  # (H, W)

    # Sample white noise in frequency domain, weight by amplitude, IFFT back
    # Real and imaginary parts are independent normals
    noise_freq = torch.randn(N, C, H, W) + 1j * torch.randn(N, C, H, W)
    noise_freq = noise_freq * amplitude
    noise = torch.fft.ifft2(noise_freq).real

    corrupted = work_x + noise.to(work_x.device)
    if _is_model_padded_shape(x):
        return _repad_like_dataset(corrupted)
    return corrupted


def apply_random_pixel_replace(x, severity, max_replace_prob=0.3):
    """
    x: torch tensor of shape (N, C, H, W)
    severity: float in [0, 1]. Maps to replacement probability in [0, 0.3].
    Replaces a random subset of interior pixels with Gaussian samples.
    """
    if severity <= 0:
        return x

    replace_prob = severity * max_replace_prob
    work_x = _crop_interior(x) if _is_model_padded_shape(x) else x

    mask = torch.rand_like(work_x) < replace_prob
    replacement = torch.randn_like(work_x)
    corrupted = torch.where(mask, replacement, work_x)

    if _is_model_padded_shape(x):
        return _repad_like_dataset(corrupted)
    return corrupted


def apply_wind_patch_shuffle(x, severity, patch_size=16):
    """
    Leaves temperature and pressure untouched while shuffling wind patches.
    severity: fraction of wind patches to permute, in [0, 1].
    """
    if severity <= 0:
        return x

    work_x = (_crop_interior(x) if _is_model_padded_shape(x) else x).clone()
    _, _, height, width = work_x.shape
    shuffled_height = (height // patch_size) * patch_size
    shuffled_width = (width // patch_size) * patch_size

    if shuffled_height == 0 or shuffled_width == 0:
        return x

    wind_region = work_x[:, U10_CHANNEL:V10_CHANNEL + 1, :shuffled_height, :shuffled_width]
    n_samples, _, _, _ = wind_region.shape
    n_patch_rows = shuffled_height // patch_size
    n_patch_cols = shuffled_width // patch_size
    n_patches = n_patch_rows * n_patch_cols
    n_shuffle = min(int(round(severity * n_patches)), n_patches)

    if n_shuffle < 2:
        return x

    wind_patches = wind_region.reshape(
        n_samples,
        2,
        n_patch_rows,
        patch_size,
        n_patch_cols,
        patch_size,
    )
    wind_patches = wind_patches.permute(0, 2, 4, 1, 3, 5).reshape(n_samples, n_patches, 2, patch_size, patch_size)
    shuffled_patches = wind_patches.clone()

    for sample_idx in range(n_samples):
        selected = torch.randperm(n_patches, device=work_x.device)[:n_shuffle]
        permuted = selected[torch.randperm(n_shuffle, device=work_x.device)]
        shuffled_patches[sample_idx, selected] = wind_patches[sample_idx, permuted]

    shuffled_region = shuffled_patches.reshape(
        n_samples,
        n_patch_rows,
        n_patch_cols,
        2,
        patch_size,
        patch_size,
    )
    shuffled_region = shuffled_region.permute(0, 3, 1, 4, 2, 5).reshape(
        n_samples,
        2,
        shuffled_height,
        shuffled_width,
    )
    work_x[:, U10_CHANNEL:V10_CHANNEL + 1, :shuffled_height, :shuffled_width] = shuffled_region
    return _finalize_like_input(x, work_x)


def apply_spectral_lowpass(x, severity, min_cutoff_ratio=0.1):
    """Attenuate high spatial wavenumbers — the *forecast-direction* corruption.

    ML weather models systematically lose small-scale (high-wavenumber) power
    relative to ERA5. This corruption mimics that by multiplying each channel's
    2-D spatial spectrum by a smooth Gaussian low-pass whose cutoff shrinks as
    ``severity`` grows.

    x: torch tensor of shape (N, C, H, W).
    severity: float in [0, 1]. 0 = identity; larger removes more high-freq power.
    min_cutoff_ratio: cutoff at severity=1, as a fraction of the Nyquist freq.
    """
    if severity <= 0:
        return x

    work_x = _crop_interior(x) if _is_model_padded_shape(x) else x
    _, _, H, W = work_x.shape

    # Radial wavenumber grid (cycles/pixel); Nyquist = 0.5.
    fy = torch.fft.fftfreq(H, device=work_x.device)
    fx = torch.fft.fftfreq(W, device=work_x.device)
    kr = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    # Cutoff decreases geometrically from Nyquist (mild) to nyq*min_ratio (strong).
    nyq = 0.5
    cutoff = nyq * (min_cutoff_ratio ** float(severity))
    filt = torch.exp(-0.5 * (kr / cutoff) ** 2).to(work_x.dtype)

    spec = torch.fft.fft2(work_x)
    corrupted = torch.fft.ifft2(spec * filt).real.to(work_x.dtype)

    if _is_model_padded_shape(x):
        return _repad_like_dataset(corrupted)
    return corrupted


def apply_wind_channel_rotation(x, severity):
    """
    Leaves temperature and pressure untouched while rotating every wind vector.
    severity: rotation angle in [0, 90] degrees.
    """
    if severity <= 0:
        return x

    work_x = (_crop_interior(x) if _is_model_padded_shape(x) else x).clone()
    angle = severity * (np.pi / 2.0)
    cos_theta = float(np.cos(angle))
    sin_theta = float(np.sin(angle))

    u = work_x[:, U10_CHANNEL].clone()
    v = work_x[:, V10_CHANNEL].clone()
    work_x[:, U10_CHANNEL] = cos_theta * u - sin_theta * v
    work_x[:, V10_CHANNEL] = sin_theta * u + cos_theta * v
    return _finalize_like_input(x, work_x)


# ---------------------------------------------------------------------------
# Contrastive realism training: a *subtle* corruption suite (~10x weaker than
# the eval ladder) sized to the real↔forecast gap rather than gross corruption.
# Spans BOTH directions so the embedding learns "realism = the right amount of
# structure" (a band): under-resolved (lowpass, blur) AND over-resolved (noise,
# grf, pixel_replace), plus a mild wind rotation. wind_patch_shuffle is excluded.
# Each entry is (fn, max_severity); the lowpass cap is retuned from the PSD
# diagnostic once the forecast spectral deficit is measured.
# ---------------------------------------------------------------------------
CONTRASTIVE_CORRUPTION_SPECS = {
    "lowpass":       (apply_spectral_lowpass,      0.50),
    "blur":          (apply_gaussian_blur,         0.20),
    "noise":         (apply_high_freq_noise,       0.20),
    "grf":           (apply_gaussian_field_noise,  0.20),
    "pixel_replace": (apply_random_pixel_replace,  0.20),
    "wind_rotation": (apply_wind_channel_rotation, 0.333),  # ≈ 30° max rotation
}


def sample_contrastive_corruption(x, generator=None, types=None):
    """Apply one randomly chosen subtle corruption to the whole batch ``x``.

    Picks a corruption type (uniformly from ``types`` or all
    ``CONTRASTIVE_CORRUPTION_SPECS`` keys) and a severity ``U(0, max_severity)``.
    Returns ``(x_corrupted, strength, name)`` where ``strength = severity /
    max_severity`` is the normalized target in [0, 1] (comparable across types).
    One type+severity per call; variety accumulates across training batches.
    """
    names = list(CONTRASTIVE_CORRUPTION_SPECS.keys()) if types is None else list(types)
    name = names[int(torch.randint(len(names), (1,), generator=generator).item())]
    fn, max_sev = CONTRASTIVE_CORRUPTION_SPECS[name]
    severity = float(torch.rand(1, generator=generator).item()) * max_sev
    x_corrupted = fn(x, severity)
    strength = severity / max_sev if max_sev > 0 else 0.0
    return x_corrupted, strength, name


if __name__ == "__main__":
    dummy_input = torch.randn(2, 4, 128, 256)
    blurred = apply_gaussian_blur(dummy_input, severity=0.5)
    noised = apply_high_freq_noise(dummy_input, severity=0.5)
    grf_noised = apply_gaussian_field_noise(dummy_input, severity=0.5)
    pixel_replaced = apply_random_pixel_replace(dummy_input, severity=0.5)
    wind_shuffled = apply_wind_patch_shuffle(dummy_input, severity=0.5)
    wind_rotated = apply_wind_channel_rotation(dummy_input, severity=0.5)
    lowpassed = apply_spectral_lowpass(dummy_input, severity=0.5)
    corr, strength, name = sample_contrastive_corruption(dummy_input)

    print(f"Clean std: {dummy_input.std():.3f}")
    print(f"Low-pass std: {lowpassed.std():.3f}")
    print(f"Sampled contrastive corruption: {name} (strength={strength:.3f}), std: {corr.std():.3f}")
    print(f"Blurred std: {blurred.std():.3f}")
    print(f"High-freq noised std: {noised.std():.3f}")
    print(f"GRF noised std: {grf_noised.std():.3f}")
    print(f"Pixel-replaced std: {pixel_replaced.std():.3f}")
    print(f"Wind-shuffled std: {wind_shuffled.std():.3f}")
    print(f"Wind-rotated std: {wind_rotated.std():.3f}")

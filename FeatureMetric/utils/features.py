import torch


def extract_features_for_loader(model, loader, device, transform_fn=None):
    """Returns (N, D) CPU tensor of mean-pooled patch features.

    Args:
        transform_fn: optional callable applied to each batch before forwarding,
                      e.g. ``lambda img: apply_corruption(img, severity)``.
    """
    feats = []
    with torch.no_grad():
        for img in loader:
            img = img.to(device, non_blocking=device.type == "cuda")
            if transform_fn is not None:
                img = transform_fn(img)
            feats.append(model.extract_features(img).cpu())
    return torch.cat(feats, dim=0)

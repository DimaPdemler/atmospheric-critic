"""Bind discriminator checkpoints to histogram-matching preprocessing artifacts."""

import json
from pathlib import Path

try:
    from .histogram_matching import artifact_dir, enabled
except ImportError:
    from histogram_matching import artifact_dir, enabled


def _manifest_hash(cfg):
    if not enabled(cfg):
        return None
    path = artifact_dir(cfg) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Histogram matching manifest not found: {path}")
    return json.loads(path.read_text()).get("sha256")


def binding_path(checkpoint):
    return Path(checkpoint).with_suffix(".histogram_matching.json")


def write_binding(checkpoint, cfg):
    payload = {"enabled": enabled(cfg), "manifest_sha256": _manifest_hash(cfg)}
    path = binding_path(checkpoint)
    path.write_text(json.dumps(payload, indent=2))
    return path


def validate_binding(checkpoint, cfg):
    path = binding_path(checkpoint)
    if not path.is_file():
        if enabled(cfg):
            raise FileNotFoundError(
                f"Histogram-matched evaluation requires checkpoint binding metadata: {path}"
            )
        return
    payload = json.loads(path.read_text())
    expected = _manifest_hash(cfg)
    if bool(payload.get("enabled")) != enabled(cfg) or payload.get("manifest_sha256") != expected:
        raise ValueError(
            f"Checkpoint {checkpoint} was trained with a different histogram-matching artifact."
        )

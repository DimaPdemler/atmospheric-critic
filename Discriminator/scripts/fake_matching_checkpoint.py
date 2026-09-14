"""Bind discriminator checkpoints to fake-distribution preprocessing artifacts."""

import json
from pathlib import Path

try:
    from .fake_matching_apply import matching_mode
    from .histogram_matching import artifact_dir as histogram_artifact_dir
    from .moment_matching import artifact_dir as moment_artifact_dir
except ImportError:
    from fake_matching_apply import matching_mode
    from histogram_matching import artifact_dir as histogram_artifact_dir
    from moment_matching import artifact_dir as moment_artifact_dir


def _manifest_hash(cfg):
    mode = matching_mode(cfg)
    if mode == "none":
        return None
    directory = histogram_artifact_dir(cfg) if mode == "histogram" else moment_artifact_dir(cfg)
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{mode.title()} matching manifest not found: {path}")
    return json.loads(path.read_text()).get("sha256")


def binding_path(checkpoint):
    # Retain the established suffix so existing raw/histogram checkpoints remain readable.
    return Path(checkpoint).with_suffix(".histogram_matching.json")


def write_binding(checkpoint, cfg):
    mode = matching_mode(cfg)
    payload = {
        "enabled": mode != "none", "mode": mode,
        "manifest_sha256": _manifest_hash(cfg),
    }
    path = binding_path(checkpoint)
    path.write_text(json.dumps(payload, indent=2))
    return path


def validate_binding(checkpoint, cfg):
    mode = matching_mode(cfg)
    path = binding_path(checkpoint)
    if not path.is_file():
        if mode != "none":
            raise FileNotFoundError(
                f"{mode.title()}-matched evaluation requires checkpoint binding metadata: {path}"
            )
        return
    payload = json.loads(path.read_text())
    recorded_mode = payload.get("mode")
    if recorded_mode is None:
        recorded_mode = "histogram" if bool(payload.get("enabled")) else "none"
    if recorded_mode != mode or payload.get("manifest_sha256") != _manifest_hash(cfg):
        raise ValueError(
            f"Checkpoint {checkpoint} was trained with different fake-matching preprocessing."
        )

#!/usr/bin/env python3
"""Inventory, migrate, and safely prune baseline-pipeline storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path


def tree_bytes(root):
    total, seen = 0, set()
    for path in Path(root).rglob("*"):
        try:
            if path.is_file():
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity not in seen:
                    seen.add(identity); total += stat.st_size
        except OSError:
            continue
    return total


def suffix_totals(root):
    totals = defaultdict(lambda: {"bytes": 0, "files": 0})
    seen = set()
    for path in Path(root).rglob("*"):
        try:
            if path.is_file():
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                key = path.suffix.lower() or "[none]"
                totals[key]["bytes"] += stat.st_size
                totals[key]["files"] += 1
        except OSError:
            continue
    return dict(sorted(totals.items(), key=lambda item: item[1]["bytes"], reverse=True))


def run_status(path):
    manifest = Path(path) / "manifest.json"
    if not manifest.is_file():
        return "missing-manifest"
    try:
        return str(json.loads(manifest.read_text()).get("status", "unknown"))
    except (OSError, json.JSONDecodeError):
        return "invalid-manifest"


def pipeline_runs(home):
    found = []
    for parent in Path(home).rglob("pipeline_runs"):
        if not parent.is_dir():
            continue
        for candidate in parent.iterdir():
            if candidate.is_dir():
                found.append(candidate)
    return sorted(set(found))


def inventory(home):
    home = Path(home).resolve()
    runs = [
        {"path": str(path), "bytes": tree_bytes(path), "status": run_status(path)}
        for path in pipeline_runs(home)
    ]
    wandb_roots = [
        home / ".cache" / "wandb",
        home / ".local" / "share" / "wandb",
    ]
    wandb_roots.extend(
        path for path in home.rglob("wandb")
        if path.is_dir() and not {"site-packages", "dist-packages"}.intersection(path.parts)
    )
    unique_wandb = []
    for path in sorted(set(wandb_roots)):
        if any(parent in path.parents for parent in unique_wandb):
            continue
        if path.exists():
            unique_wandb.append(path)
    return {
        "home": str(home),
        "home_bytes": tree_bytes(home),
        "pipeline_runs": sorted(runs, key=lambda item: item["bytes"], reverse=True),
        "wandb_roots": [
            {"path": str(path), "bytes": tree_bytes(path), "suffixes": suffix_totals(path)}
            for path in unique_wandb
        ],
    }


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(root):
    root = Path(root)
    return {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def migrate(source, destination_parent, *, apply=False, delete_source=False):
    source = Path(source).resolve()
    destination_parent = Path(destination_parent).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"No such run directory: {source}")
    destination = destination_parent / source.name
    partial = destination_parent / f".{source.name}.partial"
    result = {
        "source": str(source), "destination": str(destination),
        "bytes": tree_bytes(source), "apply": bool(apply),
        "delete_source": bool(delete_source),
    }
    if not apply:
        return result
    if destination.exists() or partial.exists():
        raise FileExistsError(f"Migration target already exists: {destination} or {partial}")
    destination_parent.mkdir(parents=True, exist_ok=True)
    copied_inodes = {}

    def copy_preserving_hardlinks(source_file, destination_file):
        stat = os.stat(source_file, follow_symlinks=False)
        identity = (stat.st_dev, stat.st_ino)
        prior = copied_inodes.get(identity)
        if prior is not None:
            os.link(prior, destination_file)
            return destination_file
        result = shutil.copy2(source_file, destination_file)
        copied_inodes[identity] = destination_file
        return result

    shutil.copytree(source, partial, symlinks=True, copy_function=copy_preserving_hardlinks)
    source_manifest = file_manifest(source)
    copied_manifest = file_manifest(partial)
    if source_manifest != copied_manifest:
        shutil.rmtree(partial)
        raise RuntimeError("Migration verification failed; partial copy was removed.")
    partial.replace(destination)
    if delete_source:
        shutil.rmtree(source)
    result["verified_files"] = len(source_manifest)
    return result


def _wandb_process_active():
    result = subprocess.run(
        ["pgrep", "-u", str(os.getuid()), "-f", "[w]andb"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    return result.returncode == 0


def clean_wandb(home, *, apply=False, cache_target="0B", staging_age_days=1, force=False,
                legacy_roots=()):
    home = Path(home).resolve()
    legacy = []
    for configured in legacy_roots:
        path = Path(configured).resolve()
        if (path.name != "wandb" or home not in path.parents
                or {"site-packages", "dist-packages"}.intersection(path.parts)):
            raise ValueError(f"Refusing unsafe legacy W&B root: {path}")
        if path in {home / ".cache" / "wandb", home / ".local" / "share" / "wandb"}:
            raise ValueError(f"Use the built-in cache/staging cleanup for {path}")
        legacy.append(path)
    cache = home / ".cache" / "wandb"
    staging = home / ".local" / "share" / "wandb" / "artifacts" / "staging"
    threshold = time.time() - float(staging_age_days) * 86400
    stale = [
        path for path in staging.glob("*")
        if path.is_file() and path.stat().st_mtime < threshold
    ] if staging.is_dir() else []
    result = {
        "cache": str(cache), "cache_bytes": tree_bytes(cache) if cache.exists() else 0,
        "staging": str(staging), "staging_bytes": sum(path.stat().st_size for path in stale),
        "stale_staging_files": len(stale), "apply": bool(apply),
        "legacy_roots": [
            {"path": str(path), "bytes": tree_bytes(path) if path.exists() else 0}
            for path in legacy
        ],
    }
    if not apply:
        return result
    if _wandb_process_active() and not force:
        raise RuntimeError("A W&B process is active; refusing cleanup without --force.")
    executable = shutil.which("wandb")
    if executable and cache.exists():
        subprocess.run(
            [executable, "artifact", "cache", "cleanup", "--remove-temp", cache_target],
            check=True,
        )
    for path in stale:
        path.unlink()
    for path in legacy:
        if path.exists():
            shutil.rmtree(path)
    return result


def print_report(payload):
    print(json.dumps(payload, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inventory")
    inspect_parser.add_argument("--home", type=Path, default=Path.home())
    inspect_parser.add_argument("--output-json", type=Path)

    migrate_parser = subparsers.add_parser("migrate")
    migrate_parser.add_argument("source", type=Path)
    migrate_parser.add_argument("destination_parent", type=Path)
    migrate_parser.add_argument("--apply", action="store_true")
    migrate_parser.add_argument("--delete-source", action="store_true")

    clean_parser = subparsers.add_parser("clean-wandb")
    clean_parser.add_argument("--home", type=Path, default=Path.home())
    clean_parser.add_argument("--cache-target", default="0B")
    clean_parser.add_argument("--staging-age-days", type=float, default=1.0)
    clean_parser.add_argument("--apply", action="store_true")
    clean_parser.add_argument("--force", action="store_true")
    clean_parser.add_argument("--legacy-root", action="append", type=Path, default=[])

    arguments = parser.parse_args()
    if arguments.command == "inventory":
        payload = inventory(arguments.home)
        if arguments.output_json:
            arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
            arguments.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    elif arguments.command == "migrate":
        if arguments.delete_source and not arguments.apply:
            parser.error("--delete-source requires --apply")
        payload = migrate(
            arguments.source, arguments.destination_parent,
            apply=arguments.apply, delete_source=arguments.delete_source,
        )
    else:
        payload = clean_wandb(
            arguments.home, apply=arguments.apply, cache_target=arguments.cache_target,
            staging_age_days=arguments.staging_age_days, force=arguments.force,
            legacy_roots=arguments.legacy_root,
        )
    print_report(payload)


if __name__ == "__main__":
    main()

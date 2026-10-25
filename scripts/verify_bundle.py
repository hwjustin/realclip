#!/usr/bin/env python3
"""Verify raw data and optionally unpack authenticated, regular-file tar shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_target(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (not name or "\\" in name or relative.is_absolute() or ".." in relative.parts
            or any(":" in part for part in relative.parts)):
        raise ValueError(f"Unsafe relative path: {name!r}")
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes data root: {name!r}")
    for parent in (target, *target.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError(f"Symlink is not allowed: {name!r}")
    return target


def extract_shard(path: Path, output: Path) -> int:
    """Never execute tar links or overwrite differing existing data."""
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if not member.isfile():
                # Exported shards contain regular files only; reject links/devices.
                raise ValueError(f"Nonregular tar member: {member.name}")
            target = safe_target(output, member.name)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read tar member: {member.name}")
            with source:
                if target.exists():
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                    if not target.is_file() or target.stat().st_size != member.size or hash_file(target) != digest.hexdigest():
                        raise ValueError(f"Existing file differs; refusing overwrite: {member.name}")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + ".extracting")
                    # Exclusive creation also rejects stale or symlink temporary paths.
                    created = False
                    try:
                        with temporary.open("xb") as destination:
                            created = True
                            shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
                        if temporary.stat().st_size != member.size:
                            raise ValueError(f"Truncated tar member: {member.name}")
                        temporary.replace(target)
                    finally:
                        if created:
                            temporary.unlink(missing_ok=True)
            count += 1
    return count


def verify_manifest(data_root: Path) -> dict:
    manifest_path = safe_target(data_root, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest.get("manifest_id")
    if stable_hash({k: v for k, v in manifest.items() if k != "manifest_id"}) != identity:
        raise ValueError("Manifest identity mismatch")
    channels = ["DNA", "RNA", "ER", "Mito", "AGP"]
    if manifest.get("channels") != channels:
        raise ValueError("Unsupported channels")
    hashes = {}
    counts = {"train": 0, "val": 0, "test": 0}
    sample_ids = set()
    plate_splits, parent_splits = {}, {}
    for sample in manifest["samples"]:
        sid = sample["sample_id"]
        if sid in sample_ids:
            raise ValueError(f"Duplicate sample ID: {sid}")
        sample_ids.add(sid)
        split = sample["split"]
        if split not in counts:
            raise ValueError(f"Invalid split: {split}")
        counts[split] += 1
        description = sample["description"]
        if not isinstance(description, str) or not description.strip() or hashlib.sha256(description.encode()).hexdigest() != sample["description_sha256"]:
            raise ValueError(f"Description checksum mismatch: {sid}")
        plate = (sample.get("source"), sample.get("plate"))
        if all(value is not None for value in plate) and plate_splits.setdefault(plate, split) != split:
            raise ValueError(f"Plate leaks across splits: {plate}")
        parent = sample.get("parent_id")
        if parent is not None and parent_splits.setdefault(parent, split) != split:
            raise ValueError(f"Parent leaks across splits: {parent}")
        seen = set()
        for image in sample["images"]:
            key = (str(image["site"]), image["channel"])
            if key in seen:
                raise ValueError(f"Duplicate site/channel in {sid}: {key}")
            seen.add(key)
            target = safe_target(data_root, image["path"])
            if not target.is_file():
                raise ValueError(f"Missing image: {image['path']}")
            if str(target) not in hashes:
                hashes[str(target)] = hash_file(target)
            if hashes[str(target)] != image["sha256"]:
                raise ValueError(f"Image checksum mismatch: {image['path']}")
        sites = {site for site, _ in seen}
        if not sites or seen != {(site, channel) for site in sites for channel in channels}:
            raise ValueError(f"Incomplete five-channel site in {sid}")
    return {"manifest_id": identity, "wells": len(sample_ids), "images": len(hashes), "splits": counts}


def load_shard_index(bundle_dir: Path) -> tuple[dict, list[dict]]:
    path = bundle_dir / "upload_index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    records = index["files"]
    if index.get("complete") is not True or not isinstance(records, list) or not records:
        raise ValueError("Bundle export is incomplete or has no shards")
    return index, records


def verify_shards(bundle_dir: Path) -> list[Path]:
    _, records = load_shard_index(bundle_dir)
    paths = []
    for record in records:
        path = safe_target(bundle_dir, record["file_name"])
        if not path.is_file() or hash_file(path) != record["sha256"]:
            raise ValueError(f"Shard checksum mismatch: {record['file_name']}")
        if path.stat().st_size != record["size"]:
            raise ValueError(f"Shard size mismatch: {record['file_name']}")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    if args.extract and (not args.bundle_dir or not args.data_root):
        parser.error("--extract requires --bundle-dir and --data-root")
    if not args.bundle_dir and not args.data_root:
        parser.error("Provide --bundle-dir and/or --data-root")
    if args.bundle_dir:
        shards = verify_shards(args.bundle_dir)
        print(f"Verified {len(shards)} shard checksums", flush=True)
        if args.extract:
            for path in shards:
                count = extract_shard(path, args.data_root)
                print(f"Extracted {path.name}: {count} files", flush=True)
    if args.data_root:
        result = verify_manifest(args.data_root)
        if args.bundle_dir:
            index, _ = load_shard_index(args.bundle_dir)
            if index.get("manifest_id") != result["manifest_id"]:
                raise ValueError("Data manifest differs from the bundle index")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export only frozen well pairs as resumable, verified, portable tar shards.

No staging copy of the TIFFs is made. Extract metadata.tar.gz and every
images-NNNNN.tar.gz into one data directory; paths inside are relative.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from realclip_h200.data import CHANNELS, FORMAT, PREPROCESS, hash_file, stable_hash, validate_manifest


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_export(source_manifest: Path, windows_path: Path) -> tuple[dict, dict]:
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source.get("manifest_id") != stable_hash({k: v for k, v in source.items() if k != "manifest_id"}):
        raise ValueError("Source manifest identity mismatch")
    windows = json.loads(windows_path.read_text(encoding="utf-8"))
    if source.get("channels") != list(CHANNELS) or source.get("preprocessing") != PREPROCESS or windows.get("preprocessing") != PREPROCESS:
        raise ValueError("Unsupported source channel/preprocessing configuration")
    samples, files = [], {}
    for original in source["samples"]:
        sample = {key: original[key] for key in ("sample_id", "well_id", "source", "plate", "well", "parent_id",
                                                "description", "description_sha256", "split") if key in original}
        sample["windows"] = windows["wells"][sample["sample_id"]]
        sample["images"] = []
        for image in original["images"]:
            path = Path(image["path"])
            if not path.is_absolute():
                path = source_manifest.parent / path
            if not path.is_file():
                raise ValueError(f"Missing source TIFF: {path}")
            sha = image["sha256"]
            relative = f"images/{sha}.tif"
            item = {key: image[key] for key in ("image_id", "site", "channel", "sha256", "shape") if key in image}
            item["path"] = relative
            sample["images"].append(item)
            entry = {"source": str(path.resolve()), "path": relative, "sha256": sha, "size": path.stat().st_size}
            previous = files.setdefault(relative, entry)
            if previous["size"] != entry["size"]:
                raise ValueError("Source files claim the same SHA256 with different sizes")
        samples.append(sample)
    payload = {"schema_version": 1, "format": FORMAT, "channels": list(CHANNELS),
               "preprocessing": PREPROCESS, "samples": samples,
               "lineage": {"source_manifest_id": source["manifest_id"],
                           "source_manifest_sha256": hash_file(source_manifest),
                           "source_windows_sha256": hash_file(windows_path),
                           "policy": "Keep frozen pairs, descriptions, splits, original raw TIFF bytes, and well/channel windows; exclude upstream generation artifacts."}}
    manifest = {**payload, "manifest_id": stable_hash(payload)}
    validate_manifest(manifest)
    return manifest, dict(sorted(files.items()))


def partition_files(files: dict, shard_size: int) -> list[list[dict]]:
    if shard_size < 20 * 1024:
        raise ValueError("Shard size must be at least 20 KiB")
    shards, current, current_size = [], [], 10240
    for entry in files.values():
        cost = 512 + ((entry["size"] + 511) // 512) * 512
        if current and current_size + cost > shard_size:
            shards.append(current)
            current, current_size = [], 10240
        current.append(entry)
        current_size += cost
    if current:
        shards.append(current)
    return shards


class HashingReader:
    def __init__(self, stream):
        self.stream, self.digest, self.size = stream, hashlib.sha256(), 0

    def read(self, size=-1):
        value = self.stream.read(size)
        self.digest.update(value)
        self.size += len(value)
        return value


def add_bytes(archive, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(value), 0o644, 0
    archive.addfile(info, io.BytesIO(value))


def write_archive(path: Path, entries: list[dict] | None = None, documents: dict | None = None) -> dict:
    temporary = path.with_name(path.name + ".partial")
    # A prior interrupted .partial is owned by this exact export; completed files
    # are never overwritten and are reused only through verified receipts.
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                for name, value in (documents or {}).items():
                    add_bytes(archive, name, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())
                for entry in entries or []:
                    with Path(entry["source"]).open("rb") as stream:
                        reader = HashingReader(stream)
                        info = tarfile.TarInfo(entry["path"])
                        info.size, info.mode, info.mtime = entry["size"], 0o644, 0
                        archive.addfile(info, reader)
                        if reader.size != entry["size"] or reader.digest.hexdigest() != entry["sha256"] or stream.read(1):
                            raise ValueError(f"Source TIFF bytes changed: {entry['source']}")
        raw.flush()
        os.fsync(raw.fileno())
    if path.exists():
        raise ValueError(f"Refusing to overwrite existing archive: {path}")
    temporary.replace(path)
    return {"file_name": path.name, "size": path.stat().st_size, "sha256": hash_file(path)}


def compression_probe(entries: list[dict], count: int) -> dict:
    count = min(count, len(entries))
    if count <= 0:
        return {}
    selected = [entries[index * len(entries) // count] for index in range(count)]
    raw_bytes, compressed_bytes = 0, 0
    for entry in selected:
        compressor = zlib.compressobj(1, zlib.DEFLATED, 31)
        with Path(entry["source"]).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                raw_bytes += len(block)
                compressed_bytes += len(compressor.compress(block))
        compressed_bytes += len(compressor.flush())
    ratio = compressed_bytes / raw_bytes if raw_bytes else 1.0
    return {"sample_images": count, "raw_bytes": raw_bytes, "gzip1_bytes": compressed_bytes,
            "gzip1_ratio": ratio, "estimated_total_gzip1_bytes": round(sum(x["size"] for x in entries) * ratio),
            "note": "Evenly spaced image sample; final compressed size varies by shard."}


def export(source_manifest: Path, windows_path: Path, output: Path, shard_size: int,
           plan_only: bool = False, probe_images: int = 0) -> dict:
    manifest, files = build_export(source_manifest.resolve(), windows_path.resolve())
    groups = partition_files(files, shard_size)
    plan = {"manifest_id": manifest["manifest_id"], "source_manifest_id": manifest["lineage"]["source_manifest_id"],
            "well_count": len(manifest["samples"]), "split_counts": dict(Counter(s["split"] for s in manifest["samples"])),
            "image_references": sum(len(s["images"]) for s in manifest["samples"]), "unique_images": len(files),
            "raw_image_bytes": sum(x["size"] for x in files.values()), "image_shards": len(groups),
            "shard_uncompressed_target_bytes": shard_size, "compression": "gzip level 1",
            "source_bytes_verified_during_export": True,
            "compression_probe": compression_probe(list(files.values()), probe_images)}
    print(json.dumps(plan, indent=2), flush=True)
    if plan_only:
        return plan
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, index_path = output / "manifest.json", output / "upload_index.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Output directory belongs to a different dataset; choose a new directory")
    else:
        write_json(manifest_path, manifest)
    write_json(output / "lineage.json", manifest["lineage"])
    identity = stable_hash({"manifest_id": manifest["manifest_id"], "shards": [[e["path"] for e in g] for g in groups],
                            "compression": "gzip1"})
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index.get("export_id") != identity:
            raise ValueError("Existing upload index has a different shard plan")
    else:
        index = {"schema_version": 1, "export_id": identity, "manifest_id": manifest["manifest_id"],
                 "complete": False, "plan": plan, "files": [],
                 "restore": "Extract metadata.tar.gz and every images-NNNNN.tar.gz into the same directory, then run python -m realclip_h200.data --data-root DATA --verify-images."}
    expected = [("metadata.tar.gz", None)] + [(f"images-{number:05d}.tar.gz", group) for number, group in enumerate(groups, 1)]
    receipts = {entry["file_name"]: entry for entry in index["files"]}
    for name, group in expected:
        target = output / name
        if name in receipts:
            receipt = receipts[name]
            if not target.is_file() or target.stat().st_size != receipt["size"] or hash_file(target) != receipt["sha256"]:
                raise ValueError(f"Completed archive changed: {target}")
            print(f"Reused verified {name}", flush=True)
            continue
        if target.exists():
            raise ValueError(f"Archive exists without a completion receipt; move it aside before resuming: {target}")
        documents = {"manifest.json": manifest, "lineage.json": manifest["lineage"]} if group is None else None
        receipt = write_archive(target, group, documents)
        receipt["image_count"] = len(group) if group else 0
        index["files"].append(receipt)
        write_json(index_path, index)
        print(f"Completed {name}: {receipt['size']} bytes, SHA256 {receipt['sha256']}", flush=True)
    index["complete"] = True
    write_json(index_path, index)
    sums = "".join(f"{entry['sha256']}  {entry['file_name']}\n" for entry in index["files"])
    (output / "SHA256SUMS").write_text(sums, encoding="ascii")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-size-gib", type=float, default=1)
    parser.add_argument("--plan-only", action="store_true", help="Inspect sources without creating output files")
    parser.add_argument("--probe-images", type=int, default=0, help="Estimate gzip1 compression from a spaced sample, without writing image copies")
    args = parser.parse_args()
    if not (args.shard_size_gib > 0 and args.shard_size_gib < float("inf")):
        parser.error("--shard-size-gib must be finite and positive")
    export(args.source_manifest, args.windows, args.output, int(args.shard_size_gib * 1024 ** 3), args.plan_only, args.probe_images)


if __name__ == "__main__":
    main()

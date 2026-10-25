#!/usr/bin/env python3
"""Create portable, reproducible public raw-image download manifests.

This is a source-machine export tool. The server downloader does not need the
upstream Observation files or any source-machine paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlsplit


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("manifest_id") != stable_hash({k: v for k, v in value.items() if k != "manifest_id"}):
        raise ValueError(f"Manifest identity mismatch: {path.name}")
    return value


def canonical_json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def validate_url(url: str) -> str:
    if not isinstance(url, str):
        raise ValueError("Missing public raw image URL")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "cellpainting-gallery.s3.amazonaws.com"
            or not parsed.path.startswith("/cpg0016-jump/") or parsed.query or parsed.fragment
            or ".." in parsed.path.split("/") or "\\" in parsed.path):
        raise ValueError("Raw URL must be an unsigned public JUMP object")
    return url


def resolved(value: str, directory: Path) -> Path:
    path = Path(value)
    return (directory / path).resolve() if not path.is_absolute() else path.resolve()


def build_downloads(portable_path: Path, source_path: Path) -> tuple[dict, list[dict]]:
    portable, source = read_manifest(portable_path), read_manifest(source_path)
    lineage = portable.get("lineage", {})
    if (lineage.get("source_manifest_id") != source["manifest_id"]
            or lineage.get("source_manifest_sha256") != file_hash(source_path)):
        raise ValueError("Portable manifest is not bound to this source manifest")
    originals = {row["sample_id"]: row for row in source["samples"]}
    portable_ids = [row["sample_id"] for row in portable["samples"]]
    if len(set(portable_ids)) != len(portable_ids) or set(portable_ids) != set(originals):
        raise ValueError("Portable/source sample sets differ")
    downloads = {}
    for sample in portable["samples"]:
        original = originals[sample["sample_id"]]
        for key in ("well_id", "source", "plate", "well", "parent_id", "description", "description_sha256", "split"):
            if sample.get(key) != original.get(key):
                raise ValueError(f"Portable sample changed original {key}")
        if hashlib.sha256(sample["description"].encode()).hexdigest() != sample["description_sha256"]:
            raise ValueError("Description SHA256 mismatch")
        provenance = original.get("provenance", {}).get("files", {})
        pairing_names = [name for name in provenance
                         if Path(name).name == "pairing.json"
                         or (Path(name).parent.name == "pairings"
                             and Path(name).stem == original["provenance"].get("compound_id"))]
        candidates = [resolved(name, source_path.parent) for name in pairing_names]
        if len(candidates) != 1:
            raise ValueError("Each source sample needs exactly one recorded pairing.json")
        pairing_path = candidates[0]
        expected = provenance[pairing_names[0]]
        if file_hash(pairing_path) != expected:
            raise ValueError("Source pairing.json checksum mismatch")
        pairing = json.loads(pairing_path.read_text(encoding="utf-8"))
        records = {}
        for record in pairing["images"]:
            if record.get("raw_path") is None:
                continue
            path = resolved(record["raw_path"], pairing_path.parent)
            row = {"url": record.get("source_url"), "sha256": record.get("raw_sha256"),
                   "size_bytes": record.get("raw_bytes"), "role": record.get("role"),
                   "well_id": record.get("well_id"), "shape": record.get("shape")}
            if record.get("url", row["url"]) != row["url"]:
                raise ValueError("Conflicting public URL in source pairing")
            if path in records and records[path] != row:
                raise ValueError("Conflicting URL or source bytes for one raw path")
            records[path] = row
        original_images = {row["image_id"]: row for row in original["images"]}
        if {row["image_id"] for row in sample["images"]} != set(original_images):
            raise ValueError("Portable/source image sets differ")
        for image in sample["images"]:
            old = original_images[image["image_id"]]
            path = resolved(old["path"], source_path.parent)
            if path not in records:
                raise ValueError(f"Missing public source URL for image {image['image_id']}")
            record = records[path]
            if (record["role"] != "treatment" or record["well_id"] != sample["well_id"]
                    or record["sha256"] != old["sha256"] or image["sha256"] != old["sha256"]
                    or record["shape"] != old["shape"] or image["shape"] != old["shape"]
                    or image["site"] != old["site"] or image["channel"] != old["channel"]):
                raise ValueError("Public raw image bytes or treatment identity differ")
            size = record["size_bytes"]
            if size is None:
                if file_hash(path) != old["sha256"]:
                    raise ValueError("Legacy raw image checksum mismatch")
                size = path.stat().st_size
            if type(size) is not int or size <= 0 or path.stat().st_size != size:
                raise ValueError("Raw image byte count mismatch")
            if image["path"] != f"images/{image['sha256']}.tif":
                raise ValueError("Portable image paths must be content-addressed")
            row = {"path": image["path"], "sha256": image["sha256"],
                   "size_bytes": size, "url": validate_url(record["url"])}
            if row["path"] in downloads and downloads[row["path"]] != row:
                raise ValueError("Conflicting URL for one portable image")
            downloads[row["path"]] = row
    return portable, [downloads[key] for key in sorted(downloads)]


def gzip_bytes(content: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=output, mode="wb", compresslevel=9, mtime=0) as compressed:
        compressed.write(content)
    return output.getvalue()


def save_bytes(path: Path, content: bytes) -> None:
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"Refusing to overwrite a different export: {path.name}")
    if not path.exists():
        temporary = path.with_name(path.name + ".partial")
        temporary.write_bytes(content)
        temporary.replace(path)


def prepare(portable_path: Path, source_path: Path, output: Path) -> dict:
    portable, downloads = build_downloads(portable_path.resolve(), source_path.resolve())
    content = {
        "dataset.json.gz": gzip_bytes(canonical_json(portable)),
        "raw_downloads.jsonl.gz": gzip_bytes(b"".join(canonical_json(row) for row in downloads)),
    }
    metadata = {
        "schema_version": 1, "manifest_id": portable["manifest_id"],
        "counts": {"wells": len(portable["samples"]), "images": len(downloads)},
        "split_counts": dict(sorted(Counter(row["split"] for row in portable["samples"]).items())),
        "raw_bytes": sum(row["size_bytes"] for row in downloads),
        "artifacts": {name: {"sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)}
                      for name, value in sorted(content.items())},
        "provenance": {"source_manifest_id": portable["lineage"]["source_manifest_id"],
                       "download_source": "Unsigned public Cell Painting Gallery S3 objects",
                       "source_registry": "https://registry.opendata.aws/cellpainting-gallery/",
                       "policy": "Preserve every frozen sample, description, split, intensity window, and raw TIFF byte; publish no source-machine paths."},
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in content.items():
        save_bytes(output / name, value)
    save_bytes(output / "metadata.json", canonical_json(metadata))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portable-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.portable_manifest, args.source_manifest, args.output), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export the current training cohort plus image-only archival prepared wells.

Source-machine tool only. No image generation, paid APIs or downloads occur.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_raw_manifest import (canonical_json, file_hash, gzip_bytes,
                                  prepare, read_manifest, save_bytes, stable_hash,
                                  validate_url)

CHANNELS = ("DNA", "RNA", "ER", "Mito", "AGP")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(value, parent):
    path = Path(value)
    return (path if path.is_absolute() else parent / path).resolve()


def archive_samples(prepared_path, inventory_path, training):
    prepared_path, inventory_path = Path(prepared_path), Path(inventory_path)
    prepared = read_json(prepared_path)
    seal_path = prepared_path.parent / "preparation_complete.json"
    if seal_path.exists() and read_json(seal_path).get("manifest_sha256") != file_hash(prepared_path):
        raise ValueError("Prepared universe manifest changed after its completion seal")
    rows = prepared["samples"]
    inventory_rows = [json.loads(line) for line in inventory_path.read_text().splitlines() if line.strip()]
    inventory = {row["compound_id"]: row for row in inventory_rows}
    if len(inventory) != len(inventory_rows) or len(rows) != len(inventory):
        raise ValueError("Prepared universe and inventory do not match")
    training_wells = {sample["well_id"]: sample for sample in training["samples"]}
    if len(training_wells) != len(training["samples"]):
        raise ValueError("Duplicate training well")
    known = {image["path"]: image["sha256"] for s in training["samples"] for image in s["images"]}
    seen_wells, seen_compounds, archived, input_hashes = set(), set(), [], []
    source_rows = []
    fallback_bytes = 0
    for row in rows:
        cid, well_id = row["compound_id"], row["dataset_sample_id"]
        if cid not in inventory or cid in seen_compounds or well_id in seen_wells:
            raise ValueError("Unknown or duplicate prepared compound/well")
        seen_compounds.add(cid)
        seen_wells.add(well_id)
        if well_id in training_wells:
            if inventory[cid]["category"] != "complete":
                raise ValueError("Training sample lacks a complete description")
            continue
        input_path = resolve(row["input_path"], prepared_path.parent)
        if file_hash(input_path) != row["input_sha256"]:
            raise ValueError(f"Prepared sample input checksum mismatch: {cid}")
        pairing_path = resolve(row["pairing_path"], prepared_path.parent)
        pairing = read_json(pairing_path)
        if pairing.get("compound_id") != cid or pairing.get("dataset_sample_id") != well_id:
            raise ValueError("Prepared pairing identity mismatch")
        input_hashes.append(file_hash(pairing_path))
        images, seen = [], set()
        for image in pairing["images"]:
            if image.get("role") != "treatment" or image.get("channel") not in CHANNELS:
                continue
            if image.get("well_id") != well_id:
                raise ValueError("Image belongs to another treatment well")
            url = validate_url(image.get("source_url", image.get("url")))
            if image.get("url", url) != url:
                raise ValueError("Conflicting raw URL")
            site, channel = str(image["field"]), image["channel"]
            if not site.isdigit() or (site, channel) in seen:
                raise ValueError("Invalid or duplicate raw site/channel")
            seen.add((site, channel))
            sha, size = image.get("raw_sha256"), image.get("raw_bytes")
            raw = resolve(image["raw_path"], pairing_path.parent)
            if not sha or not size:
                # Six legacy prepared inputs predate raw checksum fields. Hash
                # their original TIFF bytes rather than inventing an identity.
                sha, size = file_hash(raw), raw.stat().st_size
                fallback_bytes += size
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha) or type(size) is not int or size <= 0:
                raise ValueError("Invalid raw content identity")
            relative = f"images/{sha}.tif"
            if relative in known:
                raise ValueError("Archival image overlaps a training image")
            images.append({"path": relative, "sha256": sha, "size_bytes": size,
                           "url": url, "site": site, "channel": channel})
        sites = {site for site, _ in seen}
        if not sites or seen != {(site, channel) for site in sites for channel in CHANNELS}:
            raise ValueError("Prepared well needs all five channels at every site")
        record = inventory[cid]
        complete = record["category"] == "complete"
        text = None
        if complete:
            paths = record.get("result_paths", [])
            if not paths:
                raise ValueError("Complete inventory row has no sealed description")
            result_path = Path(paths[0])
            result = read_json(result_path)
            receipt = read_json(result_path.parent / "complete.json")
            if receipt["files"].get("result.json") != file_hash(result_path):
                raise ValueError("Archival description result checksum mismatch")
            text = result["description"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Missing archival description")
        archived.append({"sample_id": row["sample_id"], "compound_id": cid,
                         "well_id": well_id, "source": row["source"], "plate": row["plate"],
                         "well": row["treatment_well"], "training_eligible": False,
                         "description_status": "complete" if complete else "incomplete",
                         "description": text,
                         "description_sha256": hashlib.sha256(text.encode()).hexdigest() if text else None,
                         "reason": "excluded_from_frozen_training_cohort" if complete else "no_complete_description",
                         "images": sorted(images, key=lambda value: (int(value["site"]), CHANNELS.index(value["channel"])))})
        source_rows.extend(images)
    if not set(training_wells).issubset(seen_wells):
        raise ValueError("Training well absent from the prepared universe")
    unique = {}
    for row in source_rows:
        compact = {k: row[k] for k in ("path", "sha256", "size_bytes", "url")}
        if row["path"] in unique and unique[row["path"]] != compact:
            raise ValueError("Conflicting archival raw identity")
        unique[row["path"]] = compact
    payload = {"schema_version": 1, "format": "realclip-prepared-archive-v1",
               "training_manifest_id": training["manifest_id"],
               "total_prepared_wells": len(rows), "samples": sorted(archived, key=lambda x: x["well_id"]),
               "images": [unique[key] for key in sorted(unique)],
               "provenance": {"prepared_manifest_sha256": file_hash(prepared_path),
                              "inventory_sha256": file_hash(inventory_path),
                              "pairing_hashes_sha256": stable_hash(sorted(input_hashes)),
                              "legacy_raw_bytes_hashed": fallback_bytes,
                              "scope": "Treatment-only, every site, DNA/RNA/ER/Mito/AGP. Archive records never enter training."}}
    return {**payload, "archive_id": stable_hash(payload)}


def export_all(source, windows, prepared, inventory, previous_manifest, output):
    # The existing well exporter keeps original descriptions and preprocessing
    # while stripping machine-specific paths.
    spec = importlib.util.spec_from_file_location("well_export", Path(__file__).with_name("export_data.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    portable, _ = module.build_export(Path(source), Path(windows))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    portable_path = output / "portable.source.json"
    save_bytes(portable_path, canonical_json(portable))
    staging = output / ".training"
    metadata = prepare(portable_path, Path(source), staging)
    for name in ("dataset.json.gz", "raw_downloads.jsonl.gz"):
        save_bytes(output / name, (staging / name).read_bytes())
    archive = archive_samples(Path(prepared), Path(inventory), portable)
    value = gzip_bytes(canonical_json(archive))
    save_bytes(output / "prepared_archive.json.gz", value)
    previous = read_manifest(Path(previous_manifest))
    new_samples = {s["sample_id"]: s for s in portable["samples"]}
    if any(new_samples.get(s["sample_id"]) != s for s in previous["samples"]):
        raise ValueError("Expansion changed or removed a previous portable sample")
    metadata["compatible_previous_manifest_ids"] = [previous["manifest_id"]]
    metadata["artifacts"]["prepared_archive.json.gz"] = {
        "sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)}
    descriptions = Counter(s["description_status"] for s in archive["samples"])
    archive_bytes = sum(row["size_bytes"] for row in archive["images"])
    metadata["archive"] = {"archive_id": archive["archive_id"], "wells": len(archive["samples"]),
                           "images": len(archive["images"]), "raw_bytes": archive_bytes,
                           "complete_descriptions": descriptions["complete"],
                           "incomplete_descriptions": descriptions["incomplete"]}
    metadata["all_prepared"] = {"wells": archive["total_prepared_wells"],
                                "images": metadata["counts"]["images"] + len(archive["images"]),
                                "raw_bytes": metadata["raw_bytes"] + archive_bytes}
    save_bytes(output / "metadata.json", canonical_json(metadata))
    return metadata


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--previous-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_all(args.source_manifest, args.windows, args.prepared_manifest,
                                args.inventory, args.previous_manifest, args.output), indent=2))


if __name__ == "__main__":
    main()

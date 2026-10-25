#!/usr/bin/env python3
"""Stage portable TIFF pairs as hardlinks for direct rsync/SFTP transfer.

This creates directory entries rather than a second copy of image bytes. Do not
edit images in the staging directory: hardlinks share the original file bytes.
No network connection or transfer is made by this script.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from realclip_h200.data import hash_file, image_path, load_manifest
from export_data import build_export, write_json


def _preserve_json(path: Path, value) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink at metadata path: {path}")
    if path.exists():
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Refusing to overwrite different metadata: {path}")
    else:
        write_json(path, value)


def stage(source_manifest: Path, windows: Path, output: Path,
          plan_only: bool = False, verify_source: bool = False) -> dict:
    manifest, files = build_export(source_manifest.resolve(), windows.resolve())
    output = output.resolve()
    plan = {"manifest_id": manifest["manifest_id"], "well_count": len(manifest["samples"]),
            "split_counts": dict(Counter(s["split"] for s in manifest["samples"])),
            "unique_images": len(files), "image_bytes": sum(entry["size"] for entry in files.values()),
            "staging_method": "hardlink", "additional_image_bytes": 0,
            "source_sha256_verification_requested": verify_source, "network_transfer_performed": False}
    if plan_only:
        print(json.dumps(plan, indent=2), flush=True)
        return plan
    output.mkdir(parents=True, exist_ok=True)
    if (output / "images").is_symlink():
        raise ValueError("Refusing an images directory symlink")
    (output / "images").mkdir(exist_ok=True)
    _preserve_json(output / "manifest.json", manifest)
    _preserve_json(output / "lineage.json", manifest["lineage"])
    created, reused = 0, 0
    for number, entry in enumerate(files.values(), 1):
        source = Path(entry["source"])
        if (output / entry["path"]).is_symlink():
            raise ValueError(f"Refusing a staged-image symlink: {output / entry['path']}")
        target = image_path(output, entry["path"])
        source_stat = source.stat()
        if source_stat.st_size != entry["size"]:
            raise ValueError(f"Source TIFF size changed: {source}")
        if verify_source and hash_file(source) != entry["sha256"]:
            raise ValueError(f"Source TIFF hash mismatch: {source}")
        # Existing staged bytes are preserved; a symlink is never accepted.
        if target.is_symlink():
            raise ValueError(f"Refusing a staged-image symlink: {target}")
        if target.exists():
            target_stat = target.stat()
            same_inode = source_stat.st_dev == target_stat.st_dev and source_stat.st_ino == target_stat.st_ino
            if not target.is_file() or target_stat.st_size != entry["size"] or (
                    not same_inode and hash_file(target) != entry["sha256"]):
                raise ValueError(f"Refusing to overwrite different staged bytes: {target}")
            reused += 1
        else:
            try:
                os.link(source, target)
            except OSError as exc:
                if exc.errno == 18:
                    raise ValueError("Hardlink staging must be on the same filesystem as the source TIFFs; choose an output there") from exc
                raise
            created += 1
        if number % 1000 == 0:
            print(f"Staged {number}/{len(files)} TIFF hardlinks", flush=True)
    load_manifest(output)
    result = {**plan, "complete": True, "new_hardlinks": created, "reused_files": reused,
              "source_sha256_verified": verify_source,
              "restore_check": "python -m realclip_h200.data --data-root DATA --verify-images"}
    write_json(output / "stage_complete.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verify-source", action="store_true", help="Read all source TIFF bytes and verify SHA256 before staging")
    args = parser.parse_args()
    stage(args.source_manifest, args.windows, args.output, args.plan_only, args.verify_source)


if __name__ == "__main__":
    main()

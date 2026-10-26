"""Freeze new-plate raw-image MoA evaluation wells without encoding or captions.

Every selected parent and plate is absent from all 438 original CLIP wells.
The original MoA class sets remain fixed; this prepares image-only inference
inputs plus separate annotations, never new caption supervision.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wellclip.common import CHANNELS, PREPROCESS, hash_file, read_json, stable_hash, utc_now, write_json
from prepare_moa_probe import raw_labels, unique_lookup, well_key


def run(root: Path, output: Path) -> dict:
    import pandas as pd

    main_path = root / "CLIP/artifacts/moa_probe_v1/cohort.json"
    original = read_json(main_path)
    processed_path = root / "data/processed/dataset.parquet"
    if hash_file(processed_path) != original["inputs"]["data/processed/dataset.parquet"]:
        raise ValueError("Processed metadata changed since the primary cohort was frozen")
    raw_path = root / "data/raw/moa_labels.csv"
    if hash_file(raw_path) != original["inputs"]["data/raw/moa_labels.csv"]:
        raise ValueError("Raw annotations changed since the primary cohort was frozen")
    metadata_lookup = unique_lookup(pd.read_parquet(processed_path).to_dict("records"), ("source", "plate", "well"))
    label_lookup = unique_lookup(pd.read_csv(raw_path, dtype=str).to_dict("records"),
                                 ("Metadata_Source", "Metadata_Plate", "Metadata_Well"))
    prepared_path = root / "D/runs/observation_prepared_1806_20261023/manifest.json"
    prepared = read_json(prepared_path)
    old_parents = {row["parent_id"] for row in original["samples"]}
    old_plates = {(row["source"], row["plate"]) for row in original["samples"]}
    candidates, exclusions = [], Counter()
    for row in prepared["samples"]:
        key = well_key(row["source"], row["plate"], row["treatment_well"])
        metadata = metadata_lookup[key]
        if metadata["compound_group"] in old_parents:
            exclusions["original_parent"] += 1
            continue
        if key[:2] in old_plates:
            exclusions["original_train_val_or_test_plate"] += 1
            continue
        pairing_path = Path(row["pairing_path"])
        pairing = read_json(pairing_path)
        images = [im for im in (pairing.get("images") or []) if im["role"] == "treatment" and im["channel"] in CHANNELS]
        sites = {str(site) for site in (pairing.get("treatment_sites") or [])}
        image_keys = {(str(im["field"]), im["channel"]) for im in images}
        if not sites or image_keys != {(site, channel) for site in sites for channel in CHANNELS} or len(image_keys) != len(images):
            exclusions["incomplete_site_channel_declaration"] += 1
            continue
        if any(not Path(im["raw_path"]).is_file() for im in images):
            exclusions["missing_declared_raw_path"] += 1
            continue
        annotation = label_lookup[key]
        compound = row["compound_id"]
        if not (compound == pairing["compound_id"] == metadata["compound_id"] == annotation["Metadata_JCP2022"]):
            raise ValueError("Supplemental compound identity mismatch")
        if well_key(pairing["source"], pairing["plate"], pairing["treatment_well"]) != key:
            raise ValueError("Pairing and prepared source/plate/well disagree")
        if pairing["dataset_sample_id"] != "|".join(key):
            raise ValueError("Pairing dataset well ID disagrees with exact join")
        labels = raw_labels(annotation["Metadata_MoA"])
        if metadata["is_control"] or labels != sorted(set(metadata["moa_labels"])):
            raise ValueError("Supplemental control or annotation mismatch")
        candidates.append((compound, key, metadata, labels, pairing_path, images))
    candidates.sort(key=lambda row: row[0])
    if len(candidates) != 142:
        raise ValueError(f"The audited 142-well supplemental pool changed: {len(candidates)}")
    samples, annotation_rows, total_bytes, n_images = [], [], 0, 0
    for number, (compound, key, metadata, labels, pairing_path, images) in enumerate(candidates):
        records = []
        for image in sorted(images, key=lambda im: (int(im["field"]), CHANNELS.index(im["channel"]))):
            path = Path(image["raw_path"])
            digest = hash_file(path)
            if digest != image["raw_sha256"]:
                raise ValueError(f"Changed raw TIFF bytes: {path}")
            with Image.open(path) as opened:
                shape = [opened.height, opened.width]
            if shape != image["shape"] or image["raw_dtype"] != "uint16":
                raise ValueError(f"Changed TIFF shape or declared dtype: {path}")
            total_bytes += path.stat().st_size
            records.append({"image_id": image["image_id"], "site": str(image["field"]),
                            "channel": image["channel"], "path": str(path.resolve()),
                            "sha256": digest, "shape": shape, "role": "treatment", "well_id": "|".join(key)})
        parent = metadata["compound_group"]
        samples.append({"sample_id": compound, "well_id": "|".join(key), "parent_id": parent,
                        "source": key[0], "plate": key[1], "well": key[2], "split": "supplemental",
                        "images": records, "provenance": {"compound_id": compound,
                            "pairing_path": str(pairing_path), "pairing_sha256": hash_file(pairing_path)}})
        annotation_rows.append({"sample_id": compound, "well_id": "|".join(key), "parent_id": parent,
                                "compound_id": compound, "source": key[0], "plate": key[1], "well": key[2],
                                "split": "supplemental", "labels": labels, "scaffold": metadata["scaffold"]})
        n_images += len(records)
        if number == 0 or (number + 1) % 20 == 0 or number + 1 == len(candidates):
            print(f"Supplemental raw verification {number + 1}/{len(candidates)} wells, {n_images} TIFFs", flush=True)
    parent_ids = {row["parent_id"] for row in samples}
    plates = {(row["source"], row["plate"]) for row in samples}
    assert not parent_ids & old_parents and not plates & old_plates
    assert len(parent_ids) == len(samples)
    output.mkdir(parents=True, exist_ok=True)
    inputs = {str(path.relative_to(root)): hash_file(path) for path in [main_path, processed_path, raw_path, prepared_path]}
    payload = {"schema_version": 1, "channels": list(CHANNELS), "preprocessing": PREPROCESS,
               "purpose": "Supplemental new-plate raw-image-only inference; no descriptions or training",
               "primary_cohort_manifest_id": original["manifest_id"], "inputs": inputs, "samples": samples}
    manifest = {**payload, "manifest_id": stable_hash(payload)}
    write_json(output / "manifest.json", manifest)
    label_names = original["vocabulary"]
    y = np.asarray([[label in row["labels"] for label in label_names] for row in annotation_rows], dtype=np.int8)
    np.savez_compressed(output / "cohort_arrays.npz", y_all=y, label_names=np.asarray(label_names),
                        **{key: np.asarray([row[field] for row in annotation_rows])
                           for key, field in [("sample_ids", "sample_id"), ("well_ids", "well_id"),
                                              ("parent_ids", "parent_id"), ("compound_ids", "compound_id"),
                                              ("splits", "split"), ("sources", "source"), ("plates", "plate")]})
    supports = Counter(label for row in annotation_rows for label in row["labels"])
    old_scaffolds = {row["scaffold"] for row in original["samples"]}
    result = {"schema_version": 1, "completed_at": utc_now(), "script_sha256": hash_file(__file__),
              "inputs": inputs, "manifest_id": manifest["manifest_id"],
              "sample_count": len(samples), "image_count": n_images, "verified_raw_bytes": total_bytes,
              "vocabulary": label_names, "samples": annotation_rows,
              "fixed_label_sets": {key: {"labels": values["labels"],
                                        "supplemental_positive_support": {label: supports[label] for label in values["labels"]}}
                                   for key, values in original["label_sets"].items()},
              "source_counts": dict(Counter(row["source"] for row in samples)),
              "plate_count": len(plates), "exclusions": dict(exclusions),
              "audit": {"raw_tiffs_sha256_verified": n_images, "all_sites_have_five_channels": True,
                        "parent_overlap_with_all_original438": len(parent_ids & old_parents),
                        "plate_overlap_with_all_original438": len(plates & old_plates),
                        "scaffold_overlap_count_with_original438": len({row["scaffold"] for row in annotation_rows} & old_scaffolds),
                        "scaffolds_only_recorded_not_filtered": True,
                        "raw_processed_identity_and_labels_agree": True,
                        "caption_generation": False, "encoding_or_training": False},
              "labels_absent_from_original_vocabulary": sorted(set(supports) - set(label_names)),
              "all_label_positive_support": dict(sorted(supports.items())),
              "artifacts": {name: hash_file(output / name) for name in ["manifest.json", "cohort_arrays.npz"]}}
    write_json(output / "cohort.json", result)
    print({"wells": len(samples), "images": n_images, "plate_count": len(plates), "fixed_label_sets": result["fixed_label_sets"]}, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root / "CLIP/artifacts/moa_probe_v1/supplemental")
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())

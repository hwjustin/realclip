"""Join existing MoA annotations to the sealed 438-well CLIP cohort.

No split, source feature, caption, model or upstream data file is changed.
Identity is checked independently against processed metadata, raw well labels
and the original caption provenance compound identifier.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import re
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wellclip.common import hash_file, load_manifest, read_json, utc_now, write_json

SPLITS = ("train", "val", "test")


def well_key(source: object, plate: object, well: object) -> tuple[str, str, str]:
    source, plate, well = (str(value).strip() for value in (source, plate, well))
    match = re.fullmatch(r"([A-Za-z]+)([0-9]+)", well)
    if not source or not plate or match is None:
        raise ValueError("Missing or malformed source/plate/well key")
    return source, plate, f"{match.group(1).upper()}{int(match.group(2)):02d}"


def raw_labels(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing raw treatment MoA annotation")
    values = sorted({part.strip() for part in value.split("|") if part.strip()})
    if not values or "negative_control" in values:
        raise ValueError("Control or missing labels are not treatment MoA targets")
    return values


def unique_lookup(rows: list[dict], columns: tuple[str, str, str]) -> dict:
    result = {}
    for row in rows:
        key = well_key(*(row[name] for name in columns))
        if key in result:
            raise ValueError(f"Duplicate normalized identity key: {key}")
        result[key] = row
    return result


def support_rows(samples: list[dict], labels: list[str]) -> list[dict]:
    parents = {split: {row["parent_id"] for row in samples if row["split"] == split} for split in SPLITS}
    positives = {split: defaultdict(set) for split in SPLITS}
    for row in samples:
        for label in row["labels"]:
            positives[row["split"]][label].add(row["parent_id"])
    rows = []
    for label in labels:
        counts = {split: len(positives[split][label]) for split in SPLITS}
        negative = {split: len(parents[split]) - counts[split] for split in SPLITS}
        rows.append({"label": label,
                     **{f"{split}_positive_parents": counts[split] for split in SPLITS},
                     **{f"{split}_negative_parents": negative[split] for split in SPLITS},
                     "primary_eligible": all(counts[split] >= n for split, n in zip(SPLITS, [5, 1, 2]))
                                         and all(negative[split] >= 1 for split in SPLITS),
                     "exploratory_eligible": all(counts[split] >= n for split, n in zip(SPLITS, [3, 1, 1]))
                                             and all(negative[split] >= 1 for split in SPLITS)})
    return rows


def run(root: Path, work: Path, output: Path) -> dict:
    import pandas as pd

    manifest_path = work / "manifest.json"
    manifest = load_manifest(manifest_path)
    processed_path = root / "data/processed/dataset.parquet"
    processed_receipt = read_json(root / "data/processed/manifest.json")
    if hash_file(processed_path) != processed_receipt["metadata_sha256"]:
        raise ValueError("Processed metadata differs from its completed manifest")
    processed = pd.read_parquet(processed_path).to_dict("records")
    raw_path = root / "data/raw/moa_labels.csv"
    raw = pd.read_csv(raw_path, dtype=str).to_dict("records")
    processed_lookup = unique_lookup(processed, ("source", "plate", "well"))
    raw_lookup = unique_lookup(raw, ("Metadata_Source", "Metadata_Plate", "Metadata_Well"))
    samples = []
    parent_annotations, parent_split = {}, {}
    for sample in manifest["samples"]:
        key = well_key(*(sample[name] for name in ("source", "plate", "well")))
        if sample["well_id"] != "|".join(key):
            raise ValueError("Manifest well ID disagrees with its source/plate/well fields")
        metadata = processed_lookup[key]
        annotation = raw_lookup[key]
        compound = sample["provenance"]["compound_id"]
        if not (compound == metadata["compound_id"] == metadata["Metadata_JCP2022"] == annotation["Metadata_JCP2022"]):
            raise ValueError(f"Compound identity mismatch: {sample['sample_id']}")
        if metadata["compound_group"] != sample["parent_id"] or metadata["is_control"]:
            raise ValueError("Manifest parent mismatch or control treatment")
        labels = raw_labels(annotation["Metadata_MoA"])
        if labels != sorted(set(metadata["moa_labels"])) or labels != raw_labels(metadata["Metadata_MoA"]):
            raise ValueError(f"Processed and raw MoA disagree: {sample['sample_id']}")
        parent = sample["parent_id"]
        if parent in parent_annotations and parent_annotations[parent] != labels:
            raise ValueError("One canonical parent has conflicting labels")
        if parent in parent_split and parent_split[parent] != sample["split"]:
            raise ValueError("A canonical parent crosses the original CLIP splits")
        parent_annotations[parent], parent_split[parent] = labels, sample["split"]
        samples.append({"sample_id": sample["sample_id"], "well_id": sample["well_id"],
                        "parent_id": parent, "compound_id": compound, "source": key[0],
                        "plate": key[1], "well": key[2], "split": sample["split"],
                        "labels": labels, "scaffold": metadata["scaffold"]})
    vocabulary = sorted({label for sample in samples for label in sample["labels"]})
    support = support_rows(samples, vocabulary)
    plate_sets = {split: {(row["source"], row["plate"]) for row in samples if row["split"] == split} for split in SPLITS}
    scaffold_sets = {split: {row["scaffold"] for row in samples if row["split"] == split} for split in SPLITS}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    if any(plate_sets[a] & plate_sets[b] for a, b in pairs):
        raise ValueError("Original CLIP plate split is not isolated")
    output.mkdir(parents=True, exist_ok=True)
    y_all = np.asarray([[label in sample["labels"] for label in vocabulary] for sample in samples], dtype=np.int8)
    np.savez_compressed(output / "cohort_arrays.npz", y_all=y_all,
                        label_names=np.asarray(vocabulary),
                        **{key: np.asarray([sample[field] for sample in samples])
                           for key, field in [("sample_ids", "sample_id"), ("well_ids", "well_id"),
                                              ("parent_ids", "parent_id"), ("compound_ids", "compound_id"),
                                              ("splits", "split"), ("sources", "source"), ("plates", "plate")]})
    pd.DataFrame(support).to_csv(output / "class_support.csv", index=False)
    result = {
        "schema_version": 1, "completed_at": utc_now(), "script_sha256": hash_file(__file__),
        "purpose": "Exact-identity MoA annotation join; existing CLIP split retained; no classifier scores inspected",
        "manifest_id": manifest["manifest_id"], "sample_count": len(samples),
        "sample_order_matches_original_manifest": True,
        "inputs": {str(path.relative_to(root)): hash_file(path) for path in
                   [manifest_path, processed_path, raw_path, root / "data/processed/manifest.json"]},
        "join_audit": {
            "key": ["source", "plate", "well"], "raw_metadata_joined": len(samples),
            "processed_metadata_joined": len(samples), "compound_ids_match_provenance_processed_and_raw": True,
            "canonical_parent_matches_processed_metadata": True, "raw_processed_labels_equal": True,
            "unique_parents": len(parent_annotations), "duplicate_parent_well_counts": {
                key: count for key, count in Counter(sample["parent_id"] for sample in samples).items() if count > 1},
            "parent_label_conflicts": 0, "parent_split_conflicts": 0,
            "plate_cross_split_overlap": {a + "_" + b: len(plate_sets[a] & plate_sets[b]) for a, b in pairs},
            "scaffold_cross_split_overlap": {a + "_" + b: len(scaffold_sets[a] & scaffold_sets[b]) for a, b in pairs},
            "unlabeled_wells": sum(not sample["labels"] for sample in samples),
        },
        "split_counts": dict(Counter(sample["split"] for sample in samples)),
        "source_counts_by_split": {split: dict(Counter(sample["source"] for sample in samples if sample["split"] == split)) for split in SPLITS},
        "label_sets": {
            "primary": {"rule": "train>=5, val>=1, test>=2 positive canonical parents and >=1 negative in each split",
                        "labels": [row["label"] for row in support if row["primary_eligible"]]},
            "exploratory": {"rule": "train>=3, val>=1, test>=1 positive canonical parents and >=1 negative in each split",
                            "labels": [row["label"] for row in support if row["exploratory_eligible"]]},
        },
        "vocabulary": vocabulary, "class_support": support, "samples": samples,
        "limitations": ["Unlisted MoA is treated as a benchmark negative, not an experimentally excluded mechanism.",
                        "Eligible classes are selected by annotation support, never feature performance.",
                        "The original CLIP split isolates parents and plates but not necessarily scaffolds or sources.",
                        "Test support is small; the primary rule yields only two classes and exploratory classes are not a robust broad-MoA benchmark."],
        "artifacts": {name: hash_file(output / name) for name in ["cohort_arrays.npz", "class_support.csv"]},
    }
    write_json(output / "cohort.json", result)
    print({"samples": len(samples), "labels": len(vocabulary), "primary": result["label_sets"]["primary"]["labels"],
           "exploratory": result["label_sets"]["exploratory"]["labels"]})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--work", type=Path, default=root / "CLIP/artifacts/prototype_v1")
    parser.add_argument("--output", type=Path, default=root / "CLIP/artifacts/moa_probe_v1")
    args = parser.parse_args()
    run(args.root.resolve(), args.work.resolve(), args.output.resolve())

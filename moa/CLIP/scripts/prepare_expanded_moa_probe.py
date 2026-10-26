"""Identity-checked expanded MoA cohort and overlap-filtered supplemental wells.

Preserves the original two/eight class definitions and 264-column vocabulary.
The old142 supplemental set is audited against ALL expanded train/val/test
parents, plates and raw-image hashes before retaining any evaluation rows.
Only existing frozen DINO feature rows are copied; no encoding or fitting.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wellclip.common import CHANNELS, PREPROCESS, hash_file, load_manifest, read_json, stable_hash, utc_now, write_json
from prepare_moa_probe import raw_labels, run as prepare_identity_cohort, unique_lookup, well_key


def save_arrays(path: Path, samples: list[dict], vocabulary: list[str]) -> None:
    y = np.asarray([[label in sample["labels"] for label in vocabulary] for sample in samples], dtype=np.int8)
    np.savez_compressed(path, y_all=y, label_names=np.asarray(vocabulary),
                        **{key: np.asarray([sample[field] for sample in samples])
                           for key, field in [("sample_ids", "sample_id"), ("well_ids", "well_id"),
                                              ("parent_ids", "parent_id"), ("compound_ids", "compound_id"),
                                              ("splits", "split"), ("sources", "source"), ("plates", "plate")]})


def overlap_reasons(sample: dict, parent_sets: dict, plate_sets: dict,
                    image_hash_sets: dict, image_hashes: set) -> list[str]:
    reasons = []
    for split in ("train", "val", "test"):
        if sample["parent_id"] in parent_sets[split]:
            reasons.append(f"expanded_{split}_parent")
        if (sample["source"], sample["plate"]) in plate_sets[split]:
            reasons.append(f"expanded_{split}_plate")
        if image_hashes & image_hash_sets[split]:
            reasons.append(f"expanded_{split}_raw_image_sha")
    return reasons


def run(root: Path, output: Path) -> dict:
    import pandas as pd

    work = root / "CLIP/artifacts/expanded_samples_v1"
    legacy = root / "CLIP/artifacts/moa_probe_v1"
    old_manifest = load_manifest(root / "CLIP/artifacts/prototype_v1/manifest.json")
    manifest = load_manifest(work / "manifest.json")
    old_cohort_path = legacy / "cohort.json"
    old_cohort = read_json(old_cohort_path)
    vocabulary = old_cohort["vocabulary"]
    old_sets = copy.deepcopy(old_cohort["label_sets"])
    expanded_by_well = {sample["well_id"]: sample for sample in manifest["samples"]}
    assert len(expanded_by_well) == len(manifest["samples"])
    preserved = {split: all(sample == expanded_by_well.get(sample["well_id"])
                            for sample in old_manifest["samples"] if sample["split"] == split)
                 for split in ("train", "val", "test")}
    if not all(preserved.values()):
        raise ValueError("Original438 sample identities or splits changed in the expansion")
    for split in ("val", "test"):
        if [sample for sample in manifest["samples"] if sample["split"] == split] != [sample for sample in old_manifest["samples"] if sample["split"] == split]:
            raise ValueError(f"Original66 {split} sample list changed")
    # Reuse the existing exact-key / raw-label / provenance validation unchanged,
    # but keep its automatically derived eligibility only as audit information.
    with tempfile.TemporaryDirectory(prefix="expanded-moa-cohort-") as temporary:
        checked = prepare_identity_cohort(root, work, Path(temporary))
    output.mkdir(parents=True, exist_ok=True)
    save_arrays(output / "cohort_arrays.npz", checked["samples"], vocabulary)
    support = checked["class_support"]
    for row in support:
        row["meets_primary_support_rule"] = row.pop("primary_eligible")
        row["meets_exploratory_support_rule"] = row.pop("exploratory_eligible")
        row["fixed_primary_class"] = row["label"] in old_sets["primary"]["labels"]
        row["fixed_exploratory_class"] = row["label"] in old_sets["exploratory"]["labels"]
        row["encoded_in_fixed_vocabulary"] = row["label"] in vocabulary
    pd.DataFrame(support).to_csv(output / "class_support.csv", index=False)
    checked["all_annotation_vocabulary"] = checked["vocabulary"]
    checked["annotations_outside_fixed_vocabulary"] = sorted(set(checked["vocabulary"]) - set(vocabulary))
    checked["vocabulary"] = vocabulary
    checked["label_sets"] = old_sets
    checked["label_set_policy"] = "Keep the original2 primary /8 exploratory names, never add classes after expansion; y_all keeps original264 columns. All raw labels remain in samples and all_annotation_vocabulary."
    checked["script_sha256"] = hash_file(__file__)
    checked["original438_preserved"] = preserved
    checked["original306_train_subset_of_expanded603"] = True
    checked["original306_train_sample_ids"] = [sample["sample_id"] for sample in old_manifest["samples"] if sample["split"] == "train"]
    checked["original306_train_well_ids"] = [sample["well_id"] for sample in old_manifest["samples"] if sample["split"] == "train"]
    checked["original66_val_exactly_preserved"] = True
    checked["original66_test_exactly_preserved"] = True
    checked["inputs"][str(old_cohort_path.relative_to(root))] = hash_file(old_cohort_path)
    checked["artifacts"] = {name: hash_file(output / name) for name in ["cohort_arrays.npz", "class_support.csv"]}
    checked["limitations"] += ["The original142 supplemental wells are not automatically held out after expansion; see supplemental_eligible.json.",
                               "The fixed264 y_all columns do not encode newly observed out-of-vocabulary labels, which remain recorded in each sample."]
    write_json(output / "cohort.json", checked)

    supplemental_dir = legacy / "supplemental"
    old_sup_path = supplemental_dir / "cohort.json"
    old_sup = read_json(old_sup_path)
    old_sup_manifest = load_manifest(supplemental_dir / "manifest.json")
    if hash_file(supplemental_dir / "manifest.json") != old_sup["artifacts"]["manifest.json"]:
        raise ValueError("Old142 supplemental manifest no longer matches its cohort receipt")
    assert [s["sample_id"] for s in old_sup["samples"]] == [s["sample_id"] for s in old_sup_manifest["samples"]]
    metadata = unique_lookup(pd.read_parquet(root / "data/processed/dataset.parquet").to_dict("records"), ("source", "plate", "well"))
    raw = unique_lookup(pd.read_csv(root / "data/raw/moa_labels.csv", dtype=str).to_dict("records"),
                        ("Metadata_Source", "Metadata_Plate", "Metadata_Well"))
    parent_sets = {split: {s["parent_id"] for s in manifest["samples"] if s["split"] == split} for split in ("train", "val", "test")}
    plate_sets = {split: {(s["source"], s["plate"]) for s in manifest["samples"] if s["split"] == split} for split in parent_sets}
    image_hash_sets = {split: {im["sha256"] for s in manifest["samples"] if s["split"] == split for im in s["images"]} for split in parent_sets}
    audit_rows, retained_indices, samples, image_samples = [], [], [], []
    for index, (sample, image_sample) in enumerate(zip(old_sup["samples"], old_sup_manifest["samples"])):
        key = well_key(sample["source"], sample["plate"], sample["well"])
        processed, annotation = metadata[key], raw[key]
        if not (sample["compound_id"] == processed["compound_id"] == annotation["Metadata_JCP2022"] == image_sample["provenance"]["compound_id"]):
            raise ValueError("Old supplemental compound provenance identity mismatch")
        if sample["parent_id"] != processed["compound_group"]:
            raise ValueError("Old supplemental canonical parent identity mismatch")
        if sample["labels"] != raw_labels(annotation["Metadata_MoA"]) or sample["labels"] != sorted(set(processed["moa_labels"])):
            raise ValueError("Old supplemental raw and processed labels disagree")
        hashes = {im["sha256"] for im in image_sample["images"]}
        reasons = overlap_reasons(sample, parent_sets, plate_sets, image_hash_sets, hashes)
        audit_rows.append({"old142_index": index, **sample, "eligible": not reasons, "exclusion_reasons": reasons})
        if not reasons:
            retained_indices.append(index)
            samples.append(copy.deepcopy(sample))
            image_samples.append(copy.deepcopy(image_sample))
    assert len(samples) == 67, f"Audited supplemental retained size changed: {len(samples)}"
    counts = Counter(reason for row in audit_rows for reason in row["exclusion_reasons"])
    supp_out = output / "supplemental"
    supp_out.mkdir(parents=True, exist_ok=True)
    save_arrays(supp_out / "cohort_arrays.npz", samples, vocabulary)
    supports = Counter(label for sample in samples for label in sample["labels"])
    fixed = {name: {**value, "positive_counts": {label: supports[label] for label in value["labels"]},
                    "unique_positive_wells": sum(any(label in sample["labels"] for label in value["labels"]) for sample in samples),
                    "unsupported_labels": [label for label in value["labels"] if supports[label] == 0]}
             for name, value in old_sets.items()}
    manifest_payload = {"schema_version": 1, "channels": list(CHANNELS), "preprocessing": PREPROCESS,
                        "purpose": "Retained67 of old142 supplemental wells after excluding all expanded735 parent/plate/raw-image overlaps",
                        "expanded_manifest_id": manifest["manifest_id"], "source_manifest_id": old_sup_manifest["manifest_id"],
                        "source_old142_indices": retained_indices, "samples": image_samples}
    supp_manifest = {**manifest_payload, "manifest_id": stable_hash(manifest_payload)}
    write_json(supp_out / "manifest.json", supp_manifest)
    eligibility = {"schema_version": 1, "created_at": utc_now(), "script_sha256": hash_file(__file__),
                   "expanded_manifest_id": manifest["manifest_id"], "old142_manifest_id": old_sup_manifest["manifest_id"],
                   "inputs": {str(path.relative_to(root)): hash_file(path) for path in [work / "manifest.json", old_sup_path, supplemental_dir / "manifest.json"]},
                   "source_count": 142, "excluded_count": 142 - len(samples), "retained_count": len(samples),
                   "retained_plate_count": len({(s["source"], s["plate"]) for s in samples}),
                   "selection": "Exact parent/plate/raw-image overlap only; no MoA or prediction-score filtering; scaffolds recorded, not excluded",
                   "exclusion_reason_counts": dict(counts), "retained_old142_indices": retained_indices,
                   "fixed_label_sets": fixed, "rows": audit_rows,
                   "limitations": ["These67 wells were evaluated in the earlier142 MoA test and are historically inspected, not a fresh prospective test.",
                                   "The fixed8-class set has unsupported HDAC/tubulin labels in the retained67; report this explicitly, never silently take a6-class macro AP as the original8-class result."]}
    write_json(output / "supplemental_eligible.json", eligibility)
    supp_cohort = {"schema_version": 1, "manifest_id": supp_manifest["manifest_id"], "sample_count": len(samples),
                   "vocabulary": vocabulary, "samples": samples, "fixed_label_sets": fixed, "label_sets": fixed,
                   "old142_indices": retained_indices, "source_cohort_sha256": hash_file(old_sup_path),
                   "selection_audit_sha256": hash_file(output / "supplemental_eligible.json"),
                   "source_counts": dict(Counter(s["source"] for s in samples)),
                   "artifacts": {name: hash_file(supp_out / name) for name in ["manifest.json", "cohort_arrays.npz"]}}
    write_json(supp_out / "cohort.json", supp_cohort)
    # Copy only authenticated old frozen-image rows; no DINO execution is needed.
    legacy_features_path = supplemental_dir / "features/features.npz"
    legacy_features_metadata = read_json(supplemental_dir / "features/metadata.json")
    if hash_file(legacy_features_path) != legacy_features_metadata["artifacts"]["features.npz"]:
        raise ValueError("Legacy142 DINO archive authentication failed")
    expanded_identity = read_json(work / "features/identity.json")
    legacy_protocol = read_json(supplemental_dir / "supplemental_protocol.json")
    assert expanded_identity["model_identities"]["dino"] == legacy_features_metadata["backbone_identity"]
    assert expanded_identity["preprocessing"] == legacy_protocol["preprocessing"] == PREPROCESS
    for name, digest in expanded_identity["implementation_sha256"].items():
        assert digest == legacy_protocol["code_sha256"][name]
    with np.load(legacy_features_path, allow_pickle=False) as archive:
        assert archive["sample_ids"].tolist() == [s["sample_id"] for s in old_sup["samples"]]
        features = archive["image_features"][retained_indices]
    feature_out = supp_out / "features"
    feature_out.mkdir(exist_ok=True)
    np.savez_compressed(feature_out / "features.npz", sample_ids=np.asarray([s["sample_id"] for s in samples]), image_features=features)
    feature_metadata = {"feature_id": stable_hash({"source_feature_id": legacy_features_metadata["feature_id"], "retained_indices": retained_indices}),
                        "manifest_id": supp_manifest["manifest_id"], "image_feature_shape": list(features.shape),
                        "source_archive_sha256": hash_file(legacy_features_path), "source_feature_id": legacy_features_metadata["feature_id"],
                        "expanded_backbone_and_preprocessing_match": True, "no_encoding_or_fitting": True,
                        "sample_ids": [s["sample_id"] for s in samples], "artifacts": {"features.npz": hash_file(feature_out / "features.npz")}}
    write_json(feature_out / "metadata.json", feature_metadata)
    print({"expanded_split_counts": checked["split_counts"], "encoded_labels": len(vocabulary),
           "all_annotation_labels": len(checked["all_annotation_vocabulary"]), "old142_exclusions": dict(counts),
           "retained": len(samples), "retained_fixed_label_sets": fixed}, flush=True)
    return {"cohort": checked, "supplemental": supp_cohort, "eligibility": eligibility}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root / "CLIP/artifacts/expanded_moa_probe_v1")
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())

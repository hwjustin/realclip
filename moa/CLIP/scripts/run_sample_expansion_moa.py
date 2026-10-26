"""Frozen-vocabulary MOA readouts after adding CLIP training wells.

The original66 test wells remain the primary benchmark. The historically
retained67 image-only wells are audited again; only their identity-disjoint
subset is scored, with the old603 predictions restricted to that same subset.
The all stage uses the existing CLIP and sklearn environments separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wellclip.common import hash_file, load_manifest, read_json, stable_hash, write_json


def bound(path: Path, value) -> None:
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Previously bound expansion MOA artifact changed: {path}")
    write_json(path, value)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def supplemental_indices(primary: list[dict], supplemental: list[dict]) -> tuple[list[int], list[dict]]:
    """Exclude identity overlaps only, without labels or predictions as inputs."""
    parents = {s["parent_id"] for s in primary}
    plates = {(s["source"], s["plate"]) for s in primary}
    wells = {s["well_id"] for s in primary}
    hashes = {im["sha256"] for s in primary for im in s["images"]}
    kept, audit = [], []
    for i, sample in enumerate(supplemental):
        reasons = []
        if sample["parent_id"] in parents:
            reasons.append("primary_parent")
        if (sample["source"], sample["plate"]) in plates:
            reasons.append("primary_plate")
        if sample["well_id"] in wells:
            reasons.append("primary_well")
        if hashes & {im["sha256"] for im in sample["images"]}:
            reasons.append("primary_raw_image")
        audit.append({"old67_index": i, "sample_id": sample["sample_id"],
                      "well_id": sample["well_id"], "eligible": not reasons, "reasons": reasons})
        if not reasons:
            kept.append(i)
    return kept, audit


def prepare(work: Path, legacy: Path) -> dict:
    from prepare_moa_probe import run as join_annotations
    from prepare_expanded_moa_probe import save_arrays
    from run_finetune_moa_probe import PRIMARY_LABELS, EXPLORATORY_LABELS, validate_cohorts

    output = work / "moa/cohort"
    if (output / "complete.json").exists():
        current = validate_cohort_files(output)
        if (current["manifest_sha256"] != hash_file(work / "manifest.json")
                or current["legacy_cohort_sha256"] != hash_file(legacy / "cohort.json")
                or current["script_sha256"] != hash_file(__file__)):
            raise ValueError("Existing expansion MOA cohort belongs to changed inputs or code")
        return read_json(output / "supplemental_audit.json")
    manifest = load_manifest(work / "manifest.json")
    old = read_json(legacy / "cohort.json")
    vocabulary = old["vocabulary"]
    for split in ("val", "test"):
        now_ids = [s["sample_id"] for s in manifest["samples"] if s["split"] == split]
        old_ids = [s["sample_id"] for s in old["samples"] if s["split"] == split]
        if now_ids != old_ids or len(now_ids) != 66:
            raise ValueError("Original66 validation/test membership changed")
    with tempfile.TemporaryDirectory(prefix="expansion-v2-moa-") as temporary:
        joined = join_annotations(ROOT.parent, work, Path(temporary))
    output.mkdir(parents=True, exist_ok=True)
    save_arrays(output / "cohort_arrays.npz", joined["samples"], vocabulary)
    old_extra = read_json(legacy / "supplemental/cohort.json")
    old_extra_manifest = load_manifest(legacy / "supplemental/manifest.json")
    if [s["sample_id"] for s in old_extra["samples"]] != [s["sample_id"] for s in old_extra_manifest["samples"]]:
        raise ValueError("Legacy supplemental annotation/image identities disagree")
    if len(old_extra["samples"]) != 67:
        raise ValueError("Supplemental source must be the previously retained67")
    kept, audit = supplemental_indices(manifest["samples"], old_extra_manifest["samples"])
    extra_samples = [old_extra["samples"][i] for i in kept]
    (output / "supplemental").mkdir(exist_ok=True)
    save_arrays(output / "supplemental/cohort_arrays.npz", extra_samples, vocabulary)
    cohort = load_npz(output / "cohort_arrays.npz")
    extra = load_npz(output / "supplemental/cohort_arrays.npz")
    # save_arrays preserves [0,K] for an empty supplemental set explicitly here.
    if not kept:
        extra["y_all"] = np.empty((0, len(vocabulary)), dtype=np.int8)
        np.savez_compressed(output / "supplemental/cohort_arrays.npz", **extra)
    counts = dict(Counter(s["split"] for s in manifest["samples"]))
    validate_cohorts(cohort, extra, counts)
    joined.update(vocabulary=vocabulary, annotations_outside_fixed_vocabulary=sorted(set(joined["vocabulary"]) - set(vocabulary)),
                  label_sets={"primary": {"labels": PRIMARY_LABELS}, "exploratory": {"labels": EXPLORATORY_LABELS}})
    bound(output / "cohort.json", joined)
    bound(output / "supplemental/cohort.json", {"samples": extra_samples, "old67_indices": kept})
    info = {"old67_count": 67, "retained_count": len(kept), "excluded_count": 67 - len(kept),
            "old67_indices": kept, "rows": audit,
            "selection": "Exclude any primary parent, plate, well or raw-image overlap; no label/score filtering",
            "full67_comparable": len(kept) == 67}
    bound(output / "supplemental_audit.json", info)
    protocol = read_json(legacy / "protocol.json")
    if protocol["primary_labels"] != PRIMARY_LABELS or protocol["exploratory_labels"] != EXPLORATORY_LABELS:
        raise ValueError("Legacy fixed MOA label sets changed")
    bound(output / "protocol.json", {"primary_labels": PRIMARY_LABELS, "exploratory_labels": EXPLORATORY_LABELS,
          "linear_probe": protocol["linear_probe"], "split_counts": counts,
          "manifest_sha256": hash_file(work / "manifest.json"), "legacy_cohort": str(legacy),
          "legacy_cohort_sha256": hash_file(legacy / "cohort.json"),
          "vocabulary_policy": "Keep original264 columns and original2 primary/8 exploratory labels",
          "supplemental_policy": info["selection"], "script_sha256": hash_file(__file__)})
    bound(output / "complete.json", {"artifacts": {name: hash_file(output / name) for name in (
        "cohort.json", "cohort_arrays.npz", "supplemental/cohort.json", "supplemental/cohort_arrays.npz",
        "supplemental_audit.json", "protocol.json")}})
    print(f"MOA cohort: {counts}; supplemental retained {len(kept)}/67", flush=True)
    return info


def validate_cohort_files(directory: Path) -> dict:
    complete = read_json(directory / "complete.json")
    for name, expected in complete["artifacts"].items():
        if hash_file(directory / name) != expected:
            raise ValueError(f"MOA cohort artifact changed: {name}")
    return read_json(directory / "protocol.json")


def export(work: Path, legacy: Path) -> dict:
    import torch
    from wellclip.training import load_features
    from wellclip.lr64_extended_training import validate_extended_lr64_training, load_extended_lr64_head
    from wellclip.finetune_evaluation import _load_supplemental_features

    torch.set_num_threads(2)
    target = work / "moa/representations"
    cohort_dir = work / "moa/cohort"
    validate_cohort_files(cohort_dir)
    manifest, bank, metadata = load_features(work / "manifest.json", work / "features")
    protocol, receipts = validate_extended_lr64_training(work / "training", manifest["manifest_id"], metadata["feature_id"])
    config = protocol["config"]
    if (config["learning_rates"] != [1e-4] or config["batch_size"] != 64 or config["epochs"] != 300
            or config["seeds"] != [42, 43, 44] or len(receipts) != 6):
        raise ValueError("Expected six default128 batch64/lr1e-4/300epoch checkpoints")
    old_extra_manifest = load_manifest(legacy / "supplemental/manifest.json")
    extra_bank, extra_metadata = _load_supplemental_features(old_extra_manifest, legacy / "supplemental/features")
    kept = read_json(cohort_dir / "supplemental_audit.json")["old67_indices"]
    extra_samples = [old_extra_manifest["samples"][i] for i in kept]
    recomputed, _ = supplemental_indices(manifest["samples"], old_extra_manifest["samples"])
    if kept != recomputed:
        raise ValueError("Supplemental identity audit no longer matches the trained cohort")
    stores = {"primary": {"sample_ids": bank["sample_ids"], "dino1920": bank["image_features"].reshape(len(bank["sample_ids"]), 1920)},
              "supplemental": {"sample_ids": extra_bank["sample_ids"][kept],
                               "dino1920": extra_bank["image_features"][kept].reshape(len(kept), 1920)}}
    binding = {"manifest_id": manifest["manifest_id"], "feature_id": metadata["feature_id"],
               "protocol_id": protocol["protocol_id"], "training_complete_sha256": hash_file(work / "training/complete.json"),
               "cohort_complete_sha256": hash_file(cohort_dir / "complete.json"),
               "supplemental_features_sha256": extra_metadata["artifacts"]["features.npz"],
               "all_checkpoints_selected_before_test_export": True, "moa_labels_used_for_export": False,
               "checkpoints": {r["name"]: hash_file(work / "training" / r["name"] / "best.pt") for r in receipts},
               "script_sha256": hash_file(__file__)}
    target.mkdir(parents=True, exist_ok=True)
    bound(target / "binding.json", binding)
    specs = [{"key": "dino1920", "stage": "raw", "dimension": 1920}]
    for run in receipts:
        name = run["name"]
        model, checkpoint = load_extended_lr64_head(work / "training" / name / "best.pt")
        if checkpoint["protocol_id"] != protocol["protocol_id"]:
            raise ValueError("MOA export head belongs to another protocol")
        for scope, features in (("primary", bank["image_features"]), ("supplemental", extra_bank["image_features"][kept])):
            pixels = torch.from_numpy(np.array(features, dtype=np.float32, copy=True))
            with torch.inference_mode():
                stores[scope][name + "__embedding"] = model.encode_image(pixels).numpy()
                stores[scope][name + "__hidden160"] = model.channel_projection(pixels).flatten(start_dim=1).numpy()
        for stage, dim in (("embedding", 128), ("hidden160", 160)):
            specs.append({"key": name + "__" + stage, "stage": "embedding" if dim == 128 else "hidden",
                          "dimension": dim, "run_name": name, "alignment": run["alignment"], "seed": run["seed"]})
    for scope, store in stores.items():
        np.savez_compressed(target / f"{scope}_store.npz", **store)
    fields = ("sample_id", "well_id", "parent_id", "source", "plate", "split")
    bound(target / "samples.json", {scope: [{key: s[key] for key in fields} for s in samples]
                                    for scope, samples in (("primary", manifest["samples"]), ("supplemental", extra_samples))})
    bound(target / "features.json", specs)
    bound(target / "runs.json", [{k: run[k] for k in ("name", "alignment", "seed", "best_epoch", "validation_mrr")} for run in receipts])
    result = {"binding_sha256": hash_file(target / "binding.json"), "moa_labels_used_for_export": False,
              "artifacts": {name: hash_file(target / name) for name in (
                  "samples.json", "features.json", "runs.json", "primary_store.npz", "supplemental_store.npz")}}
    bound(target / "complete.json", result)
    print(f"Exported {len(specs)} MOA representations, frozen backbones unchanged", flush=True)
    return result


def baseline_metrics(baseline: Path, spec: dict, label_set: str, test: dict,
                     extra: dict, retained_ids: list[str]) -> dict:
    """Restrict old saved predictions to identical wells; never refit old models."""
    from run_expanded_moa_probe import safe_metrics

    result = read_json(baseline / "results.json")
    for name, digest in result["probability_archives"].items():
        if hash_file(baseline / name) != digest:
            raise ValueError("Baseline MOA probability archive changed")
    old_rep = "dino1920" if spec["stage"] == "raw" else (
        f"original128__{spec['alignment']}_seed{spec['seed']}__" +
        ("embedding" if spec["stage"] == "embedding" else "hidden160"))
    key = label_set + "__" + old_rep
    old = next(r for r in result["runs"] if r["id"] == key)
    if spec.get("labels", old["labels"]) != old["labels"]:
        raise ValueError("Baseline and new MOA label order differ")
    p = load_npz(baseline / "probabilities.npz")
    if len(set(p["sample_ids"])) != len(p["sample_ids"]) or p[key].shape != (len(p["sample_ids"]), len(old["labels"])):
        raise ValueError("Invalid baseline probability rows/columns")
    lookup = {str(x): i for i, x in enumerate(p["sample_ids"])}
    idx = [lookup[str(s)] for s in test["sample_ids"]]
    metrics = {"test": safe_metrics(test["y"], p[key][idx], old["labels"]), "supplemental": None}
    if retained_ids:
        p = load_npz(baseline / "supplemental_probabilities.npz")
        if len(set(p["sample_ids"])) != len(p["sample_ids"]) or p[key].shape != (len(p["sample_ids"]), len(old["labels"])):
            raise ValueError("Invalid baseline supplemental probability rows/columns")
        lookup = {str(x): i for i, x in enumerate(p["sample_ids"])}
        idx = [lookup[s] for s in retained_ids]
        metrics["supplemental"] = safe_metrics(extra["y"], p[key][idx], old["labels"])
    return {"representation": old_rep, "n_train": 603, "metrics": metrics,
            "results_sha256": hash_file(baseline / "results.json"), "same_test_wells_only": True}


def validate_results(work: Path) -> dict:
    """Read-only checksum/selection audit, usable from either Python environment."""
    output = Path(work) / "moa"
    complete = read_json(output / "complete.json")
    required = {"results.json", "protocol.json", "identity.json", "selection.json", "selection_seal.json",
                "probabilities.npz", "supplemental_probabilities.npz"}
    if not required <= set(complete["artifacts"]):
        raise ValueError("Incomplete expansion MOA completion coverage")
    for name, digest in complete["artifacts"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or hash_file(path) != digest:
            raise ValueError(f"Expansion MOA artifact changed: {name}")
    result = read_json(output / "results.json")
    identity = {"protocol_sha256": hash_file(output / "protocol.json")}
    selection = read_json(output / "selection.json")
    seal = read_json(output / "selection_seal.json")
    if (result["identity"] != identity or read_json(output / "identity.json") != identity
            or selection["identity"] != identity or seal["identity"] != identity
            or seal["selection_sha256"] != hash_file(output / "selection.json")
            or result["selection_seal_sha256"] != hash_file(output / "selection_seal.json")
            or seal.get("all_readouts_selected_before_test_prediction") is not True
            or seal["n_readouts"] != result["n_runs"] or len(selection["runs"]) != result["n_runs"]
            or len(result["runs"]) != result["n_runs"]):
        raise ValueError("Expansion MOA selection identity/seal changed")
    for row in selection["runs"]:
        path = (output / row["model_path"]).resolve()
        if (not path.is_relative_to(output.resolve()) or row["identity"] != identity
                or row.get("test_used_for_model_selection") is not False
                or hash_file(path) != seal["models"].get(row["id"]) or hash_file(path) != row["model_sha256"]):
            raise ValueError("Expansion MOA model changed after selection")
    protocol = read_json(output / "protocol.json")
    cohort_protocol = validate_cohort_files(output / "cohort")
    if (protocol["cohort_complete_sha256"] != hash_file(output / "cohort/complete.json")
            or protocol["export_complete_sha256"] != hash_file(output / "representations/complete.json")
            or cohort_protocol["manifest_sha256"] != hash_file(Path(work) / "manifest.json")):
        raise ValueError("Expansion MOA cohort/export binding changed")
    exported = output / "representations"
    export_complete = read_json(exported / "complete.json")
    if export_complete["binding_sha256"] != hash_file(exported / "binding.json"):
        raise ValueError("Expansion MOA export binding changed")
    for name, digest in export_complete["artifacts"].items():
        path = (exported / name).resolve()
        if not path.is_relative_to(exported.resolve()) or hash_file(path) != digest:
            raise ValueError("Expansion MOA exported features changed")
    for name, digest in result["probability_archives"].items():
        if hash_file(output / name) != digest:
            raise ValueError("Expansion MOA probability archive changed")
    if result["supplemental_audit"] != read_json(output / "cohort/supplemental_audit.json"):
        raise ValueError("Expansion MOA supplemental audit changed")
    return result


def fit(work: Path, baseline: Path, threads: int = 2) -> dict:
    import joblib
    from threadpoolctl import threadpool_limits
    from run_direct256_moa_probe import (validate_export, feature_specs, fit_direct256_readout,
                                         seal_selection, load_sealed_selection)
    from run_dimension_moa_probe import validate_store, _check_samples, predict_dimension_saved
    from run_expanded_moa_probe import safe_metrics
    from run_finetune_moa_probe import validate_cohorts

    output, cohort_dir, exported = work / "moa", work / "moa/cohort", work / "moa/representations"
    old_protocol = validate_cohort_files(cohort_dir)
    validate_export(exported)
    cohort, extra = load_npz(cohort_dir / "cohort_arrays.npz"), load_npz(cohort_dir / "supplemental/cohort_arrays.npz")
    splits = validate_cohorts(cohort, extra, old_protocol["split_counts"])
    _check_samples(read_json(exported / "samples.json"), cohort, extra)
    specs = feature_specs(exported)
    stores = {scope: validate_store(load_npz(exported / f"{scope}_store.npz"), rows, specs)
              for scope, rows in (("primary", cohort), ("supplemental", extra))}
    protocol = {**old_protocol, "experiment": "expanded training cohort, unchanged default frozen128 model",
                "export_complete_sha256": hash_file(exported / "complete.json"),
                "cohort_complete_sha256": hash_file(cohort_dir / "complete.json"),
                "baseline_results_sha256": hash_file(baseline / "results.json"), "representations": specs,
                "test_used_for_model_selection": False, "moa_labels_used_for_clip_training": False,
                "comparison": "Expanded MOA training labels versus old603; same val66/test66 and any clean old67 subset",
                "code_sha256": {name: hash_file(Path(__file__).with_name(name)) for name in (
                    "run_sample_expansion_moa.py", "run_direct256_moa_probe.py", "run_dimension_moa_probe.py",
                    "run_moa_linear_probe.py", "run_expanded_moa_probe.py", "run_finetune_moa_probe.py")}}
    bound(output / "protocol.json", protocol)
    identity = {"protocol_sha256": hash_file(output / "protocol.json")}
    bound(output / "identity.json", identity)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    train, val, test = (splits[name] for name in ("train", "val", "test"))
    train_store = {k: v[train] for k, v in stores["primary"].items()}
    val_store = {k: v[val] for k, v in stores["primary"].items()}
    labels_all = cohort["label_names"].tolist()
    selected = []
    for label_set in ("primary", "exploratory"):
        labels = protocol[label_set + "_labels"]
        columns = [labels_all.index(label) for label in labels]
        for spec in specs:
            key = label_set + "__" + spec["key"]
            path = model_dir / (key + ".joblib")
            audit_path = path.with_suffix(".json")
            if audit_path.exists():
                audit = read_json(audit_path)
                if audit["identity"] != identity or hash_file(path) != audit["model_sha256"]:
                    raise ValueError("Previously fitted expansion MOA readout changed")
            else:
                with threadpool_limits(limits=threads):
                    model, audit = fit_direct256_readout(spec, train_store, cohort["sources"][train],
                        cohort["y_all"][np.ix_(train, columns)], val_store, cohort["sources"][val],
                        cohort["y_all"][np.ix_(val, columns)], protocol["linear_probe"]["C_grid"])
                model.update(labels=labels, id=key)
                joblib.dump(model, path)
                audit = {**spec, **audit, "id": key, "identity": identity, "representation": spec["key"],
                         "label_set": label_set, "labels": labels, "n_train": len(train),
                         "model_path": str(path.relative_to(output)), "model_sha256": hash_file(path),
                         "test_used_for_model_selection": False}
                write_json(audit_path, audit)
            selected.append(audit)
            print(f"SELECTED expansionMOA {len(selected)}/{2 * len(specs)} {key}", flush=True)
    seal_selection(output, identity, selected)
    selected, seal = load_sealed_selection(output, identity)
    predictions = {"sample_ids": cohort["sample_ids"], "splits": cohort["splits"]}
    extra_predictions = {"sample_ids": extra["sample_ids"], "splits": extra["splits"]}
    results = []
    for audit in selected:
        model = joblib.load(output / audit["model_path"])
        with threadpool_limits(limits=threads):
            p = predict_dimension_saved(model, stores["primary"], cohort["sources"])
            ep = (predict_dimension_saved(model, stores["supplemental"], extra["sources"])
                  if len(extra["sample_ids"]) else np.empty((0, len(audit["labels"]))))
        predictions[audit["id"]], extra_predictions[audit["id"]] = p, ep
        columns = [labels_all.index(label) for label in audit["labels"]]
        validation = safe_metrics(cohort["y_all"][np.ix_(val, columns)], p[val], audit["labels"])
        if not np.isclose(validation["macro_ap"], audit["validation_macro_ap"], atol=1e-12, rtol=0):
            raise ValueError("Reloaded expansion MOA classifier changed validation AP")
        target = cohort["y_all"][np.ix_(test, columns)]
        extra_target = extra["y_all"][:, columns]
        old = baseline_metrics(baseline, audit, audit["label_set"],
                               {"sample_ids": cohort["sample_ids"][test], "y": target},
                               {"y": extra_target}, extra["sample_ids"].tolist())
        results.append({**audit, "baseline603": old, "metrics": {"val": validation,
                        "test": safe_metrics(target, p[test], audit["labels"]),
                        "supplemental": safe_metrics(extra_target, ep, audit["labels"]) if len(ep) else None}})
    np.savez_compressed(output / "probabilities.npz", **predictions)
    np.savez_compressed(output / "supplemental_probabilities.npz", **extra_predictions)
    result = {"identity": identity, "selection_seal_sha256": hash_file(output / "selection_seal.json"),
              "n_runs": len(results), "n_train": len(train), "n_val": len(val), "n_test": len(test),
              "n_supplemental": len(extra["sample_ids"]), "supplemental_audit": read_json(cohort_dir / "supplemental_audit.json"),
              "primary_endpoint": "Fixed two-class macroAP on unchanged66; clean supplemental subset reported separately",
              "probability_archives": {name: hash_file(output / name) for name in ("probabilities.npz", "supplemental_probabilities.npz")},
              "comparison_limitation": "Expanded MOA probes also receive more labeled training wells; changes do not isolate CLIP pair expansion alone",
              "runs": results}
    bound(output / "results.json", result)
    bound(output / "complete.json", {"artifacts": {name: hash_file(output / name) for name in (
        "results.json", "protocol.json", "identity.json", "selection.json", "selection_seal.json",
        "probabilities.npz", "supplemental_probabilities.npz")}})
    validate_results(work)
    print(f"Completed {len(results)} MOA readouts; test66 unchanged, clean supplemental {len(extra['sample_ids'])}/67", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/expanded_samples_v2")
    parser.add_argument("--legacy", type=Path, default=ROOT / "artifacts/expanded_moa_probe_v1")
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/direct256_v1/moa")
    parser.add_argument("--stage", choices=("prepare", "export", "fit", "all"), default="all")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    work, legacy, baseline = args.output.resolve(), args.legacy.resolve(), args.baseline.resolve()
    if args.threads < 1:
        raise ValueError("At least one CPU thread is required")
    if args.stage == "all":
        for stage in ("prepare", "export", "fit"):
            executable = ROOT / ".venv/bin/python" if stage == "export" else ROOT.parent / ".venv/bin/python"
            subprocess.run([str(executable), "-u", str(Path(__file__)), "--stage", stage,
                "--output", str(work), "--legacy", str(legacy), "--baseline", str(baseline),
                "--threads", str(args.threads)], check=True)
    elif args.stage == "prepare":
        prepare(work, legacy)
    elif args.stage == "export":
        export(work, legacy)
    else:
        fit(work, baseline, args.threads)


if __name__ == "__main__":
    main()

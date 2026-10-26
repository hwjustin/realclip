"""Validation-selected MOA readouts for direct256 versus original128 well heads.

Run with the repository sklearn environment. The original two primary/eight
exploratory labels and 603/66/66 plus retained67 cohorts are unchanged. All
readouts are selected and sealed before either test cohort is predicted.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from run_dimension_moa_probe import (fit_dimension_readout, predict_dimension_saved,
                                     _check_samples, validate_store)
from run_expanded_moa_probe import load_cohort, safe_metrics
from run_finetune_moa_probe import EXPLORATORY_LABELS, PRIMARY_LABELS, validate_cohorts
from run_moa_linear_probe import hash_file, read_json, write_json


REQUIRED_EXPORTS = ("features.json", "runs.json", "samples.json", "primary_store.npz",
                    "supplemental_store.npz")


def _write_bound(path: Path, value: dict) -> None:
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Existing direct256 comparison MOA binding changed: {path}")
    write_json(path, value)


def feature_specs(export_dir: Path) -> list[dict]:
    """Accept declared stages/dimensions without introducing extra PCA controls."""
    specs = read_json(export_dir / "features.json")
    if not isinstance(specs, list) or not specs:
        raise ValueError("A nonempty direct256 comparison feature list is required")
    keys = []
    for spec in specs:
        key = spec.get("key")
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError("Unsafe or missing representation key")
        if key in ("source_only", "prevalence", "dino_pca128") or key.startswith("dino_plus_"):
            raise ValueError(f"Reserved legacy representation key: {key}")
        if spec.get("stage") not in ("raw", "hidden", "embedding"):
            raise ValueError("Only exported raw, hidden and embedding stages are accepted")
        dimension = spec.get("dimension")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("Representation dimensions must be positive integers")
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate direct256 comparison representation key")
    return specs


def validate_export(export_dir: Path) -> dict:
    """Require checksum coverage of every feature and sample identity input."""
    complete = read_json(export_dir / "complete.json")
    if complete.get("binding_sha256") != hash_file(export_dir / "binding.json"):
        raise ValueError("Direct256 comparison representation binding checksum mismatch")
    if complete.get("moa_labels_used_for_export") is not False:
        raise ValueError("CLIP representation exports must not use MOA targets")
    artifacts = complete.get("artifacts", {})
    if not set(REQUIRED_EXPORTS) <= set(artifacts):
        raise ValueError("Export completion does not bind every required representation artifact")
    for name, digest in artifacts.items():
        path = (export_dir / name).resolve()
        if not path.is_relative_to(export_dir.resolve()) or hash_file(path) != digest:
            raise ValueError(f"Direct256 comparison export artifact checksum mismatch: {name}")
    return complete


def fit_direct256_readout(spec: dict, train_store: dict, train_sources: np.ndarray,
                           y_train: np.ndarray, val_store: dict, val_sources: np.ndarray,
                           y_val: np.ndarray, c_grid: list[float]) -> tuple[dict, dict]:
    """The established train/validation-only linear probe, without test arguments."""
    if spec["stage"] not in ("raw", "hidden", "embedding"):
        raise ValueError("Direct256 comparison readouts require an exported feature stage")
    return fit_dimension_readout(spec, train_store, train_sources, y_train,
                                 val_store, val_sources, y_val, c_grid)


def seal_selection(output: Path, identity: dict, selected: list[dict]) -> dict:
    if not selected or len({row["id"] for row in selected}) != len(selected):
        raise ValueError("Readout selection must be nonempty and unique")
    for row in selected:
        path = (output / row["model_path"]).resolve()
        if (row.get("identity") != identity or row.get("test_used_for_model_selection") is not False
                or not path.is_relative_to(output.resolve()) or hash_file(path) != row["model_sha256"]):
            raise ValueError("Cannot seal an unbound or changed MOA readout")
    _write_bound(output / "selection.json", {"identity": identity, "runs": selected})
    seal = {"identity": identity, "selection_sha256": hash_file(output / "selection.json"),
            "all_readouts_selected_before_test_prediction": True, "n_readouts": len(selected),
            "models": {row["id"]: row["model_sha256"] for row in selected}}
    _write_bound(output / "selection_seal.json", seal)
    return seal


def load_sealed_selection(output: Path, identity: dict) -> tuple[list[dict], dict]:
    """Reject a changed selection or model before allowing test prediction."""
    seal = read_json(output / "selection_seal.json")
    selection = read_json(output / "selection.json")
    selected = selection["runs"]
    if (seal.get("identity") != identity or selection.get("identity") != identity
            or seal.get("all_readouts_selected_before_test_prediction") is not True
            or seal.get("selection_sha256") != hash_file(output / "selection.json")
            or seal.get("n_readouts") != len(selected)
            or seal.get("models") != {row["id"]: row["model_sha256"] for row in selected}):
        raise ValueError("Direct256 comparison MOA selection seal changed")
    for row in selected:
        path = (output / row["model_path"]).resolve()
        if (not path.is_relative_to(output.resolve()) or row.get("identity") != identity
                or row.get("test_used_for_model_selection") is not False
                or hash_file(path) != seal["models"][row["id"]]):
            raise ValueError("MOA readout changed after selection seal")
    return selected, seal


def run(export_dir: Path, cohort_root: Path, output: Path, threads: int = 2) -> dict:
    export_dir, cohort_root, output = export_dir.resolve(), cohort_root.resolve(), output.resolve()
    if threads < 1:
        raise ValueError("At least one CPU thread is required")
    validate_export(export_dir)
    old_protocol = read_json(cohort_root / "protocol.json")
    if old_protocol["primary_labels"] != PRIMARY_LABELS or old_protocol["exploratory_labels"] != EXPLORATORY_LABELS:
        raise ValueError("Established MOA vocabularies changed")
    cohort = load_cohort(cohort_root / "cohort_arrays.npz")
    supplemental = load_cohort(cohort_root / "supplemental/cohort_arrays.npz")
    splits = validate_cohorts(cohort, supplemental, {"train": 603, "val": 66, "test": 66})
    if len(supplemental["sample_ids"]) != 67:
        raise ValueError("Use the unchanged retained67 supplemental cohort")
    _check_samples(read_json(export_dir / "samples.json"), cohort, supplemental)
    specs = feature_specs(export_dir)
    stores = {scope: validate_store(load_cohort(export_dir / f"{scope}_store.npz"), rows, specs)
              for scope, rows in (("primary", cohort), ("supplemental", supplemental))}
    bindings = {"export_complete": export_dir / "complete.json", "export_binding": export_dir / "binding.json",
                "features": export_dir / "features.json", "runs": export_dir / "runs.json",
                "samples": export_dir / "samples.json", "primary_store": export_dir / "primary_store.npz",
                "supplemental_store": export_dir / "supplemental_store.npz",
                "cohort": cohort_root / "cohort_arrays.npz",
                "supplemental_cohort": cohort_root / "supplemental/cohort_arrays.npz",
                "existing_moa_protocol": cohort_root / "protocol.json", "script": Path(__file__),
                "stage_helper": Path(__file__).with_name("run_dimension_moa_probe.py"),
                "fit_helper": Path(__file__).with_name("run_moa_linear_probe.py"),
                "metrics_helper": Path(__file__).with_name("run_expanded_moa_probe.py"),
                "cohort_helper": Path(__file__).with_name("run_finetune_moa_probe.py")}
    protocol = {"experiment": "direct image1920-to256 and text512-to256 versus original hidden160/output128 on frozen DINO and BiomedCLIP text",
                "bindings": {key: {"path": str(path), "sha256": hash_file(path)} for key, path in bindings.items()},
                "representations": specs, "primary_labels": PRIMARY_LABELS, "exploratory_labels": EXPLORATORY_LABELS,
                "linear_probe": old_protocol["linear_probe"],
                "split_sample_ids": {key: cohort["sample_ids"][rows].tolist() for key, rows in splits.items()},
                "supplemental_sample_ids": supplemental["sample_ids"].tolist(),
                "primary_endpoint": "fixed two-class macroAP on original66 and retained67 separately",
                "exploratory_endpoint": "fixed eight labels; unsupported fullmacro stays null",
                "test_used_for_model_selection": False,
                "moa_labels_used_for_clip_training": False,
                "default_original_unchanged": True,
                "comparison_scope": "unchanged frozen image/text feature banks and original603/66/66 plus retained67 cohorts; direct256 versus original128 at batch64/lr1e-4/300epochs; no test-based default promotion"}
    output.mkdir(parents=True, exist_ok=True)
    _write_bound(output / "protocol.json", protocol)
    identity = {"protocol_sha256": hash_file(output / "protocol.json")}
    _write_bound(output / "identity.json", identity)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    train, val, test = (splits[name] for name in ("train", "val", "test"))
    train_store = {key: value[train] for key, value in stores["primary"].items()}
    val_store = {key: value[val] for key, value in stores["primary"].items()}
    all_labels = cohort["label_names"].tolist()
    selected = []
    # This entire loop is restricted to training and validation feature/target rows.
    for label_set, labels in (("primary", PRIMARY_LABELS), ("exploratory", EXPLORATORY_LABELS)):
        columns = [all_labels.index(label) for label in labels]
        for spec in specs:
            run_id = label_set + "__" + spec["key"]
            path = model_dir / f"{run_id}.joblib"
            audit_path = path.with_suffix(".json")
            if audit_path.exists():
                audit = read_json(audit_path)
                if audit["identity"] != identity or audit["model_sha256"] != hash_file(path):
                    raise ValueError("Saved direct256 comparison readout identity changed")
            else:
                with threadpool_limits(limits=threads):
                    model, fit = fit_direct256_readout(spec, train_store, cohort["sources"][train],
                        cohort["y_all"][np.ix_(train, columns)], val_store, cohort["sources"][val],
                        cohort["y_all"][np.ix_(val, columns)], protocol["linear_probe"]["C_grid"])
                model.update(id=run_id, labels=labels)
                joblib.dump(model, path)
                audit = {**spec, "id": run_id, "identity": identity, "representation": spec["key"],
                         "label_set": label_set, "labels": labels, "n_train": len(train),
                         "model_path": str(path.relative_to(output)), "model_sha256": hash_file(path),
                         "train_sample_ids": cohort["sample_ids"][train].tolist(),
                         "validation_sample_ids": cohort["sample_ids"][val].tolist(),
                         "test_used_for_model_selection": False, **fit}
                write_json(audit_path, audit)
            selected.append(audit)
            print(f"SELECTED direct256MOA {len(selected)}/{2 * len(specs)} {run_id} C={audit['selected_C']} valAP={audit['validation_macro_ap']:.6f}", flush=True)
    seal_selection(output, identity, selected)
    # Read back the complete, immutable seal before any test predictions are made.
    selected, seal = load_sealed_selection(output, identity)
    predictions = {"sample_ids": cohort["sample_ids"], "splits": cohort["splits"]}
    extra_predictions = {"sample_ids": supplemental["sample_ids"], "splits": supplemental["splits"]}
    results = []
    for audit in selected:
        path = output / audit["model_path"]
        if hash_file(path) != seal["models"][audit["id"]]:
            raise ValueError("Direct256 comparison readout changed after selection seal")
        model = joblib.load(path)
        with threadpool_limits(limits=threads):
            probability = predict_dimension_saved(model, stores["primary"], cohort["sources"])
            extra = predict_dimension_saved(model, stores["supplemental"], supplemental["sources"])
        columns = [all_labels.index(label) for label in audit["labels"]]
        extra_columns = [supplemental["label_names"].tolist().index(label) for label in audit["labels"]]
        validation = safe_metrics(cohort["y_all"][np.ix_(val, columns)], probability[val], audit["labels"])
        if not np.isclose(validation["macro_ap"], audit["validation_macro_ap"], atol=1e-12, rtol=0):
            raise ValueError("Reloaded direct256 comparison readout changed validation selection")
        predictions[audit["id"]] = probability
        extra_predictions[audit["id"]] = extra
        results.append({**audit, "metrics": {"val": validation,
                        "test": safe_metrics(cohort["y_all"][np.ix_(test, columns)], probability[test], audit["labels"]),
                        "supplemental": safe_metrics(supplemental["y_all"][:, extra_columns], extra, audit["labels"])}})
    np.savez_compressed(output / "probabilities.npz", **predictions)
    np.savez_compressed(output / "supplemental_probabilities.npz", **extra_predictions)
    result = {"identity": identity, "selection_seal_sha256": hash_file(output / "selection_seal.json"),
              "n_runs": len(results), "n_train": len(train), "n_val": len(val), "n_test": len(test),
              "n_supplemental": len(supplemental["sample_ids"]), "representations": specs,
              "probability_archives": {name: hash_file(output / name) for name in (
                  "probabilities.npz", "supplemental_probabilities.npz")},
              "scope": protocol["comparison_scope"], "runs": results}
    write_json(output / "results.json", result)
    print(f"Completed {len(results)} direct256 comparison MOA readouts after all selections were sealed", flush=True)
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=root / "artifacts/direct256_v1/representations")
    parser.add_argument("--cohort-root", type=Path, default=root / "artifacts/expanded_moa_probe_v1")
    parser.add_argument("--output", type=Path, default=root / "artifacts/direct256_v1/moa")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.export_dir, args.cohort_root, args.output, args.threads)

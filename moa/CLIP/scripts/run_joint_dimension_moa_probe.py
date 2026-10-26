"""Validation-selected MOA probes for the paired128/512 joint DINO pilot.

Run in the repository sklearn environment. All54 readouts are sealed before any
test predictions. The fixed established labels, cohorts and C grid are reused.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from run_dimension_moa_probe import (fit_dimension_readout, predict_dimension_saved,
                                     _check_samples, validate_store)
from run_expanded_moa_probe import load_cohort, safe_metrics
from run_finetune_moa_probe import EXPLORATORY_LABELS, PRIMARY_LABELS, validate_cohorts
from run_moa_linear_probe import hash_file, read_json, write_json


def _write_bound(path: Path, value: dict) -> None:
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Existing joint-dimension MOA binding changed: {path}")
    write_json(path, value)


def feature_specs(export_dir: Path) -> list[dict]:
    specs = read_json(export_dir / "features.json")
    if len({row["key"] for row in specs}) != len(specs):
        raise ValueError("Duplicate joint-dimension representation key")
    raw = [row for row in specs if row["stage"] == "raw"]
    if len(raw) != 1 or raw[0]["key"] != "dino1920":
        raise ValueError("Joint pilot requires the fixed original DINO1920 reference")
    runs = read_json(export_dir / "runs.json")
    expected = {f"{mode}_d{dimension}_{alignment}"
                for mode in ("frozen", "joint") for dimension in (128, 512)
                for alignment in ("matched", "shuffled")}
    if len(runs) != 8 or {row["name"] for row in runs} != expected:
        raise ValueError("All eight selected models are required before MOA fitting")
    expected_features = {"dino1920"} | {name + "__" + stage for name in expected
                                         for stage in ("pre1920", "hidden160", "embedding")}
    if {row["key"] for row in specs} != expected_features:
        raise ValueError("Complete pre/hidden/post features are required for every selected arm")
    by_name = {row["name"]: row for row in runs}
    for spec in specs:
        if spec["stage"] == "raw":
            if spec["dimension"] != 1920:
                raise ValueError("Original DINO reference must remain1920-dimensional")
            continue
        arm = spec.get("arm", spec.get("run_name"))
        run = by_name[arm]
        dimension = {"pre1920": 1920, "hidden160": 160, "embedding": run["embedding_dim"]}[spec["stage"]]
        if spec["dimension"] != dimension or spec["key"] != arm + "__" + spec["stage"]:
            raise ValueError("Joint feature stage/dimension metadata differs from its arm")
        for key in ("mode", "embedding_dim", "channel_dim", "alignment", "seed"):
            if spec.get(key) != run.get(key):
                raise ValueError(f"Joint feature {key} differs from selected run")
    specs += [{"key": f"dino_pca{dimension}", "bank": "dino1920", "stage": "pca", "dimension": dimension}
              for dimension in (128, 512)]
    return specs


def fit_joint_readout(spec: dict, train_store: dict, train_sources: np.ndarray,
                      y_train: np.ndarray, val_store: dict, val_sources: np.ndarray,
                      y_val: np.ndarray, c_grid: list[float]) -> tuple[dict, dict]:
    """Fit one PCA only, without invoking legacy reserved-name dispatch.

    ``dino_pca128`` is a public control name in this experiment and a special
    instruction in the legacy linear-probe helper. Its projected matrix must
    therefore use a neutral internal name when it reaches that helper. The
    saved input_pca transform carries the alias through prediction/reloading.
    Public result and audit representation keys remain unchanged.
    """
    if spec["stage"] != "pca":
        return fit_dimension_readout(spec, train_store, train_sources, y_train,
                                      val_store, val_sources, y_val, c_grid)
    internal = {**spec, "key": f"joint_projected_control_{int(spec['dimension'])}"}
    model, audit = fit_dimension_readout(internal, train_store, train_sources,
                                         y_train, val_store, val_sources, y_val, c_grid)
    model["requested_representation"] = spec["key"]
    audit["pca_internal_feature"] = internal["key"]
    return model, audit


def run(export_dir: Path, cohort_root: Path, output: Path, threads: int = 2) -> dict:
    export_dir, cohort_root, output = export_dir.resolve(), cohort_root.resolve(), output.resolve()
    complete = read_json(export_dir / "complete.json")
    if complete["binding_sha256"] != hash_file(export_dir / "binding.json"):
        raise ValueError("Joint-dimension representation binding changed")
    if complete.get("moa_labels_used_for_export") is not False:
        raise ValueError("CLIP representations must not use MOA targets")
    for name, digest in complete["artifacts"].items():
        if hash_file(export_dir / name) != digest:
            raise ValueError(f"Joint representation artifact changed: {name}")
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
                "cohort": cohort_root / "cohort_arrays.npz",
                "supplemental_cohort": cohort_root / "supplemental/cohort_arrays.npz",
                "existing_moa_protocol": cohort_root / "protocol.json", "script": Path(__file__),
                "pca_helper": Path(__file__).with_name("run_dimension_moa_probe.py"),
                "fit_helper": Path(__file__).with_name("run_moa_linear_probe.py"),
                "metrics_helper": Path(__file__).with_name("run_expanded_moa_probe.py"),
                "cohort_helper": Path(__file__).with_name("run_finetune_moa_probe.py")}
    protocol = {"experiment": "paired output128/512 joint-last2 pilot with same-schedule frozen controls",
                "bindings": {key: {"path": str(path), "sha256": hash_file(path)} for key, path in bindings.items()},
                "representations": specs, "primary_labels": PRIMARY_LABELS, "exploratory_labels": EXPLORATORY_LABELS,
                "linear_probe": old_protocol["linear_probe"],
                "split_sample_ids": {key: cohort["sample_ids"][rows].tolist() for key, rows in splits.items()},
                "supplemental_sample_ids": supplemental["sample_ids"].tolist(),
                "pca": "train-only raw standardization, full-SVD PCA128/512, train-only poststandardization",
                "primary_endpoint": "fixed two-class macroAP on original66 and retained67 separately",
                "exploratory_endpoint": "fixed eight labels; unsupported fullmacro stays null",
                "test_used_for_model_selection": False,
                "comparison_scope": "same pretrained DINO and symmetrically pretrained dimension-specific heads; joint/frozen continuation recipe identical apart from DINO gradients"}
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
    for label_set, labels in (("primary", PRIMARY_LABELS), ("exploratory", EXPLORATORY_LABELS)):
        columns = [all_labels.index(label) for label in labels]
        for spec in specs:
            run_id = label_set + "__" + spec["key"]
            path = model_dir / f"{run_id}.joblib"
            audit_path = path.with_suffix(".json")
            if audit_path.exists():
                audit = read_json(audit_path)
                if audit["identity"] != identity or audit["model_sha256"] != hash_file(path):
                    raise ValueError("Saved joint-dimension readout identity changed")
            else:
                with threadpool_limits(limits=threads):
                    model, fit = fit_joint_readout(spec, train_store, cohort["sources"][train],
                        cohort["y_all"][np.ix_(train, columns)], val_store, cohort["sources"][val],
                        cohort["y_all"][np.ix_(val, columns)], protocol["linear_probe"]["C_grid"])
                model.update(id=run_id, labels=labels)
                joblib.dump(model, path)
                audit = {"id": run_id, "identity": identity, "representation": spec["key"], **spec,
                         "label_set": label_set, "labels": labels, "n_train": len(train),
                         "model_path": str(path.relative_to(output)), "model_sha256": hash_file(path),
                         "train_sample_ids": cohort["sample_ids"][train].tolist(),
                         "validation_sample_ids": cohort["sample_ids"][val].tolist(),
                         "test_used_for_model_selection": False, **fit}
                write_json(audit_path, audit)
            selected.append(audit)
            print(f"SELECTED jointMOA {len(selected)}/{2 * len(specs)} {run_id} C={audit['selected_C']} valAP={audit['validation_macro_ap']:.6f}", flush=True)
    _write_bound(output / "selection.json", {"identity": identity, "runs": selected})
    seal = {"identity": identity, "selection_sha256": hash_file(output / "selection.json"),
            "all_readouts_selected_before_test_prediction": True, "n_readouts": len(selected),
            "models": {row["id"]: row["model_sha256"] for row in selected}}
    _write_bound(output / "selection_seal.json", seal)
    predictions = {"sample_ids": cohort["sample_ids"], "splits": cohort["splits"]}
    extra_predictions = {"sample_ids": supplemental["sample_ids"], "splits": supplemental["splits"]}
    results = []
    for audit in selected:
        path = output / audit["model_path"]
        if hash_file(path) != seal["models"][audit["id"]]:
            raise ValueError("Joint readout changed after selection seal")
        model = joblib.load(path)
        with threadpool_limits(limits=threads):
            probability = predict_dimension_saved(model, stores["primary"], cohort["sources"])
            extra = predict_dimension_saved(model, stores["supplemental"], supplemental["sources"])
        columns = [all_labels.index(label) for label in audit["labels"]]
        extra_columns = [supplemental["label_names"].tolist().index(label) for label in audit["labels"]]
        validation = safe_metrics(cohort["y_all"][np.ix_(val, columns)], probability[val], audit["labels"])
        if not np.isclose(validation["macro_ap"], audit["validation_macro_ap"], atol=1e-12, rtol=0):
            raise ValueError("Reloaded joint readout changed validation selection")
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
    print(f"Completed {len(results)} joint128/512 MOA readouts after all selections were sealed", flush=True)
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=root / "artifacts/joint_dim512_v1/representations")
    parser.add_argument("--cohort-root", type=Path, default=root / "artifacts/expanded_moa_probe_v1")
    parser.add_argument("--output", type=Path, default=root / "artifacts/joint_dim512_v1/moa")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.export_dir, args.cohort_root, args.output, args.threads)

"""Stage-wise MOA probes for fixed-backbone CLIP projection capacity.

Run in the repository sklearn environment. Every standardizer/PCA is fit using
training wells only; all readout hyperparameters are sealed before test scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from run_expanded_moa_probe import load_cohort, safe_metrics
from run_finetune_moa_probe import (EXPLORATORY_LABELS, PRIMARY_LABELS,
                                   validate_cohorts)
from run_moa_linear_probe import (fit_and_select, hash_file, predict_saved,
                                  read_json, write_json)


def _write_bound(path: Path, value: dict) -> None:
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Existing dimension MOA binding changed: {path}")
    write_json(path, value)


def feature_specs(export_dir: Path) -> list[dict]:
    specs = read_json(export_dir / "features.json")
    keys = [row["key"] for row in specs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate exported representation key")
    banks = [row["bank"] for row in specs if row["stage"] == "raw"]
    specs += [{"key": f"{bank}__pca{dimension}", "bank": bank, "stage": "pca", "dimension": dimension}
              for bank in banks for dimension in (128, 256, 512)]
    return specs


def validate_store(store: dict, cohort: dict, specs: list[dict]) -> dict:
    store = dict(store)
    if store.pop("sample_ids").tolist() != cohort["sample_ids"].tolist():
        raise ValueError("Dimension feature rows differ from frozen cohort")
    for spec in specs:
        if spec["stage"] == "pca":
            continue
        value = store.get(spec["key"])
        if value is None or value.shape != (len(cohort["sample_ids"]), spec["dimension"]) or not np.isfinite(value).all():
            raise ValueError(f"Invalid representation: {spec['key']}")
    return store


def fit_dimension_readout(spec: dict, train_store: dict, train_sources: np.ndarray,
                          y_train: np.ndarray, val_store: dict, val_sources: np.ndarray,
                          y_val: np.ndarray, c_grid: list[float]) -> tuple[dict, dict]:
    """No test arguments. PCA happens before the existing post-PCA scaler."""
    if spec["stage"] != "pca":
        return fit_and_select(spec["key"], train_store, train_sources, y_train,
                              val_store, val_sources, y_val, c_grid)
    raw = np.asarray(train_store[spec["bank"]], dtype=float)
    dimension = int(spec["dimension"])
    if dimension < 1 or dimension > min(raw.shape[1], len(raw) - 1):
        raise ValueError("PCA dimension must fit only the training matrix")
    scaler = StandardScaler().fit(raw)
    pca = PCA(n_components=dimension, svd_solver="full").fit(scaler.transform(raw))
    train_projected = {spec["key"]: pca.transform(scaler.transform(raw))}
    val_projected = {spec["key"]: pca.transform(scaler.transform(val_store[spec["bank"]]))}
    model, audit = fit_and_select(spec["key"], train_projected, train_sources, y_train,
                                  val_projected, val_sources, y_val, c_grid)
    model["input_pca"] = {"feature": spec["bank"], "output_feature": spec["key"],
                           "scaler": scaler, "pca": pca}
    audit.update(pca_fit_split="train", pca_n_fit_rows=len(raw), n_components=dimension,
                 pca_input_dimension=raw.shape[1],
                 pca_explained_variance_ratio_sum=float(pca.explained_variance_ratio_.sum()))
    return model, audit


def predict_dimension_saved(model: dict, store: dict, sources: np.ndarray) -> np.ndarray:
    if "input_pca" in model:
        transform = model["input_pca"]
        projected = transform["pca"].transform(transform["scaler"].transform(store[transform["feature"]]))
        store = {transform["output_feature"]: projected}
    return predict_saved(model, store, sources)


def _check_samples(samples: dict, cohort: dict, supplemental: dict) -> None:
    for scope, rows in (("primary", cohort), ("supplemental", supplemental)):
        for field, key in (("sample_id", "sample_ids"), ("well_id", "well_ids"),
                           ("parent_id", "parent_ids"), ("source", "sources"), ("plate", "plates")):
            if [row[field] for row in samples[scope]] != rows[key].tolist():
                raise ValueError(f"Exported {scope} {field} differs from MOA cohort")


def run(export_dir: Path, cohort_root: Path, output: Path, threads: int = 2) -> dict:
    export_dir, cohort_root, output = export_dir.resolve(), cohort_root.resolve(), output.resolve()
    complete = read_json(export_dir / "complete.json")
    if complete["binding_sha256"] != hash_file(export_dir / "binding.json"):
        raise ValueError("Dimension representation binding changed")
    for name, digest in complete["artifacts"].items():
        if hash_file(export_dir / name) != digest:
            raise ValueError(f"Dimension export artifact changed: {name}")
    old_protocol = read_json(cohort_root / "protocol.json")
    if old_protocol["primary_labels"] != PRIMARY_LABELS or old_protocol["exploratory_labels"] != EXPLORATORY_LABELS:
        raise ValueError("Established MOA vocabularies changed")
    cohort = load_cohort(cohort_root / "cohort_arrays.npz")
    supplemental = load_cohort(cohort_root / "supplemental/cohort_arrays.npz")
    splits = validate_cohorts(cohort, supplemental, {"train": 603, "val": 66, "test": 66})
    if len(supplemental["sample_ids"]) != 67:
        raise ValueError("Retained supplemental cohort must stay67")
    _check_samples(read_json(export_dir / "samples.json"), cohort, supplemental)
    specs = feature_specs(export_dir)
    stores = {scope: validate_store(load_cohort(export_dir / f"{scope}_store.npz"), rows, specs)
              for scope, rows in (("primary", cohort), ("supplemental", supplemental))}
    bindings = {"export_complete": export_dir / "complete.json", "export_binding": export_dir / "binding.json",
                "features": export_dir / "features.json", "runs": export_dir / "runs.json",
                "cohort": cohort_root / "cohort_arrays.npz",
                "supplemental_cohort": cohort_root / "supplemental/cohort_arrays.npz",
                "existing_moa_protocol": cohort_root / "protocol.json", "script": Path(__file__),
                "fit_helper": Path(__file__).with_name("run_moa_linear_probe.py"),
                "metrics_helper": Path(__file__).with_name("run_expanded_moa_probe.py"),
                "cohort_helper": Path(__file__).with_name("run_finetune_moa_probe.py")}
    protocol = {"experiment": "fixed-input projection width/output dimension diagnostic",
                "bindings": {key: {"path": str(path), "sha256": hash_file(path)} for key, path in bindings.items()},
                "representations": specs, "primary_labels": PRIMARY_LABELS, "exploratory_labels": EXPLORATORY_LABELS,
                "linear_probe": old_protocol["linear_probe"],
                "split_sample_ids": {key: cohort["sample_ids"][rows].tolist() for key, rows in splits.items()},
                "supplemental_sample_ids": supplemental["sample_ids"].tolist(),
                "pca": "train-only raw standardization, full-SVD PCA to exact requested dimension, train-only poststandardization",
                "primary_endpoint": "fixed two-class macroAP on original66 and retained67 separately",
                "exploratory_endpoint": "fixed eight labels, unsupported fullmacro stays null",
                "test_used_for_model_selection": False,
                "comparison_scope": "heads retrained identically on each fixed feature bank; last2 bank inherited prior correct-caption training"}
    output.mkdir(parents=True, exist_ok=True)
    _write_bound(output / "protocol.json", protocol)
    identity = {"protocol_sha256": hash_file(output / "protocol.json")}
    _write_bound(output / "identity.json", identity)
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    train, val, test = (splits[name] for name in ("train", "val", "test"))
    train_store = {key: value[train] for key, value in stores["primary"].items()}
    val_store = {key: value[val] for key, value in stores["primary"].items()}
    labels_all = cohort["label_names"].tolist()
    selected = []
    for label_set, labels in (("primary", PRIMARY_LABELS), ("exploratory", EXPLORATORY_LABELS)):
        columns = [labels_all.index(label) for label in labels]
        for spec in specs:
            run_id = label_set + "__" + spec["key"]
            path = model_dir / f"{run_id}.joblib"
            audit_path = path.with_suffix(".json")
            if audit_path.exists():
                audit = read_json(audit_path)
                if audit["identity"] != identity or audit["model_sha256"] != hash_file(path):
                    raise ValueError("Saved dimension readout identity changed")
            else:
                with threadpool_limits(limits=threads):
                    model, fit = fit_dimension_readout(spec, train_store, cohort["sources"][train],
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
            print(f"SELECTED dimensionMOA {len(selected)}/{2 * len(specs)} {run_id} C={audit['selected_C']} valAP={audit['validation_macro_ap']:.6f}", flush=True)
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
            raise ValueError("Dimension readout changed after selection seal")
        model = joblib.load(path)
        with threadpool_limits(limits=threads):
            probability = predict_dimension_saved(model, stores["primary"], cohort["sources"])
            extra = predict_dimension_saved(model, stores["supplemental"], supplemental["sources"])
        columns = [labels_all.index(label) for label in audit["labels"]]
        extra_columns = [supplemental["label_names"].tolist().index(label) for label in audit["labels"]]
        validation = safe_metrics(cohort["y_all"][np.ix_(val, columns)], probability[val], audit["labels"])
        if not np.isclose(validation["macro_ap"], audit["validation_macro_ap"], atol=1e-12, rtol=0):
            raise ValueError("Reloaded dimension readout changed validation selection")
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
              "probability_archives": {name: hash_file(output / name) for name in
                  ("probabilities.npz", "supplemental_probabilities.npz")},
              "scope": protocol["comparison_scope"], "runs": results}
    write_json(output / "results.json", result)
    print(f"Completed {len(results)} dimension readouts after all selections were sealed", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--export-dir", type=Path, default=root / "artifacts/projection_dimension_v1/representations")
    parser.add_argument("--cohort-root", type=Path, default=root / "artifacts/expanded_moa_probe_v1")
    parser.add_argument("--output", type=Path, default=root / "artifacts/projection_dimension_v1/moa")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.export_dir, args.cohort_root, args.output, args.threads)

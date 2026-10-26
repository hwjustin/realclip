"""Fixed306 and expanded603 MoA probes on old/new frozen CLIP representations.

Uses the original sklearn fitting helpers; all 88 choices are sealed before
the original or retained supplemental test is scored. No old artifact is written.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from run_moa_linear_probe import fit_and_select, hash_file, predict_saved, read_json, write_json


def safe_metrics(y: np.ndarray, probability: np.ndarray, labels: list[str]) -> dict:
    """Preserve the fixed vocabulary and explicitly mark unsupported classes."""
    rows = []
    for i, label in enumerate(labels):
        positive = int(y[:, i].sum())
        negative = len(y) - positive
        supported = positive > 0 and negative > 0
        rows.append({"label": label, "positive_count": positive, "negative_count": negative,
                     "supported": supported,
                     "average_precision": float(average_precision_score(y[:, i], probability[:, i])) if supported else None,
                     "roc_auc": float(roc_auc_score(y[:, i], probability[:, i])) if supported else None})
    supported = [r for r in rows if r["supported"]]
    complete = len(supported) == len(labels)
    return {"n_samples": len(y), "n_labels": len(labels), "all_labels_supported": complete,
            "unsupported_labels": [r["label"] for r in rows if not r["supported"]],
            "n_supported_labels": len(supported),
            "macro_ap": float(np.mean([r["average_precision"] for r in rows])) if complete else None,
            "micro_ap": float(average_precision_score(y, probability, average="micro")) if y.sum() else None,
            "macro_roc_auc": float(np.mean([r["roc_auc"] for r in rows])) if complete else None,
            "supported_only_macro_ap": float(np.mean([r["average_precision"] for r in supported])) if supported else None,
            "supported_only_macro_scope": [r["label"] for r in supported], "per_label": rows}


def load_cohort(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def stage_indices(cohort: dict, original_samples: list[dict]) -> dict[str, np.ndarray]:
    lookup = {sample: i for i, sample in enumerate(cohort["sample_ids"])}
    fixed = np.asarray([lookup[s["sample_id"]] for s in original_samples if s["split"] == "train"])
    expanded = np.flatnonzero(cohort["splits"] == "train")
    if len(fixed) != 306 or len(expanded) != 603 or not set(fixed) <= set(expanded):
        raise ValueError("Original306 and expanded603 training membership differs from the frozen plan")
    return {"fixed306": fixed, "expanded603": expanded}


def check_original_reproduction(results: list[dict], probabilities: dict, cohort: dict, output: Path) -> dict:
    old_root = output.parent / "moa_probe_v1/linear"
    old_results = read_json(old_root / "results.json")
    old_by_id = {row["id"]: row for row in old_results["runs"]}
    with np.load(old_root / "probabilities.npz", allow_pickle=False) as archive:
        old_ids = archive["sample_ids"].copy()
        old_probabilities = {key: archive[key].copy() for key in archive.files if key.startswith(("primary__", "exploratory__"))}
    lookup = {sample: i for i, sample in enumerate(cohort["sample_ids"])}
    rows = np.asarray([lookup[s] for s in old_ids])
    checked = []
    for result in results:
        if result["stage"] != "fixed306":
            continue
        rep = result["representation"]
        if rep.startswith("new_") or "plus_new_" in rep:
            continue
        old_rep = rep.replace("dino_plus_old_", "dino_plus_").removeprefix("old_")
        old_id = f"{result['label_set']}__full__{old_rep}"
        old = old_by_id[old_id]
        error = float(np.max(np.abs(probabilities[result["probability_key"]][rows] - old_probabilities[old_id])))
        assert result["selected_C"] == old["selected_C"]
        assert result["validation_macro_ap"] == old["validation_macro_ap"]
        assert result["metrics"]["test"]["macro_ap"] == old["metrics"]["test"]["macro_ap"]
        assert error < 1e-10, f"Old prediction drift {result['id']}: {error}"
        checked.append({"id": result["id"], "old_id": old_id,
                        "max_probability_error_all_original438": error,
                        "selected_C_validation_and_test_AP_exactly_equal": True})
    assert len(checked) == 26
    return {"n_reproduced_old_methods": len(checked), "all_exactly_reproduced": True, "runs": checked}


def run(output: Path) -> dict:
    protocol_path = output / "protocol.json"
    protocol = read_json(protocol_path)
    bindings = protocol["bindings"]
    for name, binding in bindings.items():
        assert hash_file(binding["path"]) == binding["sha256"], f"Frozen expanded MoA input changed: {name}"
    precomputed = read_json(output / "precomputed/complete.json")
    assert precomputed["protocol_sha256"] == hash_file(protocol_path)
    stores = {}
    for scope in ["primary", "supplemental"]:
        path = output / f"precomputed/{scope}_store.npz"
        assert hash_file(path) == precomputed[f"{scope}_store_sha256"]
        stores[scope] = load_cohort(path)
    cohort = load_cohort(output / "cohort_arrays.npz")
    supplemental = load_cohort(output / "supplemental/cohort_arrays.npz")
    for scope, rows in [("primary", cohort), ("supplemental", supplemental)]:
        assert stores[scope].pop("sample_ids").tolist() == rows["sample_ids"].tolist()
    original_samples = read_json(bindings["old_manifest"]["path"])["samples"]
    stages = stage_indices(cohort, original_samples)
    lookup = {sample: i for i, sample in enumerate(cohort["sample_ids"])}
    val = np.asarray([lookup[s["sample_id"]] for s in original_samples if s["split"] == "val"])
    test = np.asarray([lookup[s["sample_id"]] for s in original_samples if s["split"] == "test"])
    assert len(val) == len(test) == 66
    directory = output / "linear"
    directory.mkdir(exist_ok=True)
    model_directory = directory / "models"
    model_directory.mkdir(exist_ok=True)
    identity = {"protocol_sha256": hash_file(protocol_path), "precomputed_sha256": hash_file(output / "precomputed/complete.json"),
                "script_sha256": hash_file(__file__), "fit_helper_sha256": bindings["fit_helper"]["sha256"]}
    identity_path = directory / "identity.json"
    if identity_path.exists():
        assert read_json(identity_path) == identity
    write_json(identity_path, identity)
    selected = []
    all_labels = cohort["label_names"].tolist()
    supplemental_labels = supplemental["label_names"].tolist()
    for stage, train in stages.items():
        train_store = {key: value[train] for key, value in stores["primary"].items()}
        val_store = {key: value[val] for key, value in stores["primary"].items()}
        for label_set in ["primary", "exploratory"]:
            labels = protocol[f"{label_set}_labels"]
            columns = [all_labels.index(label) for label in labels]
            for representation in protocol["representations"]:
                run_id = f"{stage}__{label_set}__{representation}"
                model_path = model_directory / f"{run_id}.joblib"
                audit_path = model_path.with_suffix(".json")
                if model_path.exists() and audit_path.exists():
                    audit = read_json(audit_path)
                    assert audit["identity"] == identity and hash_file(model_path) == audit["model_sha256"]
                else:
                    model, fit = fit_and_select(representation, train_store, cohort["sources"][train],
                                                cohort["y_all"][np.ix_(train, columns)], val_store, cohort["sources"][val],
                                                cohort["y_all"][np.ix_(val, columns)], protocol["linear_probe"]["C_grid"])
                    model.update(labels=labels, id=run_id)
                    joblib.dump(model, model_path)
                    audit = {"id": run_id, "identity": identity, "stage": stage, "label_set": label_set,
                             "representation": representation, "n_train": len(train),
                             "train_sample_ids": cohort["sample_ids"][train].tolist(),
                             "validation_sample_ids": cohort["sample_ids"][val].tolist(),
                             "labels": labels, **fit, "model_path": str(model_path.relative_to(directory)),
                             "model_sha256": hash_file(model_path), "probability_key": run_id,
                             "test_used_for_model_selection": False}
                    write_json(audit_path, audit)
                selected.append(audit)
                print(f"SELECTED {len(selected)}/88 {run_id} C={audit['selected_C']} valAP={audit['validation_macro_ap']:.6f}", flush=True)
    assert len(selected) == 88
    write_json(directory / "selection.json", {"identity": identity, "runs": selected})
    seal = {"identity": identity, "all_88_choices_sealed_before_test_prediction": True,
            "selection_sha256": hash_file(directory / "selection.json"),
            "models": {row["id"]: row["model_sha256"] for row in selected}}
    write_json(directory / "selection_seal.json", seal)
    probabilities = {"sample_ids": cohort["sample_ids"], "splits": cohort["splits"],
                     "primary_labels": np.asarray(protocol["primary_labels"]), "exploratory_labels": np.asarray(protocol["exploratory_labels"])}
    supplemental_probabilities = {"sample_ids": supplemental["sample_ids"], "splits": supplemental["splits"],
                                "primary_labels": np.asarray(protocol["primary_labels"]), "exploratory_labels": np.asarray(protocol["exploratory_labels"])}
    evaluated = []
    for audit in selected:
        assert hash_file(directory / audit["model_path"]) == seal["models"][audit["id"]]
        model = joblib.load(directory / audit["model_path"])
        probability = predict_saved(model, stores["primary"], cohort["sources"])
        extra = predict_saved(model, stores["supplemental"], supplemental["sources"])
        probabilities[audit["probability_key"]] = probability
        supplemental_probabilities[audit["probability_key"]] = extra
        columns = [all_labels.index(label) for label in audit["labels"]]
        extra_columns = [supplemental_labels.index(label) for label in audit["labels"]]
        validation = safe_metrics(cohort["y_all"][np.ix_(val, columns)], probability[val], audit["labels"])
        assert abs(validation["macro_ap"] - audit["validation_macro_ap"]) < 1e-12
        evaluated.append({**audit, "metrics": {"val": validation,
                         "test": safe_metrics(cohort["y_all"][np.ix_(test, columns)], probability[test], audit["labels"]),
                         "supplemental": safe_metrics(supplemental["y_all"][:, extra_columns], extra, audit["labels"])}})
    reproduction = check_original_reproduction(evaluated, probabilities, cohort, output)
    np.savez_compressed(directory / "probabilities.npz", **probabilities)
    np.savez_compressed(directory / "supplemental_probabilities.npz", **supplemental_probabilities)
    result = {"identity": identity, "selection_seal_sha256": hash_file(directory / "selection_seal.json"),
              "probability_archive": "probabilities.npz", "probability_archive_sha256": hash_file(directory / "probabilities.npz"),
              "supplemental_probability_archive": "supplemental_probabilities.npz",
              "supplemental_probability_archive_sha256": hash_file(directory / "supplemental_probabilities.npz"),
              "schema": "All735 or held67 sample_ids/splits rows; each run id names [rows,K] probabilities; fixed primary/exploratory labels specify column order",
              "n_runs": len(evaluated), "original_fixed306_reproduction": reproduction, "runs": evaluated}
    write_json(directory / "results.json", result)
    print(f"Completed {len(evaluated)} probes, 26 old-method results exactly reproduced; held67 eight-class macro unsupported and null", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/expanded_moa_probe_v1")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)

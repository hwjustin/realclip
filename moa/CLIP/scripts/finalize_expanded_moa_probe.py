"""Finish expanded MoA inference after fixing numerical ties in source controls.

The initial 88 selections are retained byte-for-byte under numerical_v1.
Only four source-only controls are validated again using mathematically identical
design rows evaluated once at a fixed one-row batch shape. No image model changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from run_moa_linear_probe import fit_design, hash_file, predict_saved, read_json, transform_design, write_json
from run_expanded_moa_probe import load_cohort, safe_metrics, stage_indices


def stable_source_prediction(model: dict, sources: np.ndarray) -> np.ndarray:
    """Evaluate each exact source-only design row once, always as a single row."""
    design = transform_design(model, {}, np.asarray(sources))
    unique, inverse = np.unique(design, axis=0, return_inverse=True)
    probability = np.asarray([[float(classifier.predict_proba(row[None, :])[0, 1])
                               for classifier in model["classifiers"]] for row in unique])
    result = probability[inverse]
    for i in range(len(unique)):
        assert np.all(np.ptp(result[inverse == i], axis=0) == 0)
    return result


def fit_stable_source(train_sources: np.ndarray, y_train: np.ndarray,
                      val_sources: np.ndarray, y_val: np.ndarray, c_grid: list[float]) -> tuple[dict, dict]:
    """The correction remains train/validation-only; test inputs are not accepted."""
    design, base = fit_design("source_only", {}, train_sources)
    candidates, best_score, best_c, best_model = [], -float("inf"), None, None
    for c in sorted(c_grid):
        classifiers = [LogisticRegression(C=c, class_weight="balanced", solver="lbfgs", max_iter=4000, tol=1e-7)
                       .fit(design, y_train[:, i]) for i in range(y_train.shape[1])]
        model = {"kind": "logistic", **base, "classifiers": classifiers}
        score = float(average_precision_score(y_val, stable_source_prediction(model, val_sources), average="macro"))
        candidates.append({"C": c, "validation_macro_ap": score,
                           "iterations_per_label": [int(classifier.n_iter_[0]) for classifier in classifiers]})
        if score > best_score:
            best_score, best_c, best_model = score, c, model
    return best_model, {"selected_C": best_c, "validation_macro_ap": best_score, "c_candidates": candidates}


def check_old_non_source_reproduction(evaluated: list[dict], probabilities: dict,
                                     cohort: dict, output: Path) -> dict:
    old_root = output.parent / "moa_probe_v1/linear"
    old_by_id = {r["id"]: r for r in read_json(old_root / "results.json")["runs"]}
    with np.load(old_root / "probabilities.npz", allow_pickle=False) as archive:
        old_ids = archive["sample_ids"].tolist()
        old_probabilities = {r["probability_key"]: archive[r["probability_key"]].copy() for r in old_by_id.values()}
    lookup = {s: i for i, s in enumerate(cohort["sample_ids"])}
    indices = np.asarray([lookup[s] for s in old_ids])
    rows = []
    for r in evaluated:
        if (r["stage"] != "fixed306" or r["representation"] == "source_only"
                or r["representation"].startswith("new_") or "plus_new_" in r["representation"]):
            continue
        old_rep = r["representation"].replace("dino_plus_old_", "dino_plus_").removeprefix("old_")
        old_id = f"{r['label_set']}__full__{old_rep}"
        old = old_by_id[old_id]
        error = float(np.max(np.abs(probabilities[r["probability_key"]][indices] - old_probabilities[old_id])))
        assert r["selected_C"] == old["selected_C"]
        assert r["validation_macro_ap"] == old["validation_macro_ap"]
        assert r["metrics"]["test"]["macro_ap"] == old["metrics"]["test"]["macro_ap"]
        assert error < 1e-10
        rows.append({"id": r["id"], "old_id": old_id, "max_probability_error_all_original438": error,
                     "selected_C_validation_and_test_AP_exactly_equal": True})
    assert len(rows) == 24
    return {"n_reproduced_old_methods": len(rows), "all_non_source_methods_exactly_reproduced": True,
            "source_controls_excluded_from_exact_reproduction": "Four current source-only controls use corrected exact-tie inference; old artifacts are unchanged",
            "runs": rows}


def run(output: Path) -> dict:
    directory = output / "linear"
    backup = directory / "numerical_v1"
    backup.mkdir(exist_ok=True)
    for filename in ["selection.json", "selection_seal.json"]:
        if not (backup / filename).exists():
            shutil.copy2(directory / filename, backup / filename)
    original_selection = read_json(backup / "selection.json")
    original_seal = read_json(backup / "selection_seal.json")
    assert hash_file(backup / "selection.json") == original_seal["selection_sha256"]
    for row in original_selection["runs"]:
        assert hash_file(directory / row["model_path"]) == original_seal["models"][row["id"]]
    protocol = read_json(output / "protocol.json")
    for binding in protocol["bindings"].values():
        assert hash_file(binding["path"]) == binding["sha256"]
    correction_plan = {
        "type": "principled numerical tie correction, independent of test labels or scores",
        "problem_run": "expanded603__exploratory__source_only",
        "observed_problem": "18 identical source_7 validation design rows had two probabilities differing by 1ulp in the tubulin class; batch66 AP 0.03450488331347387 versus batch735/unique-row AP 0.0343006349474608",
        "policy": "For all four source-only controls, evaluate each exact source design row once at fixed batch size one and broadcast; recheck same six C values using validation only, smallest C wins exact ties",
        "unchanged": "All 84 non-source fitted models, original protocol, CLIP heads, cohorts, class sets and old artifacts",
        "original_selection_seal_sha256": hash_file(backup / "selection_seal.json"),
        "protocol_sha256": hash_file(output / "protocol.json"), "script_sha256": hash_file(__file__),
        "training_boundary": "fit_stable_source accepts train/validation sources and targets only; no test inputs",
    }
    correction_path = output / "protocol_numerical_correction.json"
    if correction_path.exists():
        assert read_json(correction_path) == correction_plan
    else:
        write_json(correction_path, correction_plan)
    cohort = load_cohort(output / "cohort_arrays.npz")
    supplemental = load_cohort(output / "supplemental/cohort_arrays.npz")
    stores = {key: load_cohort(output / f"precomputed/{key}_store.npz") for key in ["primary", "supplemental"]}
    for key, rows in [("primary", cohort), ("supplemental", supplemental)]:
        assert stores[key].pop("sample_ids").tolist() == rows["sample_ids"].tolist()
    original_samples = read_json(protocol["bindings"]["old_manifest"]["path"])["samples"]
    stages = stage_indices(cohort, original_samples)
    lookup = {s: i for i, s in enumerate(cohort["sample_ids"])}
    val = np.asarray([lookup[s["sample_id"]] for s in original_samples if s["split"] == "val"])
    test = np.asarray([lookup[s["sample_id"]] for s in original_samples if s["split"] == "test"])
    labels_all = cohort["label_names"].tolist()
    extra_labels_all = supplemental["label_names"].tolist()
    corrected, changes = [], []
    source_directory = directory / "source_numerical_correction"
    source_directory.mkdir(exist_ok=True)
    for old in original_selection["runs"]:
        if old["representation"] != "source_only":
            corrected.append(old)
            continue
        train = stages[old["stage"]]
        columns = [labels_all.index(label) for label in old["labels"]]
        model, fit = fit_stable_source(cohort["sources"][train], cohort["y_all"][np.ix_(train, columns)],
                                       cohort["sources"][val], cohort["y_all"][np.ix_(val, columns)], protocol["linear_probe"]["C_grid"])
        model.update(id=old["id"], labels=old["labels"])
        path = source_directory / f"{old['id']}.joblib"
        joblib.dump(model, path)
        row = {**old, **fit, "model_path": str(path.relative_to(directory)), "model_sha256": hash_file(path),
               "source_numerical_correction": True, "correction_plan_sha256": hash_file(correction_path),
               "numerical_v1_selected_C": old["selected_C"], "numerical_v1_validation_macro_ap": old["validation_macro_ap"]}
        write_json(path.with_suffix(".json"), row)
        corrected.append(row)
        changes.append({"id": row["id"], "old_C": old["selected_C"], "corrected_C": row["selected_C"],
                        "old_validation_macro_ap": old["validation_macro_ap"], "corrected_validation_macro_ap": row["validation_macro_ap"]})
        print(f"SOURCE CORRECTED {row['id']} C {old['selected_C']} -> {row['selected_C']}, val {old['validation_macro_ap']:.8f} -> {row['validation_macro_ap']:.8f}", flush=True)
    assert len(changes) == 4 and len(corrected) == 88
    for r in corrected:
        if r["representation"] != "source_only":
            assert hash_file(directory / r["model_path"]) == original_seal["models"][r["id"]]
    selection = {**original_selection, "runs": corrected, "source_numerical_correction_plan_sha256": hash_file(correction_path)}
    write_json(directory / "selection.json", selection)
    seal = {**original_seal, "selection_sha256": hash_file(directory / "selection.json"),
            "models": {r["id"]: r["model_sha256"] for r in corrected},
            "all_88_choices_sealed_before_test_prediction": True,
            "84_non_source_models_unchanged": True, "original_selection_seal_sha256": hash_file(backup / "selection_seal.json"),
            "numerical_correction_plan_sha256": hash_file(correction_path)}
    write_json(directory / "selection_seal.json", seal)
    # All four corrected source choices and the other 84 unchanged choices are
    # now sealed. No further model fitting or selection occurs below.
    probabilities = {"sample_ids": cohort["sample_ids"], "splits": cohort["splits"],
                     "primary_labels": np.asarray(protocol["primary_labels"]), "exploratory_labels": np.asarray(protocol["exploratory_labels"])}
    extra_probabilities = {"sample_ids": supplemental["sample_ids"], "splits": supplemental["splits"],
                           "primary_labels": np.asarray(protocol["primary_labels"]), "exploratory_labels": np.asarray(protocol["exploratory_labels"])}
    evaluated = []
    for row in corrected:
        assert hash_file(directory / row["model_path"]) == seal["models"][row["id"]]
        model = joblib.load(directory / row["model_path"])
        if row["representation"] == "source_only":
            probability = stable_source_prediction(model, cohort["sources"])
            extra = stable_source_prediction(model, supplemental["sources"])
        else:
            probability = predict_saved(model, stores["primary"], cohort["sources"])
            extra = predict_saved(model, stores["supplemental"], supplemental["sources"])
        probabilities[row["probability_key"]] = probability
        extra_probabilities[row["probability_key"]] = extra
        columns = [labels_all.index(label) for label in row["labels"]]
        extra_columns = [extra_labels_all.index(label) for label in row["labels"]]
        validation = safe_metrics(cohort["y_all"][np.ix_(val, columns)], probability[val], row["labels"])
        assert abs(validation["macro_ap"] - row["validation_macro_ap"]) < 1e-12
        evaluated.append({**row, "metrics": {"val": validation,
                         "test": safe_metrics(cohort["y_all"][np.ix_(test, columns)], probability[test], row["labels"]),
                         "supplemental": safe_metrics(supplemental["y_all"][:, extra_columns], extra, row["labels"])}})
    reproduction = check_old_non_source_reproduction(evaluated, probabilities, cohort, output)
    np.savez_compressed(directory / "probabilities.npz", **probabilities)
    np.savez_compressed(directory / "supplemental_probabilities.npz", **extra_probabilities)
    correction = {"plan_sha256": hash_file(correction_path), "all_84_non_source_models_unchanged": True,
                  "all_source_groups_exactly_tied": True, "source_choices": changes,
                  "initial_seal_sha256": hash_file(backup / "selection_seal.json"),
                  "final_seal_sha256": hash_file(directory / "selection_seal.json"),
                  "finalizer_script_sha256": hash_file(__file__)}
    write_json(directory / "source_numerical_correction.json", correction)
    results = {"identity": original_selection["identity"], "selection_seal_sha256": hash_file(directory / "selection_seal.json"),
               "probability_archive": "probabilities.npz", "probability_archive_sha256": hash_file(directory / "probabilities.npz"),
               "supplemental_probability_archive": "supplemental_probabilities.npz",
               "supplemental_probability_archive_sha256": hash_file(directory / "supplemental_probabilities.npz"),
               "schema": "All735 or held67 sample_ids/splits rows; each run id names [rows,K] probabilities; fixed primary/exploratory labels specify column order",
               "n_runs": 88, "source_numerical_correction": correction,
               "original_fixed306_reproduction": reproduction, "runs": evaluated}
    write_json(directory / "results.json", results)
    print("Finalized 88 probes; 84 image models unchanged, source controls corrected; 24 old non-source methods exactly reproduced", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/expanded_moa_probe_v1")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)

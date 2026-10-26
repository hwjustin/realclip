"""Frozen-representation MoA probes with all choices sealed before test scoring.

Uses the root environment (numpy / sklearn / joblib), never imports PyTorch.
Saved models contain ordinary dictionaries and sklearn objects, so they remain
loadable by another process without relying on __main__ class definitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_blocks(representation: str) -> list[tuple[str, bool]]:
    if representation == "dino_pca128":
        return [("dino", True)]
    if representation.startswith("dino_plus_"):
        return [("dino", False), (representation[len("dino_plus_"):], False)]
    return [(representation, False)]


def onehot(sources: np.ndarray, vocabulary: list[str]) -> np.ndarray:
    return (np.asarray(sources)[:, None] == np.asarray(vocabulary)[None, :]).astype(float)


def fit_design(representation: str, store: dict, sources: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit every transformation using training rows supplied by the caller only."""
    preprocessors, arrays = [], []
    vocabulary = sorted(set(sources.tolist())) if representation == "source_only" else []
    for name, use_pca in feature_blocks(representation):
        raw = onehot(sources, vocabulary) if name == "source_only" else np.asarray(store[name], dtype=float)
        scaler = StandardScaler().fit(raw)
        scaled = scaler.transform(raw)
        pca = post_scaler = None
        if use_pca:
            components = min(128, raw.shape[1], len(raw) - 1)
            if components < 1:
                raise ValueError("PCA requires at least two training rows")
            pca = PCA(n_components=components, svd_solver="full").fit(scaled)
            transformed = pca.transform(scaled)
            post_scaler = StandardScaler().fit(transformed)
            scaled = post_scaler.transform(transformed)
        dimension = scaled.shape[1]
        arrays.append(scaled / np.sqrt(dimension))
        preprocessors.append({"feature": name, "scaler": scaler, "pca": pca,
                              "post_scaler": post_scaler, "dimension": dimension,
                              "input_dimension": raw.shape[1]})
    return np.concatenate(arrays, axis=1) / np.sqrt(len(arrays)), {
        "representation": representation, "source_vocabulary": vocabulary,
        "blocks": preprocessors, "block_count": len(preprocessors),
    }


def transform_design(model: dict, store: dict, sources: np.ndarray) -> np.ndarray:
    arrays = []
    for block in model["blocks"]:
        raw = (onehot(sources, model["source_vocabulary"]) if block["feature"] == "source_only"
               else np.asarray(store[block["feature"]], dtype=float))
        scaled = block["scaler"].transform(raw)
        if block["pca"] is not None:
            scaled = block["post_scaler"].transform(block["pca"].transform(scaled))
        arrays.append(scaled / np.sqrt(block["dimension"]))
    return np.concatenate(arrays, axis=1) / np.sqrt(len(arrays))


def predict_saved(model: dict, store: dict, sources: np.ndarray, indices=None) -> np.ndarray:
    """Predict saved dictionary models on original or new feature rows.

    store keys: dino [n,1920], matched_seed42 etc. [n,128]. sources has
    length n. indices may select any subset without refitting a transformation.
    """
    sources = np.asarray(sources)
    indices = np.arange(len(sources)) if indices is None else np.asarray(indices, dtype=int)
    if model["kind"] == "prevalence":
        return np.tile(model["probability"], (len(indices), 1))
    selected = {key: np.asarray(value)[indices] for key, value in store.items()}
    design = transform_design(model, selected, sources[indices])
    return np.column_stack([classifier.predict_proba(design)[:, 1] for classifier in model["classifiers"]])


def fit_and_select(representation: str, train_store: dict, train_sources: np.ndarray, y_train: np.ndarray,
                   val_store: dict, val_sources: np.ndarray, y_val: np.ndarray,
                   c_grid: list[float]) -> tuple[dict, dict]:
    """No test feature/target arguments; one global C selected per label set."""
    if y_train.ndim != 2 or y_val.shape[1] != y_train.shape[1]:
        raise ValueError("Expected aligned train/validation label vocabularies")
    if not np.all((y_train.sum(axis=0) > 0) & (y_train.sum(axis=0) < len(y_train))):
        raise ValueError("Every training label needs positives and negatives")
    if representation == "prevalence":
        probability = y_train.mean(axis=0)
        model = {"kind": "prevalence", "representation": representation, "probability": probability}
        validation = np.tile(probability, (len(y_val), 1))
        return model, {"selected_C": None,
                       "validation_macro_ap": float(average_precision_score(y_val, validation, average="macro")),
                       "c_candidates": [], "preprocessing": [], "n_components": None}
    design, base = fit_design(representation, train_store, train_sources)
    validation_design = transform_design(base, val_store, val_sources)
    best_score, best_c, best_classifiers = -float("inf"), None, None
    candidates = []
    for c in sorted(c_grid):
        classifiers = [LogisticRegression(C=c, class_weight="balanced", solver="lbfgs",
                                          max_iter=4000, tol=1e-7).fit(design, y_train[:, i])
                       for i in range(y_train.shape[1])]
        probabilities = np.column_stack([classifier.predict_proba(validation_design)[:, 1]
                                          for classifier in classifiers])
        score = float(average_precision_score(y_val, probabilities, average="macro"))
        candidates.append({"C": c, "validation_macro_ap": score,
                           "iterations_per_label": [int(classifier.n_iter_[0]) for classifier in classifiers]})
        if score > best_score:
            best_score, best_c, best_classifiers = score, c, classifiers
    model = {"kind": "logistic", **base, "classifiers": best_classifiers}
    audit = {"selected_C": best_c, "validation_macro_ap": best_score, "c_candidates": candidates,
             "preprocessing": [{"feature": block["feature"], "input_dimension": block["input_dimension"],
                                "output_dimension": block["dimension"], "fit_split": "train",
                                "n_fit_rows": len(y_train), "pca_fit_split": "train" if block["pca"] is not None else None,
                                "pca_explained_variance_ratio_sum": float(block["pca"].explained_variance_ratio_.sum()) if block["pca"] is not None else None}
                               for block in base["blocks"]],
             "n_components": next((block["dimension"] for block in base["blocks"] if block["pca"] is not None), None)}
    return model, audit


def nested_subsets(train_indices: np.ndarray, primary_targets: np.ndarray, seeds: list[int],
                   fractions: list[float]) -> list[dict]:
    minimum = round(len(train_indices) * min(fractions))
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for attempt in range(100000):
            order = rng.permutation(len(train_indices))
            y = primary_targets[order[:minimum]]
            if np.all((y.sum(axis=0) >= 2) & ((len(y) - y.sum(axis=0)) >= 2)):
                break
        else:
            raise RuntimeError("Could not find a low-label training subset with fixed support")
        for fraction in fractions:
            count = round(len(train_indices) * fraction)
            selected = np.sort(train_indices[order[:count]])
            rows.append({"name": f"low_seed{seed}_n{count:03d}", "seed": seed, "fraction": fraction,
                         "indices": selected, "attempts": attempt + 1})
    return rows


def metrics(y: np.ndarray, probability: np.ndarray, labels: list[str]) -> dict:
    aps = [float(average_precision_score(y[:, i], probability[:, i])) for i in range(len(labels))]
    aucs = [float(roc_auc_score(y[:, i], probability[:, i])) for i in range(len(labels))]
    return {"macro_ap": float(np.mean(aps)), "micro_ap": float(average_precision_score(y, probability, average="micro")),
            "macro_roc_auc": float(np.mean(aucs)), "n_samples": len(y),
            "per_label": [{"label": label, "average_precision": ap, "roc_auc": auc,
                           "positive_count": int(y[:, i].sum()), "negative_count": int(len(y) - y[:, i].sum())}
                          for i, (label, ap, auc) in enumerate(zip(labels, aps, aucs))]}


def load_store(work: Path) -> tuple[dict, dict, dict]:
    manifest = read_json(work / "manifest.json")
    metadata = read_json(work / "features/metadata.json")
    assert hash_file(work / "features/features.npz") == metadata["artifacts"]["features.npz"]
    evaluation = read_json(work / "evaluation/evaluation.json")
    seal = read_json(work / "evaluation/complete.json")
    assert hash_file(work / "evaluation/evaluation.json") == seal["evaluation_sha256"]
    with np.load(work / "features/features.npz", allow_pickle=False) as archive:
        ids = archive["sample_ids"].copy()
        store = {"dino": archive["image_features"].reshape(len(ids), -1).astype(float)}
    assert ids.tolist() == [sample["sample_id"] for sample in manifest["samples"]]
    hashes = {"features.npz": hash_file(work / "features/features.npz")}
    for arm in ["matched", "shuffled"]:
        for seed in [42, 43, 44]:
            name = f"{arm}_seed{seed}"
            path = work / f"evaluation/{name}_gallery.npz"
            assert hash_file(path) == evaluation["artifacts"][path.name]
            with np.load(path, allow_pickle=False) as archive:
                assert archive["sample_ids"].tolist() == ids.tolist()
                store[name] = archive["image_embeddings"].astype(float)
            hashes[path.name] = hash_file(path)
    return manifest, store, hashes


def run(work: Path, parent: Path, output: Path) -> dict:
    protocol = read_json(parent / "protocol.json")
    manifest, store, hashes = load_store(work)
    assert hash_file(parent / "cohort_arrays.npz") == protocol["cohort_arrays_sha256"]
    with np.load(parent / "cohort_arrays.npz", allow_pickle=False) as archive:
        sample_ids, splits, sources = archive["sample_ids"].copy(), archive["splits"].copy(), archive["sources"].copy()
        all_labels, y_all = archive["label_names"].tolist(), archive["y_all"].copy()
    assert sample_ids.tolist() == [sample["sample_id"] for sample in manifest["samples"]]
    assert splits.tolist() == [sample["split"] for sample in manifest["samples"]]
    train, val = [np.flatnonzero(splits == name) for name in ["train", "val"]]
    reps = ["dino", "dino_pca128"] + [f"{arm}_seed{seed}" for arm in ["matched", "shuffled"] for seed in [42, 43, 44]]
    reps += [f"dino_plus_{arm}_seed{seed}" for arm in ["matched", "shuffled"] for seed in [42, 43, 44]]
    reps += ["source_only", "prevalence"]
    jobs = [{"label_set": label_set, "subset": "full", "representation": rep, "indices": train,
             "fraction": 1.0, "subset_seed": None} for label_set in ["primary", "exploratory"] for rep in reps]
    low = protocol["low_label_diagnostic"]
    primary_columns = [all_labels.index(label) for label in protocol["primary_labels"]]
    subsets = nested_subsets(train, y_all[np.ix_(train, primary_columns)], low["subset_seeds"], low["fractions"])
    for subset in subsets:
        jobs.extend({"label_set": "primary", "subset": subset["name"], "representation": rep,
                     "indices": subset["indices"], "fraction": subset["fraction"], "subset_seed": subset["seed"]}
                    for rep in low["representations"])
    assert len(jobs) == 68
    output.mkdir(parents=True, exist_ok=True)
    identity = {"protocol_sha256": hash_file(parent / "protocol.json"), "cohort_arrays_sha256": hash_file(parent / "cohort_arrays.npz"),
                "source_artifact_sha256": hashes, "script_sha256": hash_file(__file__)}
    identity_path = output / "identity.json"
    if identity_path.exists():
        assert read_json(identity_path) == identity, "Linear probe identity changed; use a separate output directory"
    write_json(identity_path, identity)
    selected = []
    for job in jobs:
        label_set, subset, rep, indices = [job[key] for key in ["label_set", "subset", "representation", "indices"]]
        run_id = f"{label_set}__{subset}__{rep}"
        directory = output / "models"
        directory.mkdir(exist_ok=True)
        model_path = directory / f"{run_id}.joblib"
        audit_path = directory / f"{run_id}.json"
        labels = protocol[f"{label_set}_labels"]
        columns = [all_labels.index(label) for label in labels]
        train_store = {key: value[indices] for key, value in store.items()}
        val_store = {key: value[val] for key, value in store.items()}
        if model_path.exists() and audit_path.exists():
            audit = read_json(audit_path)
            assert audit["identity"] == identity and hash_file(model_path) == audit["model_sha256"]
        else:
            model, fit = fit_and_select(rep, train_store, sources[indices], y_all[np.ix_(indices, columns)],
                                        val_store, sources[val], y_all[np.ix_(val, columns)], protocol["linear_probe"]["C_grid"])
            model.update(labels=labels, id=run_id)
            joblib.dump(model, model_path)
            audit = {"id": run_id, "identity": identity, "label_set": label_set, "subset": subset,
                     "representation": rep, "n_train": len(indices), "train_sample_ids": sample_ids[indices].tolist(),
                     "validation_sample_ids": sample_ids[val].tolist(), "labels": labels,
                     "subset_seed": job["subset_seed"], "fraction": job["fraction"], **fit,
                     "model_path": str(model_path.relative_to(output)), "model_sha256": hash_file(model_path),
                     "probability_key": run_id, "test_used_for_model_selection": False}
            write_json(audit_path, audit)
        selected.append(audit)
        print(f"SELECTED {len(selected)}/68 {run_id} C={audit['selected_C']} valAP={audit['validation_macro_ap']:.6f}", flush=True)
    selection = {"identity": identity, "runs": selected,
                 "low_label_subsets": [{**{key: value for key, value in subset.items() if key != "indices"},
                                          "sample_ids": sample_ids[subset["indices"]].tolist()} for subset in subsets]}
    write_json(output / "selection.json", selection)
    sealed = {"identity": identity, "all_68_choices_sealed_before_test_prediction": True,
              "selection_sha256": hash_file(output / "selection.json"),
              "models": {row["id"]: row["model_sha256"] for row in selected}}
    write_json(output / "selection_seal.json", sealed)
    # First test scoring begins after the complete set of fitting choices exists.
    test = np.flatnonzero(splits == "test")
    probability_arrays = {"sample_ids": sample_ids, "splits": splits,
                          "primary_labels": np.asarray(protocol["primary_labels"]),
                          "exploratory_labels": np.asarray(protocol["exploratory_labels"])}
    evaluated = []
    for audit in selected:
        assert hash_file(output / audit["model_path"]) == sealed["models"][audit["id"]]
        model = joblib.load(output / audit["model_path"])
        probability = predict_saved(model, store, sources)
        probability_arrays[audit["probability_key"]] = probability
        columns = [all_labels.index(label) for label in audit["labels"]]
        val_metrics = metrics(y_all[np.ix_(val, columns)], probability[val], audit["labels"])
        assert abs(val_metrics["macro_ap"] - audit["validation_macro_ap"]) < 1e-12
        evaluated.append({**audit, "metrics": {"val": val_metrics,
                          "test": metrics(y_all[np.ix_(test, columns)], probability[test], audit["labels"])}})
    np.savez_compressed(output / "probabilities.npz", **probability_arrays)
    result = {"identity": identity, "selection_seal_sha256": hash_file(output / "selection_seal.json"),
              "probability_archive": "probabilities.npz", "probability_archive_sha256": hash_file(output / "probabilities.npz"),
              "probability_schema": "All 438 original rows aligned with sample_ids/splits; each run id names [438,K] probabilities. Primary/exploratory label arrays give column order.",
              "n_full_data_runs": 32, "n_low_label_runs": 36, "runs": evaluated}
    write_json(output / "results.json", result)
    print(f"Completed {len(evaluated)} MoA probes; all validation selections exactly reproduced after serialization", flush=True)
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "artifacts"
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=root / "prototype_v1")
    parser.add_argument("--parent", type=Path, default=root / "moa_probe_v1")
    parser.add_argument("--output", type=Path, default=root / "moa_probe_v1/linear")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.work, args.parent, args.output)

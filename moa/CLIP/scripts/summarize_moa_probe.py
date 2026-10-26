"""Paired, plate-clustered MoA comparisons and frozen supplemental inference."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wellclip.common import hash_file, read_json, utc_now, write_json


def weighted_ap(y: np.ndarray, score: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Threshold AP with tied scores, for many paired bootstrap weight vectors."""
    order = np.argsort(-score, kind="stable")
    ends = np.flatnonzero(np.r_[np.diff(score[order]) != 0, True])
    weighted_y = weights[:, order] * y[order][None]
    tp = np.cumsum(weighted_y, axis=1)[:, ends]
    total = np.cumsum(weights[:, order], axis=1)[:, ends]
    precision = np.divide(tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0)
    increments = np.diff(tp, axis=1, prepend=0)
    return (precision * increments).sum(axis=1) / weighted_y.sum(axis=1)


def bootstrap_weights(groups: np.ndarray, draws: int, seed: int) -> tuple[np.ndarray, int]:
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, len(unique), size=(draws, len(unique)))
    counts = np.zeros((draws, len(unique)), dtype=np.int16)
    np.add.at(counts, (np.arange(draws)[:, None], chosen), 1)
    return counts[:, inverse], len(unique)


def metrics(y, p):
    aps = [float(average_precision_score(y[:, j], p[:, j])) if y[:, j].sum() else None for j in range(y.shape[1])]
    aucs = [float(roc_auc_score(y[:, j], p[:, j])) if len(np.unique(y[:, j])) == 2 else None for j in range(y.shape[1])]
    return {"macro_ap": float(np.mean([v for v in aps if v is not None])), "per_label_ap": aps,
            "micro_ap": float(average_precision_score(y.ravel(), p.ravel())),
            "macro_roc_auc": float(np.mean([v for v in aucs if v is not None])) if any(v is not None for v in aucs) else None,
            "positive_counts": y.sum(axis=0).astype(int).tolist(), "n": len(y)}


def model_family(name):
    if name.startswith("dino_plus_matched_"):
        return "fusion_matched"
    if name.startswith("dino_plus_shuffled_"):
        return "fusion_shuffled"
    if name.startswith("matched_seed"):
        return "matched"
    if name.startswith("shuffled_seed"):
        return "shuffled"
    return name


def paired_uncertainty(y, predictions, families, group_ids, draws, seed):
    weights, n_groups = bootstrap_weights(group_ids, draws, seed)
    pos = weights @ y
    neg = weights @ (1 - y)
    keep = np.all((pos > 0) & (neg > 0), axis=1)
    retained = weights[keep].astype(float)
    if not len(retained):
        return {"attempted_draws": draws, "retained_draws": 0, "n_clusters": n_groups}
    distributions = {}
    for name, p in predictions.items():
        distributions[name] = np.mean([weighted_ap(y[:, j], p[:, j], retained) for j in range(y.shape[1])], axis=0)
    grouped = {family: np.mean([distributions[name] for name in names], axis=0) for family, names in families.items()}
    comparisons = [("matched", "dino"), ("matched", "shuffled"), ("matched", "dino_pca128"),
                   ("fusion_matched", "dino"), ("matched", "capacity"), ("fusion_matched", "fusion_shuffled")]
    rows = []
    for a, b in comparisons:
        delta = grouped[a] - grouped[b]
        point = np.mean([metrics(y, predictions[name])["macro_ap"] for name in families[a]]) - np.mean([metrics(y, predictions[name])["macro_ap"] for name in families[b]])
        rows.append({"a": a, "b": b, "observed_macro_ap_difference": float(point),
                     "conditional_paired_bootstrap_95_interval": np.quantile(delta, [.025, .975]).tolist()})
    return {"attempted_draws": draws, "retained_draws": int(keep.sum()), "excluded_missing_class_support": int((~keep).sum()),
            "n_clusters": n_groups, "comparisons": rows,
            "limitation": "Only draws with positives and negatives for every fixed class contribute. Interval conditions on this support, fixed models and this cohort; excludes retraining/new splits. Exploratory and no multiplicity correction."}


def supplemental_linear(output):
    from run_moa_linear_probe import predict_saved
    directory, sup = output / "linear", output / "supplemental"
    selection = read_json(directory / "selection.json")
    seal = read_json(directory / "selection_seal.json")
    assert seal.get("selection_sha256", seal.get("selection_json_sha256")) == hash_file(directory / "selection.json")
    with np.load(sup / "features/features.npz", allow_pickle=False) as a:
        ids = a["sample_ids"]
        store = {"dino": a["image_features"].reshape(len(ids), -1).astype(float)}
    with np.load(sup / "projected.npz", allow_pickle=False) as a:
        np.testing.assert_array_equal(a["sample_ids"], ids)
        for arm in ["matched", "shuffled"]:
            for seed in [42, 43, 44]:
                key = f"{arm}_seed{seed}"
                store[key] = a[key].astype(float)
    with np.load(sup / "cohort_arrays.npz", allow_pickle=False) as a:
        np.testing.assert_array_equal(a["sample_ids"], ids)
        sources, y_all, vocabulary = a["sources"], a["y_all"], a["label_names"].tolist()
    protocol = read_json(output / "protocol.json")
    probabilities, scored = {}, []
    for row in selection["runs"]:
        path = directory / row["model_path"]
        assert hash_file(path) == row["model_sha256"]
        model = joblib.load(path)
        p = predict_saved(model, store, sources, np.arange(len(ids)))
        probabilities[row["probability_key"]] = p
        labels = protocol[row["label_set"] + "_labels"]
        y = y_all[:, [vocabulary.index(label) for label in labels]]
        scored.append({"id": row["id"], "metrics": metrics(y, p)})
    np.savez_compressed(sup / "linear_probabilities.npz", sample_ids=ids, **probabilities)
    write_json(sup / "linear_results.json", {"selection_seal_sha256": hash_file(directory / "selection_seal.json"),
                "protocol_sha256": hash_file(output / "protocol.json"), "runs": scored,
                "probabilities_sha256": hash_file(sup / "linear_probabilities.npz"), "no_refit_or_tuning": True})


def summarize(output: Path):
    linear = read_json(output / "linear/results.json")
    protocol = read_json(output / "protocol.json")
    archives = {}
    for dataset, cohort_path, linear_path, capacity_path in [
        ("original_test", output, output / "linear/probabilities.npz", output / "capacity/probabilities.npz"),
        ("new_plates", output / "supplemental", output / "supplemental/linear_probabilities.npz", output / "capacity/supplemental_probabilities.npz")]:
        with np.load(cohort_path / "cohort_arrays.npz", allow_pickle=False) as a:
            cohort = {key: a[key] for key in a.files}
        with np.load(linear_path, allow_pickle=False) as a:
            lp = {key: a[key] for key in a.files}
        with np.load(capacity_path, allow_pickle=False) as a:
            cp = {key: a[key] for key in a.files}
        np.testing.assert_array_equal(lp["sample_ids"], cohort["sample_ids"])
        np.testing.assert_array_equal(cp["sample_ids"], cohort["sample_ids"])
        index = np.flatnonzero(cohort["splits"] == "test") if dataset == "original_test" else np.arange(len(cohort["sample_ids"]))
        vocabulary = cohort["label_names"].tolist()
        result = {"n_wells": len(index), "n_plates": len(set(zip(cohort["sources"][index], cohort["plates"][index]))),
                  "source_counts": {s: int(np.sum(cohort["sources"][index] == s)) for s in np.unique(cohort["sources"][index])},
                  "label_sets": {}, "low_label": []}
        for label_set in ["primary", "exploratory"]:
            labels = protocol[label_set + "_labels"]
            y = cohort["y_all"][index][:, [vocabulary.index(label) for label in labels]]
            predictions, families = {}, {}
            for row in linear["runs"]:
                if row["label_set"] != label_set:
                    continue
                p = lp[row["probability_key"]][index]
                if row["subset"] != "full":
                    result["low_label"].append({"representation": row["representation"], "n_train": row["n_train"],
                                                "subset": row["subset"], "metrics": metrics(y, p)})
                    continue
                name = row["representation"]
                predictions[name] = p
                families.setdefault(model_family(name), []).append(name)
            for seed in [42, 43, 44]:
                name = f"capacity_seed{seed}"
                predictions[name] = cp[f"{label_set}_seed{seed}"][index]
                families.setdefault("capacity", []).append(name)
            values = {}
            for family, names in families.items():
                rows = [metrics(y, predictions[name]) for name in names]
                values[family] = {"n_models": len(names), "models": names,
                    "mean_macro_ap": float(np.mean([row["macro_ap"] for row in rows])),
                    "seed_std_macro_ap": float(np.std([row["macro_ap"] for row in rows], ddof=1)) if len(rows) > 1 else 0.,
                    "per_label_mean_ap": np.mean([row["per_label_ap"] for row in rows], axis=0).tolist(),
                    "run_metrics": dict(zip(names, rows))}
            entry = {"labels": labels, "positive_counts": y.sum(axis=0).astype(int).tolist(),
                     "unique_positive_wells": int(np.any(y, axis=1).sum()), "families": values}
            if label_set == "primary":
                groups = np.asarray([str(s) + "|" + str(p) for s, p in zip(cohort["sources"][index], cohort["plates"][index])])
                entry["plate_bootstrap"] = paired_uncertainty(y, predictions, families, groups, 5000, 20261024)
                entry["well_bootstrap"] = paired_uncertainty(y, predictions, families, cohort["parent_ids"][index], 2000, 20261024)
            result["label_sets"][label_set] = entry
        archives[dataset] = result
    report = {"created_at": utc_now(), "protocol_sha256": hash_file(output / "protocol.json"),
              "script_sha256": hash_file(__file__), "datasets": archives,
              "no_test_tuning": True, "inference_input": "image features only; source-only is an explicitly named nuisance control"}
    write_json(output / "summary.json", report)
    print({dataset: {label_set: {family: round(row["mean_macro_ap"], 4) for family, row in result["families"].items()}
                     for label_set, result in values["label_sets"].items()} for dataset, values in archives.items()})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/moa_probe_v1")
    parser.add_argument("--infer-supplemental", action="store_true")
    args = parser.parse_args()
    if args.infer_supplemental:
        supplemental_linear(args.output)
    summarize(args.output)

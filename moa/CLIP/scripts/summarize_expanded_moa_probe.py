"""Fixed-label, two-stage MoA statistics for the expanded image-text cohort.

No fitting or model selection occurs here. Missing classes never silently shrink
an eight-class macro: aggregate metrics are null and per-class support is retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COMPARISONS = [("new_matched", "old_matched"), ("new_matched", "dino"),
               ("new_matched", "new_shuffled"), ("fusion_new_matched", "dino")]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(path):
    with np.load(path, allow_pickle=False) as a:
        return {key: a[key] for key in a.files}


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def binary_ap(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or y.shape != p.shape:
        raise ValueError("Invalid aligned binary targets or scores")
    n = int(y.sum())
    if not n:
        return None
    order = np.argsort(-p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[order])), len(p) - 1]
    tp = np.cumsum(y[order])[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / n)


def binary_auc(y, p):
    positive, negative = np.asarray(p)[np.asarray(y) == 1], np.asarray(p)[np.asarray(y) == 0]
    if not len(positive) or not len(negative):
        return None
    differences = positive[:, None] - negative[None]
    return float(np.mean((differences > 0) + .5 * (differences == 0)))


def metrics(y, probabilities):
    y, p = np.asarray(y), np.asarray(probabilities)
    if y.ndim != 2 or y.shape != p.shape or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Invalid aligned probability matrix")
    positive = y.sum(axis=0).astype(int)
    negative = len(y) - positive
    complete = bool(np.all((positive > 0) & (negative > 0)))
    aps = [binary_ap(y[:, i], p[:, i]) for i in range(y.shape[1])]
    aucs = [binary_auc(y[:, i], p[:, i]) for i in range(y.shape[1])]
    return {"status": "complete" if complete else "unsupported", "n": len(y),
            "positive_counts": positive.tolist(), "negative_counts": negative.tolist(),
            "macro_ap": float(np.mean(aps)) if complete else None,
            "micro_ap": binary_ap(y.ravel(), p.ravel()) if complete else None,
            "macro_roc_auc": float(np.mean(aucs)) if complete else None,
            "per_label_ap": aps, "per_label_roc_auc": aucs}


def model_family(name):
    aliases = {"dino_pca128": "pca", "source_only": "source", "dino": "dino", "prevalence": "prevalence"}
    if name in aliases:
        return aliases[name]
    if name.startswith("dino_plus_"):
        return "fusion_" + name[len("dino_plus_"):].rsplit("_seed", 1)[0]
    return name.rsplit("_seed", 1)[0]


def primary_model(names):
    if len(names) == 1:
        return names[0]
    chosen = [name for name in names if name.endswith("_seed42")]
    if len(chosen) != 1:
        raise ValueError("Expected the predeclared seed42 model in each seeded family")
    return chosen[0]


def bootstrap_weights(groups, draws, seed):
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(len(unique), size=(draws, len(unique)))
    counts = np.stack([np.bincount(row, minlength=len(unique)) for row in sampled])
    return counts[:, inverse], len(unique)


def weighted_ap(y, p, weights):
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(-p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[order])), len(p) - 1]
    cumulative_positive = np.cumsum(weights[:, order] * y[order], axis=1)[:, ends]
    cumulative_total = np.cumsum(weights[:, order], axis=1)[:, ends]
    precision = np.divide(cumulative_positive, cumulative_total, out=np.zeros_like(cumulative_positive), where=cumulative_total > 0)
    increments = np.diff(cumulative_positive, axis=1, prepend=0)
    return np.sum(increments * precision, axis=1) / cumulative_positive[:, -1]


def paired_bootstrap(y, predictions, families, groups, draws, seed=20261025):
    weights, n_clusters = bootstrap_weights(groups, draws, seed)
    positives, negatives = weights @ y, weights @ (1 - y)
    keep = np.all((positives > 0) & (negatives > 0), axis=1)
    retained = weights[keep].astype(float)
    result = {"attempted_draws": draws, "retained_draws": int(keep.sum()),
              "excluded_missing_class_support": int((~keep).sum()), "n_clusters": n_clusters,
              "seed": seed, "comparisons": [],
              "limitation": "Fixed models and gallery; paired source+plate or parent resampling. Only draws supporting every fixed class enter intervals. Three-head means average APs, not predictions. Exploratory, unadjusted, excludes refitting or new-split uncertainty."}
    if not len(retained):
        return result
    distributions = {name: np.mean([weighted_ap(y[:, j], p[:, j], retained) for j in range(y.shape[1])], axis=0)
                     for name, p in predictions.items()}
    points = {name: metrics(y, p)["macro_ap"] for name, p in predictions.items()}
    family_distributions = {family: np.mean([distributions[name] for name in names], axis=0) for family, names in families.items()}
    family_points = {family: float(np.mean([points[name] for name in names])) for family, names in families.items()}
    for a, b in COMPARISONS:
        pa, pb = primary_model(families[a]), primary_model(families[b])
        result["comparisons"].append({
            "a": a, "b": b, "observed_macro_ap_difference": family_points[a] - family_points[b],
            "conditional_paired_bootstrap_95_interval": np.quantile(family_distributions[a] - family_distributions[b], [.025, .975]).tolist(),
            "seed42_observed_difference": points[pa] - points[pb],
            "seed42_conditional_paired_bootstrap_95_interval": np.quantile(distributions[pa] - distributions[pb], [.025, .975]).tolist()})
    return result


def family_summary(y, predictions, families):
    complete = bool(np.all((y.sum(axis=0) > 0) & (y.sum(axis=0) < len(y))))
    result, seed42 = {}, {}
    for family, names in families.items():
        values = {name: metrics(y, predictions[name]) for name in names}
        mean_per_label = []
        for j in range(y.shape[1]):
            entries = [row["per_label_ap"][j] for row in values.values()]
            mean_per_label.append(float(np.mean(entries)) if all(value is not None for value in entries) else None)
        macro = [row["macro_ap"] for row in values.values()]
        result[family] = {"models": names, "n_models": len(names), "status": "complete" if complete else "unsupported",
                          "mean_macro_ap": float(np.mean(macro)) if complete else None,
                          "seed_std_macro_ap": (float(np.std(macro, ddof=1)) if len(macro) > 1 else 0.) if complete else None,
                          "per_label_mean_ap": mean_per_label, "run_metrics": values}
        selected = primary_model(names)
        seed42[family] = {"model": selected, **values[selected]}
    return result, seed42


def run(output):
    protocol = read(output / "protocol.json")
    old_protocol = read(ROOT / "artifacts/moa_probe_v1/protocol.json")
    selected, seal = read(output / "linear/selection.json"), read(output / "linear/selection_seal.json")
    assert seal["selection_sha256"] == sha(output / "linear/selection.json")
    linear = read(output / "linear/results.json")
    assert len(linear["runs"]) == len(selected["runs"]) == 88
    results = {}
    for stage, n_train in [("fixed306", 306), ("expanded603", 603)]:
        stage_result = {"n_moa_train": n_train, "datasets": {}}
        for dataset, cohort_directory, probability_path in [
            ("original66", output, output / "linear/probabilities.npz"),
            ("retained67", output / "supplemental", output / "linear/supplemental_probabilities.npz")]:
            cohort, probabilities = archive(cohort_directory / "cohort_arrays.npz"), archive(probability_path)
            np.testing.assert_array_equal(cohort["sample_ids"], probabilities["sample_ids"])
            indices = np.flatnonzero(cohort["splits"] == "test") if dataset == "original66" else np.arange(len(cohort["sample_ids"]))
            plates = np.asarray([str(s) + "|" + str(p) for s, p in zip(cohort["sources"][indices], cohort["plates"][indices])])
            dataset_result = {"n_wells": len(indices), "n_plates": len(set(plates)),
                              "source_counts": {source: int(np.sum(cohort["sources"][indices] == source)) for source in sorted(set(cohort["sources"][indices]))},
                              "label_sets": {}}
            for label_set in ["primary", "exploratory"]:
                labels = old_protocol[label_set + "_labels"]
                relevant = [row for row in linear["runs"] if row["stage"] == stage and row["label_set"] == label_set]
                assert len(relevant) == 22 and all(row["labels"] == labels and row["n_train"] == n_train for row in relevant)
                vocabulary = cohort["label_names"].tolist()
                y = cohort["y_all"][indices][:, [vocabulary.index(label) for label in labels]]
                predictions = {row["representation"]: probabilities[row["probability_key"]][indices] for row in relevant}
                families = {}
                for name in predictions:
                    families.setdefault(model_family(name), []).append(name)
                expected = {"dino", "pca", "old_matched", "old_shuffled", "new_matched", "new_shuffled",
                            "fusion_old_matched", "fusion_new_matched", "source", "prevalence"}
                assert set(families) == expected
                values, seed42 = family_summary(y, predictions, families)
                positive, negative = y.sum(axis=0).astype(int), len(y) - y.sum(axis=0).astype(int)
                complete = bool(np.all((positive > 0) & (negative > 0)))
                entry = {"status": "complete" if complete else "unsupported", "labels": labels,
                         "support": {"positive_counts": positive.tolist(), "negative_counts": negative.tolist(),
                                     "supported_classes": [label for label, pos, neg in zip(labels, positive, negative) if pos > 0 and neg > 0],
                                     "unsupported_classes": [label for label, pos, neg in zip(labels, positive, negative) if pos == 0 or neg == 0],
                                     "unique_positive_wells": int(np.any(y, axis=1).sum()),
                                     "positive_plate_count": len(set(plates[np.any(y, axis=1)]))},
                         "families": values, "seed42": seed42}
                if label_set == "primary" and complete:
                    entry["plate_bootstrap"] = paired_bootstrap(y, predictions, families, plates, 5000)
                    entry["well_bootstrap"] = paired_bootstrap(y, predictions, families, cohort["parent_ids"][indices], 2000)
                elif not complete:
                    entry["aggregate_metric_note"] = "Fixed eight-class metric is unsupported; aggregate AP/AUC are null. Per-class AP is shown only where positives exist, never relabeled as an eight-class result."
                dataset_result["label_sets"][label_set] = entry
            stage_result["datasets"][dataset] = dataset_result
        results[stage] = stage_result
    summary = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
               "protocol_sha256": sha(output / "protocol.json"), "script_sha256": sha(__file__),
               "linear_results_sha256": sha(output / "linear/results.json"),
               "selection_seal_sha256": sha(output / "linear/selection_seal.json"),
               "retrieval_gain": read(output / "retrieval_gain.json"), "stages": results,
               "verification_reference": "verification.json",
               "limitations": ["Original66 test data were already inspected; retained67 is an availability-selected internal subset, not external validation.",
                               "Stage A fixes only explicit downstream MoA labels; added caption supervision may contain mechanism priors.",
                               "All current labels and class vocabularies remain fixed; missing supplemental classes are not silently omitted.",
                               "Head seeds share biological samples and do not increase effective sample size.",
                               "All confidence intervals condition on fixed fitted models and class-supporting bootstrap draws."]}
    write(output / "summary.json", summary)
    print({stage: {dataset: {label_set: {family: row["mean_macro_ap"] for family, row in entry["families"].items()}
                             for label_set, entry in data["label_sets"].items()} for dataset, data in stage_result["datasets"].items()}
           for stage, stage_result in results.items()})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/expanded_moa_probe_v1")
    args = parser.parse_args()
    run(args.output)

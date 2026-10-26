"""Independent identities and probability-metric verification for MoA probes.

Run with the repository root environment (NumPy/pandas/scikit-learn). This file
does not load a model, select hyperparameters, or change analysis protocols.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import joblib


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "CLIP/artifacts/moa_probe_v1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def average_precision(y, probability):
    """Threshold-group AP, independently implemented with correct tie handling."""
    y = np.asarray(y)
    probability = np.asarray(probability, dtype=float)
    if y.ndim != 1 or y.shape != probability.shape or not len(y):
        raise ValueError("Expected aligned nonempty binary outcomes and probabilities")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(probability).all():
        raise ValueError("Invalid outcomes or predictions")
    n_positive = int(y.sum())
    if not n_positive:
        return None
    order = np.argsort(-probability, kind="stable")
    ordered_y, ordered_p = y[order], probability[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_p)), len(y) - 1]
    positives = np.cumsum(ordered_y)[ends]
    increments = np.diff(np.r_[0, positives]) / n_positive
    return float(np.sum(increments * positives / (ends + 1)))


def auc(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    positives, negatives = probability[y == 1], probability[y == 0]
    if not len(positives) or not len(negatives):
        return None
    difference = positives[:, None] - negatives[None]
    return float(np.mean((difference > 0) + .5 * (difference == 0)))


def classification_metrics(y, probabilities):
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    if y.ndim != 2 or y.shape != probabilities.shape:
        raise ValueError("Expected aligned binary and probability matrices")
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Probabilities are outside [0,1]")
    per_label_ap = [average_precision(y[:, i], probabilities[:, i]) for i in range(y.shape[1])]
    per_label_auc = [auc(y[:, i], probabilities[:, i]) for i in range(y.shape[1])]
    return {"macro_ap": float(np.mean([v for v in per_label_ap if v is not None])),
            "micro_ap": average_precision(y.ravel(), probabilities.ravel()),
            "macro_roc_auc": float(np.mean([v for v in per_label_auc if v is not None])),
            "per_label_ap": per_label_ap, "per_label_roc_auc": per_label_auc}


def self_checks():
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.array([1, 0, 0, 1, 0, 1])
    for p in [np.array([.5] * 6), np.array([.9, .8, .5, .5, .5, .1]),
              np.array([.1, .3, .4, .8, .2, .7])]:
        assert abs(average_precision(y, p) - average_precision_score(y, p)) < 1e-12
        assert abs(auc(y, p) - roc_auc_score(y, p)) < 1e-12
    assert average_precision(np.zeros(3), np.ones(3)) is None
    assert auc(np.ones(3), np.ones(3)) is None
    return {"constant_score_ties": True, "mixed_ties": True,
            "untied_scores": True, "unsupported_class_guard": True}


def verify_inputs(output=OUT):
    work = ROOT / "CLIP/artifacts/prototype_v1"
    protocol = read(output / "protocol.json")
    assert sha(output / "cohort.json") == protocol["cohort_sha256"]
    assert sha(output / "cohort_arrays.npz") == protocol["cohort_arrays_sha256"]
    assert sha(work / "features/features.npz") == protocol["feature_archive_sha256"]
    cohort, manifest = read(output / "cohort.json"), read(work / "manifest.json")
    assert cohort["manifest_id"] == manifest["manifest_id"]
    with np.load(output / "cohort_arrays.npz", allow_pickle=False) as a:
        labels, y, splits = a["label_names"].tolist(), a["y_all"], a["splits"]
        assert a["sample_ids"].tolist() == [s["sample_id"] for s in manifest["samples"]]
        assert a["well_ids"].tolist() == [s["well_id"] for s in manifest["samples"]]
        assert a["parent_ids"].tolist() == [s["parent_id"] for s in manifest["samples"]]
        assert splits.tolist() == [s["split"] for s in manifest["samples"]]
        assert len(set(a["parent_ids"].tolist())) == len(y) == 438
        for group in ["primary", "exploratory"]:
            chosen = protocol[group + "_labels"]
            assert chosen == cohort["label_sets"][group]["labels"]
            indices = [labels.index(label) for label in chosen]
            for split in ["train", "val", "test"]:
                values = y[splits == split][:, indices]
                assert (values.sum(axis=0) > 0).all() and (values.sum(axis=0) < len(values)).all()
        primary = y[splits == "test"][:, [labels.index(label) for label in protocol["primary_labels"]]]
        positive_primary_wells = int(np.any(primary, axis=1).sum())
    for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
        a = [s for s in manifest["samples"] if s["split"] == first]
        b = [s for s in manifest["samples"] if s["split"] == second]
        assert not {s["parent_id"] for s in a} & {s["parent_id"] for s in b}
        assert not {(s["source"], s["plate"]) for s in a} & {(s["source"], s["plate"]) for s in b}
    snapshot = read(work / "verification.json")
    for name, expected in snapshot["artifacts"].items():
        assert sha(ROOT / "CLIP" / name) == expected
    complete = read(work / "training/complete.json")
    for name, expected in complete["runs"].items():
        directory = work / "training" / name
        assert sha(directory / "complete.json") == expected
        for artifact, digest in read(directory / "complete.json")["artifacts"].items():
            assert sha(directory / artifact) == digest
    return {"protocol_sha256": sha(output / "protocol.json"), "cohort_authenticated": True,
            "feature_archive_authenticated": True, "original_snapshot_unchanged": True,
            "all_six_original_checkpoints_authenticated": True,
            "sample_id_well_parent_split_alignment": True, "parent_plate_isolation": True,
            "n_primary_labels": len(protocol["primary_labels"]),
            "primary_independent_positive_test_wells": positive_primary_wells,
            "metric_self_checks": self_checks()}


def verify_capacity(selection_only=True, output=OUT):
    directory = output / "capacity"
    result = read(directory / "results.json")
    protocol = read(output / "protocol.json")
    seal = read(directory / "selection_seal.json")
    assert result["identity"]["protocol_sha256"] == sha(output / "protocol.json")
    assert result["selection_seal_sha256"] == sha(directory / "selection_seal.json")
    assert result["probability_archive_sha256"] == sha(directory / "probabilities.npz")
    assert seal["all_six_sealed_before_test_predictions"]
    with np.load(output / "cohort_arrays.npz", allow_pickle=False) as a:
        ids, splits, labels, y = a["sample_ids"].copy(), a["splits"].copy(), a["label_names"].tolist(), a["y_all"].copy()
    reports = []
    with np.load(directory / "probabilities.npz", allow_pickle=False) as a:
        assert a["sample_ids"].tolist() == ids.tolist()
        assert a["splits"].tolist() == splits.tolist()
        for record in result["runs"]:
            name, group = record["name"], record["label_set"]
            run_dir = directory / name
            assert sha(run_dir / "complete.json") == seal["runs"][name] == record["receipt_sha256"]
            receipt = read(run_dir / "complete.json")
            for filename, expected in receipt["artifacts"].items():
                assert sha(run_dir / filename) == expected
            history, selection = read(run_dir / "history.json"), read(run_dir / "selection.json")
            validation = [row for row in history if "validation_macro_ap" in row]
            best = max(row["validation_macro_ap"] for row in validation)
            earliest = next(row["step"] for row in validation if row["validation_macro_ap"] == best)
            assert selection["validation_macro_ap"] == record["validation_macro_ap"] == best
            assert selection["best_step"] == record["best_step"] == earliest
            assert selection["training_sample_ids"] == ids[splits == "train"].tolist()
            assert selection["validation_sample_ids"] == ids[splits == "val"].tolist()
            assert not selection["test_labels_used_for_selection"]
            chosen = protocol[group + "_labels"]
            assert a[group + "_labels"].tolist() == chosen == record["labels"]
            targets = y[:, [labels.index(label) for label in chosen]]
            probability = a[name]
            assert probability.shape == targets.shape
            assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
            metrics = {"val": classification_metrics(targets[splits == "val"], probability[splits == "val"])}
            assert abs(metrics["val"]["macro_ap"] - selection["validation_macro_ap"]) < 1e-12
            if not selection_only:
                metrics["test"] = classification_metrics(targets[splits == "test"], probability[splits == "test"])
            reports.append({"id": name, "earliest_best_step": earliest, "metrics": metrics})
    assert len(reports) == 6
    return {"selection_only": selection_only, "all_six_authenticated": True,
            "all_validation_AP_reproduced": True, "all_earliest_ties_verified": True, "runs": reports}


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def original_feature_store():
    work = ROOT / "CLIP/artifacts/prototype_v1"
    features = load_archive(work / "features/features.npz")
    evaluation = read(work / "evaluation/evaluation.json")
    store = {"dino": features["image_features"].reshape(len(features["sample_ids"]), -1).astype(float)}
    for arm in ["matched", "shuffled"]:
        for seed in [42, 43, 44]:
            name = f"{arm}_seed{seed}"
            path = work / f"evaluation/{name}_gallery.npz"
            assert sha(path) == evaluation["artifacts"][path.name]
            arrays = load_archive(path)
            np.testing.assert_array_equal(arrays["sample_ids"], features["sample_ids"])
            store[name] = arrays["image_embeddings"].astype(float)
    return store


def independent_predict(model, store, sources):
    """Compute saved logistic models directly, independently of production helpers."""
    if model["kind"] == "prevalence":
        return np.tile(model["probability"], (len(sources), 1))
    blocks = []
    for block in model["blocks"]:
        raw = ((sources[:, None] == np.asarray(model["source_vocabulary"])[None]).astype(float)
               if block["feature"] == "source_only" else store[block["feature"]])
        scaler = block["scaler"]
        value = (raw - scaler.mean_) / scaler.scale_
        if block["pca"] is not None:
            pca = block["pca"]
            value = (value - pca.mean_) @ pca.components_.T
            post = block["post_scaler"]
            value = (value - post.mean_) / post.scale_
        blocks.append(value / np.sqrt(block["dimension"]))
    design = np.concatenate(blocks, axis=1) / np.sqrt(len(blocks))
    logits = np.column_stack([design @ classifier.coef_[0] + classifier.intercept_[0]
                              for classifier in model["classifiers"]])
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


def verify_linear(output=OUT):
    directory = output / "linear"
    protocol, result = read(output / "protocol.json"), read(directory / "results.json")
    seal, selected = read(directory / "selection_seal.json"), read(directory / "selection.json")
    assert result["selection_seal_sha256"] == sha(directory / "selection_seal.json")
    assert seal["selection_sha256"] == sha(directory / "selection.json")
    assert seal["all_68_choices_sealed_before_test_prediction"]
    assert result["probability_archive_sha256"] == sha(directory / "probabilities.npz")
    assert result["identity"]["protocol_sha256"] == sha(output / "protocol.json")
    assert result["identity"]["script_sha256"] == sha(ROOT / "CLIP/scripts/run_moa_linear_probe.py")
    cohort, probability = load_archive(output / "cohort_arrays.npz"), load_archive(directory / "probabilities.npz")
    ids, splits, sources = cohort["sample_ids"], cohort["splits"], cohort["sources"]
    np.testing.assert_array_equal(ids, probability["sample_ids"])
    np.testing.assert_array_equal(splits, probability["splits"])
    lookup = {name: i for i, name in enumerate(ids)}
    store = original_feature_store()
    records, subsets = [], {}
    for row, choice in zip(result["runs"], selected["runs"]):
        assert all(row[key] == value for key, value in choice.items())
        assert not choice["test_used_for_model_selection"]
        assert choice["validation_sample_ids"] == ids[splits == "val"].tolist()
        indices = np.asarray([lookup[name] for name in choice["train_sample_ids"]])
        assert len(indices) == len(set(indices)) == row["n_train"]
        assert (splits[indices] == "train").all()
        if row["subset"] == "full":
            np.testing.assert_array_equal(indices, np.flatnonzero(splits == "train"))
        key = (row["label_set"], row["subset"])
        assert key not in subsets or subsets[key] == choice["train_sample_ids"]
        subsets[key] = choice["train_sample_ids"]
        labels = protocol[row["label_set"] + "_labels"]
        assert labels == row["labels"] == probability[row["label_set"] + "_labels"].tolist()
        targets = cohort["y_all"][:, [cohort["label_names"].tolist().index(label) for label in labels]]
        path = directory / row["model_path"]
        assert sha(path) == row["model_sha256"] == seal["models"][row["id"]]
        model = joblib.load(path)
        assert model["labels"] == labels and model["id"] == row["id"]
        candidates = row["c_candidates"]
        if candidates:
            assert [c["C"] for c in candidates] == protocol["linear_probe"]["C_grid"]
            best = max(c["validation_macro_ap"] for c in candidates)
            best_c = next(c["C"] for c in candidates if c["validation_macro_ap"] == best)
            assert row["selected_C"] == best_c and row["validation_macro_ap"] == best
            assert all(classifier.C == best_c for classifier in model["classifiers"])
            for block in model["blocks"]:
                raw = ((sources[indices, None] == np.asarray(model["source_vocabulary"])[None]).astype(float)
                       if block["feature"] == "source_only" else store[block["feature"]][indices])
                if block["feature"] == "source_only":
                    assert model["source_vocabulary"] == sorted(set(sources[indices]))
                scaler = block["scaler"]
                assert int(scaler.n_samples_seen_) == len(indices)
                np.testing.assert_allclose(scaler.mean_, raw.mean(axis=0), atol=1e-12)
                np.testing.assert_allclose(scaler.var_, raw.var(axis=0), atol=1e-12)
                if block["pca"] is not None:
                    pca = block["pca"]
                    assert pca.n_samples_ == len(indices)
                    assert block["dimension"] == min(128, len(indices) - 1)
                    train_scaled = scaler.transform(raw)
                    np.testing.assert_allclose(pca.mean_, train_scaled.mean(axis=0), atol=1e-12)
                    np.testing.assert_allclose(block["post_scaler"].mean_, pca.transform(train_scaled).mean(axis=0), atol=1e-12)
        else:
            assert model["kind"] == "prevalence" and row["selected_C"] is None
            np.testing.assert_allclose(model["probability"], targets[indices].mean(axis=0), atol=0)
        actual = independent_predict(model, store, sources)
        saved = probability[row["probability_key"]]
        np.testing.assert_allclose(actual, saved, atol=1e-10, rtol=1e-10)
        values = {}
        for split in ["val", "test"]:
            metric = classification_metrics(targets[splits == split], saved[splits == split])
            for key in ["macro_ap", "micro_ap", "macro_roc_auc"]:
                assert abs(metric[key] - row["metrics"][split][key]) < 1e-12
            np.testing.assert_allclose(metric["per_label_ap"], [r["average_precision"] for r in row["metrics"][split]["per_label"]], atol=1e-12)
            values[split] = metric
        records.append({"id": row["id"], "selected_C": row["selected_C"], "metrics": values})
    for seed in [42, 43, 44]:
        low = [r for r in selected["low_label_subsets"] if r["seed"] == seed]
        low.sort(key=lambda row: row["fraction"])
        assert len(low) == 2 and set(low[0]["sample_ids"]) <= set(low[1]["sample_ids"])
        for row in low:
            assert len(row["sample_ids"]) == round(306 * row["fraction"])
            assert row["sample_ids"] == subsets[("primary", row["name"])]
    assert len(records) == 68 == len(selected["runs"])
    return {"all_68_models_sealed_and_authenticated": True, "all_C_choices_earliest_best_validation": True,
            "all_training_preprocessor_statistics_verified": True, "all_saved_predictions_reproduced_independently": True,
            "all_val_test_metrics_recomputed": True, "low_label_methods_share_nested_training_rows": True,
            "runs": records}


def verify_supplemental(output=OUT):
    directory = output / "supplemental"
    original = read(ROOT / "CLIP/artifacts/prototype_v1/manifest.json")
    manifest, cohort = read(directory / "manifest.json"), load_archive(directory / "cohort_arrays.npz")
    protocol, receipt = read(directory / "supplemental_protocol.json"), read(directory / "extraction_complete.json")
    assert sha(directory / "manifest.json") == protocol["manifest_sha256"]
    assert sha(directory / "cohort.json") == protocol["cohort_sha256"]
    for filename, expected in read(directory / "cohort.json")["artifacts"].items():
        assert sha(directory / filename) == expected
    assert receipt["protocol_id"] == protocol["protocol_id"]
    assert protocol["sample_ids"] == [s["sample_id"] for s in manifest["samples"]] == cohort["sample_ids"].tolist()
    for filename, expected in receipt["artifacts"].items():
        assert sha(directory / filename) == expected
    old_parents = {s["parent_id"] for s in original["samples"]}
    old_plates = {(s["source"], s["plate"]) for s in original["samples"]}
    new_parents = {s["parent_id"] for s in manifest["samples"]}
    new_plates = {(s["source"], s["plate"]) for s in manifest["samples"]}
    assert not new_parents & old_parents and not new_plates & old_plates
    old_sha = {image["sha256"] for s in original["samples"] for image in s["images"]}
    new_sha = {image["sha256"] for s in manifest["samples"] for image in s["images"]}
    assert not old_sha & new_sha
    features, projected = load_archive(directory / "features/features.npz"), load_archive(directory / "projected.npz")
    np.testing.assert_array_equal(features["sample_ids"], cohort["sample_ids"])
    np.testing.assert_array_equal(projected["sample_ids"], cohort["sample_ids"])
    store = {"dino": features["image_features"].reshape(len(cohort["sample_ids"]), -1).astype(float)}
    for arm in ["matched", "shuffled"]:
        for seed in [42, 43, 44]:
            store[f"{arm}_seed{seed}"] = projected[f"{arm}_seed{seed}"].astype(float)
    values, probability = read(directory / "linear_results.json"), load_archive(directory / "linear_probabilities.npz")
    assert sha(directory / "linear_probabilities.npz") == values["probabilities_sha256"]
    assert values["selection_seal_sha256"] == sha(output / "linear/selection_seal.json")
    assert values["no_refit_or_tuning"]
    selection = read(output / "linear/selection.json")
    for row in selection["runs"]:
        model = joblib.load(output / "linear" / row["model_path"])
        actual = independent_predict(model, store, cohort["sources"])
        np.testing.assert_allclose(actual, probability[row["probability_key"]], atol=1e-10, rtol=1e-10)
    capacity_receipt = read(output / "capacity/supplemental_prediction_receipt.json")
    assert capacity_receipt["original_capacity_results_sha256"] == sha(output / "capacity/results.json")
    assert capacity_receipt["original_selection_seal_sha256"] == sha(output / "capacity/selection_seal.json")
    assert capacity_receipt["feature_archive_sha256"] == sha(directory / "features/features.npz")
    assert capacity_receipt["probability_archive_sha256"] == sha(output / "capacity" / capacity_receipt["probability_archive"])
    for name, expected in capacity_receipt["checkpoint_sha256"].items():
        assert sha(output / "capacity" / name / "best.pt") == expected
    return {"n_wells": len(new_parents), "n_new_plates": len(new_plates), "raw_sha_overlap": 0,
            "no_parent_or_plate_overlap_with_original_438": True,
            "supplemental_extraction_receipt_authenticated": True,
            "six_supplemental_capacity_checkpoint_and_probability_hashes_authenticated": True,
            "all_68_supplemental_predictions_reproduced_without_refitting": True}


def independent_bootstrap(y, predictions, families, groups, draws, seed):
    unique = sorted(set(groups.tolist()))
    rng = np.random.default_rng(seed)
    chosen = rng.integers(len(unique), size=(draws, len(unique)))
    group_counts = np.stack([np.bincount(row, minlength=len(unique)) for row in chosen])
    weights = group_counts[:, [unique.index(group) for group in groups]]
    positives, negatives = weights @ y, weights @ (1 - y)
    retained = weights[np.all((positives > 0) & (negatives > 0), axis=1)].astype(float)
    distributions = {}
    for name, probability in predictions.items():
        label_values = []
        for column in range(y.shape[1]):
            p, target = probability[:, column], y[:, column]
            order = np.argsort(-p, kind="stable")
            last = np.r_[np.flatnonzero(np.diff(p[order])), len(p) - 1]
            total = np.cumsum(retained[:, order], axis=1)[:, last]
            tp = np.cumsum(retained[:, order] * target[order], axis=1)[:, last]
            recall_increment = np.diff(np.column_stack([np.zeros(len(tp)), tp]), axis=1) / tp[:, -1:]
            precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
            label_values.append(np.sum(recall_increment * precision, axis=1))
        distributions[name] = np.mean(label_values, axis=0)
    grouped = {family: np.mean([distributions[name] for name in members], axis=0) for family, members in families.items()}
    return len(retained), grouped


def verify_summary(output=OUT):
    summary, protocol = read(output / "summary.json"), read(output / "protocol.json")
    assert summary["protocol_sha256"] == sha(output / "protocol.json")
    assert summary["script_sha256"] == sha(ROOT / "CLIP/scripts/summarize_moa_probe.py")
    linear = read(output / "linear/results.json")
    records = {}
    for dataset, base, lp_path, cp_path in [
        ("original_test", output, output / "linear/probabilities.npz", output / "capacity/probabilities.npz"),
        ("new_plates", output / "supplemental", output / "supplemental/linear_probabilities.npz", output / "capacity/supplemental_probabilities.npz")]:
        cohort, lp, cp = load_archive(base / "cohort_arrays.npz"), load_archive(lp_path), load_archive(cp_path)
        np.testing.assert_array_equal(lp["sample_ids"], cohort["sample_ids"])
        np.testing.assert_array_equal(cp["sample_ids"], cohort["sample_ids"])
        index = np.flatnonzero(cohort["splits"] == "test") if dataset == "original_test" else np.arange(len(cohort["sample_ids"]))
        result = summary["datasets"][dataset]
        records[dataset] = {}
        for label_set in ["primary", "exploratory"]:
            labels = protocol[label_set + "_labels"]
            y = cohort["y_all"][index][:, [cohort["label_names"].tolist().index(label) for label in labels]]
            entry = result["label_sets"][label_set]
            assert entry["labels"] == labels and entry["positive_counts"] == y.sum(axis=0).tolist()
            assert entry["unique_positive_wells"] == int(np.any(y, axis=1).sum())
            predictions = {row["representation"]: lp[row["probability_key"]][index] for row in linear["runs"]
                           if row["label_set"] == label_set and row["subset"] == "full"}
            predictions.update({f"capacity_seed{seed}": cp[f"{label_set}_seed{seed}"][index] for seed in [42, 43, 44]})
            families = {family: row["models"] for family, row in entry["families"].items()}
            family_metrics = {}
            for family, names in families.items():
                metrics = [classification_metrics(y, predictions[name]) for name in names]
                aps = [m["macro_ap"] for m in metrics]
                expected = entry["families"][family]
                assert abs(np.mean(aps) - expected["mean_macro_ap"]) < 1e-12
                assert abs((np.std(aps, ddof=1) if len(aps) > 1 else 0) - expected["seed_std_macro_ap"]) < 1e-12
                np.testing.assert_allclose(np.mean([m["per_label_ap"] for m in metrics], axis=0), expected["per_label_mean_ap"], atol=1e-12)
                family_metrics[family] = float(np.mean(aps))
            records[dataset][label_set] = {"mean_macro_ap": family_metrics, "per_label_points_verified": True}
            if label_set == "primary":
                for name, groups, draws in [("plate_bootstrap", np.array([str(s) + "|" + str(p) for s, p in zip(cohort["sources"][index], cohort["plates"][index])]), 5000),
                                            ("well_bootstrap", cohort["parent_ids"][index], 2000)]:
                    n, distributions = independent_bootstrap(y, predictions, families, groups, draws, 20261024)
                    expected = entry[name]
                    assert n == expected["retained_draws"]
                    assert expected["n_clusters"] == len(set(groups)) and expected["attempted_draws"] == draws
                    assert expected["excluded_missing_class_support"] == draws - n
                    for comparison in expected["comparisons"]:
                        a, b = comparison["a"], comparison["b"]
                        np.testing.assert_allclose(np.quantile(distributions[a] - distributions[b], [.025, .975]), comparison["conditional_paired_bootstrap_95_interval"], atol=1e-12)
                        assert abs(family_metrics[a] - family_metrics[b] - comparison["observed_macro_ap_difference"]) < 1e-12
                    records[dataset][label_set][name] = {"retained": n, "attempted": draws, "all_six_comparison_intervals_reproduced": True}
    return records


def run(inputs_only=False, selection_only=False):
    result = verify_inputs()
    if selection_only:
        result["capacity"] = verify_capacity(selection_only=True)
    elif not inputs_only:
        result["capacity"] = verify_capacity(selection_only=False)
        result["linear"] = verify_linear()
        result["supplemental"] = verify_supplemental()
        result["summary"] = verify_summary()
    result["verification_script_sha256"] = sha(__file__)
    if (OUT / "test_verification.json").exists():
        test_result = read(OUT / "test_verification.json")
        for filename, expected in test_result["test_files_sha256"].items():
            assert sha(ROOT / filename) == expected
        result["recorded_tests"] = test_result
    filename = "selection_verification.json" if selection_only else ("inputs_verification.json" if inputs_only else "finalverification.json")
    (OUT / filename).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ["linear", "capacity"]}, indent=2))
    if "linear" in result:
        print("Verified all 68 linear choices, train-only transforms, independent predictions and val/test metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-only", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    args = parser.parse_args()
    run(args.inputs_only, args.selection_only)

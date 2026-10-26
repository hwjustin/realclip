"""Independent audit of expanded-cohort probes and two-stage MoA statistics.

Reuses prior independent matrix and threshold-metric implementations, never fits
a model and never writes an old artifact. Only new verification JSON is written.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from verify_moa_probe import (average_precision, auc, classification_metrics,
                              independent_bootstrap, independent_predict)
from summarize_expanded_moa_probe import archive, read, sha, write


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/expanded_moa_probe_v1"


def core_audit(output):
    protocol = read(output / "protocol.json")
    for name, binding in protocol["bindings"].items():
        assert sha(binding["path"]) == binding["sha256"], name
    cohort, held = archive(output / "cohort_arrays.npz"), archive(output / "supplemental/cohort_arrays.npz")
    original = read(ROOT / "artifacts/prototype_v1/manifest.json")["samples"]
    expanded = read(ROOT / "artifacts/expanded_samples_v1/manifest.json")["samples"]
    metadata, held_metadata = read(output / "cohort.json"), read(output / "supplemental/cohort.json")
    for directory, record in [(output, metadata), (output / "supplemental", held_metadata)]:
        for name, expected in record["artifacts"].items():
            assert sha(directory / name) == expected
    assert cohort["sample_ids"].tolist() == [s["sample_id"] for s in expanded]
    assert cohort["parent_ids"].tolist() == [s["parent_id"] for s in expanded]
    assert len(set(cohort["parent_ids"])) == len(expanded) == 735
    lookup = {name: i for i, name in enumerate(cohort["sample_ids"])}
    original306 = [s["sample_id"] for s in original if s["split"] == "train"]
    assert original306 == metadata["original306_train_sample_ids"]
    assert all(cohort["splits"][lookup[name]] == "train" for name in original306)
    for split in ["val", "test"]:
        assert [s["sample_id"] for s in original if s["split"] == split] == cohort["sample_ids"][cohort["splits"] == split].tolist()
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ia, ib = cohort["splits"] == a, cohort["splits"] == b
        assert not set(cohort["parent_ids"][ia]) & set(cohort["parent_ids"][ib])
        assert not set(zip(cohort["sources"][ia], cohort["plates"][ia])) & set(zip(cohort["sources"][ib], cohort["plates"][ib]))
    assert not set(held["parent_ids"]) & set(cohort["parent_ids"])
    assert not set(zip(held["sources"], held["plates"])) & set(zip(cohort["sources"], cohort["plates"]))
    old142 = archive(ROOT / "artifacts/moa_probe_v1/supplemental/cohort_arrays.npz")
    fit = np.isin(cohort["splits"], ["train", "val"])
    fit_parents, fit_plates = set(cohort["parent_ids"][fit]), set(zip(cohort["sources"][fit], cohort["plates"][fit]))
    retained = [i for i in range(len(old142["sample_ids"])) if old142["parent_ids"][i] not in fit_parents
                and (old142["sources"][i], old142["plates"][i]) not in fit_plates]
    assert old142["sample_ids"][retained].tolist() == held["sample_ids"].tolist()
    assert len(retained) == 67
    old_vocabulary = archive(ROOT / "artifacts/moa_probe_v1/cohort_arrays.npz")["label_names"].tolist()
    assert cohort["label_names"].tolist() == held["label_names"].tolist() == old_vocabulary
    old_protocol = read(ROOT / "artifacts/moa_probe_v1/protocol.json")
    for label_set in ["primary", "exploratory"]:
        assert protocol[label_set + "_labels"] == old_protocol[label_set + "_labels"]
    precomputed = read(output / "precomputed/complete.json")
    assert precomputed["protocol_sha256"] == sha(output / "protocol.json")
    for scope in ["primary", "supplemental"]:
        assert sha(output / f"precomputed/{scope}_store.npz") == precomputed[f"{scope}_store_sha256"]
    stores = {scope: archive(output / f"precomputed/{scope}_store.npz") for scope in ["primary", "supplemental"]}
    for scope, rows in [("primary", cohort), ("supplemental", held)]:
        np.testing.assert_array_equal(stores[scope].pop("sample_ids"), rows["sample_ids"])
    for model in precomputed["models"]:
        checkpoint = Path(model["checkpoint"])
        assert sha(checkpoint) == model["checkpoint_sha256"]
        receipt = read(checkpoint.parent / "complete.json")
        assert receipt["artifacts"]["best.pt"] == model["checkpoint_sha256"]
        sealed = read(checkpoint.parent.parent / "complete.json")
        assert sealed["runs"][checkpoint.parent.name] == sha(checkpoint.parent / "complete.json")
        directory = checkpoint.parent.parent.parent
        gallery_path = directory / "evaluation" / f"{checkpoint.parent.name}_gallery.npz"
        assert sha(gallery_path) == model["gallery_sha256"]
        gallery = archive(gallery_path)
        current = {sample: i for i, sample in enumerate(cohort["sample_ids"])}
        indices = [current[name] for name in gallery["sample_ids"]]
        np.testing.assert_array_equal(stores["primary"][model["name"]][indices], gallery["image_embeddings"])
    old_dino = archive(ROOT / "artifacts/prototype_v1/features/features.npz")
    indices = [lookup[name] for name in old_dino["sample_ids"]]
    np.testing.assert_array_equal(stores["primary"]["dino"][indices], old_dino["image_features"].reshape(len(indices), -1))
    return {"protocol_sha256": sha(output / "protocol.json"), "bindings_authenticated": True,
            "original306_in_expanded603_and_original66_holdouts_preserved": True,
            "735_parent_plate_isolation": True, "retained67_exactly_matches_metadata_exclusion_rule": True,
            "retained67_parent_plate_isolated_from_all735": True,
            "all_twelve_checkpoints_and_published_gallery_rows_authenticated": True,
            "original438_dino_features_preserved_exactly": True}, cohort, held, stores


def validate_metric(y, probability, expected, allow_unsupported=False):
    complete = bool(np.all((y.sum(axis=0) > 0) & (y.sum(axis=0) < len(y))))
    ap = [average_precision(y[:, i], probability[:, i]) for i in range(y.shape[1])]
    aucs = [auc(y[:, i], probability[:, i]) for i in range(y.shape[1])]
    if complete:
        computed = classification_metrics(y, probability)
        for key in ["macro_ap", "micro_ap", "macro_roc_auc"]:
            assert abs(computed[key] - expected[key]) < 1e-12
    else:
        assert allow_unsupported and expected["macro_ap"] is None and expected["macro_roc_auc"] is None
    for i, row in enumerate(expected["per_label"]):
        if ap[i] is None:
            assert row["average_precision"] is None
        else:
            assert abs(ap[i] - row["average_precision"]) < 1e-12
        if aucs[i] is None:
            assert row["roc_auc"] is None
        else:
            assert abs(aucs[i] - row["roc_auc"]) < 1e-12


def fitted_models_audit(output, cohort, held, stores):
    directory = output / "linear"
    result, selected, seal = [read(directory / filename) for filename in ["results.json", "selection.json", "selection_seal.json"]]
    protocol = read(output / "protocol.json")
    assert len(result["runs"]) == len(selected["runs"]) == 88
    assert seal["all_88_choices_sealed_before_test_prediction"]
    assert sha(directory / "selection.json") == seal["selection_sha256"]
    assert sha(directory / "selection_seal.json") == result["selection_seal_sha256"]
    assert sha(directory / "probabilities.npz") == result["probability_archive_sha256"]
    assert sha(directory / "supplemental_probabilities.npz") == result["supplemental_probability_archive_sha256"]
    assert result["identity"]["protocol_sha256"] == sha(output / "protocol.json")
    assert result["identity"]["script_sha256"] == sha(ROOT / "scripts/run_expanded_moa_probe.py")
    correction = read(output / "protocol_numerical_correction.json")
    assert correction["protocol_sha256"] == sha(output / "protocol.json")
    assert correction["script_sha256"] == sha(ROOT / "scripts/finalize_expanded_moa_probe.py")
    assert seal["numerical_correction_plan_sha256"] == sha(output / "protocol_numerical_correction.json")
    assert seal["original_selection_seal_sha256"] == sha(directory / "numerical_v1/selection_seal.json")
    previous = read(directory / "numerical_v1/selection_seal.json")
    assert correction["original_selection_seal_sha256"] == sha(directory / "numerical_v1/selection_seal.json")
    assert previous["selection_sha256"] == sha(directory / "numerical_v1/selection.json")
    unchanged = [row for row in selected["runs"] if row["representation"] != "source_only"]
    assert len(unchanged) == 84 and all(seal["models"][row["id"]] == previous["models"][row["id"]] for row in unchanged)
    probabilities = {"primary": archive(directory / "probabilities.npz"), "supplemental": archive(directory / "supplemental_probabilities.npz")}
    lookup = {name: i for i, name in enumerate(cohort["sample_ids"])}
    metadata = read(output / "cohort.json")
    stage_ids = {"fixed306": metadata["original306_train_sample_ids"],
                 "expanded603": cohort["sample_ids"][cohort["splits"] == "train"].tolist()}
    audits = []
    for row, choice in zip(result["runs"], selected["runs"]):
        assert all(row[key] == value for key, value in choice.items())
        assert not row["test_used_for_model_selection"]
        assert row["train_sample_ids"] == stage_ids[row["stage"]]
        assert row["validation_sample_ids"] == cohort["sample_ids"][cohort["splits"] == "val"].tolist()
        indices = np.array([lookup[name] for name in row["train_sample_ids"]])
        assert len(indices) == row["n_train"] == (306 if row["stage"] == "fixed306" else 603)
        assert np.all(cohort["splits"][indices] == "train")
        labels = protocol[row["label_set"] + "_labels"]
        assert labels == row["labels"]
        path = directory / row["model_path"]
        assert sha(path) == row["model_sha256"] == seal["models"][row["id"]]
        model = joblib.load(path)
        assert model["labels"] == labels and model["id"] == row["id"]
        candidates = row["c_candidates"]
        if candidates:
            assert [entry["C"] for entry in candidates] == protocol["linear_probe"]["C_grid"]
            best = max(entry["validation_macro_ap"] for entry in candidates)
            earliest = next(entry["C"] for entry in candidates if entry["validation_macro_ap"] == best)
            assert row["selected_C"] == earliest and row["validation_macro_ap"] == best
            for block in model["blocks"]:
                raw = ((cohort["sources"][indices, None] == np.asarray(model["source_vocabulary"])[None]).astype(float)
                       if block["feature"] == "source_only" else stores["primary"][block["feature"]][indices].astype(float))
                scaler = block["scaler"]
                assert int(scaler.n_samples_seen_) == len(indices)
                np.testing.assert_allclose(scaler.mean_, raw.mean(axis=0), atol=1e-12)
                np.testing.assert_allclose(scaler.var_, raw.var(axis=0), atol=1e-12)
                if block["pca"] is not None:
                    pca = block["pca"]
                    assert pca.n_samples_ == len(indices) and block["dimension"] == 128
                    scaled = scaler.transform(raw)
                    np.testing.assert_allclose(pca.mean_, scaled.mean(axis=0), atol=1e-12)
                    np.testing.assert_allclose(block["post_scaler"].mean_, pca.transform(scaled).mean(axis=0), atol=1e-12)
        else:
            assert model["kind"] == "prevalence" and row["selected_C"] is None
            y = cohort["y_all"][:, [cohort["label_names"].tolist().index(label) for label in labels]]
            np.testing.assert_array_equal(model["probability"], y[indices].mean(axis=0))
        for scope, rows in [("primary", cohort), ("supplemental", held)]:
            np.testing.assert_array_equal(probabilities[scope]["sample_ids"], rows["sample_ids"])
            actual = independent_predict(model, stores[scope], rows["sources"])
            saved = probabilities[scope][row["probability_key"]]
            np.testing.assert_allclose(actual, saved, atol=1e-10, rtol=1e-10)
            if row["representation"] == "source_only":
                for source in np.unique(rows["sources"]):
                    group = saved[rows["sources"] == source]
                    assert np.array_equal(group, np.repeat(group[:1], len(group), axis=0))
            y = rows["y_all"][:, [rows["label_names"].tolist().index(label) for label in labels]]
            if scope == "primary":
                for split, name in [("val", "val"), ("test", "test")]:
                    mask = rows["splits"] == split
                    validate_metric(y[mask], saved[mask], row["metrics"][name])
            else:
                validate_metric(y, saved, row["metrics"]["supplemental"], allow_unsupported=True)
        audits.append(row["id"])
    old = read(ROOT / "artifacts/moa_probe_v1/linear/results.json")
    old_probabilities = archive(ROOT / "artifacts/moa_probe_v1/linear/probabilities.npz")
    old_rows = [lookup[name] for name in old_probabilities["sample_ids"]]
    old_by_id = {row["id"]: row for row in old["runs"]}
    n_reproduced = 0
    for row in result["runs"]:
        if row["stage"] != "fixed306" or row["representation"] == "source_only" or row["representation"].startswith("new_") or "plus_new_" in row["representation"]:
            continue
        old_rep = row["representation"].replace("dino_plus_old_", "dino_plus_").removeprefix("old_")
        old_id = f"{row['label_set']}__full__{old_rep}"
        original = old_by_id[old_id]
        assert row["selected_C"] == original["selected_C"]
        assert row["validation_macro_ap"] == original["validation_macro_ap"]
        assert row["metrics"]["test"]["macro_ap"] == original["metrics"]["test"]["macro_ap"]
        np.testing.assert_allclose(probabilities["primary"][row["probability_key"]][old_rows], old_probabilities[old_id], atol=1e-10, rtol=0)
        n_reproduced += 1
    assert n_reproduced == 24
    return {"n_models_authenticated": len(audits), "all_best_validation_C_and_earliest_ties_verified": True,
            "all_train_only_scaler_PCA_statistics_verified": True,
            "all735_and_held67_probabilities_recomputed_independently": True,
            "all_recorded_AP_AUC_recomputed_including_unsupported_guards": True,
            "n_old_fixed306_non_source_methods_reproduced": n_reproduced,
            "source_only_validation_numerical_correction_authenticated": True,
            "all_84_non_source_model_hashes_unchanged_from_v1_seal": True,
            "identical_source_rows_have_bitwise_identical_probability_vectors": True}, result, probabilities


def summary_audit(output, cohort, held, result, probabilities):
    summary, protocol = read(output / "summary.json"), read(output / "protocol.json")
    assert summary["protocol_sha256"] == sha(output / "protocol.json")
    assert summary["script_sha256"] == sha(ROOT / "scripts/summarize_expanded_moa_probe.py")
    record = {}
    for stage in ["fixed306", "expanded603"]:
        record[stage] = {}
        for dataset, scope, rows in [("original66", "primary", cohort), ("retained67", "supplemental", held)]:
            indices = np.flatnonzero(rows["splits"] == "test") if dataset == "original66" else np.arange(len(rows["sample_ids"]))
            record[stage][dataset] = {}
            for label_set in ["primary", "exploratory"]:
                labels = protocol[label_set + "_labels"]
                y = rows["y_all"][indices][:, [rows["label_names"].tolist().index(label) for label in labels]]
                entry = summary["stages"][stage]["datasets"][dataset]["label_sets"][label_set]
                assert entry["labels"] == labels and entry["support"]["positive_counts"] == y.sum(axis=0).tolist()
                complete = bool(np.all((y.sum(axis=0) > 0) & (y.sum(axis=0) < len(y))))
                assert (entry["status"] == "complete") == complete
                predictions = {row["representation"]: probabilities[scope][row["probability_key"]][indices]
                               for row in result["runs"] if row["stage"] == stage and row["label_set"] == label_set}
                families = {family: value["models"] for family, value in entry["families"].items()}
                for family, members in families.items():
                    aps = [[average_precision(y[:, j], predictions[name][:, j]) for j in range(y.shape[1])] for name in members]
                    expected = entry["families"][family]
                    for j in range(y.shape[1]):
                        if aps[0][j] is None:
                            assert expected["per_label_mean_ap"][j] is None
                        else:
                            assert abs(float(np.mean([row[j] for row in aps])) - expected["per_label_mean_ap"][j]) < 1e-12
                    if complete:
                        macros = [float(np.mean(ap)) for ap in aps]
                        assert abs(float(np.mean(macros)) - expected["mean_macro_ap"]) < 1e-12
                        assert abs((float(np.std(macros, ddof=1)) if len(macros) > 1 else 0) - expected["seed_std_macro_ap"]) < 1e-12
                    else:
                        assert expected["mean_macro_ap"] is None and expected["seed_std_macro_ap"] is None
                    selected = entry["seed42"][family]["model"]
                    assert selected == (members[0] if len(members) == 1 else next(name for name in members if name.endswith("_seed42")))
                    if complete:
                        assert abs(classification_metrics(y, predictions[selected])["macro_ap"] - entry["seed42"][family]["macro_ap"]) < 1e-12
                    else:
                        assert entry["seed42"][family]["macro_ap"] is None
                checks = {"complete_fixed_vocabulary": complete, "all_family_and_per_class_points_verified": True}
                if label_set == "primary":
                    for kind, groups, draws in [("plate_bootstrap", np.asarray([str(s) + "|" + str(p) for s, p in zip(rows["sources"][indices], rows["plates"][indices])]), 5000),
                                                ("well_bootstrap", rows["parent_ids"][indices], 2000)]:
                        n, distributions = independent_bootstrap(y, predictions, families, groups, draws, 20261025)
                        selected_members = {family: [entry["seed42"][family]["model"]] for family in families}
                        selected_predictions = {name: predictions[name] for members in selected_members.values() for name in members}
                        _, primary_distributions = independent_bootstrap(y, selected_predictions, selected_members, groups, draws, 20261025)
                        expected = entry[kind]
                        assert n == expected["retained_draws"] and len(set(groups)) == expected["n_clusters"]
                        assert expected["attempted_draws"] == draws and expected["excluded_missing_class_support"] == draws - n
                        for comparison in expected["comparisons"]:
                            a, b = comparison["a"], comparison["b"]
                            np.testing.assert_allclose(np.quantile(distributions[a] - distributions[b], [.025, .975]), comparison["conditional_paired_bootstrap_95_interval"], atol=1e-12)
                            np.testing.assert_allclose(np.quantile(primary_distributions[a] - primary_distributions[b], [.025, .975]), comparison["seed42_conditional_paired_bootstrap_95_interval"], atol=1e-12)
                        checks[kind] = {"retained_draws": n, "attempted_draws": draws, "all_four_seedmean_and_seed42_intervals_verified": True}
                record[stage][dataset][label_set] = checks
    return record


def run(output, inputs_only=False):
    checks, cohort, held, stores = core_audit(output)
    if not inputs_only:
        model_checks, result, probabilities = fitted_models_audit(output, cohort, held, stores)
        checks["model_verification"] = model_checks
        checks["statistics_verification"] = summary_audit(output, cohort, held, result, probabilities)
    checks["verification_script_sha256"] = sha(__file__)
    checks["independent_helper_sha256"] = sha(ROOT / "scripts/verify_moa_probe.py")
    write(output / ("inputs_verification.json" if inputs_only else "verification.json"), checks)
    print(checks)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    run(args.output, args.inputs_only)

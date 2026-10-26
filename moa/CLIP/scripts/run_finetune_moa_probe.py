"""Same603-label, validation-selected MOA probes for the DINO last-two pilot.

Run with the existing root sklearn environment. Feature extraction uses the CLIP
environment separately. No test score is computed until every readout is sealed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from run_expanded_moa_probe import load_cohort, safe_metrics
from run_moa_linear_probe import fit_and_select, hash_file, predict_saved, read_json, write_json

ARMS = ("frozen_matched", "frozen_shuffled", "tail_matched", "tail_shuffled")
PRIMARY_LABELS = ["dopamine receptor antagonist", "serotonin receptor antagonist"]
EXPLORATORY_LABELS = ["EGFR inhibitor", "HDAC inhibitor", "SRC inhibitor", "cyclooxygenase inhibitor",
                      "dopamine receptor antagonist", "phosphodiesterase inhibitor",
                      "serotonin receptor antagonist", "tubulin polymerization inhibitor"]


def validate_cohorts(cohort: dict, supplemental: dict,
                     expected_counts: dict | None = None) -> dict[str, np.ndarray]:
    """Keep all existing labels, and reject plate/parent leakage before fitting."""
    splits = {name: np.flatnonzero(cohort["splits"] == name) for name in ("train", "val", "test")}
    if sum(map(len, splits.values())) != len(cohort["sample_ids"]) or any(not len(v) for v in splits.values()):
        raise ValueError("Unknown or empty primary split")
    if expected_counts is not None and {k: len(v) for k, v in splits.items()} != expected_counts:
        raise ValueError("Primary membership differs from the fixed603/66/66 experiment")
    for rows in (cohort, supplemental):
        n = len(rows["sample_ids"])
        if len(set(rows["sample_ids"])) != n or rows["y_all"].shape != (n, len(rows["label_names"])):
            raise ValueError("Invalid cohort row/label membership")
        if not np.isin(rows["y_all"], (0, 1)).all():
            raise ValueError("MOA targets must remain binary")
    if cohort["label_names"].tolist() != supplemental["label_names"].tolist():
        raise ValueError("Supplemental labels changed vocabulary/order")
    for field in ("parent_ids", "well_ids"):
        identities = [set(cohort[field][splits[name]]) for name in splits]
        if any(identities[a] & identities[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f"Cross-split {field} overlap")
    plates = np.asarray([f"{source}\0{plate}" for source, plate in zip(cohort["sources"], cohort["plates"])])
    groups = [set(plates[splits[name]]) for name in splits]
    if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Cross-split plate overlap")
    fitted = np.concatenate((splits["train"], splits["val"]))
    if set(cohort["parent_ids"][fitted]) & set(supplemental["parent_ids"]):
        raise ValueError("Supplemental parents overlap train/validation")
    extra_plates = {f"{source}\0{plate}" for source, plate in zip(supplemental["sources"], supplemental["plates"])}
    if set(plates[fitted]) & extra_plates:
        raise ValueError("Supplemental plates overlap train/validation")
    if set(cohort["sample_ids"]) & set(supplemental["sample_ids"]):
        raise ValueError("Duplicate primary/supplemental samples")
    if set(cohort["well_ids"]) & set(supplemental["well_ids"]):
        raise ValueError("Duplicate primary/supplemental wells")
    return splits


def validate_store(store: dict, cohort: dict, representations: list[str]) -> dict:
    if store.pop("sample_ids").tolist() != cohort["sample_ids"].tolist():
        raise ValueError("Representations do not follow the frozen cohort well order")
    for representation in representations:
        key = "dino" if representation == "dino_pca128" else representation
        dim = 1920 if key == "dino" or key.endswith("_pre1920") else 128
        value = store.get(key)
        if value is None or value.shape != (len(cohort["sample_ids"]), dim) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {representation} feature shape/values")
    return store


def _write_bound(path: Path, value: dict) -> None:
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Existing frozen evaluation binding changed: {path}")
    write_json(path, value)


def run(export_dir: Path, cohort_root: Path, output: Path, threads: int = 2) -> dict:
    export_dir, cohort_root, output = export_dir.resolve(), cohort_root.resolve(), output.resolve()
    complete = read_json(export_dir / "complete.json")
    if complete["binding_sha256"] != hash_file(export_dir / "binding.json"):
        raise ValueError("Representation export binding checksum mismatch")
    for name, digest in complete["artifacts"].items():
        if hash_file(export_dir / name) != digest:
            raise ValueError(f"Exported representation changed: {name}")
    old_protocol = read_json(cohort_root / "protocol.json")
    if old_protocol["primary_labels"] != PRIMARY_LABELS or old_protocol["exploratory_labels"] != EXPLORATORY_LABELS:
        raise ValueError("The established primary/exploratory label sets changed")
    representations = ["dino", "dino_pca128", "legacy_matched_seed42", "legacy_shuffled_seed42"]
    representations += [f"{arm}_{stage}" for arm in ARMS for stage in ("pre1920", "post128")]
    cohort = load_cohort(cohort_root / "cohort_arrays.npz")
    supplemental = load_cohort(cohort_root / "supplemental/cohort_arrays.npz")
    splits = validate_cohorts(cohort, supplemental, {"train": 603, "val": 66, "test": 66})
    if len(supplemental["sample_ids"]) != 67:
        raise ValueError("Use the existing retained67 supplemental evaluation cohort")
    samples = read_json(export_dir / "samples.json")
    for scope, rows in (("primary", cohort), ("supplemental", supplemental)):
        if [s["sample_id"] for s in samples[scope]] != rows["sample_ids"].tolist():
            raise ValueError("Export scope differs from the existing MOA cohort")
        for field, array_key in (("well_id", "well_ids"), ("parent_id", "parent_ids"), ("source", "sources"), ("plate", "plates")):
            if [s[field] for s in samples[scope]] != rows[array_key].tolist():
                raise ValueError(f"Exported {field} differs from frozen MOA identity")
    stores = {scope: validate_store(load_cohort(export_dir / f"{scope}_store.npz"), rows, representations)
              for scope, rows in (("primary", cohort), ("supplemental", supplemental))}
    bindings = {"export_complete": export_dir / "complete.json", "export_binding": export_dir / "binding.json",
                "cohort": cohort_root / "cohort_arrays.npz", "supplemental_cohort": cohort_root / "supplemental/cohort_arrays.npz",
                "existing_moa_protocol": cohort_root / "protocol.json", "script": Path(__file__),
                "fit_helper": Path(__file__).with_name("run_moa_linear_probe.py"),
                "metrics_helper": Path(__file__).with_name("run_expanded_moa_probe.py")}
    protocol = {"experiment": "DINO last-two-block fixed-seed pilot; frozen text; unchanged descriptions",
                "bindings": {key: {"path": str(path), "sha256": hash_file(path)} for key, path in bindings.items()},
                "representations": representations, "primary_labels": PRIMARY_LABELS, "exploratory_labels": EXPLORATORY_LABELS,
                "linear_probe": old_protocol["linear_probe"],
                "split_sample_ids": {name: cohort["sample_ids"][idx].tolist() for name, idx in splits.items()},
                "supplemental_sample_ids": supplemental["sample_ids"].tolist(),
                "primary_endpoint": "two-class macro average precision on original66 and retained67 separately",
                "exploratory_endpoint": "fixed eight classes, original66; retained67 full macro null if unsupported",
                "test_used_for_model_selection": False,
                "comparison_scope": "frozen/tail pilot arms share training schedule; legacy full-batch heads are reference only"}
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
    # Nothing in this loop accesses primary test targets or supplemental targets.
    for label_set, labels in (("primary", PRIMARY_LABELS), ("exploratory", EXPLORATORY_LABELS)):
        columns = [all_labels.index(label) for label in labels]
        for representation in representations:
            run_id = f"{label_set}__{representation}"
            path = model_dir / f"{run_id}.joblib"
            audit_path = path.with_suffix(".json")
            if audit_path.exists():
                audit = read_json(audit_path)
                if audit["identity"] != identity or audit["model_sha256"] != hash_file(path):
                    raise ValueError("Saved downstream readout identity changed")
            else:
                with threadpool_limits(limits=threads):
                    model, fit = fit_and_select(representation, train_store, cohort["sources"][train],
                                                cohort["y_all"][np.ix_(train, columns)], val_store,
                                                cohort["sources"][val], cohort["y_all"][np.ix_(val, columns)],
                                                protocol["linear_probe"]["C_grid"])
                model.update(id=run_id, labels=labels)
                joblib.dump(model, path)
                audit = {"id": run_id, "identity": identity, "representation": representation,
                         "label_set": label_set, "labels": labels, "n_train": len(train),
                         "model_path": str(path.relative_to(output)), "model_sha256": hash_file(path),
                         "train_sample_ids": cohort["sample_ids"][train].tolist(),
                         "validation_sample_ids": cohort["sample_ids"][val].tolist(),
                         "test_used_for_model_selection": False, **fit}
                write_json(audit_path, audit)
            selected.append(audit)
            print(f"SELECTED {len(selected)}/{len(representations) * 2} {run_id} C={audit['selected_C']} valAP={audit['validation_macro_ap']:.6f}", flush=True)
    selection = {"identity": identity, "runs": selected}
    _write_bound(output / "selection.json", selection)
    seal = {"identity": identity, "selection_sha256": hash_file(output / "selection.json"),
            "all_readouts_selected_before_test_prediction": True,
            "n_readouts": len(selected), "models": {row["id"]: row["model_sha256"] for row in selected}}
    _write_bound(output / "selection_seal.json", seal)
    predictions = {"sample_ids": cohort["sample_ids"], "splits": cohort["splits"],
                   "primary_labels": np.asarray(PRIMARY_LABELS), "exploratory_labels": np.asarray(EXPLORATORY_LABELS)}
    extra_predictions = {"sample_ids": supplemental["sample_ids"], "splits": supplemental["splits"],
                         "primary_labels": np.asarray(PRIMARY_LABELS), "exploratory_labels": np.asarray(EXPLORATORY_LABELS)}
    results = []
    for audit in selected:
        path = output / audit["model_path"]
        if hash_file(path) != seal["models"][audit["id"]]:
            raise ValueError("Readout changed after selection seal")
        model = joblib.load(path)
        with threadpool_limits(limits=threads):
            probability = predict_saved(model, stores["primary"], cohort["sources"])
            extra = predict_saved(model, stores["supplemental"], supplemental["sources"])
        columns = [all_labels.index(label) for label in audit["labels"]]
        extra_columns = [supplemental["label_names"].tolist().index(label) for label in audit["labels"]]
        validation = safe_metrics(cohort["y_all"][np.ix_(val, columns)], probability[val], audit["labels"])
        if not np.isclose(validation["macro_ap"], audit["validation_macro_ap"], atol=1e-12, rtol=0):
            raise ValueError("Reloaded readout changed validation selection score")
        predictions[audit["id"]] = probability
        extra_predictions[audit["id"]] = extra
        results.append({**audit, "metrics": {"val": validation,
                        "test": safe_metrics(cohort["y_all"][np.ix_(test, columns)], probability[test], audit["labels"]),
                        "supplemental": safe_metrics(supplemental["y_all"][:, extra_columns], extra, audit["labels"])}})
    np.savez_compressed(output / "probabilities.npz", **predictions)
    np.savez_compressed(output / "supplemental_probabilities.npz", **extra_predictions)
    result = {"identity": identity, "selection_seal_sha256": hash_file(output / "selection_seal.json"),
              "n_runs": len(results), "n_train": len(train), "n_val": len(val), "n_test": len(test), "n_supplemental": len(supplemental["sample_ids"]),
              "probability_archives": {name: hash_file(output / name) for name in ("probabilities.npz", "supplemental_probabilities.npz")},
              "scope": protocol["comparison_scope"], "runs": results}
    write_json(output / "results.json", result)
    print(f"Completed {len(results)} frozen/tail pre/post MOA readouts; original66 and retained67 separately", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--export-dir", type=Path, default=root / "artifacts/dino_last2_v1/representations")
    parser.add_argument("--cohort-root", type=Path, default=root / "artifacts/expanded_moa_probe_v1")
    parser.add_argument("--output", type=Path, default=root / "artifacts/dino_last2_v1/moa")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.export_dir, args.cohort_root, args.output, args.threads)

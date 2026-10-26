#!/usr/bin/env python3
"""Separate CLIP representation changes from MOA probe training-set changes."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wellclip.common import hash_file, read_json, write_json

OLD = ROOT / "artifacts/direct256_v1"
NEW = ROOT / "artifacts/expanded_samples_v2"
OUT = ROOT / "artifacts/expansion_diagnosis_v2/moa_cross"


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def bound(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Diagnostic identity changed: {path}")
    write_json(path, value)


def export(output):
    import torch
    from wellclip.lr64_extended_training import load_extended_lr64_head
    from wellclip.training import load_features

    torch.set_num_threads(2)
    manifest, bank, metadata = load_features(NEW / "manifest.json", NEW / "features")
    old_manifest = read_json(ROOT / "artifacts/expanded_samples_v1/manifest.json")
    old_store = arrays(OLD / "representations/primary_store.npz")
    new_store = arrays(NEW / "moa/representations/primary_store.npz")
    ids = bank["sample_ids"]
    if (old_store["sample_ids"].tolist() != [s["sample_id"] for s in old_manifest["samples"]]
            or new_store["sample_ids"].tolist() != ids.tolist()
            or ids[:len(old_manifest["samples"] )].tolist() != old_store["sample_ids"].tolist()):
        raise ValueError("Old/new feature rows disagree")
    output.mkdir(parents=True, exist_ok=True)
    n_old = len(old_store["sample_ids"])
    store = {"sample_ids": ids, "dino1920": bank["image_features"].reshape(len(ids), 1920)}
    specs = [{"key": "dino1920", "stage": "raw", "dimension": 1920, "clip_cohort": "shared"}]
    bindings = {str(p): hash_file(p) for p in (
        NEW / "manifest.json", NEW / "features/metadata.json", NEW / "features/features.npz",
        OLD / "representations/primary_store.npz", NEW / "moa/representations/primary_store.npz",
        NEW / "moa/cohort/cohort_arrays.npz", OLD / "moa/results.json", OLD / "moa/probabilities.npz",
        NEW / "moa/results.json", NEW / "moa/probabilities.npz", Path(__file__))}
    for seed in (42, 43, 44):
        for alignment in ("matched", "shuffled"):
            old_name = f"original128__{alignment}_seed{seed}"
            new_name = f"lr1e-04__{alignment}_seed{seed}"
            path = OLD / "training" / old_name / "best.pt"
            bindings[str(path)] = hash_file(path)
            model, _ = load_extended_lr64_head(path)
            with torch.inference_mode():
                added = model.encode_image(torch.from_numpy(bank["image_features"][n_old:].copy())).numpy()
                check = model.encode_image(torch.from_numpy(bank["image_features"][:n_old].copy())).numpy()
            original = old_store[old_name + "__embedding"]
            if not np.allclose(check, original, atol=1e-6):
                raise ValueError("Reloaded old head disagrees with authenticated old features")
            for label, value in (("old", np.concatenate([original, added])),
                                 ("new", new_store[new_name + "__embedding"])):
                key = f"{label}__{alignment}_seed{seed}"
                if value.shape != (len(ids), 128) or not np.isfinite(value).all():
                    raise ValueError("Invalid cross-readout embedding")
                store[key] = value
                specs.append({"key": key, "stage": "embedding", "dimension": 128,
                              "clip_cohort": label, "alignment": alignment, "seed": seed})
    protocol = {"experiment": "2x2 CLIP old603/new1074 representations x MOA probe603/probe1074",
                "bindings": bindings, "manifest_id": manifest["manifest_id"], "feature_id": metadata["feature_id"],
                "specs": specs, "train_sizes": [603, 1074], "primary_labels": [
                    "dopamine receptor antagonist", "serotonin receptor antagonist"],
                "C_grid": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "selection": "Established train-only scaler, balanced logistic, global C by original val66 macroAP; all26 sealed before prediction",
                "test_used_for_selection": False, "retrained_clip_backbone_or_head": False,
                "scope": "Posthoc controlled diagnostic on previously examined test66; no independent external confirmation"}
    bound(output / "protocol.json", protocol)
    np.savez_compressed(output / "features.npz", **store)
    bound(output / "export_complete.json", {"protocol_sha256": hash_file(output / "protocol.json"),
           "features_sha256": hash_file(output / "features.npz")})
    print("EXPORTED old/new heads on identical1206 wells; CLIP models unchanged", flush=True)


def fit(output):
    import joblib
    from threadpoolctl import threadpool_limits
    from run_direct256_moa_probe import fit_direct256_readout, seal_selection, load_sealed_selection
    from run_dimension_moa_probe import predict_dimension_saved
    from run_expanded_moa_probe import safe_metrics

    complete = read_json(output / "export_complete.json")
    if (complete["protocol_sha256"] != hash_file(output / "protocol.json")
            or complete["features_sha256"] != hash_file(output / "features.npz")):
        raise ValueError("Cross-readout export changed")
    protocol = read_json(output / "protocol.json")
    for path, digest in protocol["bindings"].items():
        if hash_file(path) != digest:
            raise ValueError(f"Diagnostic source changed: {path}")
    store = arrays(output / "features.npz")
    cohort = arrays(NEW / "moa/cohort/cohort_arrays.npz")
    old_manifest = read_json(ROOT / "artifacts/expanded_samples_v1/manifest.json")
    old_train_ids = {s["sample_id"] for s in old_manifest["samples"] if s["split"] == "train"}
    if store.pop("sample_ids").tolist() != cohort["sample_ids"].tolist():
        raise ValueError("Cross-readout cohort order changed")
    labels = protocol["primary_labels"]
    columns = [cohort["label_names"].tolist().index(label) for label in labels]
    y = cohort["y_all"][:, columns]
    selections = {603: np.flatnonzero(np.isin(cohort["sample_ids"], sorted(old_train_ids))),
                  1074: np.flatnonzero(cohort["splits"] == "train")}
    val, test = (np.flatnonzero(cohort["splits"] == split) for split in ("val", "test"))
    assert all(len(rows) == n and np.all(cohort["splits"][rows] == "train") for n, rows in selections.items())
    identity = {"protocol_sha256": hash_file(output / "protocol.json"),
                "export_sha256": hash_file(output / "export_complete.json")}
    (output / "models").mkdir(exist_ok=True)
    selected = []
    for spec in protocol["specs"]:
        for n, train in selections.items():
            key = spec["key"] + f"__probe{n}"
            with threadpool_limits(limits=2):
                model, audit = fit_direct256_readout(spec, {k:v[train] for k,v in store.items()},
                    cohort["sources"][train], y[train], {k:v[val] for k,v in store.items()},
                    cohort["sources"][val], y[val], protocol["C_grid"])
            model.update(id=key, labels=labels)
            path = output / "models" / (key + ".joblib")
            joblib.dump(model, path)
            selected.append({**spec, **audit, "id": key, "n_train": n, "identity": identity,
                "labels": labels, "model_path": str(path.relative_to(output)), "model_sha256": hash_file(path),
                "test_used_for_model_selection": False})
    seal_selection(output, identity, selected)
    selected, seal = load_sealed_selection(output, identity)
    old_probs, new_probs = (arrays(path / "moa/probabilities.npz") for path in (OLD, NEW))
    old_results, new_results = (read_json(path / "moa/results.json") for path in (OLD, NEW))
    results, probabilities = [], {"sample_ids": cohort["sample_ids"], "splits": cohort["splits"]}
    for audit in selected:
        model = joblib.load(output / audit["model_path"])
        with threadpool_limits(limits=2):
            p = predict_dimension_saved(model, store, cohort["sources"])
        probabilities[audit["id"]] = p
        metrics = {split: safe_metrics(y[rows], p[rows], labels) for split, rows in (("val",val),("test",test))}
        if not np.isclose(metrics["val"]["macro_ap"], audit["validation_macro_ap"], atol=1e-12):
            raise ValueError("Sealed readout changed validation score")
        reproduced = None
        diagonal = ((audit["clip_cohort"] == "old" and audit["n_train"] == 603)
                    or (audit["clip_cohort"] == "new" and audit["n_train"] == 1074)
                    or audit["clip_cohort"] == "shared")
        if diagonal:
            old = audit["n_train"] == 603
            archive, saved = (old_probs, old_results) if old else (new_probs, new_results)
            rep = "dino1920" if audit["stage"] == "raw" else (
                f"original128__{audit['alignment']}_seed{audit['seed']}__embedding" if old else
                f"lr1e-04__{audit['alignment']}_seed{audit['seed']}__embedding")
            baseline_id = "primary__" + rep
            reference = next(r for r in saved["runs"] if r["id"] == baseline_id)
            lookup = {str(x):i for i,x in enumerate(archive["sample_ids"])}
            idx = [lookup[str(cohort["sample_ids"][i])] for i in test]
            if (not np.allclose(p[test], archive[baseline_id][idx], atol=1e-6)
                    or metrics["test"] != reference["metrics"]["test"]):
                raise ValueError("Cross-readout diagonal did not reproduce original predictions")
            reproduced = baseline_id
        results.append({**audit, "metrics": metrics, "original_diagonal_reproduced": reproduced})
    np.savez_compressed(output / "probabilities.npz", **probabilities)
    result = {"identity": identity, "selection_seal_sha256": hash_file(output / "selection_seal.json"),
              "n_readouts": len(results), "test_count": len(test), "primary_labels": labels,
              "all26_selected_before_prediction": True, "runs": results,
              "probabilities_sha256": hash_file(output / "probabilities.npz")}
    bound(output / "results.json", result)
    bound(output / "complete.json", {"results_sha256": hash_file(output / "results.json"),
          "selection_seal_sha256": hash_file(output / "selection_seal.json")})
    for size in (603,1074):
        for family in ("old","new"):
            group=[r["metrics"]["test"]["macro_ap"] for r in results
                   if r["n_train"]==size and r["clip_cohort"]==family and r.get("alignment")=="matched"]
            print(f"CLIP={family}, MOAtrain={size}: {np.mean(group):.6f} +/- {np.std(group,ddof=1):.6f}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--stage", choices=("export","fit","all"), default="all")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == "all":
        for stage, executable in (("export",ROOT/".venv/bin/python"),("fit",ROOT.parent/".venv/bin/python")):
            subprocess.run([str(executable),"-u",str(Path(__file__)),"--stage",stage,"--output",str(args.output)],check=True)
    elif args.stage == "export":
        export(args.output)
    else:
        fit(args.output)

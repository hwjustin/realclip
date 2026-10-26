from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


def module():
    directory = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(directory))
    import run_sample_expansion_moa
    return run_sample_expansion_moa


def sample(identifier, parent=None, source="s", plate=None, well=None, digest=None):
    return {"sample_id": identifier, "parent_id": parent or identifier,
            "source": source, "plate": plate or identifier,
            "well_id": well or identifier, "images": [{"sha256": digest or identifier}]}


def test_supplemental_filter_uses_identity_only_and_keeps_source_in_plate_key():
    probe = module()
    primary = [sample("training", parent="parent", plate="plate", well="well", digest="raw")]
    extra = [sample("parent_hit", parent="parent"), sample("plate_hit", plate="plate"),
             sample("well_hit", well="well"), sample("raw_hit", digest="raw"),
             sample("different_source", source="s2", plate="plate"), sample("clean")]
    kept, audit = probe.supplemental_indices(primary, extra)
    assert kept == [4, 5]
    assert [row["reasons"] for row in audit[:4]] == [["primary_parent"], ["primary_plate"], ["primary_well"], ["primary_raw_image"]]
    # Labels and predictions are deliberately absent from the filter contract.
    assert all(row["eligible"] for row in audit[4:])
    assert probe.supplemental_indices(primary, [primary[0]])[0] == []


def test_cohort_completion_rejects_annotation_mutation(tmp_path):
    probe = module()
    probe.write_json(tmp_path / "protocol.json", {"split_counts": {"train": 1000, "val": 66, "test": 66}})
    probe.write_json(tmp_path / "cohort.json", {"sample_ids": ["a"]})
    probe.write_json(tmp_path / "complete.json", {"artifacts": {name: probe.hash_file(tmp_path / name)
                     for name in ("protocol.json", "cohort.json")}})
    assert probe.validate_cohort_files(tmp_path)["split_counts"]["train"] == 1000
    probe.write_json(tmp_path / "cohort.json", {"sample_ids": ["changed"]})
    with pytest.raises(ValueError, match="cohort artifact changed"):
        probe.validate_cohort_files(tmp_path)


def test_old_probability_comparison_uses_same_clean_subset_and_fixed_labels(tmp_path):
    pytest.importorskip("sklearn")
    probe = module()
    labels = ["first", "second"]
    rep = "original128__matched_seed42__embedding"
    key = "primary__" + rep
    np.savez_compressed(tmp_path / "probabilities.npz", sample_ids=np.asarray(["t2", "t1"]),
                        **{key: np.asarray([[.1, .9], [.9, .1]])})
    np.savez_compressed(tmp_path / "supplemental_probabilities.npz", sample_ids=np.asarray(["s1", "s2", "contaminated"]),
                        **{key: np.asarray([[.9, .1], [.1, .9], [1., 1.]])})
    probe.write_json(tmp_path / "results.json", {"probability_archives": {name: probe.hash_file(tmp_path / name)
        for name in ("probabilities.npz", "supplemental_probabilities.npz")},
        "runs": [{"id": key, "labels": labels}]})
    spec = {"stage": "embedding", "alignment": "matched", "seed": 42}
    result = probe.baseline_metrics(tmp_path, spec, "primary",
        {"sample_ids": np.asarray(["t1", "t2"]), "y": np.asarray([[1, 0], [0, 1]])},
        {"y": np.asarray([[1, 0], [0, 1]])}, ["s1", "s2"])
    assert result["metrics"]["test"]["macro_ap"] == 1.
    assert result["metrics"]["supplemental"]["n_samples"] == 2
    assert result["metrics"]["supplemental"]["macro_ap"] == 1.
    unsupported = probe.baseline_metrics(tmp_path, spec, "primary",
        {"sample_ids": np.asarray(["t1", "t2"]), "y": np.asarray([[1, 0], [0, 1]])},
        {"y": np.asarray([[1, 0]])}, ["s1"])
    assert unsupported["metrics"]["supplemental"]["macro_ap"] is None
    assert unsupported["metrics"]["supplemental"]["n_labels"] == 2
    with pytest.raises(ValueError, match="label order differ"):
        probe.baseline_metrics(tmp_path, {**spec, "labels": list(reversed(labels))}, "primary",
            {"sample_ids": np.asarray(["t1", "t2"]), "y": np.asarray([[1, 0], [0, 1]])},
            {"y": np.empty((0, 2))}, [])


def test_no_clean_supplemental_rows_yields_no_baseline_score(tmp_path):
    pytest.importorskip("sklearn")
    probe = module()
    key = "primary__dino1920"
    np.savez_compressed(tmp_path / "probabilities.npz", sample_ids=np.asarray(["t1", "t2"]),
                        **{key: np.asarray([[.9], [.1]])})
    probe.write_json(tmp_path / "results.json", {"probability_archives": {
        "probabilities.npz": probe.hash_file(tmp_path / "probabilities.npz")},
        "runs": [{"id": key, "labels": ["first"]}]})
    result = probe.baseline_metrics(tmp_path, {"stage": "raw"}, "primary",
        {"sample_ids": np.asarray(["t1", "t2"]), "y": np.asarray([[1], [0]])},
        {"y": np.empty((0, 1))}, [])
    assert result["metrics"]["supplemental"] is None

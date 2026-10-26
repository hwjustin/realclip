"""Projection-only, full-well contrastive training with a sealed test split."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .common import hash_file, load_manifest, read_json, stable_hash, utc_now, write_json
from .model import WellCLIP, symmetric_loss


def retrieval_metrics(similarities: np.ndarray) -> dict:
    """Paired square gallery metrics, with stable manifest-order tie resolution."""
    similarities = np.asarray(similarities, dtype=np.float64)
    if similarities.ndim != 2 or similarities.shape[0] != similarities.shape[1]:
        raise ValueError("Paired retrieval requires a square similarity matrix")
    if not len(similarities) or not np.isfinite(similarities).all():
        raise ValueError("Empty or nonfinite retrieval similarities")

    def direction(scores: np.ndarray) -> dict:
        order = np.argsort(-scores, axis=1, kind="stable")
        ranks = np.argmax(order == np.arange(len(scores))[:, None], axis=1) + 1
        return {
            "n_queries": int(len(scores)), "candidate_count": int(len(scores)),
            **{f"R@{k}": float(np.mean(ranks <= k)) for k in (1, 5, 10)},
            "MRR": float(np.mean(1.0 / ranks)), "median_rank": float(np.median(ranks)),
        }

    image_to_text = direction(similarities)
    text_to_image = direction(similarities.T)
    return {"image_to_text": image_to_text, "text_to_image": text_to_image,
            "mean_mrr": (image_to_text["MRR"] + text_to_image["MRR"]) / 2}


def random_expectation(n: int) -> dict:
    if n < 1:
        raise ValueError("Gallery must be nonempty")
    return {"candidate_count": n, **{f"R@{k}": min(k, n) / n for k in (1, 5, 10)},
            "MRR": sum(1.0 / rank for rank in range(1, n + 1)) / n}


def derangement(n: int, seed: int) -> np.ndarray:
    if n < 2:
        raise ValueError("A shuffled control needs at least two training wells")
    rng = np.random.default_rng(seed)
    original = np.arange(n)
    for _ in range(10000):
        assignment = rng.permutation(n)
        if np.all(assignment != original):
            return assignment
    raise RuntimeError("Unable to construct a derangement")


def split_indices(manifest: dict) -> dict[str, np.ndarray]:
    samples = manifest["samples"]
    splits = {name: np.asarray([i for i, row in enumerate(samples) if row["split"] == name],
                               dtype=np.int64) for name in ("train", "val", "test")}
    if any(not len(indices) for indices in splits.values()):
        raise ValueError("Training, validation and test splits must be nonempty")
    if sum(map(len, splits.values())) != len(samples):
        raise ValueError("Unknown split label")
    for field in ("well_id", "parent_id"):
        identities = [{samples[i][field] for i in splits[name]} for name in splits]
        if any(identities[a] & identities[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f"Cross-split {field} overlap")
    plates = [{(samples[i]["source"], samples[i]["plate"]) for i in splits[name]}
              for name in splits]
    if any(plates[a] & plates[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Cross-split plate overlap")
    return splits


def load_features(manifest_path: str | Path, features_dir: str | Path) -> tuple[dict, dict, dict]:
    manifest = load_manifest(manifest_path)
    features_dir = Path(features_dir)
    metadata = read_json(features_dir / "metadata.json")
    feature_path = features_dir / "features.npz"
    if metadata.get("manifest_id") != manifest["manifest_id"]:
        raise ValueError("Features belong to a different manifest")
    expected_hash = metadata.get("artifacts", {}).get("features.npz")
    if not expected_hash or expected_hash != hash_file(feature_path):
        raise ValueError("Feature archive checksum mismatch")
    if not metadata.get("feature_id"):
        raise ValueError("Missing frozen-feature identity")
    if "identity" in metadata and stable_hash(metadata["identity"]) != metadata["feature_id"]:
        raise ValueError("Frozen-feature identity checksum mismatch")
    with np.load(feature_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    expected_ids = [sample["sample_id"] for sample in manifest["samples"]]
    if arrays["sample_ids"].tolist() != expected_ids:
        raise ValueError("Feature rows do not match manifest sample ordering")
    n = len(expected_ids)
    for name, shape in (("image_features", (n, 5, 384)), ("text_features", (n, 512))):
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"Invalid {name} shape or values")
        arrays[name] = np.asarray(arrays[name], dtype=np.float32)
    split_indices(manifest)
    return manifest, arrays, metadata


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def load_run_model(checkpoint_path: str | Path, device: str = "cpu") -> tuple[WellCLIP, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = WellCLIP()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def _validate_run(directory: Path, protocol_id: str) -> dict:
    receipt = read_json(directory / "complete.json")
    if receipt.get("protocol_id") != protocol_id:
        raise ValueError(f"Run identity mismatch: {directory}")
    for filename, digest in receipt["artifacts"].items():
        if hash_file(directory / filename) != digest:
            raise ValueError(f"Run artifact checksum mismatch: {directory / filename}")
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    if (checkpoint.get("protocol_id") != protocol_id or
            checkpoint.get("arm") != receipt["arm"] or checkpoint.get("seed") != receipt["seed"]):
        raise ValueError(f"Checkpoint identity mismatch: {directory}")
    return receipt


def validate_training(training_dir: str | Path, manifest_id: str, feature_id: str) -> tuple[dict, list[dict]]:
    """Validate every finalized run before any caller reads test embeddings."""
    training_dir = Path(training_dir)
    protocol = read_json(training_dir / "protocol.json")
    payload = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if protocol["protocol_id"] != stable_hash(payload):
        raise ValueError("Training protocol checksum mismatch")
    if protocol["manifest_id"] != manifest_id or protocol["feature_id"] != feature_id:
        raise ValueError("Training input identity mismatch")
    complete = read_json(training_dir / "complete.json")
    if complete.get("protocol_id") != protocol["protocol_id"] or not complete.get("all_complete"):
        raise ValueError("Training has not finalized all checkpoints")
    expected_names = [f"{arm}_seed{seed}" for seed in protocol["seeds"]
                      for arm in ("matched", "shuffled")]
    if sorted(complete["runs"]) != sorted(expected_names):
        raise ValueError("Missing or extra training runs")
    receipts = []
    for name in expected_names:
        if hash_file(training_dir / name / "complete.json") != complete["runs"][name]:
            raise ValueError(f"Finalized run receipt changed: {name}")
        receipt = _validate_run(training_dir / name, protocol["protocol_id"])
        receipts.append({"name": name, **receipt})
    return protocol, receipts


def train(manifest_path: str | Path, features_dir: str | Path, output_dir: str | Path,
          device: str = "cuda", steps: int = 1000, seeds: tuple[int, ...] = (42, 43, 44),
          eval_every: int = 20) -> dict:
    if steps < 1 or eval_every < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid training steps, evaluation interval or seeds")
    manifest, arrays, metadata = load_features(manifest_path, features_dir)
    indices = split_indices(manifest)
    descriptions = [manifest["samples"][i]["description"] for i in indices["train"]]
    if len(set(descriptions)) != len(descriptions):
        raise ValueError("Duplicate training captions require an explicit multi-positive protocol")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1, "manifest_id": manifest["manifest_id"],
        "feature_id": metadata["feature_id"],
        "features_sha256": metadata["artifacts"]["features.npz"],
        "seeds": list(seeds), "steps": steps, "eval_every": eval_every,
        "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.01},
        "batch": "entire_train_split_one_row_per_well",
        "selection": "maximum_mean_bidirectional_val_MRR_earliest_tie",
        "split_ids": {name: [manifest["samples"][i]["sample_id"] for i in values]
                      for name, values in indices.items()},
        "code": {name: hash_file(Path(__file__).with_name(name))
                 for name in ("training.py", "model.py")},
    }
    protocol = {**payload, "protocol_id": stable_hash(payload)}
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("Existing training directory uses a different frozen protocol")
    write_json(protocol_path, protocol)
    if (output_dir / "complete.json").exists():
        validate_training(output_dir, manifest["manifest_id"], metadata["feature_id"])
        print("Reused all authenticated training checkpoints", flush=True)
        return read_json(output_dir / "complete.json")
    image_features = torch.from_numpy(arrays["image_features"]).to(device)
    text_features = torch.from_numpy(arrays["text_features"]).to(device)
    train_idx = torch.as_tensor(indices["train"], device=device)
    val_idx = torch.as_tensor(indices["val"], device=device)
    finalized = {}
    for seed in seeds:
        initial_hash = None
        for arm in ("matched", "shuffled"):
            name = f"{arm}_seed{seed}"
            directory = output_dir / name
            directory.mkdir(exist_ok=True)
            if (directory / "complete.json").exists():
                receipt = _validate_run(directory, protocol["protocol_id"])
                if initial_hash is not None and receipt["initial_state_sha256"] != initial_hash:
                    raise ValueError("Matched and shuffled initialization differs")
                initial_hash = receipt["initial_state_sha256"]
                finalized[name] = hash_file(directory / "complete.json")
                print(f"Reused {name}: step={receipt['best_step']} val_MRR={receipt['validation_mrr']:.5f}", flush=True)
                continue
            seed_everything(seed)
            model = WellCLIP().to(device)
            run_initial_hash = state_sha256(model.state_dict())
            if initial_hash is not None and run_initial_hash != initial_hash:
                raise ValueError("Matched and shuffled initialization differs")
            initial_hash = run_initial_hash
            assignment = np.arange(len(train_idx)) if arm == "matched" else derangement(len(train_idx), seed)
            text_idx = train_idx[torch.as_tensor(assignment, device=device)]
            assignment_rows = [{"image_sample_id": manifest["samples"][int(i)]["sample_id"],
                                "text_sample_id": manifest["samples"][int(j)]["sample_id"]}
                               for i, j in zip(indices["train"], indices["train"][assignment])]
            write_json(directory / "assignment.json", assignment_rows)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            history, best_state, best_score, best_step = [], None, -1.0, 0
            for step in range(1, steps + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                image_embeddings, text_embeddings, scale = model(image_features[train_idx], text_features[text_idx])
                loss = symmetric_loss(scale * image_embeddings @ text_embeddings.T)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in {name} at step {step}")
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                       for parameter in model.parameters()):
                    raise FloatingPointError(f"Nonfinite gradient in {name} at step {step}")
                optimizer.step()
                row: dict[str, Any] = {"step": step, "train_loss": float(loss.detach().cpu())}
                if step % eval_every == 0 or step == steps:
                    model.eval()
                    with torch.no_grad():
                        val_images = model.encode_image(image_features[val_idx])
                        val_texts = model.encode_text(text_features[val_idx])
                        metrics = retrieval_metrics((val_images @ val_texts.T).cpu().numpy())
                    row["validation"] = metrics
                    score = metrics["mean_mrr"]
                    if score > best_score:
                        best_score, best_step = score, step
                        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    if step % max(eval_every, 100) == 0 or step == steps:
                        print(f"{name} step={step}/{steps} loss={row['train_loss']:.5f} val_MRR={score:.5f}", flush=True)
                history.append(row)
            checkpoint = {
                "schema_version": 1, "protocol_id": protocol["protocol_id"],
                "manifest_id": manifest["manifest_id"], "feature_id": metadata["feature_id"],
                "arm": arm, "seed": seed, "best_step": best_step,
                "validation_mrr": best_score, "initial_state_sha256": initial_hash,
                "state_dict": best_state,
            }
            temporary = directory / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(directory / "best.pt")
            write_json(directory / "history.json", history)
            reloaded, _ = load_run_model(directory / "best.pt", device)
            with torch.no_grad():
                loaded_score = retrieval_metrics((reloaded.encode_image(image_features[val_idx]) @
                                                  reloaded.encode_text(text_features[val_idx]).T).cpu().numpy())["mean_mrr"]
            if loaded_score != best_score:
                raise ValueError("Checkpoint reload changed validation retrieval")
            receipt = {
                "schema_version": 1, "protocol_id": protocol["protocol_id"], "arm": arm,
                "seed": seed, "best_step": best_step, "validation_mrr": best_score,
                "initial_state_sha256": initial_hash, "completed_at": utc_now(),
                "trainable_parameters": sum(parameter.numel() for parameter in reloaded.parameters() if parameter.requires_grad),
                "artifacts": {filename: hash_file(directory / filename)
                              for filename in ("best.pt", "history.json", "assignment.json")},
            }
            write_json(directory / "complete.json", receipt)
            finalized[name] = hash_file(directory / "complete.json")
    complete = {"schema_version": 1, "protocol_id": protocol["protocol_id"],
                "all_complete": True, "completed_at": utc_now(), "runs": finalized}
    write_json(output_dir / "complete.json", complete)
    validate_training(output_dir, manifest["manifest_id"], metadata["feature_id"])
    return complete

"""Epoch-based batch64/full603 comparison on unchanged frozen well features.

All minibatches contain complete wells. The original 128-dimensional WellCLIP
head is unchanged, and neither image nor text backbone participates in training.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .common import hash_file, read_json, stable_hash, utc_now, write_json
from .model import WellCLIP, symmetric_loss
from .training import (derangement, load_features, retrieval_metrics,
                       seed_everything, split_indices, state_sha256)

DEFAULT_CONFIG = {
    "batch_sizes": [64, 603], "epochs": 100, "seeds": [42, 43, 44],
    "lr": 1e-3, "weight_decay": 0.01,
    "expected_split_counts": {"train": 603, "val": 66, "test": 66},
}
ALIGNMENTS = ("matched", "shuffled")


def run_name(batch_size: int, alignment: str, seed: int) -> str:
    if alignment not in ALIGNMENTS:
        raise ValueError("Unknown minibatch alignment")
    return f"bs{batch_size}__{alignment}_seed{seed}"


def expected_runs(config: dict) -> list[str]:
    return [run_name(batch, alignment, seed) for batch in config["batch_sizes"]
            for seed in config["seeds"] for alignment in ALIGNMENTS]


def epoch_orders(n: int, epochs: int, seed: int) -> list[np.ndarray]:
    """Independent of torch/dropout RNG, batching, and caption assignment."""
    if n < 2 or epochs < 1:
        raise ValueError("At least two wells and a positive epoch count required")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 640603]))
    return [rng.permutation(n) for _ in range(epochs)]


def minibatches(order: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Keep every complete well, including the final partial batch."""
    order = np.asarray(order, dtype=np.int64)
    if order.ndim != 1 or batch_size < 2:
        raise ValueError("A one-dimensional order and batch size >=2 are required")
    if not np.array_equal(np.sort(order), np.arange(len(order))):
        raise ValueError("Epoch order must contain every training well exactly once")
    batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
    if any(len(batch) < 2 for batch in batches):
        raise ValueError("Singleton final contrastive batch needs a separately declared policy")
    return batches


def _config(config: dict | None) -> dict:
    supplied = config or {}
    if set(supplied) - set(DEFAULT_CONFIG):
        raise ValueError("Unknown minibatch training configuration")
    result = {**DEFAULT_CONFIG, **supplied}
    for key in ("batch_sizes", "seeds"):
        result[key] = list(result[key])
        if (not result[key] or len(set(result[key])) != len(result[key])
                or any(not isinstance(x, int) or isinstance(x, bool) for x in result[key])):
            raise ValueError(f"Invalid integer {key}")
    if any(x < 2 for x in result["batch_sizes"]):
        raise ValueError("Contrastive batch sizes must be >=2")
    if not isinstance(result["epochs"], int) or isinstance(result["epochs"], bool) or result["epochs"] < 1:
        raise ValueError("Positive integer epochs required")
    if (not math.isfinite(result["lr"]) or result["lr"] <= 0
            or not math.isfinite(result["weight_decay"]) or result["weight_decay"] < 0):
        raise ValueError("Invalid optimizer configuration")
    counts = result["expected_split_counts"]
    if (set(counts) != {"train", "val", "test"} or any(
            not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in counts.values())):
        raise ValueError("Invalid expected split counts")
    return result


def load_minibatch_head(checkpoint_path: str | Path, device: str = "cpu") -> tuple[WellCLIP, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = WellCLIP()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def _validate_run(directory: Path, protocol: dict) -> dict:
    receipt = read_json(directory / "complete.json")
    for key in ("protocol_id", "manifest_id", "feature_id"):
        if receipt.get(key) != protocol.get(key):
            raise ValueError(f"Minibatch run {key} mismatch")
    if (receipt["name"] != directory.name or receipt["name"] != run_name(
            receipt["batch_size"], receipt["alignment"], receipt["seed"])):
        raise ValueError("Minibatch run identity mismatch")
    for name, digest in receipt["artifacts"].items():
        if hash_file(directory / name) != digest:
            raise ValueError(f"Minibatch run artifact checksum mismatch: {name}")
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    for key in ("name", "batch_size", "alignment", "seed", "protocol_id", "manifest_id",
                "feature_id", "best_epoch", "best_step", "validation_mrr",
                "initial_state_sha256", "selected_state_sha256"):
        if checkpoint.get(key) != receipt.get(key):
            raise ValueError(f"Minibatch checkpoint/receipt {key} mismatch")
    if state_sha256(checkpoint["state_dict"]) != receipt["selected_state_sha256"]:
        raise ValueError("Minibatch selected state checksum mismatch")
    order_hashes = protocol["epoch_order_sha256"][str(receipt["seed"])]
    if receipt["epoch_order_sha256"] != order_hashes:
        raise ValueError("Minibatch epoch order differs from shared protocol")
    n = protocol["config"]["expected_split_counts"]["train"]
    sizes = [len(batch) for batch in minibatches(np.arange(n), receipt["batch_size"])]
    if (receipt["batch_sizes_per_epoch"] != sizes or receipt["steps_per_epoch"] != len(sizes)
            or receipt["total_steps"] != len(sizes) * protocol["config"]["epochs"]
            or receipt["examples_seen"] != n * protocol["config"]["epochs"]):
        raise ValueError("Minibatch epoch coverage or update count differs")
    return {**receipt, "checkpoint_path": f"{directory.name}/best.pt"}


def validate_minibatch_training(training_dir: str | Path, manifest_id: str | None = None,
                                feature_id: str | None = None) -> tuple[dict, list[dict]]:
    directory = Path(training_dir)
    protocol = read_json(directory / "protocol.json")
    if stable_hash({k: v for k, v in protocol.items() if k != "protocol_id"}) != protocol["protocol_id"]:
        raise ValueError("Minibatch protocol checksum mismatch")
    if manifest_id is not None and protocol["manifest_id"] != manifest_id:
        raise ValueError("Minibatch manifest identity mismatch")
    if feature_id is not None and protocol["feature_id"] != feature_id:
        raise ValueError("Minibatch feature identity mismatch")
    complete = read_json(directory / "complete.json")
    names = expected_runs(protocol["config"])
    if (complete.get("protocol_id") != protocol["protocol_id"] or complete.get("all_complete") is not True
            or set(complete["runs"]) != set(names)):
        raise ValueError("All minibatch checkpoints must complete before evaluation")
    rows, initial_states = [], {}
    for name in names:
        if hash_file(directory / name / "complete.json") != complete["runs"][name]:
            raise ValueError("Finalized minibatch receipt changed")
        row = _validate_run(directory / name, protocol)
        if initial_states.setdefault(row["seed"], row["initial_state_sha256"]) != row["initial_state_sha256"]:
            raise ValueError("Batch/alignment initializations differ within seed")
        rows.append(row)
    seal = read_json(directory / "selection_seal.json")
    if (complete["selection_seal_sha256"] != hash_file(directory / "selection_seal.json")
            or seal.get("protocol_id") != protocol["protocol_id"] or seal.get("runs") != complete["runs"]
            or seal.get("all_checkpoints_selected_before_test_export") is not True):
        raise ValueError("Minibatch model-selection seal mismatch")
    return protocol, rows


def _load_inputs(manifest_path, features_dir):
    return load_features(manifest_path, features_dir)


def run_minibatch_training(manifest_path: str | Path, features_dir: str | Path,
                           output_dir: str | Path, config: dict | None = None,
                           device: str = "cuda") -> dict:
    config = _config(config)
    manifest, arrays, metadata = _load_inputs(manifest_path, features_dir)
    indices = split_indices(manifest)
    if {key: len(rows) for key, rows in indices.items()} != config["expected_split_counts"]:
        raise ValueError("Minibatch split counts differ from the declared fixed cohort")
    n = len(manifest["samples"])
    if arrays["sample_ids"].tolist() != [s["sample_id"] for s in manifest["samples"]]:
        raise ValueError("Minibatch feature row ordering changed")
    for key, shape in (("image_features", (n, 5, 384)), ("text_features", (n, 512))):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid minibatch {key}")
    train_descriptions = [manifest["samples"][i]["description"] for i in indices["train"]]
    if len(set(train_descriptions)) != len(train_descriptions):
        raise ValueError("Duplicate captions require a declared multi-positive protocol")
    n_train = len(indices["train"])
    orders = {seed: epoch_orders(n_train, config["epochs"], seed) for seed in config["seeds"]}
    sizes = {str(batch): [len(x) for x in minibatches(orders[config["seeds"][0]][0], batch)]
             for batch in config["batch_sizes"]}
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1, "experiment": "batch64 versus whole603 on unchanged frozen128 WellCLIP",
        "manifest_id": manifest["manifest_id"], "feature_id": metadata["feature_id"],
        "features_sha256": metadata.get("artifacts", {}).get("features.npz"),
        "config": config, "embedding_dim": 128, "drop_last": False,
        "batch_sizes_per_epoch": sizes,
        "epoch_order": "fresh permutation each epoch; separate NumPy Generator(SeedSequence([seed,640603])); shared across batches and alignments",
        "epoch_order_sha256": {str(seed): [stable_hash(order.tolist()) for order in values]
                               for seed, values in orders.items()},
        "caption_assignment": "matched identity or fixed global training derangement; never re-shuffled within a minibatch",
        "optimizer": {"name": "AdamW", "lr": config["lr"], "weight_decay": config["weight_decay"]},
        "backbones": "original frozen DINOv2-small and BiomedCLIP text; cached features detached; original WellCLIP head only",
        "selection": "every epoch maximum validation mean bidirectional MRR; earliest exact tie; epoch0 reference only",
        "comparison": "same100 epochs and well exposures; batch64 has10updates/epoch versus full603 one; negative pool differs",
        "resume": "completed authenticated runs reused; incomplete run deterministically restarts",
        "test_used_for_model_selection": False, "moa_labels_used_for_training": False,
        "split_ids": {key: [manifest["samples"][i]["sample_id"] for i in rows] for key, rows in indices.items()},
        "code": {name: hash_file(Path(__file__).with_name(name))
                 for name in ("minibatch_training.py", "model.py", "training.py")},
    }
    protocol = {**payload, "protocol_id": stable_hash(payload)}
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("Existing minibatch training uses another immutable protocol")
    if not protocol_path.exists():
        write_json(protocol_path, protocol)
    if (output_dir / "complete.json").exists():
        validate_minibatch_training(output_dir, manifest["manifest_id"], metadata["feature_id"])
        return read_json(output_dir / "complete.json")
    images = torch.from_numpy(np.array(arrays["image_features"], dtype=np.float32, copy=True)).to(device).detach()
    texts = torch.from_numpy(np.array(arrays["text_features"], dtype=np.float32, copy=True)).to(device).detach()
    train_idx, val_idx = (torch.as_tensor(indices[key], device=device) for key in ("train", "val"))
    finalized = {}
    for batch_size in config["batch_sizes"]:
        for seed in config["seeds"]:
            for alignment in ALIGNMENTS:
                name = run_name(batch_size, alignment, seed)
                directory = output_dir / name
                directory.mkdir(exist_ok=True)
                if (directory / "complete.json").exists():
                    _validate_run(directory, protocol)
                    finalized[name] = hash_file(directory / "complete.json")
                    print(f"REUSED {name}", flush=True)
                    continue
                seed_everything(seed)
                model = WellCLIP().to(device)
                initial_hash = state_sha256(model.state_dict())
                assignment = np.arange(n_train) if alignment == "matched" else derangement(n_train, seed)
                write_json(directory / "assignment.json", [
                    {"image_sample_id": manifest["samples"][int(indices["train"][i])]["sample_id"],
                     "text_sample_id": manifest["samples"][int(indices["train"][j])]["sample_id"]}
                    for i, j in enumerate(assignment)])
                if not np.all((assignment == np.arange(n_train)) == (alignment == "matched")):
                    raise ValueError("Caption assignment violates matched/deranged condition")
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
                model.eval()
                with torch.no_grad():
                    reference = retrieval_metrics((model.encode_image(images[val_idx]) @
                                                   model.encode_text(texts[val_idx]).T).cpu().numpy())
                history = [{"epoch": 0, "global_step": 0, "examples_seen": 0,
                            "validation": reference, "eligible_for_selection": False}]
                best_score, best_epoch, best_step, best_state = -math.inf, 0, 0, None
                global_step, examples_seen = 0, 0
                if str(device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                for epoch, order in enumerate(orders[seed], 1):
                    weighted_loss = 0.0
                    batches = minibatches(order, batch_size)
                    for positions in batches:
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        image_rows = train_idx[torch.as_tensor(positions, device=device)]
                        text_rows = train_idx[torch.as_tensor(assignment[positions], device=device)]
                        vi, vt, scale = model(images[image_rows], texts[text_rows])
                        logits = scale * vi @ vt.T
                        if logits.shape != (len(positions), len(positions)):
                            raise RuntimeError("Contrastive loss must have one row per complete well")
                        loss = symmetric_loss(logits)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite minibatch loss")
                        loss.backward()
                        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                            raise FloatingPointError("Missing/nonfinite minibatch head gradient")
                        optimizer.step()
                        global_step += 1
                        examples_seen += len(positions)
                        weighted_loss += float(loss.detach().cpu()) * len(positions)
                    model.eval()
                    with torch.no_grad():
                        validation = retrieval_metrics((model.encode_image(images[val_idx]) @
                                                        model.encode_text(texts[val_idx]).T).cpu().numpy())
                    history.append({"epoch": epoch, "global_step": global_step, "updates": len(batches),
                                    "examples_seen": examples_seen, "train_loss": weighted_loss / n_train,
                                    "batch_sizes": [len(x) for x in batches],
                                    "epoch_order_sha256": stable_hash(order.tolist()), "validation": validation})
                    if validation["mean_mrr"] > best_score:
                        best_score, best_epoch, best_step = validation["mean_mrr"], epoch, global_step
                        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                if best_state is None:
                    raise RuntimeError("No validation-selected minibatch checkpoint")
                elapsed = time.perf_counter() - started
                details = {"name": name, "protocol_id": protocol["protocol_id"],
                           "manifest_id": manifest["manifest_id"], "feature_id": metadata["feature_id"],
                           "batch_size": batch_size, "alignment": alignment, "arm": alignment,
                           "seed": seed, "embedding_dim": 128, "best_epoch": best_epoch,
                           "best_step": best_step, "validation_mrr": best_score,
                           "initial_state_sha256": initial_hash, "selected_state_sha256": state_sha256(best_state)}
                temporary = directory / "best.pt.tmp"
                torch.save({**details, "state_dict": best_state}, temporary)
                temporary.replace(directory / "best.pt")
                write_json(directory / "history.json", history)
                receipt = {**details, "epochs": config["epochs"], "steps_per_epoch": len(sizes[str(batch_size)]),
                           "total_steps": global_step, "examples_seen": examples_seen,
                           "batch_sizes_per_epoch": sizes[str(batch_size)],
                           "epoch_order_sha256": protocol["epoch_order_sha256"][str(seed)],
                           "frozen_backbones": True, "backbone_parameter_count_in_optimizer": 0,
                           "trainable_parameters": sum(p.numel() for p in model.parameters()),
                           "elapsed_seconds": elapsed,
                           "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else 0,
                           "artifacts": {key: hash_file(directory / key) for key in ("best.pt", "history.json", "assignment.json")},
                           "completed_at": utc_now()}
                write_json(directory / "complete.json", receipt)
                finalized[name] = hash_file(directory / "complete.json")
                print(f"SELECTED {name}: epoch={best_epoch} step={best_step} valMRR={best_score:.6f} seconds={elapsed:.2f}", flush=True)
    write_json(output_dir / "selection_seal.json", {"protocol_id": protocol["protocol_id"], "runs": finalized,
               "all_checkpoints_selected_before_test_export": True})
    complete = {"protocol_id": protocol["protocol_id"], "manifest_id": manifest["manifest_id"],
                "feature_id": metadata["feature_id"], "all_complete": True, "runs": finalized,
                "selection_seal_sha256": hash_file(output_dir / "selection_seal.json"), "completed_at": utc_now()}
    write_json(output_dir / "complete.json", complete)
    validate_minibatch_training(output_dir, manifest["manifest_id"], metadata["feature_id"])
    return complete

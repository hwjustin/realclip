"""Extended lower learning rates on the unchanged batch64 frozen model.

Only AdamW's constant learning rate differs across conditions. Every run uses
the same per-seed initialization, epoch permutations, caption assignment and
dropout RNG stream as the existing minibatch64 experiment.
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
from .minibatch_training import epoch_orders, minibatches
from .model import WellCLIP, symmetric_loss
from .training import (derangement, load_features, retrieval_metrics,
                       seed_everything, split_indices, state_sha256)

DEFAULT_CONFIG = {
    "learning_rates": [1e-5, 3e-5, 1e-4, 3e-4], "batch_size": 64,
    "epochs": 300, "seeds": [42, 43, 44], "weight_decay": 0.01,
    "expected_split_counts": {"train": 603, "val": 66, "test": 66},
}
ALIGNMENTS = ("matched", "shuffled")


def lr_key(learning_rate: float) -> str:
    rate = float(learning_rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("A positive finite learning rate is required")
    mantissa, exponent = f"{rate:.12e}".split("e")
    return "lr" + mantissa.rstrip("0").rstrip(".") + "e" + f"{int(exponent):+03d}"


def run_name(learning_rate: float, alignment: str, seed: int) -> str:
    if alignment not in ALIGNMENTS:
        raise ValueError("Unknown learning-rate alignment")
    return f"{lr_key(learning_rate)}__{alignment}_seed{seed}"


def expected_runs(config: dict) -> list[str]:
    return [run_name(rate, alignment, seed) for rate in config["learning_rates"]
            for seed in config["seeds"] for alignment in ALIGNMENTS]


def _config(config: dict | None) -> dict:
    supplied = config or {}
    if set(supplied) - set(DEFAULT_CONFIG):
        raise ValueError("Unknown LR64 training configuration")
    result = {**DEFAULT_CONFIG, **supplied}
    result["learning_rates"] = list(result["learning_rates"])
    result["seeds"] = list(result["seeds"])
    rates, seeds = result["learning_rates"], result["seeds"]
    if (not rates or any(isinstance(x, bool) or not isinstance(x, (int, float))
                         or not math.isfinite(x) or x <= 0 for x in rates)
            or len(set(rates)) != len(rates)
            or len({lr_key(rate) for rate in rates}) != len(rates)):
        raise ValueError("Invalid distinct positive learning rates")
    result["learning_rates"] = [float(x) for x in rates]
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(not isinstance(x, int) or isinstance(x, bool) for x in seeds)):
        raise ValueError("Invalid integer seeds")
    for key in ("epochs", "batch_size"):
        if (not isinstance(result[key], int) or isinstance(result[key], bool)
                or result[key] < (2 if key == "batch_size" else 1)):
            raise ValueError("Invalid epochs or contrastive batch size")
    if not math.isfinite(result["weight_decay"]) or result["weight_decay"] < 0:
        raise ValueError("Invalid weight decay")
    counts = result["expected_split_counts"]
    if (set(counts) != {"train", "val", "test"} or any(
            not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in counts.values())):
        raise ValueError("Invalid expected split counts")
    return result


def load_extended_lr64_head(checkpoint_path: str | Path, device: str = "cpu") -> tuple[WellCLIP, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = WellCLIP()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def select_learning_rate(receipts: list[dict], config: dict) -> dict:
    """Use matched validation scores only; neither test nor shuffled scores enter."""
    scores = []
    for rate in config["learning_rates"]:
        rows = [r for r in receipts if r["learning_rate"] == rate and r["alignment"] == "matched"]
        if (len(rows) != len(config["seeds"])
                or sorted(r["seed"] for r in rows) != sorted(config["seeds"])):
            raise ValueError("Every matched validation seed is required for LR selection")
        values = [float(r["validation_mrr"]) for r in rows]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite LR validation scores")
        scores.append({"learning_rate": rate, "lr_key": lr_key(rate),
                       "mean_validation_mrr": float(np.mean(values)),
                       "std_validation_mrr": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                       "seed_validation_mrr": {str(r["seed"]): float(r["validation_mrr"]) for r in rows}})
    selected = max(scores, key=lambda row: row["mean_validation_mrr"])
    return {"selected_learning_rate": selected["learning_rate"], "selected_lr_key": selected["lr_key"],
            "scores": scores,
            "selected_checkpoints": [run_name(selected["learning_rate"], "matched", seed) for seed in config["seeds"]],
            "selection_rule": "maximum mean of matched per-seed best validation MRR; exact LR ties choose first configured rate",
            "test_used_for_selection": False, "shuffled_scores_used_for_selection": False}


def _validate_run(directory: Path, protocol: dict) -> dict:
    receipt = read_json(directory / "complete.json")
    for key in ("protocol_id", "manifest_id", "feature_id"):
        if receipt.get(key) != protocol.get(key):
            raise ValueError(f"LR64 run {key} mismatch")
    if (receipt["name"] != directory.name or receipt["name"] != run_name(
            receipt["learning_rate"], receipt["alignment"], receipt["seed"])
            or receipt["batch_size"] != protocol["config"]["batch_size"]):
        raise ValueError("LR64 run identity mismatch")
    for name, digest in receipt["artifacts"].items():
        if hash_file(directory / name) != digest:
            raise ValueError(f"LR64 run artifact checksum mismatch: {name}")
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    for key in ("name", "learning_rate", "batch_size", "alignment", "seed", "protocol_id", "manifest_id",
                "feature_id", "best_epoch", "best_step", "validation_mrr",
                "initial_state_sha256", "selected_state_sha256"):
        if checkpoint.get(key) != receipt.get(key):
            raise ValueError(f"LR64 checkpoint/receipt {key} mismatch")
    if state_sha256(checkpoint["state_dict"]) != receipt["selected_state_sha256"]:
        raise ValueError("LR64 selected state checksum mismatch")
    prefix = torch.load(directory / "prefix100.pt", map_location="cpu", weights_only=True)
    for key in ("name", "learning_rate", "batch_size", "alignment", "seed", "protocol_id", "manifest_id",
                "feature_id", "initial_state_sha256"):
        if prefix.get(key) != receipt.get(key):
            raise ValueError(f"Extended LR64 prefix identity {key} mismatch")
    prefix_epochs = min(100, protocol["config"]["epochs"])
    if (prefix.get("prefix_epochs") != prefix_epochs or receipt.get("prefix100_epochs") != prefix_epochs
            or state_sha256(prefix["state_dict"]) != receipt.get("prefix100_best_state_sha256")
            or prefix.get("selected_state_sha256") != receipt.get("prefix100_best_state_sha256")
            or prefix.get("epoch_end_state_sha256") != receipt.get("prefix100_epoch_end_state_sha256")):
        raise ValueError("Extended LR64 prefix checkpoint state mismatch")
    history = read_json(directory / "history.json")
    prefix_history = history[:prefix_epochs + 1]
    selected_prefix = max((row for row in prefix_history if row["epoch"] > 0),
                          key=lambda row: row["validation"]["mean_mrr"])
    if (stable_hash(prefix_history) != receipt.get("prefix100_history_sha256")
            or prefix["best_epoch"] != selected_prefix["epoch"]
            or prefix["best_step"] != selected_prefix["global_step"]
            or prefix["validation_mrr"] != selected_prefix["validation"]["mean_mrr"]):
        raise ValueError("Extended LR64 prefix validation selection mismatch")
    order_hashes = protocol["epoch_order_sha256"][str(receipt["seed"])]
    n = protocol["config"]["expected_split_counts"]["train"]
    sizes = [len(batch) for batch in minibatches(np.arange(n), receipt["batch_size"])]
    if (receipt["epoch_order_sha256"] != order_hashes
            or receipt["batch_sizes_per_epoch"] != sizes or receipt["steps_per_epoch"] != len(sizes)
            or receipt["total_steps"] != len(sizes) * protocol["config"]["epochs"]
            or receipt["examples_seen"] != n * protocol["config"]["epochs"]):
        raise ValueError("LR64 epoch coverage or shared order differs")
    return {**receipt, "checkpoint_path": f"{directory.name}/best.pt"}


def validate_extended_lr64_training(training_dir: str | Path, manifest_id: str | None = None,
                           feature_id: str | None = None) -> tuple[dict, list[dict]]:
    directory = Path(training_dir)
    protocol = read_json(directory / "protocol.json")
    if stable_hash({k: v for k, v in protocol.items() if k != "protocol_id"}) != protocol["protocol_id"]:
        raise ValueError("LR64 protocol checksum mismatch")
    if manifest_id is not None and protocol["manifest_id"] != manifest_id:
        raise ValueError("LR64 manifest identity mismatch")
    if feature_id is not None and protocol["feature_id"] != feature_id:
        raise ValueError("LR64 feature identity mismatch")
    complete = read_json(directory / "complete.json")
    names = expected_runs(protocol["config"])
    if (complete.get("protocol_id") != protocol["protocol_id"] or complete.get("all_complete") is not True
            or set(complete["runs"]) != set(names)):
        raise ValueError("All LR64 checkpoints must complete before evaluation")
    rows, initial_states = [], {}
    for name in names:
        if hash_file(directory / name / "complete.json") != complete["runs"][name]:
            raise ValueError("Finalized LR64 receipt changed")
        row = _validate_run(directory / name, protocol)
        if initial_states.setdefault(row["seed"], row["initial_state_sha256"]) != row["initial_state_sha256"]:
            raise ValueError("LR/alignment initializations differ within seed")
        rows.append(row)
    selection = read_json(directory / "lr_selection.json")
    if (selection != select_learning_rate(rows, protocol["config"])
            or complete["lr_selection_sha256"] != hash_file(directory / "lr_selection.json")):
        raise ValueError("LR64 validation-only learning-rate selection changed")
    seal = read_json(directory / "selection_seal.json")
    if (complete["selection_seal_sha256"] != hash_file(directory / "selection_seal.json")
            or seal.get("protocol_id") != protocol["protocol_id"] or seal.get("runs") != complete["runs"]
            or seal.get("lr_selection_sha256") != complete["lr_selection_sha256"]
            or seal.get("all_checkpoints_selected_before_test_export") is not True
            or seal.get("learning_rate_selected_before_test_export") is not True):
        raise ValueError("LR64 model-selection seal mismatch")
    return protocol, rows


def _load_inputs(manifest_path, features_dir):
    return load_features(manifest_path, features_dir)


def run_extended_lr64_training(manifest_path: str | Path, features_dir: str | Path,
                      output_dir: str | Path, config: dict | None = None,
                      device: str = "cuda") -> dict:
    config = _config(config)
    manifest, arrays, metadata = _load_inputs(manifest_path, features_dir)
    indices = split_indices(manifest)
    if {key: len(rows) for key, rows in indices.items()} != config["expected_split_counts"]:
        raise ValueError("LR64 split counts differ from the declared fixed cohort")
    n = len(manifest["samples"])
    if arrays["sample_ids"].tolist() != [s["sample_id"] for s in manifest["samples"]]:
        raise ValueError("LR64 feature row ordering changed")
    for key, shape in (("image_features", (n, 5, 384)), ("text_features", (n, 512))):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid LR64 {key}")
    train_descriptions = [manifest["samples"][i]["description"] for i in indices["train"]]
    if len(set(train_descriptions)) != len(train_descriptions):
        raise ValueError("Duplicate captions require a declared multi-positive protocol")
    n_train = len(indices["train"])
    orders = {seed: epoch_orders(n_train, config["epochs"], seed) for seed in config["seeds"]}
    sizes = [len(x) for x in minibatches(orders[config["seeds"][0]][0], config["batch_size"])]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1, "experiment": "extended constant lower-LR grid, fixed300epochs on unchanged batch64 frozen128 WellCLIP",
        "manifest_id": manifest["manifest_id"], "feature_id": metadata["feature_id"],
        "features_sha256": metadata.get("artifacts", {}).get("features.npz"),
        "config": config, "embedding_dim": 128, "drop_last": False,
        "batch_sizes_per_epoch": sizes,
        "epoch_order": "fresh permutation; NumPy SeedSequence([seed,640603]); same across LR and alignment",
        "epoch_order_sha256": {str(seed): [stable_hash(order.tolist()) for order in values]
                               for seed, values in orders.items()},
        "caption_assignment": "identity or fixed global training derangement from the existing seed protocol",
        "optimizer": {"name": "AdamW", "learning_rates": config["learning_rates"], "weight_decay": config["weight_decay"]},
        "schedule": "constant LR; no warmup, scheduler or early stopping; fixed epoch budget",
        "prefix100": "save best-through100 state and exact first100epoch history for prior shared-LR reproduction",
        "backbones": "original frozen DINOv2-small and BiomedCLIP text; detached cached features; WellCLIP head only",
        "checkpoint_selection": "every epoch maximum validation mean bidirectional MRR; earliest exact tie; epoch0 ineligible",
        "learning_rate_selection": "maximum mean of matched best validation MRR across seeds; first configured LR wins exact ties",
        "test_used_for_model_selection": False, "moa_labels_used_for_training": False,
        "split_ids": {key: [manifest["samples"][i]["sample_id"] for i in rows] for key, rows in indices.items()},
        "code": {name: hash_file(Path(__file__).with_name(name)) for name in
                 ("lr64_extended_training.py", "minibatch_training.py", "model.py", "training.py")},
    }
    protocol = {**payload, "protocol_id": stable_hash(payload)}
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError("Existing LR64 training uses another immutable protocol")
    if not protocol_path.exists():
        write_json(protocol_path, protocol)
    if (output_dir / "complete.json").exists():
        validate_extended_lr64_training(output_dir, manifest["manifest_id"], metadata["feature_id"])
        return read_json(output_dir / "complete.json")
    images = torch.from_numpy(np.array(arrays["image_features"], dtype=np.float32, copy=True)).to(device).detach()
    texts = torch.from_numpy(np.array(arrays["text_features"], dtype=np.float32, copy=True)).to(device).detach()
    train_idx, val_idx = (torch.as_tensor(indices[key], device=device) for key in ("train", "val"))
    finalized = {}
    for rate in config["learning_rates"]:
        for seed in config["seeds"]:
            for alignment in ALIGNMENTS:
                name = run_name(rate, alignment, seed)
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
                optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=config["weight_decay"])
                model.eval()
                with torch.no_grad():
                    reference = retrieval_metrics((model.encode_image(images[val_idx]) @
                                                   model.encode_text(texts[val_idx]).T).cpu().numpy())
                history = [{"epoch": 0, "global_step": 0, "examples_seen": 0,
                            "validation": reference, "eligible_for_selection": False}]
                best_score, best_epoch, best_step, best_state = -math.inf, 0, 0, None
                prefix_epochs, prefix_snapshot = min(100, config["epochs"]), None
                global_step, examples_seen = 0, 0
                if str(device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                for epoch, order in enumerate(orders[seed], 1):
                    weighted_loss = 0.0
                    batches = minibatches(order, config["batch_size"])
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
                            raise FloatingPointError("Nonfinite LR64 loss")
                        loss.backward()
                        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                            raise FloatingPointError("Missing/nonfinite LR64 head gradient")
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
                                    "batch_sizes": [len(x) for x in batches], "learning_rate": rate,
                                    "epoch_order_sha256": stable_hash(order.tolist()), "validation": validation})
                    if validation["mean_mrr"] > best_score:
                        best_score, best_epoch, best_step = validation["mean_mrr"], epoch, global_step
                        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    if epoch == prefix_epochs:
                        prefix_snapshot = {
                            "prefix_epochs": prefix_epochs, "best_epoch": best_epoch,
                            "best_step": best_step, "validation_mrr": best_score,
                            "selected_state_sha256": state_sha256(best_state),
                            "epoch_end_state_sha256": state_sha256(model.state_dict()),
                            "state_dict": {key: value.clone() for key, value in best_state.items()},
                        }
                if best_state is None:
                    raise RuntimeError("No validation-selected LR64 checkpoint")
                if prefix_snapshot is None:
                    raise RuntimeError("Missing extended LR64 prefix checkpoint")
                elapsed = time.perf_counter() - started
                details = {"name": name, "protocol_id": protocol["protocol_id"],
                           "manifest_id": manifest["manifest_id"], "feature_id": metadata["feature_id"],
                           "learning_rate": rate, "lr_key": lr_key(rate), "batch_size": config["batch_size"],
                           "alignment": alignment, "arm": alignment, "seed": seed, "embedding_dim": 128,
                           "best_epoch": best_epoch, "best_step": best_step, "validation_mrr": best_score,
                           "initial_state_sha256": initial_hash, "selected_state_sha256": state_sha256(best_state)}
                temporary = directory / "best.pt.tmp"
                torch.save({**details, "state_dict": best_state}, temporary)
                temporary.replace(directory / "best.pt")
                temporary_prefix = directory / "prefix100.pt.tmp"
                torch.save({**details, **prefix_snapshot}, temporary_prefix)
                temporary_prefix.replace(directory / "prefix100.pt")
                write_json(directory / "history.json", history)
                receipt = {**details, "epochs": config["epochs"], "steps_per_epoch": len(sizes),
                           "total_steps": global_step, "examples_seen": examples_seen,
                           "batch_sizes_per_epoch": sizes,
                           "epoch_order_sha256": protocol["epoch_order_sha256"][str(seed)],
                           "prefix100_epochs": prefix_epochs,
                           "prefix100_best_state_sha256": prefix_snapshot["selected_state_sha256"],
                           "prefix100_epoch_end_state_sha256": prefix_snapshot["epoch_end_state_sha256"],
                           "prefix100_history_sha256": stable_hash(history[:prefix_epochs + 1]),
                           "frozen_backbones": True, "backbone_parameter_count_in_optimizer": 0,
                           "trainable_parameters": sum(p.numel() for p in model.parameters()),
                           "elapsed_seconds": elapsed,
                           "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else 0,
                           "artifacts": {key: hash_file(directory / key) for key in ("best.pt", "prefix100.pt", "history.json", "assignment.json")},
                           "completed_at": utc_now()}
                write_json(directory / "complete.json", receipt)
                finalized[name] = hash_file(directory / "complete.json")
                print(f"SELECTED {name}: epoch={best_epoch} step={best_step} valMRR={best_score:.6f} seconds={elapsed:.2f}", flush=True)
    if set(finalized) != set(expected_runs(config)):
        raise RuntimeError("All LR64 models must finish before learning-rate selection")
    rows = [_validate_run(output_dir / name, protocol) for name in expected_runs(config)]
    selection = select_learning_rate(rows, config)
    write_json(output_dir / "lr_selection.json", selection)
    selection_hash = hash_file(output_dir / "lr_selection.json")
    write_json(output_dir / "selection_seal.json", {"protocol_id": protocol["protocol_id"], "runs": finalized,
               "lr_selection_sha256": selection_hash, "all_checkpoints_selected_before_test_export": True,
               "learning_rate_selected_before_test_export": True})
    complete = {"protocol_id": protocol["protocol_id"], "manifest_id": manifest["manifest_id"],
                "feature_id": metadata["feature_id"], "all_complete": True, "runs": finalized,
                "lr_selection_sha256": selection_hash,
                "selection_seal_sha256": hash_file(output_dir / "selection_seal.json"), "completed_at": utc_now()}
    write_json(output_dir / "complete.json", complete)
    validate_extended_lr64_training(output_dir, manifest["manifest_id"], metadata["feature_id"])
    return complete

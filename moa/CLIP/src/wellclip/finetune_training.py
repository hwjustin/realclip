"""Controlled last-two-block DINO adaptation with complete-well GradCache.

The representation-gradient pass uses one loss row per complete treatment well.
The replay pass visits image microbatches, retaining the exact well/channel mean
and normalization derivatives. It is not ordinary gradient accumulation over
independent image or well losses.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import CHANNELS, hash_file, model_identity, read_json, stable_hash, utc_now, write_json
from .model import WellCLIP, symmetric_loss
from .training import (derangement, load_features, retrieval_metrics, seed_everything,
                       split_indices, state_sha256, validate_training)


ARMS = ("frozen_matched", "frozen_shuffled", "tail_matched", "tail_shuffled")
DEFAULT_CONFIG = {
    "epochs": 10, "well_batch_size": 32, "view_batch_size": 8,
    "head_lr": 1e-4, "tail_lr": 1e-5, "weight_decay": 0.01,
    "seed": 42, "grad_clip_norm": 1.0, "include_epoch0_candidate": False,
}


def well_batches(indices, batch_size: int, seed: int, epoch: int) -> list[np.ndarray]:
    """Each training well appears exactly once; never create a singleton loss."""
    indices = np.asarray(indices, dtype=np.int64)
    if batch_size < 2 or len(indices) < 2 or len(set(indices.tolist())) != len(indices):
        raise ValueError("At least two distinct training wells and batch size >=2 are required")
    order = np.random.default_rng(seed + epoch * 100003).permutation(indices)
    batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
    if len(batches[-1]) == 1:
        last = batches.pop()
        batches[-1] = np.concatenate((batches[-1], last))
    assert sorted(np.concatenate(batches).tolist()) == sorted(indices.tolist())
    return batches


def pool_view_cls(values: torch.Tensor, images: list[dict]) -> torch.Tensor:
    """Differentiable all-site mean per channel, then channel L2 normalization."""
    if values.ndim != 2 or len(values) != len(images) or not len(images):
        raise ValueError("One CLS vector is required for every site/channel image")
    keys = [(str(row["site"]), row["channel"]) for row in images]
    sites = {site for site, _ in keys}
    if len(set(keys)) != len(keys) or set(keys) != {(s, c) for s in sites for c in CHANNELS}:
        raise ValueError("Each complete treatment site must contain the five distinct channels")
    means = torch.stack([values[[i for i, image in enumerate(images) if image["channel"] == channel]].mean(0)
                         for channel in CHANNELS])
    return F.normalize(means, dim=-1)


def _tokens_tensor(tokens, start: int, stop: int, device: torch.device) -> torch.Tensor:
    if isinstance(tokens, torch.Tensor):
        return tokens[start:stop].to(device)
    # A read-only mmap must not be used as a writable tensor view.
    return torch.from_numpy(np.array(tokens[start:stop], copy=True)).to(device)


def _tail_device(tail: nn.Module) -> torch.device:
    return next(tail.parameters()).device


def _assert_replay_deterministic(tail: nn.Module) -> None:
    if tail.training:
        raise ValueError("Tail must remain in deterministic eval mode during replay")


def gradcache_backward(tail: nn.Module, head: nn.Module, tokens_by_well: list,
                       images_by_well: list[list[dict]], text_features: torch.Tensor,
                       view_batch_size: int = 8) -> dict:
    """Populate exact tail/head gradients without retaining all image graphs.

    Caller zeros gradients once and performs one optimizer step afterwards.
    Tail evaluation is deterministic; head dropout is sampled only once, in the
    complete representation-loss pass. The loss already contains its batch
    mean, so replay gradients must not be divided by image microbatch count.
    """
    if view_batch_size < 1 or len(tokens_by_well) != len(images_by_well) or len(tokens_by_well) < 2:
        raise ValueError("A nonempty image microbatch and at least two complete wells are required")
    _assert_replay_deterministic(tail)
    device = _tail_device(tail)
    leaves = []
    with torch.no_grad():
        for tokens, images in zip(tokens_by_well, images_by_well):
            if len(tokens) != len(images):
                raise ValueError("Prefix token order/count does not match well images")
            chunks = [tail(_tokens_tensor(tokens, start, start + view_batch_size, device))
                      for start in range(0, len(tokens), view_batch_size)]
            leaves.append(torch.cat(chunks).detach().requires_grad_(True))
    pooled = torch.stack([pool_view_cls(values, images)
                          for values, images in zip(leaves, images_by_well)])
    image_embeddings, text_embeddings, scale = head(pooled, text_features.to(device))
    loss = symmetric_loss(scale * image_embeddings @ text_embeddings.T)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite complete-well contrastive loss")
    loss.backward()
    derivatives = [value.grad.detach().clone() for value in leaves]
    for tokens, derivative in zip(tokens_by_well, derivatives):
        for start in range(0, len(tokens), view_batch_size):
            end = min(start + view_batch_size, len(tokens))
            cls = tail(_tokens_tensor(tokens, start, end, device))
            torch.autograd.backward(cls, derivative[start:end])
    return {"loss": float(loss.detach().cpu()), "well_count": len(leaves),
            "view_count": sum(len(tokens) for tokens in tokens_by_well),
            "image_features": pooled.detach()}


@torch.no_grad()
def encode_well_features(tail: nn.Module, samples: list[dict], cache_dir,
                         view_batch_size: int = 8) -> torch.Tensor:
    """Return only the requested wells; validation calls pass validation rows."""
    from .finetune_cache import load_well_tokens

    tail.eval()
    device = _tail_device(tail)
    result = []
    for sample in samples:
        tokens = load_well_tokens(cache_dir, sample["sample_id"])
        parts = [tail(_tokens_tensor(tokens, start, start + view_batch_size, device))
                 for start in range(0, len(tokens), view_batch_size)]
        result.append(pool_view_cls(torch.cat(parts), sample["images"]).cpu())
    if not result:
        raise ValueError("No wells were supplied for feature extraction")
    return torch.stack(result)


def _cpu_state(module: nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _atomic_torch_save(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _rng_state() -> dict:
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_finetune_run(checkpoint_path, dino_path, device: str = "cpu"):
    """Load a selected/last checkpoint; prefix remains on CPU unless moved."""
    from .finetune_backbone import SplitDINO

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    split = SplitDINO(dino_path, device="cpu")
    split.tail.load_state_dict(checkpoint["tail_state_dict"], strict=True)
    split.tail.to(device).eval()
    head = WellCLIP().to(device)
    head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    head.eval()
    return split, head, checkpoint


def validate_finetune_training(training_dir) -> tuple[dict, list[dict]]:
    directory = Path(training_dir)
    protocol = read_json(directory / "protocol.json")
    payload = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if stable_hash(payload) != protocol.get("protocol_id"):
        raise ValueError("Fine-tuning protocol identity mismatch")
    complete = read_json(directory / "complete.json")
    if complete.get("protocol_id") != protocol["protocol_id"] or not complete.get("all_complete"):
        raise ValueError("Not all fine-tuning arms have been finalized")
    if set(complete["runs"]) != set(protocol["arms"]):
        raise ValueError("Fine-tuning completed arm set differs from its protocol")
    receipts = []
    for name in protocol["arms"]:
        run = directory / name
        if hash_file(run / "complete.json") != complete["runs"][name]:
            raise ValueError(f"Fine-tuning run receipt changed: {name}")
        receipt = read_json(run / "complete.json")
        if receipt["protocol_id"] != protocol["protocol_id"] or receipt["arm"] != name:
            raise ValueError("Fine-tuning run identity mismatch")
        for filename, expected in receipt["artifacts"].items():
            if hash_file(run / filename) != expected:
                raise ValueError(f"Fine-tuning run artifact changed: {name}/{filename}")
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        for key in ("protocol_id", "manifest_id", "feature_id", "prefix_cache_id", "arm", "seed",
                    "best_epoch", "validation_mrr"):
            if checkpoint[key] != receipt[key]:
                raise ValueError(f"Fine-tuning checkpoint identity differs: {key}")
        for key in ("manifest_id", "feature_id", "prefix_cache_id"):
            if receipt[key] != protocol[key]:
                raise ValueError(f"Fine-tuning receipt identity differs: {key}")
        if checkpoint["best_epoch"] < 1:
            raise ValueError("Epoch zero is reference-only and cannot be selected")
        receipts.append({"name": name, **receipt})
    return protocol, receipts


def run_finetuning(manifest_path, features_dir, prefix_cache_dir, dino_path,
                   baseline_training_dir, output_dir, config: dict | None = None,
                   device: str = "cuda") -> dict:
    """Train same-schedule frozen/tail × matched/shuffled pilot arms."""
    from .finetune_backbone import SplitDINO
    from .finetune_cache import load_well_tokens, validate_prefix_cache

    config = {**DEFAULT_CONFIG, **(config or {})}
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown or config["epochs"] < 1 or config["well_batch_size"] < 2 or config["view_batch_size"] < 1:
        raise ValueError(f"Unsupported fine-tuning configuration: {sorted(unknown)}")
    if config["include_epoch0_candidate"]:
        raise ValueError("Epoch zero is a reference only in this experiment")
    if min(config["head_lr"], config["tail_lr"], config["grad_clip_norm"]) <= 0 or config["weight_decay"] < 0:
        raise ValueError("Fine-tuning learning rates and clipping must be positive")
    manifest_path, features_dir, prefix_cache_dir, dino_path, baseline_training_dir, output_dir = (
        Path(value).resolve() for value in (manifest_path, features_dir, prefix_cache_dir,
                                          dino_path, baseline_training_dir, output_dir))
    manifest, arrays, metadata = load_features(manifest_path, features_dir)
    indices = split_indices(manifest)
    if {key: len(value) for key, value in indices.items()} != {"train": 603, "val": 66, "test": 66}:
        raise ValueError("The authorized fine-tuning experiment requires the fixed 603/66/66 split")
    validate_prefix_cache(prefix_cache_dir, manifest_path=manifest_path, verify_arrays=True)
    prefix = read_json(prefix_cache_dir / "metadata.json")
    if prefix["manifest_id"] != manifest["manifest_id"]:
        raise ValueError("Prefix cache belongs to a different manifest")
    if model_identity(dino_path) != prefix["identity"]["model_identity"]:
        raise ValueError("Original DINO tail weights differ from the frozen prefix model")
    _, baseline_receipts = validate_training(baseline_training_dir, manifest["manifest_id"], metadata["feature_id"])
    baseline_by_name = {row["name"]: row for row in baseline_receipts}
    seed = int(config["seed"])
    warmstarts = {}
    for alignment in ("matched", "shuffled"):
        path = baseline_training_dir / f"{alignment}_seed{seed}" / "best.pt"
        if f"{alignment}_seed{seed}" not in baseline_by_name:
            raise ValueError("Requested warm-start seed was not fixed in the original experiment")
        warmstarts[alignment] = {"path": str(path), "sha256": hash_file(path)}
    payload = {
        "schema_version": 1, "manifest_id": manifest["manifest_id"], "feature_id": metadata["feature_id"],
        "prefix_cache_id": prefix["cache_id"], "config": config, "arms": list(ARMS),
        "split_ids": {key: [manifest["samples"][i]["sample_id"] for i in rows] for key, rows in indices.items()},
        "selection": "maximum validation bidirectional MRR after epochs >=1; earliest exact tie",
        "epoch_zero": "reference only, never selectable", "precision": "float32",
        "text": "frozen original BiomedCLIP512 cached features; existing text projection remains trainable",
        "loss": "one complete treatment well per row, in-well all-site/channel mean+L2, symmetric InfoNCE",
        "backward": "representation gradients cached over full well batch, exact tail replay over view microbatches",
        "tail": "DINOv2-small blocks10/11 and final LayerNorm; prefix unchanged",
        "warmstarts": warmstarts,
        "bindings": {"manifest_path": str(manifest_path), "features_dir": str(features_dir),
                     "prefix_cache_dir": str(prefix_cache_dir), "dino_path": str(dino_path),
                     "features_metadata_sha256": hash_file(features_dir / "metadata.json"),
                     "prefix_metadata_sha256": hash_file(prefix_cache_dir / "metadata.json")},
        "code": {name: hash_file(Path(__file__).with_name(name)) for name in
                 ("finetune_training.py", "finetune_backbone.py", "model.py")},
    }
    protocol = {**payload, "protocol_id": stable_hash(payload)}
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "protocol.json").exists() and read_json(output_dir / "protocol.json") != protocol:
        raise ValueError("Fine-tuning output already uses a different protocol")
    if not (output_dir / "protocol.json").exists():
        write_json(output_dir / "protocol.json", protocol)
    if (output_dir / "complete.json").exists():
        validate_finetune_training(output_dir)
        return read_json(output_dir / "complete.json")
    samples = manifest["samples"]
    train_indices, val_indices = indices["train"], indices["val"]
    text_features = torch.from_numpy(arrays["text_features"]).to(device)
    frozen_features = torch.from_numpy(arrays["image_features"]).to(device)
    train_position = {int(index): position for position, index in enumerate(train_indices)}
    shuffled = derangement(len(train_indices), seed)
    finalized = {}
    for arm in ARMS:
        directory = output_dir / arm
        directory.mkdir(exist_ok=True)
        if (directory / "complete.json").exists():
            receipt = read_json(directory / "complete.json")
            if receipt["protocol_id"] != protocol["protocol_id"]:
                raise ValueError("Existing fine-tuning arm has another identity")
            for filename, digest in receipt["artifacts"].items():
                if hash_file(directory / filename) != digest:
                    raise ValueError("Existing fine-tuning arm artifact changed")
            finalized[arm] = hash_file(directory / "complete.json")
            continue
        seed_everything(seed)
        alignment = "shuffled" if arm.endswith("shuffled") else "matched"
        adapt_tail = arm.startswith("tail_")
        split = SplitDINO(dino_path, device="cpu")
        tail = split.tail.to(device).eval().requires_grad_(adapt_tail)
        head = WellCLIP().to(device)
        original = torch.load(warmstarts[alignment]["path"], map_location="cpu", weights_only=True)
        head.load_state_dict(original["state_dict"], strict=True)
        initial_head = state_sha256(head.state_dict())
        initial_tail = state_sha256(tail.state_dict())
        groups = [{"params": list(head.parameters()), "lr": config["head_lr"]}]
        if adapt_tail:
            groups.append({"params": list(tail.parameters()), "lr": config["tail_lr"]})
        optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
        assignment = train_indices[shuffled] if alignment == "shuffled" else train_indices
        write_json(directory / "assignment.json", [
            {"image_sample_id": samples[int(i)]["sample_id"], "text_sample_id": samples[int(j)]["sample_id"]}
            for i, j in zip(train_indices, assignment)])
        history, best_score, best_epoch, best_head, best_tail = [], -math.inf, 0, None, None
        start_epoch = 1
        elapsed_before_resume = 0.0
        resume_path = directory / "resume.pt"
        if resume_path.exists():
            resumed = torch.load(resume_path, map_location="cpu", weights_only=True)
            if resumed["protocol_id"] != protocol["protocol_id"] or resumed["arm"] != arm:
                raise ValueError("Partial fine-tuning resume identity mismatch")
            head.load_state_dict(resumed["head_state_dict"])
            tail.load_state_dict(resumed["tail_state_dict"])
            optimizer.load_state_dict(resumed["optimizer_state_dict"])
            history, best_score, best_epoch = resumed["history"], resumed["best_score"], resumed["best_epoch"]
            best_head, best_tail = resumed["best_head"], resumed["best_tail"]
            start_epoch = resumed["epoch"] + 1
            epoch_zero = resumed["epoch_zero"]
            elapsed_before_resume = resumed["elapsed_seconds"]
            _restore_rng(resumed["rng"])
        else:
            head.eval()
            with torch.no_grad():
                score = retrieval_metrics((head.encode_image(frozen_features[val_indices]) @
                                           head.encode_text(text_features[val_indices]).T).cpu().numpy())
            epoch_zero = {"validation": score, "initial_head_sha256": initial_head,
                          "initial_tail_sha256": initial_tail, "selectable": False}
            write_json(directory / "epoch_zero.json", epoch_zero)
        clock_start = time.perf_counter()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            epoch_start = time.perf_counter()
            losses = []
            view_count = 0
            batches = well_batches(train_indices, int(config["well_batch_size"]), seed, epoch)
            for step, batch in enumerate(batches, 1):
                step_start = time.perf_counter()
                head.train()
                tail.eval()
                optimizer.zero_grad(set_to_none=True)
                text_rows = np.asarray([assignment[train_position[int(i)]] for i in batch])
                if adapt_tail:
                    tokens = [load_well_tokens(prefix_cache_dir, samples[int(i)]["sample_id"]) for i in batch]
                    gradients = gradcache_backward(tail, head, tokens, [samples[int(i)]["images"] for i in batch],
                                                   text_features[text_rows], int(config["view_batch_size"]))
                    loss_value = gradients["loss"]
                    view_count += gradients["view_count"]
                    del tokens, gradients
                else:
                    image_vectors, text_vectors, scale = head(frozen_features[batch], text_features[text_rows])
                    loss = symmetric_loss(scale * image_vectors @ text_vectors.T)
                    loss.backward()
                    loss_value = float(loss.detach().cpu())
                    view_count += sum(len(samples[int(i)]["images"]) for i in batch)
                parameters = [p for group in optimizer.param_groups for p in group["params"]]
                if not np.isfinite(loss_value) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError(f"Nonfinite loss or gradient in {arm}, epoch {epoch}, batch {step}")
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(config["grad_clip_norm"]))
                optimizer.step()
                losses.append({"step": step, "well_count": len(batch), "loss": loss_value,
                               "grad_norm": float(grad_norm), "seconds": time.perf_counter() - step_start})
                if adapt_tail and (step == 1 or step % 5 == 0 or step == len(batches)):
                    print(f"{arm} epoch={epoch}/{config['epochs']} batch={step}/{len(batches)} loss={loss_value:.5f}", flush=True)
            head.eval()
            tail.eval()
            with torch.no_grad():
                val_features = (encode_well_features(tail, [samples[int(i)] for i in val_indices], prefix_cache_dir,
                                                    int(config["view_batch_size"])).to(device)
                                if adapt_tail else frozen_features[val_indices])
                validation = retrieval_metrics((head.encode_image(val_features) @
                                                head.encode_text(text_features[val_indices]).T).cpu().numpy())
            score = validation["mean_mrr"]
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_head, best_tail = _cpu_state(head), _cpu_state(tail)
            row = {"epoch": epoch, "training_wells": len(train_indices), "image_views": view_count,
                   "batches": len(batches), "mean_train_loss": sum(r["loss"] * r["well_count"] for r in losses) / len(train_indices),
                   "steps": losses,
                   "validation": validation, "seconds": time.perf_counter() - epoch_start,
                   "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
                   "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.startswith("cuda") else 0}
            history.append(row)
            elapsed = elapsed_before_resume + time.perf_counter() - clock_start
            _atomic_torch_save(resume_path, {"protocol_id": protocol["protocol_id"], "arm": arm, "epoch": epoch,
                "head_state_dict": _cpu_state(head), "tail_state_dict": _cpu_state(tail),
                "optimizer_state_dict": optimizer.state_dict(), "rng": _rng_state(), "history": history,
                "best_score": best_score, "best_epoch": best_epoch, "best_head": best_head, "best_tail": best_tail,
                "epoch_zero": epoch_zero, "elapsed_seconds": elapsed})
            write_json(directory / "history.json", history)
            print(f"{arm} epoch={epoch} val_MRR={score:.5f} best={best_score:.5f}@{best_epoch} seconds={row['seconds']:.1f}", flush=True)
        common = {"schema_version": 1, "protocol_id": protocol["protocol_id"],
                  "manifest_id": manifest["manifest_id"], "feature_id": metadata["feature_id"],
                  "prefix_cache_id": prefix["cache_id"], "arm": arm, "seed": seed,
                  "best_epoch": best_epoch, "initial_head_sha256": initial_head,
                  "initial_tail_sha256": initial_tail}
        _atomic_torch_save(directory / "best.pt", {**common, "epoch": best_epoch, "validation_mrr": best_score,
                                                   "head_state_dict": best_head, "tail_state_dict": best_tail})
        _atomic_torch_save(directory / "last.pt", {**common, "epoch": int(config["epochs"]),
                                                   "validation_mrr": history[-1]["validation"]["mean_mrr"],
                                                   "head_state_dict": _cpu_state(head), "tail_state_dict": _cpu_state(tail)})
        tail_changed = state_sha256(tail.state_dict()) != initial_tail
        if tail_changed != adapt_tail:
            raise ValueError("Frozen/adapted tail change audit failed")
        receipt = {**common, "validation_mrr": best_score, "epochs_completed": int(config["epochs"]),
                   "tail_adapted": adapt_tail, "tail_last_state_changed": tail_changed,
                   "trainable_tail_parameters": sum(p.numel() for p in tail.parameters() if p.requires_grad),
                   "trainable_head_parameters": sum(p.numel() for p in head.parameters() if p.requires_grad),
                   "epoch_zero_validation_mrr": epoch_zero["validation"]["mean_mrr"],
                   "last_validation_mrr": history[-1]["validation"]["mean_mrr"],
                   "elapsed_seconds": elapsed_before_resume + time.perf_counter() - clock_start,
                   "completed_at": utc_now(),
                   "artifacts": {name: hash_file(directory / name) for name in
                                 ("best.pt", "last.pt", "history.json", "assignment.json", "epoch_zero.json")}}
        write_json(directory / "complete.json", receipt)
        finalized[arm] = hash_file(directory / "complete.json")
        del split, tail, head, optimizer
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    complete = {"schema_version": 1, "protocol_id": protocol["protocol_id"], "all_complete": True,
                "completed_at": utc_now(), "runs": finalized}
    write_json(output_dir / "complete.json", complete)
    validate_finetune_training(output_dir)
    return complete

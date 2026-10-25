"""Single-GPU full DINOv2 + BiomedCLIP-text training and explicit held-out evaluation."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import WellDataset, collate_wells, load_manifest
from .model import WellCLIP, contrastive_loss


DEFAULT_CONFIG = {
    "epochs": 20, "well_batch_size": 8, "image_microbatch": 32,
    "eval_well_batch_size": 8, "backbone_lr": 1e-5, "head_lr": 1e-4,
    "weight_decay": 0.01, "warmup_fraction": 0.1, "grad_clip_norm": 1.0,
    "embedding_dim": 128, "dropout": 0.1, "precision": "bf16",
    "gradient_checkpointing": True, "num_workers": 0, "seed": 42,
    "checkpoint_every_steps": 100,
}


def validated_config(supplied):
    if set(supplied) - set(DEFAULT_CONFIG):
        raise ValueError(f"Unknown config keys: {sorted(set(supplied) - set(DEFAULT_CONFIG))}")
    value = {**DEFAULT_CONFIG, **supplied}
    for key in ("epochs", "well_batch_size", "image_microbatch", "eval_well_batch_size",
                "embedding_dim", "checkpoint_every_steps"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if value["well_batch_size"] < 2:
        raise ValueError("Contrastive well_batch_size must be >= 2")
    if type(value["num_workers"]) is not int or value["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(value["seed"]) is not int or not 0 <= value["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(value["gradient_checkpointing"]) is not bool:
        raise ValueError("gradient_checkpointing must be boolean")
    for key in ("backbone_lr", "head_lr", "grad_clip_norm"):
        if not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(value["weight_decay"]) or value["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    for key in ("warmup_fraction", "dropout"):
        if not math.isfinite(value[key]) or not 0 <= value[key] < 1:
            raise ValueError(f"{key} must be in [0, 1)")
    if value["precision"] not in ("bf16", "fp32"):
        raise ValueError("precision must be bf16 or fp32")
    return value


def epoch_batches(count, size, seed, epoch):
    if count < 2 or size < 2:
        raise ValueError("At least two training wells and batch size >= 2 are required")
    order = np.random.default_rng(seed + epoch * 100003).permutation(count).tolist()
    batches = [order[start:start + size] for start in range(0, count, size)]
    if len(batches[-1]) == 1:
        last = batches.pop()
        batches[-1].extend(last)
    return batches


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same visible CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(path, model, optimizer, scheduler, config, manifest_id,
                    epoch, next_batch, global_step, best_mrr, history):
    _atomic_save(path, {
        "schema_version": 1, "architecture": "full-dinov2-small+biomedclip-text-well-v1",
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "rng": rng_state(), "config": config,
        "manifest_id": manifest_id, "model_provenance": model.provenance,
        "epoch": epoch, "next_batch": next_batch, "global_step": global_step,
        "best_mrr": best_mrr, "history": history,
        "trainable_parameters": {"image_encoder": sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad),
                                 "text_encoder": sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)},
    })


def load_checkpoint(path, model, optimizer, scheduler, config, manifest_id):
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Evaluation may choose CPU/FP32 or different memory/workers settings. These
    # never change the saved architecture; training resumes remain exact.
    runtime_keys = ({"precision", "num_workers", "eval_well_batch_size", "image_microbatch",
                     "gradient_checkpointing"} if optimizer is None else set())
    saved_config = {key: value for key, value in state.get("config", {}).items() if key not in runtime_keys}
    requested_config = {key: value for key, value in config.items() if key not in runtime_keys}
    if (state.get("schema_version") != 1
            or state.get("architecture") != "full-dinov2-small+biomedclip-text-well-v1"
            or saved_config != requested_config or state.get("manifest_id") != manifest_id
            or state.get("model_provenance") != model.provenance):
        raise ValueError("Checkpoint config, dataset, tokenizer, or architecture is incompatible")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
    return state


def make_optimizer(model, config):
    backbone = list(model.image_encoder.parameters()) + list(model.text_encoder.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    head = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    if not backbone or any(not p.requires_grad for p in backbone):
        raise ValueError("Both complete encoders must be unfrozen")
    return torch.optim.AdamW([
        {"params": backbone, "lr": config["backbone_lr"], "name": "both_complete_encoders"},
        {"params": head, "lr": config["head_lr"], "name": "well_heads"},
    ], weight_decay=config["weight_decay"])


def make_scheduler(optimizer, total_steps, warmup_fraction):
    warmup = min(total_steps, max(0, round(total_steps * warmup_fraction)))

    def multiplier(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1, max(0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _autocast(device, precision):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16" and torch.device(device).type == "cuda" else nullcontext())


def retrieval_metrics(image, text, descriptions):
    scores = image.float() @ text.float().T
    positives = torch.tensor([[a == b for b in descriptions] for a in descriptions], dtype=torch.bool)

    def direction(values, mask):
        order = torch.argsort(values, dim=1, descending=True, stable=True)
        ranked = mask.gather(1, order)
        ranks = ranked.long().argmax(1) + 1
        return {"MRR": float((1 / ranks.float()).mean()),
                **{f"R@{k}": float((ranks <= k).float().mean()) for k in (1, 5, 10)}}

    i2t, t2i = direction(scores, positives), direction(scores.T, positives.T)
    return {"image_to_text": i2t, "text_to_image": t2i,
            "mean_mrr": (i2t["MRR"] + t2i["MRR"]) / 2,
            "candidate_count": len(image), "positives": "identical full description"}


@torch.no_grad()
def evaluate(model, dataset, config, device):
    if len(dataset) < 1:
        raise ValueError("Evaluation split is empty")
    previous = model.training
    model.eval()
    images, texts, descriptions = [], [], []
    loader = DataLoader(dataset, batch_size=config["eval_well_batch_size"], shuffle=False,
                        collate_fn=collate_wells, num_workers=config["num_workers"],
                        generator=torch.Generator().manual_seed(config["seed"]))
    for wells in loader:
        with _autocast(device, config["precision"]):
            image, text, _ = model(wells, config["image_microbatch"])
        images.append(image.cpu())
        texts.append(text.cpu())
        descriptions.extend(row["text"] for row in wells)
    result = retrieval_metrics(torch.cat(images), torch.cat(texts), descriptions)
    model.train(previous)
    return result


def run_training(config, data_root, output, model_root, device="cuda", resume=None,
                 smoke_steps=None):
    config = validated_config(config)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu --precision fp32 for a CPU smoke run")
    if config["precision"] == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 requires a compatible CUDA GPU; select precision fp32 for CPU")
    if smoke_steps is not None and smoke_steps < 1:
        raise ValueError("smoke_steps must be positive")
    manifest = load_manifest(data_root)
    train_data = WellDataset(data_root, "train")
    val_data = WellDataset(data_root, "val")
    if len(train_data) < 2 or len(val_data) < 1:
        raise ValueError("Training requires >=2 train wells and >=1 validation well")
    output = Path(output).resolve()
    if (output / "last.pt").exists() and resume is None:
        raise FileExistsError("Output already contains last.pt; use --resume or a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    _seed(config["seed"])
    model = WellCLIP.from_pretrained(model_root, config["embedding_dim"], config["dropout"],
                                    config["gradient_checkpointing"]).to(device)
    optimizer = make_optimizer(model, config)
    steps_per_epoch = len(epoch_batches(len(train_data), config["well_batch_size"], config["seed"], 0))
    scheduler = make_scheduler(optimizer, config["epochs"] * steps_per_epoch, config["warmup_fraction"])
    epoch, next_batch, global_step, best_mrr, history = 0, 0, 0, -1.0, []
    if resume:
        state = load_checkpoint(resume, model, optimizer, scheduler, config, manifest["manifest_id"])
        epoch, next_batch, global_step = state["epoch"], state["next_batch"], state["global_step"]
        best_mrr, history = state["best_mrr"], state["history"]
        if not 0 <= epoch <= config["epochs"]:
            raise ValueError("Checkpoint epoch is outside the configured training schedule")
        if Path(resume).resolve().parent != output:
            raise ValueError("Resume checkpoint must be inside --output; copy the complete run directory before moving machines")
        if best_mrr >= 0 and not (output / "best.pt").is_file():
            raise FileNotFoundError("Resume requires the run's best.pt alongside last.pt to preserve prior model selection")
        del state
    protocol = {"config": config, "manifest_id": manifest["manifest_id"],
                "model_provenance": model.provenance, "model_selection": "validation mean bidirectional MRR; earliest tie",
                "test_used_for_training_or_selection": False,
                "image_views": "all sites, five ordered channels; whole-encoder checkpoint per microbatch",
                "parameters": {"image_encoder": sum(p.numel() for p in model.image_encoder.parameters()),
                               "text_encoder": sum(p.numel() for p in model.text_encoder.parameters()),
                               "all": sum(p.numel() for p in model.parameters())}}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    started_at_step = global_step

    def save(path):
        save_checkpoint(path, model, optimizer, scheduler, config, manifest["manifest_id"],
                        epoch, next_batch, global_step, best_mrr, history)

    while epoch < config["epochs"]:
        batches = epoch_batches(len(train_data), config["well_batch_size"], config["seed"], epoch)
        if not 0 <= next_batch <= len(batches):
            raise ValueError("Checkpoint batch cursor is outside the epoch")
        loader = DataLoader(train_data, batch_sampler=batches[next_batch:], collate_fn=collate_wells,
                            num_workers=config["num_workers"], pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(config["seed"] + epoch))
        model.train()
        for wells in loader:
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, config["precision"]):
                image, text, scale = model(wells, config["image_microbatch"])
            # Normalization and logits/loss use float32 even under BF16 encoding.
            loss = contrastive_loss(image, text, scale, [row["text"] for row in wells])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite contrastive loss")
            loss.backward()
            if global_step == 0:
                for name in ("image_encoder", "text_encoder"):
                    if not any(p.grad is not None and bool(torch.count_nonzero(p.grad))
                               for p in getattr(model, name).parameters()):
                        raise RuntimeError(f"No gradient reached {name}")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                model.logit_scale.clamp_(max=math.log(100))
            global_step += 1
            next_batch += 1
            row = {"epoch": epoch + 1, "step": global_step, "batch": next_batch,
                   "well_count": len(wells), "loss": float(loss.detach()), "grad_norm": float(norm)}
            with (output / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if global_step % config["checkpoint_every_steps"] == 0:
                save(output / "last.pt")
            if smoke_steps is not None and global_step - started_at_step >= smoke_steps:
                save(output / "last.pt")
                return {"status": "smoke_complete", "global_step": global_step,
                        "checkpoint": str(output / "last.pt")}
        validation = evaluate(model, val_data, config, device)
        history.append({"epoch": epoch + 1, "step": global_step, "validation": validation})
        epoch += 1
        next_batch = 0
        if validation["mean_mrr"] > best_mrr:
            best_mrr = validation["mean_mrr"]
            save(output / "best.pt")
        save(output / "last.pt")
        (output / "validation.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(history[-1]), flush=True)
    return {"status": "complete", "global_step": global_step, "best_validation_mrr": best_mrr,
            "checkpoint": str(output / "best.pt")}


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--evaluate", choices=("val", "test"), help="Explicit evaluation only; never optimize")
    parser.add_argument("--checkpoint", type=Path, help="Evaluation checkpoint (default output/best.pt)")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.precision:
        config["precision"] = args.precision
    config = validated_config(config)
    if args.evaluate:
        if args.resume or args.smoke_steps:
            parser.error("--resume/--smoke-steps cannot be combined with --evaluate")
        if config["precision"] == "bf16" and (not args.device.startswith("cuda") or not torch.cuda.is_available()
                                               or not torch.cuda.is_bf16_supported()):
            raise RuntimeError("BF16 evaluation requires compatible CUDA")
        manifest = load_manifest(args.data_root)
        model = WellCLIP.from_pretrained(args.model_root, config["embedding_dim"], config["dropout"],
                                        config["gradient_checkpointing"]).to(args.device)
        checkpoint_path = args.checkpoint or args.output / "best.pt"
        load_checkpoint(checkpoint_path, model, None, None, config, manifest["manifest_id"])
        result = {"split": args.evaluate, "checkpoint": str(checkpoint_path.resolve()),
                  **evaluate(model, WellDataset(args.data_root, args.evaluate), config, args.device)}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"{args.evaluate}_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    else:
        if args.checkpoint:
            parser.error("--checkpoint requires --evaluate")
        result = run_training(config, args.data_root, args.output, args.model_root,
                              args.device, args.resume, args.smoke_steps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

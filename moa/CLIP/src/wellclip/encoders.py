"""Local frozen DINOv2 and the official contrastively trained BiomedCLIP text tower."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .common import read_json


class _FrozenEncoder(nn.Module):
    def train(self, mode: bool = True):
        # Parent training calls must never enable dropout in a frozen backbone.
        super().train(False)
        return self


class FrozenDINO(_FrozenEncoder):
    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            use_safetensors=True,
        ).to(device).requires_grad_(False)
        self.device = torch.device(device)
        self.eval()

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4 or pixels.shape[1:] != (3, 518, 518):
            raise ValueError("DINO expects preprocessed [images,3,518,518]")
        self.eval()
        features = self.encoder(pixel_values=pixels.to(self.device), return_dict=True).last_hidden_state[:, 0]
        if features.shape != (len(pixels), 384) or not torch.isfinite(features).all():
            raise ValueError("Invalid frozen DINO CLS features")
        return features.float()

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encode(pixels)


def _load_biomedclip(model_dir: Path, device: str):
    """Use the official OpenCLIP loader; both checkpoint and tokenizer stay local."""
    import open_clip
    from open_clip.factory import _MODEL_CONFIGS

    config = copy.deepcopy(read_json(model_dir / "open_clip_config.json"))["model_cfg"]
    text_path = model_dir / "text_config"
    if not (text_path / "config.json").is_file():
        raise ValueError("BiomedCLIP requires a local text_config/config.json and tokenizer")
    config["text_cfg"]["hf_model_name"] = str(text_path)
    config["text_cfg"]["hf_tokenizer_name"] = str(model_dir)
    if config["embed_dim"] != 512 or config["text_cfg"]["context_length"] != 256:
        raise ValueError("Expected the official 512-dimensional, 256-context BiomedCLIP")
    model_name = "wellclip_biomedclip_local"
    _MODEL_CONFIGS[model_name] = config
    checkpoints = [model_dir / name for name in ("open_clip_pytorch_model.safetensors", "open_clip_pytorch_model.bin")]
    checkpoint = next((path for path in checkpoints if path.is_file()), None)
    if checkpoint is None:
        raise ValueError("The complete official BiomedCLIP checkpoint is missing")
    model = open_clip.create_model(
        model_name, pretrained=str(checkpoint), pretrained_hf=False, device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name, context_length=256)
    return model, tokenizer


class FrozenBiomedCLIP(_FrozenEncoder):
    context_length = 256

    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        super().__init__()
        model, self.tokenizer = _load_biomedclip(Path(model_dir).resolve(), device)
        self.encoder = model.requires_grad_(False)
        self.device = torch.device(device)
        self.eval()

    def token_counts(self, texts: list[str]) -> list[int]:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Nonempty descriptions are required")
        # The OpenCLIP tokenizer truncates by default, so inspect its underlying
        # HF tokenizer first and fail before any words can be discarded.
        counts = [len(ids) for ids in self.tokenizer.tokenizer(
            texts, add_special_tokens=True, truncation=False, padding=False,
        )["input_ids"]]
        if max(counts) > self.context_length:
            raise ValueError(f"Description exceeds BiomedCLIP's 256-token context ({max(counts)}); no truncation")
        return counts

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        self.token_counts(texts)
        self.eval()
        tokens = self.tokenizer(texts, context_length=self.context_length).to(self.device)
        features = self.encoder.encode_text(tokens, normalize=True).float()
        if features.shape != (len(texts), 512) or not torch.isfinite(features).all():
            raise ValueError("Invalid frozen BiomedCLIP text features")
        return F.normalize(features, dim=-1)

    def forward(self, texts: list[str]) -> torch.Tensor:
        return self.encode(texts)

"""Both complete encoders are trainable; no cached backbone features are used."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WellCLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, tokenizer, embedding_dim=128,
                 dropout=0.1, image_checkpoint=True):
        super().__init__()
        self.image_encoder = image_encoder.requires_grad_(True)
        self.text_encoder = text_encoder.requires_grad_(True)
        self.tokenizer = tokenizer
        self.image_checkpoint = image_checkpoint
        self.channel_projection = nn.Sequential(nn.Linear(384, 32), nn.GELU())
        self.image_projection = nn.Sequential(nn.Dropout(dropout), nn.Linear(160, embedding_dim))
        self.text_projection = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, embedding_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.provenance = {}

    @classmethod
    def from_pretrained(cls, model_root, embedding_dim=128, dropout=0.1,
                        gradient_checkpointing=True):
        from transformers import AutoModel
        import open_clip
        from open_clip.factory import _MODEL_CONFIGS

        root = Path(model_root).resolve()
        dino = AutoModel.from_pretrained(str(root / "dinov2-small"), local_files_only=True,
                                         trust_remote_code=False, use_safetensors=True)
        biomed = root / "biomedclip"
        config = copy.deepcopy(json.loads((biomed / "open_clip_config.json").read_text()))["model_cfg"]
        if config["embed_dim"] != 512 or config["text_cfg"]["context_length"] != 256:
            raise ValueError("Expected the official 512-dimensional, context256 BiomedCLIP")
        config["text_cfg"]["hf_model_name"] = str(biomed / "text_config")
        config["text_cfg"]["hf_tokenizer_name"] = str(biomed)
        config["text_cfg"]["hf_model_pretrained"] = False
        name = "realclip_h200_local_biomedclip"
        _MODEL_CONFIGS[name] = config
        weights = next((biomed / name for name in ("open_clip_pytorch_model.safetensors",
                                                   "open_clip_pytorch_model.bin")
                        if (biomed / name).is_file()), None)
        if weights is None:
            raise FileNotFoundError(f"Missing official BiomedCLIP checkpoint in {biomed}")
        full = open_clip.create_model(name, pretrained=str(weights), pretrained_hf=False, device="cpu")
        tokenizer = open_clip.get_tokenizer(name, context_length=256)
        text = full.text
        del full  # The unused BiomedCLIP vision tower never enters the optimizer/checkpoint.
        if gradient_checkpointing:
            text.transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = cls(dino, text, tokenizer, embedding_dim, dropout, gradient_checkpointing)
        # Initial weights and tokenizer/config bytes are audited independently of
        # their location. Stream weights rather than allocating a second huge blob.
        names = ["dinov2-small/config.json", "biomedclip/open_clip_config.json",
                 "biomedclip/tokenizer.json", "biomedclip/tokenizer_config.json",
                 "biomedclip/vocab.txt", "biomedclip/special_tokens_map.json",
                 "biomedclip/text_config/config.json", "dinov2-small/model.safetensors",
                 str(weights.relative_to(root))]
        model.provenance = {name: _file_sha256(root / name) for name in names}
        return model

    def _image_cls(self, pixels):
        values = self.image_encoder(pixel_values=pixels, return_dict=True).last_hidden_state[:, 0]
        if values.shape != (len(pixels), 384):
            raise ValueError("DINOv2-small must return one CLS384 per view")
        return values

    def encode_wells(self, wells, image_microbatch=32):
        if image_microbatch < 1:
            raise ValueError("image_microbatch must be positive")
        device = next(self.parameters()).device
        pooled = []
        for well in wells:
            pixels, channels = well["pixels"], well["channels"].to(device)
            if len(pixels) != len(channels) or set(channels.tolist()) != set(range(5)):
                raise ValueError("Every well needs its complete five-channel views")
            pieces = []
            for start in range(0, len(pixels), image_microbatch):
                part = pixels[start:start + image_microbatch].to(device, non_blocking=True)
                # Checkpoint the WHOLE encoder, not just each block: otherwise the
                # graphs retained for all wells/microbatches accumulate in memory.
                # Non-reentrant checkpoint supports non-grad pixels and preserves
                # the forward autocast context and stochastic RNG on recomputation.
                encoded = (checkpoint(self._image_cls, part, use_reentrant=False,
                                      preserve_rng_state=True)
                           if self.training and self.image_checkpoint and torch.is_grad_enabled()
                           else self._image_cls(part))
                pieces.append(encoded.float())
            per_view = torch.cat(pieces)
            per_channel = torch.stack([per_view[channels == channel].mean(0) for channel in range(5)])
            pooled.append(F.normalize(per_channel, dim=-1))
        hidden = self.channel_projection(torch.stack(pooled)).flatten(start_dim=1)
        return F.normalize(self.image_projection(hidden).float(), dim=-1)

    def encode_texts(self, texts):
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Nonempty descriptions are required")
        counts = [len(ids) for ids in self.tokenizer.tokenizer(
            texts, add_special_tokens=True, truncation=False, padding=False)["input_ids"]]
        if max(counts) > 256:
            raise ValueError(f"Description exceeds 256 tokens ({max(counts)}); no silent truncation")
        tokens = self.tokenizer(texts, context_length=256).to(next(self.parameters()).device)
        # Keep the original trained CLS pooling and pretrained text projection.
        vectors = F.normalize(self.text_encoder(tokens).float(), dim=-1)
        return F.normalize(self.text_projection(vectors).float(), dim=-1)

    def forward(self, wells, image_microbatch=32):
        image = self.encode_wells(wells, image_microbatch)
        text = self.encode_texts([row["text"] for row in wells])
        return image, text, self.logit_scale.float().clamp(max=math.log(100)).exp()


def contrastive_loss(image, text, scale, descriptions):
    """Symmetric InfoNCE; exactly repeated full captions count as positives."""
    if len(image) < 2 or len(image) != len(text) or len(image) != len(descriptions):
        raise ValueError("At least two complete well/description pairs are required")
    logits = scale.float() * (image.float() @ text.float().T)
    mask = torch.tensor([[a == b for b in descriptions] for a in descriptions],
                        device=logits.device, dtype=torch.bool)
    positive = logits.masked_fill(~mask, -torch.inf)
    return ((torch.logsumexp(logits, 1) - torch.logsumexp(positive, 1)).mean()
            + (torch.logsumexp(logits, 0) - torch.logsumexp(positive, 0)).mean()) / 2

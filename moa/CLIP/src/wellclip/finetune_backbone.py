"""An exact frozen-prefix/trainable-tail split of the local DINOv2 backbone."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn


class DINOTrainableTail(nn.Module):
    """Last transformer blocks and the original final normalization.

    Input contains CLS and *all* patch tokens. Returning only CLS before these
    blocks would prevent their attention from adapting to spatial structure.
    """

    def __init__(self, blocks: Iterable[nn.Module], layernorm: nn.Module, hidden_size: int):
        super().__init__()
        self.blocks = nn.ModuleList(list(blocks))
        self.layernorm = layernorm
        self.hidden_size = hidden_size

    def tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_size:
            raise ValueError("Expected full DINO tokens [views,tokens,hidden_size]")
        parameter = next(self.parameters())
        hidden = tokens.to(device=parameter.device, dtype=parameter.dtype)
        for block in self.blocks:
            hidden = block(hidden, head_mask=None)
        return self.layernorm(hidden)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.tokens(tokens)[:, 0]


class SplitDINO(nn.Module):
    """Keep the original embeddings/first ten blocks fixed, tune the last two.

    Construct on CPU and move only ``split.tail`` to CUDA when training cached
    prefix tokens. The original image prefix then consumes no accelerator RAM.
    ``model`` is an injectable Hugging Face Dinov2Model for small exactness tests.
    """

    def __init__(self, model_dir: str | Path | None = None, device: str = "cpu", *,
                 model: nn.Module | None = None, train_last: int = 2):
        super().__init__()
        if model is None:
            from transformers import AutoModel
            if model_dir is None:
                raise ValueError("A local pretrained DINO directory is required")
            model = AutoModel.from_pretrained(str(model_dir), local_files_only=True,
                                             trust_remote_code=False, use_safetensors=True)
        layers = list(model.encoder.layer)
        if train_last < 1 or train_last >= len(layers):
            raise ValueError("A nonempty frozen prefix and trainable tail are required")
        self.config = model.config
        self.split_index = len(layers) - train_last
        self.train_last = train_last
        self.embeddings = model.embeddings.requires_grad_(False)
        self.prefix_blocks = nn.ModuleList(layers[:self.split_index]).requires_grad_(False)
        self.tail = DINOTrainableTail(layers[self.split_index:], model.layernorm,
                                      self.config.hidden_size).requires_grad_(True)
        self.to(device)
        self.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.embeddings.eval()
        self.prefix_blocks.eval()
        self.tail.train(mode)
        return self

    @torch.no_grad()
    def prefix(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4 or pixels.shape[1] != self.config.num_channels:
            raise ValueError("Expected RGB image batch [views,3,height,width]")
        self.embeddings.eval()
        self.prefix_blocks.eval()
        parameter = next(self.embeddings.parameters())
        # Prefix caches deliberately preserve the original FP32 computation.
        # Outer mixed precision for the trainable tail must not alter them.
        with torch.autocast(device_type=parameter.device.type, enabled=False):
            hidden = self.embeddings(pixels.to(device=parameter.device, dtype=torch.float32),
                                     bool_masked_pos=None)
            for block in self.prefix_blocks:
                hidden = block(hidden, head_mask=None)
        return hidden.detach().float()

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.tail(self.prefix(pixels))

    def tail_parameters(self):
        return self.tail.parameters()

    def trainable_parameters(self):
        return self.tail.parameters()

    def tail_state_dict(self) -> dict:
        return self.tail.state_dict()

    def load_tail_state_dict(self, state: dict, strict: bool = True):
        return self.tail.load_state_dict(state, strict=strict)

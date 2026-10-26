"""Small trainable well/text adapters on top of frozen feature caches."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class WellCLIP(nn.Module):
    """Preserve five channel slots while treating sites as an unordered set."""

    def __init__(self) -> None:
        super().__init__()
        self.channel_projection = nn.Sequential(nn.Linear(384, 32), nn.GELU())
        self.image_projection = nn.Sequential(nn.Dropout(0.1), nn.Linear(160, 128))
        self.text_projection = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, 128))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, image_features: torch.Tensor) -> torch.Tensor:
        if image_features.ndim != 3 or image_features.shape[1:] != (5, 384):
            raise ValueError("Expected image features [wells,5,384]")
        hidden = self.channel_projection(image_features).flatten(start_dim=1)
        return F.normalize(self.image_projection(hidden), dim=-1)

    def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
        if text_features.ndim != 2 or text_features.shape[1] != 512:
            raise ValueError("Expected BiomedCLIP text features [descriptions,512]")
        return F.normalize(self.text_projection(text_features), dim=-1)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor):
        return (self.encode_image(image_features), self.encode_text(text_features),
                self.logit_scale.clamp(max=math.log(100.0)).exp())


def symmetric_loss(logits: torch.Tensor, positive_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetric InfoNCE; identical descriptions can have multiple positives."""
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or not logits.numel():
        raise ValueError("Expected a nonempty square well/description logit matrix")
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite contrastive logits")
    if positive_mask is None:
        positive_mask = torch.eye(len(logits), dtype=torch.bool, device=logits.device)
    if positive_mask.shape != logits.shape or positive_mask.dtype != torch.bool:
        raise ValueError("Positive mask must be boolean and match logits")
    positive_mask = positive_mask.to(logits.device)
    if not positive_mask.any(dim=0).all() or not positive_mask.any(dim=1).all():
        raise ValueError("Each image and description must have at least one positive")
    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    image_loss = torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
    text_loss = torch.logsumexp(logits, dim=0) - torch.logsumexp(positive_logits, dim=0)
    return (image_loss.mean() + text_loss.mean()) / 2

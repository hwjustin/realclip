"""Independent checks of full-view gradients and the contrastive objective."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from realclip_h200.model import WellCLIP, contrastive_loss
from realclip_h200.train import make_optimizer, validated_config


class ImageTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 24), nn.GELU(), nn.Linear(24, 384))

    def forward(self, pixel_values, return_dict=True):
        return SimpleNamespace(last_hidden_state=self.layers(pixel_values.mean((-2, -1)))[:, None])


class TextTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 32)
        self.projection = nn.Linear(32, 512)

    def forward(self, tokens):
        return self.projection(self.embedding(tokens).mean(1))


class Tokenizer:
    @property
    def tokenizer(self):
        return self

    def __call__(self, texts, context_length=None, **kwargs):
        ids = [[1, index + 2] for index, _ in enumerate(texts)]
        return torch.tensor(ids) if context_length else {"input_ids": ids}


def test_every_view_receives_gradient_and_optimizer_owns_both_complete_towers():
    torch.manual_seed(122)
    image = ImageTower().requires_grad_(False)
    text = TextTower().requires_grad_(False)
    model = WellCLIP(image, text, Tokenizer(), embedding_dim=12, dropout=0, image_checkpoint=True).train()
    data = [{"pixels": torch.randn(15, 3, 3, 5, requires_grad=True),
             "channels": torch.tensor(list(range(5)) * 3), "text": f"caption {i}"}
            for i in range(3)]
    config = validated_config({"precision": "fp32", "num_workers": 0})
    optimizer = make_optimizer(model, config)
    owned = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert owned == {id(parameter) for parameter in model.parameters()}
    image_vectors, text_vectors, scale = model(data, image_microbatch=4)
    contrastive_loss(image_vectors, text_vectors, scale, [row["text"] for row in data]).backward()
    for row in data:
        gradients = row["pixels"].grad
        assert gradients is not None and torch.isfinite(gradients).all()
        assert torch.all(gradients.flatten(1).abs().sum(1) > 0)
    for tower in (image, text):
        assert all(parameter.requires_grad and parameter.grad is not None
                   and torch.isfinite(parameter.grad).all() for parameter in tower.parameters())


def test_distinct_descriptions_use_standard_symmetric_infonce():
    torch.manual_seed(199)
    image = F.normalize(torch.randn(5, 8), dim=-1)
    text = F.normalize(torch.randn(5, 8), dim=-1)
    scale = torch.tensor(4.0)
    logits = scale * image @ text.T
    expected = (F.cross_entropy(logits, torch.arange(5)) + F.cross_entropy(logits.T, torch.arange(5))) / 2
    actual = contrastive_loss(image, text, scale, [str(index) for index in range(5)])
    torch.testing.assert_close(actual, expected)

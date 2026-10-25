from types import SimpleNamespace

import pytest
import torch
from torch import nn

from realclip_h200.model import WellCLIP, contrastive_loss
from realclip_h200 import train


class TinyDINO(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 384)

    def forward(self, pixel_values, return_dict=True):
        values = self.projection(pixel_values.mean(dim=(-2, -1)))
        return SimpleNamespace(last_hidden_state=values[:, None])


class TinyTokenizer:
    @property
    def tokenizer(self):
        return self

    def __call__(self, texts, context_length=None, **kwargs):
        tokens = [[sum(text.encode()) % 29 + 1, len(text) % 29 + 1] for text in texts]
        return torch.tensor(tokens) if context_length else {"input_ids": tokens}


class TinyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 512)

    def forward(self, tokens):
        return self.embedding(tokens).mean(1)


def tiny_model(checkpoint=True, dropout=0.1):
    return WellCLIP(TinyDINO(), TinyText(), TinyTokenizer(), embedding_dim=8,
                    dropout=dropout, image_checkpoint=checkpoint)


def wells(count=4):
    generator = torch.Generator().manual_seed(123)
    return [{"sample_id": str(i), "pixels": torch.rand(10, 3, 4, 4, generator=generator),
             "channels": torch.tensor(list(range(5)) * 2), "text": f"description {i}"}
            for i in range(count)]


def test_both_complete_towers_update_with_whole_encoder_checkpoint():
    torch.manual_seed(4)
    model = tiny_model().train()
    before_image = model.image_encoder.projection.weight.detach().clone()
    before_text = model.text_encoder.embedding.weight.detach().clone()
    config = train.validated_config({"precision": "fp32", "num_workers": 0})
    optimizer = train.make_optimizer(model, config)
    data = wells()
    image, text, scale = model(data, image_microbatch=3)
    contrastive_loss(image, text, scale, [row["text"] for row in data]).backward()
    assert all(parameter.requires_grad for parameter in model.image_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.text_encoder.parameters())
    optimizer.step()
    assert not torch.equal(before_image, model.image_encoder.projection.weight)
    assert not torch.equal(before_text, model.text_encoder.embedding.weight)


def test_checkpointed_microbatches_preserve_full_forward_gradients():
    torch.manual_seed(9)
    ordinary, checkpointed = tiny_model(False, 0), tiny_model(True, 0)
    checkpointed.load_state_dict(ordinary.state_dict())
    data = wells()
    for model, microbatch in ((ordinary, 10), (checkpointed, 3)):
        model.train()
        image, text, scale = model(data, microbatch)
        contrastive_loss(image, text, scale, [row["text"] for row in data]).backward()
    for left, right in zip(ordinary.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-6)


def test_all_sites_pooling_is_order_independent():
    model = tiny_model(dropout=0).eval()
    data = wells(2)
    first = model.encode_wells(data, image_microbatch=4)
    permutation = torch.tensor([9, 2, 6, 1, 5, 0, 3, 8, 7, 4])
    second = model.encode_wells([{**row, "pixels": row["pixels"][permutation],
                                 "channels": row["channels"][permutation]} for row in data], 3)
    torch.testing.assert_close(first, second)


def test_resume_matches_uninterrupted_training_and_never_loads_test(tmp_path, monkeypatch):
    data = wells(4)
    splits_seen = []

    class Dataset:
        def __init__(self, root, split):
            splits_seen.append(split)
            assert split != "test"
            self.rows = data if split == "train" else data[:2]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    monkeypatch.setattr(train, "load_manifest", lambda root: {"manifest_id": "tiny-data"})
    monkeypatch.setattr(train, "WellDataset", Dataset)
    monkeypatch.setattr(train.WellCLIP, "from_pretrained", lambda *args: tiny_model())
    config = train.validated_config({"precision": "fp32", "num_workers": 0,
                                    "epochs": 2, "well_batch_size": 2,
                                    "image_microbatch": 3, "warmup_fraction": 0.25})
    uninterrupted = tmp_path / "uninterrupted"
    interrupted = tmp_path / "interrupted"
    train.run_training(config, tmp_path, uninterrupted, tmp_path, device="cpu")
    train.run_training(config, tmp_path, interrupted, tmp_path, device="cpu", smoke_steps=1)
    train.run_training(config, tmp_path, interrupted, tmp_path, device="cpu", resume=interrupted / "last.pt")
    first = torch.load(uninterrupted / "last.pt", weights_only=True)
    resumed = torch.load(interrupted / "last.pt", weights_only=True)
    assert first["global_step"] == resumed["global_step"] == 4
    assert first["history"] == resumed["history"]
    for key in first["model"]:
        torch.testing.assert_close(first["model"][key], resumed["model"][key], rtol=0, atol=0)
    assert set(splits_seen) == {"train", "val"}
    bad_config = {**config, "backbone_lr": 2e-5}
    with pytest.raises(ValueError, match="incompatible"):
        train.run_training(bad_config, tmp_path, tmp_path / "bad", tmp_path,
                           device="cpu", resume=interrupted / "last.pt")
    with pytest.raises(ValueError, match="inside --output"):
        train.run_training(config, tmp_path, tmp_path / "other-run", tmp_path,
                           device="cpu", resume=interrupted / "last.pt")
    # Explicit evaluation may change runtime precision; resume may not.
    restored_model = tiny_model()
    eval_config = {**config, "precision": "bf16", "image_microbatch": 1}
    train.load_checkpoint(interrupted / "best.pt", restored_model, None, None,
                          eval_config, "tiny-data")
    (interrupted / "best.pt").unlink()
    with pytest.raises(FileNotFoundError, match="best.pt"):
        train.run_training(config, tmp_path, interrupted, tmp_path,
                           device="cpu", resume=interrupted / "last.pt")


def test_batching_never_creates_singleton_and_keeps_every_well():
    batches = train.epoch_batches(17, 8, 42, 0)
    assert [len(batch) for batch in batches] == [8, 9]
    assert sorted(index for batch in batches for index in batch) == list(range(17))


def test_repeated_descriptions_are_positives_and_ranking_uses_full_gallery():
    image = torch.eye(3)
    descriptions = ["one", "two", "two"]
    assert torch.isfinite(contrastive_loss(image, image, torch.tensor(10.), descriptions))
    metrics = train.retrieval_metrics(image, image, descriptions)
    assert metrics["mean_mrr"] == 1
    assert metrics["candidate_count"] == 3

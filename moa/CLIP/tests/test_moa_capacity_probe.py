"""Meaningful checks on AP ties, early selection and the MoA training boundary."""
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_moa_capacity_probe import ImageMoA, average_precision, better_checkpoint, fit_adapter


def test_tied_ap_uses_threshold_groups():
    assert average_precision(np.array([1, 0, 1, 0]), np.ones(4)) == .5
    assert average_precision(np.array([1, 0, 1]), np.array([.9, .8, .8])) == pytest.approx(5 / 6)
    assert average_precision(np.array([0, 0]), np.array([.8, .1])) == 0


def test_checkpoint_ties_keep_earliest_and_train_has_no_test_input():
    assert not better_checkpoint(.4, .4)
    assert better_checkpoint(.4001, .4)
    assert not any("test" in name for name in inspect.signature(fit_adapter).parameters)


def test_image_capacity_matches_original_and_normalizes():
    model = ImageMoA(2).eval()
    count = sum(p.numel() for name, p in model.named_parameters() if not name.startswith("classifier."))
    assert count == 32928
    with torch.no_grad():
        encoded = model.encode(torch.randn(3, 5, 384))
        torch.testing.assert_close(encoded.norm(dim=-1), torch.ones(3))
        assert model(torch.randn(3, 5, 384)).shape == (3, 2)

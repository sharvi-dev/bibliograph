"""Tests for model architecture."""
import torch
import pytest
from src.model.model import BibliographGT


def test_forward_pass():
    model = BibliographGT(in_channels=32, hidden_channels=16, out_channels=8, heads=2)
    x = torch.randn(10, 32)
    edge_index = torch.randint(0, 10, (2, 20))
    z = model(x, edge_index)
    assert z.shape == (10, 8)


def test_predict_link():
    model = BibliographGT(in_channels=32, hidden_channels=16, out_channels=8, heads=2)
    z = torch.randn(10, 8)
    edge_index = torch.tensor([[0, 1, 2], [3, 4, 5]])
    scores = model.predict_link(z, edge_index)
    assert scores.shape == (3,)

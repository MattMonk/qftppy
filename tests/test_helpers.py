"""Tests for the weakly-covered pieces of tests/helpers.py itself."""

import torch

from helpers import METRIC, from_cpp, trace_pair


def test_from_cpp_rank2_axis_order():
    # qft++ stores element (mu, nu) at mu + 4*nu, so element (1, 2) is at 1 + 4*2 = 9.
    t = from_cpp(torch.arange(16.0), 2)
    assert t[1, 2] == 1 + 4 * 2


def test_from_cpp_rank3_axis_order():
    # element (1, 2, 3) is at 1 + 4*2 + 16*3.
    t = from_cpp(torch.arange(64.0), 3)
    assert t[1, 2, 3] == 1 + 4 * 2 + 16 * 3


def test_from_cpp_batched():
    flat = torch.arange(32.0).reshape(2, 16)
    t = from_cpp(flat, 2)
    assert t.shape == (2, 4, 4)
    assert t[0, 1, 2] == 1 + 4 * 2
    assert t[1, 1, 2] == 16 + 1 + 4 * 2


def test_trace_pair_metric():
    metric = METRIC.unsqueeze(0)
    assert trace_pair(metric, 1, 2).item() == 4

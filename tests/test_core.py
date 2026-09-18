import itertools

import pytest
import torch

from helpers import METRIC, assert_close, on_shell
from qftppy import QFTNative


def _rand(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_metric(qft):
    assert torch.equal(qft.metric(), METRIC)


def test_metric_returns_copy(qft):
    g = qft.metric()
    g[0, 0] = 5.0
    assert qft.g[0, 0] == 1.0


def test_dtypes():
    assert QFTNative().dtype == torch.float64
    assert QFTNative(dtype=torch.float32).metric().dtype == torch.float32


def test_levi_civita(qft):
    eps = qft.levi_civita()
    assert eps[0, 1, 2, 3] == 1.0
    assert int((eps != 0).sum()) == 24
    for a, b in itertools.combinations(range(4), 2):
        assert torch.equal(eps, -eps.transpose(a, b))


def test_dot(qft):
    p = on_shell(0.5)
    k = on_shell(1.0, seed=7)
    expected = p[:, 0] * k[:, 0] - (p[:, 1:] * k[:, 1:]).sum(dim=1)
    assert_close(qft.dot(p, k), expected)
    assert_close(qft.dot(p, p), torch.full((p.shape[0],), 0.25, dtype=torch.float64))


def test_outer_product(qft):
    a = _rand(5, 4, seed=1)
    b = _rand(5, 4, 4, seed=2)
    res = qft.outer_product(a, b)
    assert res.shape == (5, 4, 4, 4)
    assert_close(res, a[:, :, None, None] * b[:, None, :, :])


@pytest.mark.parametrize("n", [1, 3, 4, 5])
def test_contract_vectors_with_flags(qft, n):
    a = _rand(n, 4, seed=1)
    b = _rand(n, 4, seed=2)
    expected = torch.einsum("ni,ij,nj->n", a, METRIC, b)
    assert_close(qft.contract(a, b, batched1=True, batched2=True), expected)


@pytest.mark.parametrize("n", [1, 3, 5])
def test_contract_vectors_guessed_batching(qft, n):
    a = _rand(n, 4, seed=1)
    b = _rand(n, 4, seed=2)
    expected = torch.einsum("ni,ij,nj->n", a, METRIC, b)
    assert_close(qft.contract(a, b), expected)


def test_contract_unbatched_pair(qft):
    a = _rand(4, seed=1)
    b = _rand(4, 4, seed=2)
    expected = torch.einsum("i,ij,jk->k", a, METRIC, b)
    assert_close(qft.contract(a, b), expected)
    assert_close(qft.contract(a, b, batched1=False, batched2=False), expected)


@pytest.mark.parametrize("n", [3, 4])
def test_contract_static_with_batched(qft, n):
    eps = qft.levi_civita()
    v = _rand(n, 4, seed=3)
    expected = torch.einsum("ijkl,lm,nm->nijk", eps, METRIC, v)
    assert_close(qft.contract(eps, v, batched1=False, batched2=True), expected)


def test_contract_two_indices(qft):
    t1 = _rand(3, 4, 4, seed=1)
    t2 = _rand(3, 4, 4, seed=2)
    expected = torch.einsum("nab,ac,bd,ncd->n", t1, METRIC, METRIC, t2)
    assert_close(qft.contract(t1, t2, n=2), expected)


def test_contract_zero_is_outer_product(qft):
    a = _rand(3, 4, seed=1)
    b = _rand(3, 4, seed=2)
    assert_close(qft.contract(a, b, n=0), qft.outer_product(a, b))


def test_contract_all_indices(qft):
    t1 = _rand(3, 4, 4, seed=1)
    t2 = _rand(3, 4, 4, seed=2)
    assert_close(qft.contract(t1, t2, n=-1), qft.contract(t1, t2, n=2))


def test_symmetrize(qft):
    t = _rand(2, 4, 4, 4, seed=4)
    s = qft.symmetrize(t)
    for a, b in itertools.combinations(range(1, 4), 2):
        assert_close(s, s.transpose(a, b))
    assert_close(qft.symmetrize(s), s)
    v = _rand(2, 4, seed=5)
    assert torch.equal(qft.symmetrize(v), v)

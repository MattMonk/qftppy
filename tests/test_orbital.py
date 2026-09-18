import itertools

import pytest
import torch

from helpers import assert_close, contract_index, lower, momentum_pairs, trace_pair

RANKS = range(0, 7)


def _kperp(qft, pa, pb):
    P = pa + pb
    p_ab = 0.5 * (pa - pb)
    return p_ab - P * (qft.dot(P, p_ab) / qft.dot(P, P)).unsqueeze(-1)


@pytest.mark.parametrize("rank", RANKS)
def test_shape(qft, rank):
    pa, pb = momentum_pairs()
    assert qft.orbital_tensor(pa, pb, rank).shape == (pa.shape[0],) + (4,) * rank


def test_rank0_is_one(qft):
    pa, pb = momentum_pairs()
    assert torch.equal(qft.orbital_tensor(pa, pb, 0), torch.ones(pa.shape[0], dtype=torch.float64))


def test_rank1_is_kperp(qft):
    pa, pb = momentum_pairs()
    assert_close(qft.orbital_tensor(pa, pb, 1), _kperp(qft, pa, pb))


@pytest.mark.parametrize("rank", range(2, 7))
def test_symmetric(qft, rank):
    pa, pb = momentum_pairs()
    L = qft.orbital_tensor(pa, pb, rank)
    for a, b in itertools.combinations(range(1, rank + 1), 2):
        assert_close(L.transpose(a, b), L)


@pytest.mark.parametrize("rank", range(2, 7))
def test_traceless(qft, rank):
    pa, pb = momentum_pairs()
    L = qft.orbital_tensor(pa, pb, rank)
    scale = L.abs().max()
    for a, b in itertools.combinations(range(1, rank + 1), 2):
        assert trace_pair(L, a, b).abs().max() <= 1e-10 * scale


@pytest.mark.parametrize("rank", range(1, 7))
def test_transverse_to_total_momentum(qft, rank):
    pa, pb = momentum_pairs()
    L = qft.orbital_tensor(pa, pb, rank)
    P = pa + pb
    scale = L.abs().max() * P.abs().max()
    for axis in range(1, rank + 1):
        assert contract_index(L, axis, P).abs().max() <= 1e-10 * scale


@pytest.mark.parametrize("rank", range(1, 7))
def test_normalisation(qft, rank):
    """L^(l) fully contracted with kperp^l equals (kperp^2)^l (the qft++ normalisation)."""
    pa, pb = momentum_pairs()
    L = qft.orbital_tensor(pa, pb, rank)
    kperp = _kperp(qft, pa, pb)
    contracted = L
    for _ in range(rank):
        contracted = contract_index(contracted, contracted.ndim - 1, kperp)
    kperp_sq = qft.dot(kperp, kperp)
    assert_close(contracted, kperp_sq**rank)


def test_lower_helper_consistent_with_dot(qft):
    pa, pb = momentum_pairs()
    assert_close((lower(pa, 1) * pb).sum(-1), qft.dot(pa, pb))

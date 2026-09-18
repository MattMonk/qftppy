import itertools

import pytest
import torch

from helpers import assert_close, contract_index, lower, on_shell, trace_pair
from qftppy import PolVectorNative

MASS = 0.892


def _norm(eps):
    """eps* . eps with all Lorentz indices contracted through the metric."""
    lowered = eps
    for axis in range(1, eps.ndim):
        lowered = lower(lowered, axis)
    return (eps.conj() * lowered).reshape(eps.shape[0], -1).sum(-1)


@pytest.mark.parametrize("spin", [1, 2, 3])
def test_shape_transverse_normalised(qft, spin):
    p = on_shell(MASS)
    pv = PolVectorNative(qft)
    for mz in range(-spin, spin + 1):
        eps = pv.epsilon(p, MASS, spin, mz)
        assert eps.shape == (p.shape[0],) + (4,) * spin
        for axis in range(1, spin + 1):
            assert contract_index(eps, axis, p).abs().max() <= 1e-10, (mz, axis)
        norm = _norm(eps)
        assert_close(norm, torch.full_like(norm, (-1) ** spin))


@pytest.mark.parametrize("spin", [2, 3])
def test_symmetric_traceless(qft, spin):
    p = on_shell(MASS)
    pv = PolVectorNative(qft)
    for mz in range(-spin, spin + 1):
        eps = pv.epsilon(p, MASS, spin, mz)
        for a, b in itertools.combinations(range(1, spin + 1), 2):
            assert_close(eps.transpose(a, b), eps)
            assert trace_pair(eps, a, b).abs().max() <= 1e-10


def test_rest_frame_vectors(qft):
    p = on_shell(MASS, n=1)  # the single event is at rest
    pv = PolVectorNative(qft)
    s = 2**-0.5
    expected = {
        1: torch.tensor([[0, -s, -1j * s, 0]], dtype=pv.dtype),
        0: torch.tensor([[0, 0, 0, 1]], dtype=pv.dtype),
        -1: torch.tensor([[0, s, -1j * s, 0]], dtype=pv.dtype),
    }
    for mz, vec in expected.items():
        assert_close(pv.epsilon(p, MASS, 1, mz), vec)


def test_spin1_projector_is_sum_over_states(qft):
    p = on_shell(MASS)
    pv = PolVectorNative(qft)
    total = sum(qft.outer_product(pv.epsilon(p, MASS, 1, mz), pv.epsilon(p, MASS, 1, mz).conj()) for mz in (-1, 0, 1))
    assert_close(total, pv.projector(p, MASS, 1).to(total.dtype))


def test_spin2_projector_idempotent_on_states(qft):
    """P^(2) contracted with eps(mz) over two indices returns eps(mz)."""
    p = on_shell(MASS)
    pv = PolVectorNative(qft)
    proj = pv.projector(p, MASS, 2)
    for mz in range(-2, 3):
        eps = pv.epsilon(p, MASS, 2, mz)
        applied = torch.einsum("nabcd,ncd->nab", proj, lower(lower(eps, 1), 2))
        assert_close(applied, eps)


@pytest.mark.xfail(strict=True, reason="massless polarization vectors skip the helicity rotation (qft++ rotates them)")
def test_massless_vectors_transverse(qft):
    p = torch.tensor([[1.0, 0.6, 0.0, 0.8]], dtype=torch.float64)  # photon not along z
    pv = PolVectorNative(qft)
    for mz in (1, -1):
        eps = pv.epsilon(p, 0.0, 1, mz)
        assert eps[0, 1:].real @ p[0, 1:] == pytest.approx(0.0, abs=1e-12)
        assert eps[0, 1:].imag @ p[0, 1:] == pytest.approx(0.0, abs=1e-12)

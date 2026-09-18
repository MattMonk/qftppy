import pytest
import torch

from helpers import assert_close, momentum_pairs, on_shell
from qftppy import DiracAntiSpinorNative, DiracSpinorNative, PolVectorNative, QFTNative

BATCH_SIZES = [1, 3, 4, 5]


def _per_event(fn, *batched):
    n = batched[0].shape[0]
    return torch.cat([fn(*(b[i : i + 1] for b in batched)) for i in range(n)])


@pytest.mark.parametrize("n", BATCH_SIZES)
@pytest.mark.parametrize("rank", range(0, 7))
def test_orbital_batch_matches_per_event(qft, n, rank):
    pa, pb = momentum_pairs(n=n)
    assert_close(qft.orbital_tensor(pa, pb, rank), _per_event(lambda a, b: qft.orbital_tensor(a, b, rank), pa, pb))


@pytest.mark.parametrize("n", BATCH_SIZES)
@pytest.mark.parametrize("spin", [1, 2])
def test_polarization_batch_matches_per_event(qft, n, spin):
    p = on_shell(0.892, n=n)
    pv = PolVectorNative(qft)
    for mz in range(-spin, spin + 1):
        assert_close(pv.epsilon(p, 0.892, spin, mz), _per_event(lambda q: pv.epsilon(q, 0.892, spin, mz), p))


@pytest.mark.parametrize("n", BATCH_SIZES)
def test_spinor_batch_matches_per_event(qft, n):
    p = on_shell(0.938, n=n)
    ds, das = DiracSpinorNative(qft), DiracAntiSpinorNative(qft)
    for mz in (0.5, -0.5):
        assert_close(ds.u(p, 0.938, mz), _per_event(lambda q: ds.u(q, 0.938, mz), p))
        assert_close(das.v(p, 0.938, mz), _per_event(lambda q: das.v(q, 0.938, mz), p))


@pytest.mark.parametrize("n", BATCH_SIZES)
def test_contract_with_flags_matches_per_event(qft, n):
    pa, pb = momentum_pairs(n=n)
    L1 = qft.orbital_tensor(pa, pb, 1)
    L2 = qft.orbital_tensor(pa, pb, 2)
    got = qft.contract(L2, L1, batched1=True, batched2=True)
    expected = _per_event(lambda a, b: qft.contract(a, b, batched1=True, batched2=True), L2, L1)
    assert_close(got, expected)


@pytest.mark.parametrize("rank", range(1, 5))
def test_orbital_float32_matches_float64(rank):
    pa, pb = momentum_pairs()
    q64, q32 = QFTNative(dtype=torch.float64), QFTNative(dtype=torch.float32)
    L64 = q64.orbital_tensor(pa, pb, rank)
    L32 = q32.orbital_tensor(pa.float(), pb.float(), rank)
    assert L32.dtype == torch.float32
    assert_close(L32.double(), L64, rtol=1e-5)


def test_polarization_float32_matches_float64():
    p = on_shell(0.892)
    pv64 = PolVectorNative(QFTNative(dtype=torch.float64))
    pv32 = PolVectorNative(QFTNative(dtype=torch.float32))
    e32 = pv32.epsilon(p.float(), 0.892, 2, 1)
    assert e32.dtype == torch.complex64
    assert_close(e32.to(torch.complex128), pv64.epsilon(p, 0.892, 2, 1), rtol=1e-5)

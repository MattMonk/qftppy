"""Numerical comparison of qftppy against the original C++ qft++ (via the qftcpp_ref module).

Run with: pixi run -e cpp test-cpp
Skipped automatically when qftcpp_ref is not installed (e.g. in the dev environment).
"""

import math

import pytest
import torch

import qftppy
from helpers import assert_close, from_cpp, momentum_pairs, on_shell

ref = pytest.importorskip("qftcpp_ref")

RTOL = 1e-10
N = 50


def test_metric(qft):
    assert_close(qft.metric(), from_cpp(ref.metric(), 2), rtol=0)


def test_levi_civita(qft):
    assert_close(qft.levi_civita(), from_cpp(ref.levi_civita(), 4), rtol=0)


def test_dirac_matrices():
    assert_close(qftppy.dirac_gamma(), torch.as_tensor(ref.dirac_gamma()), rtol=0)
    assert_close(qftppy.dirac_gamma5(), torch.as_tensor(ref.dirac_gamma5()).to(torch.complex128), rtol=0)
    assert_close(qftppy.dirac_sigma(), torch.as_tensor(ref.dirac_sigma()), rtol=0)
    assert_close(qftppy.pauli_sigma(), torch.as_tensor(ref.pauli_sigma()), rtol=0)


@pytest.mark.parametrize("rank", range(0, 7))
def test_orbital_tensor(qft, rank):
    pa, pb = momentum_pairs(n=N)
    expected = from_cpp(ref.orbital_tensor(pa.numpy(), pb.numpy(), rank), rank)
    assert_close(qft.orbital_tensor(pa, pb, rank), expected, rtol=RTOL)


POL_MASS = 0.892


@pytest.mark.parametrize("spin", [1, 2, 3])
def test_polarization_vectors(qft, spin):
    # moving frames only: the at-rest case is covered by tests/test_polarization.py
    p = on_shell(POL_MASS, n=N, include_rest=False)
    pv = qftppy.PolVectorNative(qft)
    for mz in range(-spin, spin + 1):
        expected = from_cpp(ref.pol_vector(p.numpy(), POL_MASS, spin, mz), spin)
        assert_close(pv.epsilon(p, POL_MASS, spin, mz), expected, rtol=RTOL)


@pytest.mark.parametrize("spin", [1, 2])
def test_polarization_projector(qft, spin):
    p = on_shell(POL_MASS, n=N, include_rest=False)
    pv = qftppy.PolVectorNative(qft)
    expected = from_cpp(ref.pol_projector(p.numpy(), POL_MASS, spin), 2 * spin)
    assert_close(pv.projector(p, POL_MASS, spin).to(torch.complex128), expected, rtol=RTOL)


FERMION_MASS = 0.938


@pytest.mark.parametrize("mz", [0.5, -0.5])
def test_dirac_spinors(qft, mz):
    p = on_shell(FERMION_MASS, n=N, include_rest=False)
    ds, das = qftppy.DiracSpinorNative(qft), qftppy.DiracAntiSpinorNative(qft)
    assert_close(ds.u(p, FERMION_MASS, mz).squeeze(-1), torch.as_tensor(ref.dirac_u(p.numpy(), FERMION_MASS, mz)))
    assert_close(das.v(p, FERMION_MASS, mz).squeeze(-1), torch.as_tensor(ref.dirac_v(p.numpy(), FERMION_MASS, mz)))


def test_dirac_projectors(qft):
    p = on_shell(FERMION_MASS, n=N, include_rest=False)
    ds, das = qftppy.DiracSpinorNative(qft), qftppy.DiracAntiSpinorNative(qft)
    assert_close(ds.projector(p, FERMION_MASS), torch.as_tensor(ref.dirac_projector(p.numpy(), FERMION_MASS)))
    assert_close(das.projector(p, FERMION_MASS), torch.as_tensor(ref.dirac_anti_projector(p.numpy(), FERMION_MASS)))


def test_spd_amplitudes(qft):
    """S, P and D-wave spin amplitudes for P0 -> V1 (p1 p2) V2 (p3 p4), built with qftppy's contract."""
    p1 = on_shell(0.494, n=N, seed=11, include_rest=False)
    p2 = on_shell(0.140, n=N, seed=12, include_rest=False)
    p3 = on_shell(0.494, n=N, seed=13, include_rest=False)
    p4 = on_shell(0.140, n=N, seed=14, include_rest=False)
    pV1, pV2 = p1 + p2, p3 + p4
    p0 = pV1 + pV2
    L1_V1 = qft.orbital_tensor(p1, p2, 1)
    L1_V2 = qft.orbital_tensor(p3, p4, 1)
    L1_P0 = qft.orbital_tensor(pV1, pV2, 1)
    L2_P0 = qft.orbital_tensor(pV1, pV2, 2)

    def c(a, b, static_first=False):
        return qft.contract(a, b, batched1=not static_first, batched2=True)

    spin_s = c(L1_V1, L1_V2)
    spin_p = c(c(c(c(qft.levi_civita(), L1_P0, static_first=True), L1_V1), L1_V2), p0)
    spin_d = c(c(L2_P0, L1_V1), L1_V2)

    expected = torch.as_tensor(ref.spd_amplitudes(p1.numpy(), p2.numpy(), p3.numpy(), p4.numpy()))
    assert_close(spin_s, expected[:, 0], rtol=RTOL)
    assert_close(spin_p, expected[:, 1], rtol=RTOL)
    assert_close(spin_d, expected[:, 2], rtol=RTOL)


def _projections(j):
    return [x - j for x in range(int(round(2 * j)) + 1)]


SPINS = [0, 0.5, 1, 1.5, 2]


def test_clebsch_all_combinations():
    count = 0
    for j1 in SPINS:
        for j2 in SPINS:
            J = abs(j1 - j2)
            while J <= j1 + j2 + 1e-9:
                for m1 in _projections(j1):
                    for m2 in _projections(j2):
                        M = m1 + m2
                        if abs(M) <= J + 1e-9:
                            got = qftppy.clebsch(j1, m1, j2, m2, J, M)
                            want = ref.clebsch(j1, m1, j2, m2, J, M)
                            assert got == pytest.approx(want, abs=1e-12), (j1, m1, j2, m2, J, M)
                            count += 1
                J += 1
    assert count > 100


def test_wigner_d_all_projections():
    beta = torch.linspace(0.0, math.pi, 17, dtype=torch.float64)
    for j in [0.5, 1, 1.5, 2]:
        for m in _projections(j):
            for n in _projections(j):
                want = torch.as_tensor(ref.wigner_d(j, m, n, beta.numpy()))
                assert_close(qftppy.wigner_d(j, m, n, beta), want, rtol=1e-12)


@pytest.mark.parametrize(
    "a, b, spin, sig, exp_fact",
    [(0.9, 0.5, 1, -1, 1), (0.9, 0.5, 1, 1, 1), (0.7, -0.01, 0, 1, 1), (0.9, 0.5, 2, 1, 0), (0.25, 0.1, 0, -1, 1)],
)
def test_regge_propagator(a, b, spin, sig, exp_fact):
    t = torch.linspace(-1.5, -0.05, 9, dtype=torch.float64)
    s = torch.linspace(3.0, 12.0, 9, dtype=torch.float64)
    want = torch.as_tensor(ref.regge_propagator(t.numpy(), s.numpy(), a, b, spin, sig, exp_fact))
    # qft++ uses pi = 3.1415926535 and its own Gamma approximation: ~3e-8 relative differences
    assert_close(qftppy.regge_propagator(t, s, a, b, spin, sig, exp_fact), want, rtol=1e-7)

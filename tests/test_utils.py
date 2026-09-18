import itertools
import math

import pytest
import torch

from helpers import assert_close
from qftppy import clebsch, regge_propagator, wigner_d


def _projections(j):
    return [x - j for x in range(int(round(2 * j)) + 1)]


@pytest.mark.parametrize(
    "args, expected",
    [
        ((0.5, 0.5, 0.5, 0.5, 1, 1), 1.0),
        ((0.5, 0.5, 0.5, -0.5, 1, 0), 1 / math.sqrt(2)),
        ((0.5, 0.5, 0.5, -0.5, 0, 0), 1 / math.sqrt(2)),
        ((0.5, -0.5, 0.5, 0.5, 0, 0), -1 / math.sqrt(2)),
        ((1, 1, 1, -1, 2, 0), 1 / math.sqrt(6)),
        ((1, 0, 1, 0, 0, 0), -1 / math.sqrt(3)),
        ((1, 1, 0.5, 0.5, 1.5, 1.5), 1.0),
        ((0.5, 0.5, 0.5, 0.5, 0, 0), 0.0),
    ],
)
def test_clebsch_values(args, expected):
    assert clebsch(*args) == pytest.approx(expected, abs=1e-12)


def test_clebsch_m_mismatch_is_zero():
    assert clebsch(1, 1, 1, 0, 2, 0) == 0.0


@pytest.mark.parametrize("j1, j2", [(0.5, 0.5), (1, 0.5), (1, 1), (1.5, 1), (2, 1)])
def test_clebsch_normalisation(j1, j2):
    """sum_{m1} <j1 m1 j2 M-m1 | J M>^2 = 1 for every allowed J, M"""
    n_j = int(round(j1 + j2 - abs(j1 - j2))) + 1
    for J in [abs(j1 - j2) + k for k in range(n_j)]:
        for M in _projections(J):
            total = sum(clebsch(j1, m1, j2, M - m1, J, M) ** 2 for m1 in _projections(j1) if abs(M - m1) <= j2 + 1e-9)
            assert total == pytest.approx(1.0, abs=1e-12), (J, M)


@pytest.mark.parametrize("j1, j2", [(0.5, 0.5), (1, 0.5), (1, 1), (2, 1)])
def test_clebsch_orthogonality_between_different_j(j1, j2):
    """sum_{m1} <j1 m1 j2 M-m1 | J M> <j1 m1 j2 M-m1 | J' M> = 0 for J != J'"""
    n_j = int(round(j1 + j2 - abs(j1 - j2))) + 1
    allowed_j = [abs(j1 - j2) + k for k in range(n_j)]
    for J, J_prime in itertools.permutations(allowed_j, 2):
        for M in set(_projections(J)) & set(_projections(J_prime)):
            total = sum(
                clebsch(j1, m1, j2, M - m1, J, M) * clebsch(j1, m1, j2, M - m1, J_prime, M)
                for m1 in _projections(j1)
                if abs(M - m1) <= j2 + 1e-9
            )
            assert total == pytest.approx(0.0, abs=1e-12), (J, J_prime, M)


BETA = torch.linspace(0.0, math.pi, 13, dtype=torch.float64)


@pytest.mark.parametrize(
    "jmn, closed_form",
    [
        ((0.5, 0.5, 0.5), lambda b: torch.cos(b / 2)),
        ((0.5, 0.5, -0.5), lambda b: -torch.sin(b / 2)),
        ((1, 0, 0), torch.cos),
        ((1, 1, 1), lambda b: (1 + torch.cos(b)) / 2),
        ((1, 1, 0), lambda b: -torch.sin(b) / math.sqrt(2)),
        ((1, 1, -1), lambda b: (1 - torch.cos(b)) / 2),
    ],
)
def test_wigner_d_closed_forms(jmn, closed_form):
    assert_close(wigner_d(*jmn, BETA), closed_form(BETA), rtol=1e-12)


@pytest.mark.parametrize("j", [0.5, 1, 1.5, 2])
def test_wigner_d_unitarity_and_identity(j):
    for n in _projections(j):
        total = sum(wigner_d(j, m, n, BETA) ** 2 for m in _projections(j))
        assert_close(total, torch.ones_like(BETA), rtol=1e-12)
        for m in _projections(j):
            expected = 1.0 if m == n else 0.0
            assert wigner_d(j, m, n, torch.zeros(1, dtype=torch.float64)).item() == pytest.approx(expected, abs=1e-12)


def _regge_reference(t, s, a, b, spin, sig, exp_fact):
    """Regge propagator via the reflection formula 2 sin(pi z) Gamma(z) = 2 pi / Gamma(1 - z)
    (math.gamma keeps signs)."""
    out = []
    for ti, si in zip(t.tolist(), s.tolist()):
        alpha = a * ti + b
        numerator = si ** (alpha - spin) * math.pi * a
        numerator *= sig + exp_fact * complex(math.cos(math.pi * alpha), -math.sin(math.pi * alpha))
        z = alpha + 1.0 - spin
        out.append(numerator * math.gamma(1.0 - z) / (2.0 * math.pi))
    return torch.tensor(out, dtype=torch.complex128)


@pytest.mark.parametrize(
    "a, b, spin, sig, exp_fact",
    [(0.9, 0.5, 1, -1, 1), (0.9, 0.5, 1, 1, 1), (0.7, -0.01, 0, 1, 1), (0.9, 0.5, 2, 1, 0), (0.25, 0.1, 0, -1, 1)],
)
def test_regge_propagator_matches_reflection_formula(a, b, spin, sig, exp_fact):
    # t grid chosen so gamma_arg = a t + b + 1 - spin takes both signs but is never a positive integer
    t = torch.linspace(-1.5, -0.05, 9, dtype=torch.float64)
    s = torch.linspace(3.0, 12.0, 9, dtype=torch.float64)
    got = regge_propagator(t, s, a, b, spin, sig, exp_fact)
    assert got.dtype == torch.complex128
    assert torch.isfinite(got.real).all() and torch.isfinite(got.imag).all()
    assert_close(got, _regge_reference(t, s, a, b, spin, sig, exp_fact), rtol=1e-12)


def test_regge_propagator_zero_gamma_argument():
    # a t + b + 1 - spin = 0 exactly: 2 sin(pi z) Gamma(z) -> 2 pi
    t = torch.tensor([-2.0], dtype=torch.float64)
    s = torch.tensor([5.0], dtype=torch.float64)
    got = regge_propagator(t, s, 0.5, 0.0, 0, 1, 1)
    assert_close(got, _regge_reference(t, s, 0.5, 0.0, 0, 1, 1), rtol=1e-12)


def test_regge_propagator_float32_dtype():
    t = torch.linspace(-1.0, -0.1, 4)
    s = torch.full_like(t, 5.0)
    assert regge_propagator(t, s, 0.9, 0.5, 1, -1).dtype == torch.complex64

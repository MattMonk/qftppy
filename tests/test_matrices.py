import torch

from helpers import METRIC, assert_close
from qftppy import dirac_gamma, dirac_gamma5, dirac_sigma, pauli_sigma

LEVI_3 = {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}


def test_pauli_algebra():
    """sigma_i sigma_j = delta_ij I + i eps_ijk sigma_k"""
    s = pauli_sigma()
    eye = torch.eye(2, dtype=s.dtype)
    for i in range(3):
        for j in range(3):
            expected = (1.0 if i == j else 0.0) * eye
            for k in range(3):
                expected = expected + 1j * LEVI_3.get((i, j, k), 0) * s[k]
            assert_close(s[i] @ s[j], expected)


def test_gamma_anticommutator():
    """{gamma^mu, gamma^nu} = 2 g^{mu nu} I"""
    gamma = dirac_gamma()
    eye = torch.eye(4, dtype=gamma.dtype)
    for mu in range(4):
        for nu in range(4):
            anti = gamma[mu] @ gamma[nu] + gamma[nu] @ gamma[mu]
            assert_close(anti, 2 * METRIC[mu, nu] * eye)


def test_gamma5_definition():
    """gamma^5 = i gamma^0 gamma^1 gamma^2 gamma^3"""
    gamma = dirac_gamma()
    assert_close(dirac_gamma5(), 1j * gamma[0] @ gamma[1] @ gamma[2] @ gamma[3])


def test_gamma5_properties():
    gamma = dirac_gamma()
    g5 = dirac_gamma5()
    for mu in range(4):
        assert_close(g5 @ gamma[mu] + gamma[mu] @ g5, torch.zeros(4, 4, dtype=g5.dtype))
    assert_close(g5 @ g5, torch.eye(4, dtype=g5.dtype))


def test_dirac_sigma():
    """sigma^{mu nu} = i/2 [gamma^mu, gamma^nu], antisymmetric in mu nu"""
    gamma = dirac_gamma()
    sigma = dirac_sigma()
    assert_close(sigma, -sigma.transpose(0, 1))
    for mu in range(4):
        for nu in range(4):
            assert_close(sigma[mu, nu], 0.5j * (gamma[mu] @ gamma[nu] - gamma[nu] @ gamma[mu]))


def test_default_dtype_and_override():
    assert dirac_gamma().dtype == torch.complex128
    assert dirac_gamma(dtype=torch.complex64).dtype == torch.complex64
    assert pauli_sigma().shape == (3, 2, 2)

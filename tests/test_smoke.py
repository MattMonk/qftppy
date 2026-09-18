import math

import pytest
import torch

import qftppy
from qftppy import QFTNative, clebsch, dirac_gamma, dirac_gamma5


def test_version():
    assert qftppy.__version__ == "0.1.0"


def test_gamma_anticommutator():
    """{gamma^mu, gamma^nu} = 2 g^{mu nu} * I"""
    gamma = dirac_gamma()
    g = QFTNative().metric().to(gamma.dtype)
    eye = torch.eye(4, dtype=gamma.dtype)
    for mu in range(4):
        for nu in range(4):
            anti = gamma[mu] @ gamma[nu] + gamma[nu] @ gamma[mu]
            assert torch.allclose(anti, 2 * g[mu, nu] * eye)


def test_gamma5():
    """gamma^5 = i gamma^0 gamma^1 gamma^2 gamma^3"""
    gamma = dirac_gamma()
    expected = 1j * gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]
    assert torch.allclose(dirac_gamma5(), expected)


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
def test_clebsch(args, expected):
    assert clebsch(*args) == pytest.approx(expected)

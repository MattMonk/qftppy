import pytest
import torch

from helpers import METRIC, assert_close, on_shell
from qftppy import DiracAntiSpinorNative, DiracSpinorNative

MASS = 0.938


def _pslash(p4, gamma):
    gamma_lower = torch.einsum("ij,jab->iab", METRIC.to(gamma.dtype), gamma)
    return torch.einsum("ni,iab->nab", p4.to(gamma.dtype), gamma_lower)


@pytest.fixture
def spinors(qft):
    return DiracSpinorNative(qft), DiracAntiSpinorNative(qft)


@pytest.mark.parametrize("mz", [0.5, -0.5])
def test_dirac_equation(spinors, mz):
    ds, das = spinors
    p = on_shell(MASS)
    eye = torch.eye(4, dtype=ds.dtype)
    ps = _pslash(p, ds.gamma)
    u = ds.u(p, MASS, mz)
    v = das.v(p, MASS, mz)
    assert u.shape == (p.shape[0], 4, 1)
    assert_close((ps - MASS * eye) @ u, torch.zeros_like(u))
    assert_close((ps + MASS * eye) @ v, torch.zeros_like(v))


@pytest.mark.parametrize("mz", [0.5, -0.5])
def test_normalisation(spinors, mz):
    ds, das = spinors
    p = on_shell(MASS)
    u = ds.u(p, MASS, mz)
    v = das.v(p, MASS, mz)
    ubar_u = (ds.bar(u) @ u).reshape(-1)
    vbar_v = (das.bar(v) @ v).reshape(-1)
    assert_close(ubar_u, torch.full_like(ubar_u, 2 * MASS))
    assert_close(vbar_v, torch.full_like(vbar_v, -2 * MASS))


def test_orthogonality(spinors):
    ds, das = spinors
    p = on_shell(MASS)
    u_up, u_down = ds.u(p, MASS, 0.5), ds.u(p, MASS, -0.5)
    zero = torch.zeros(p.shape[0], 1, 1, dtype=ds.dtype)
    assert_close(ds.bar(u_up) @ u_down, zero, rtol=1e-12)
    assert_close(das.bar(das.v(p, MASS, 0.5)) @ u_up, zero, rtol=1e-12)


def test_completeness(spinors):
    """sum u ubar = pslash + m = 2m P_u and sum v vbar = pslash - m = 2m P_v"""
    ds, das = spinors
    p = on_shell(MASS)
    eye = torch.eye(4, dtype=ds.dtype)
    ps = _pslash(p, ds.gamma)
    sum_u = sum(ds.u(p, MASS, mz) @ ds.bar(ds.u(p, MASS, mz)) for mz in (0.5, -0.5))
    sum_v = sum(das.v(p, MASS, mz) @ das.bar(das.v(p, MASS, mz)) for mz in (0.5, -0.5))
    assert_close(sum_u, ps + MASS * eye)
    assert_close(sum_v, ps - MASS * eye)
    assert_close(sum_u, 2 * MASS * ds.projector(p, MASS))
    assert_close(sum_v, 2 * MASS * das.projector(p, MASS))

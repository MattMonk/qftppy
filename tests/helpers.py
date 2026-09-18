"""Shared helpers for the qftppy test suite."""

import torch

SEED = 12345
N_EVENTS = 20
METRIC = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float64))


def on_shell(mass, n=N_EVENTS, seed=SEED, pmax=2.0, include_rest=True, dtype=torch.float64):
    """Random on-shell four-momenta (E, px, py, pz), shape (n, 4).

    If include_rest is True the first event has zero three-momentum, which exercises the
    beta^2 ~ 0 branches of the boost code.
    """
    gen = torch.Generator().manual_seed(seed)
    p3 = (2.0 * torch.rand((n, 3), generator=gen, dtype=torch.float64) - 1.0) * pmax
    if include_rest:
        p3[0] = 0.0
    energy = torch.sqrt(mass**2 + (p3**2).sum(dim=1, keepdim=True))
    return torch.cat([energy, p3], dim=1).to(dtype)


def momentum_pairs(n=N_EVENTS, seed=SEED, dtype=torch.float64):
    """Pairs (pa, pb) of on-shell momenta (pion and kaon masses), not in the pair rest frame."""
    pa = on_shell(0.14, n=n, seed=seed + 1, include_rest=False, dtype=dtype)
    pb = on_shell(0.49, n=n, seed=seed + 2, include_rest=False, dtype=dtype)
    return pa, pb


def lower(t, axis):
    """Lower the Lorentz index at position `axis` of `t` with the metric diag(1, -1, -1, -1)."""
    g = METRIC.to(t.dtype)
    return (t.movedim(axis, -1) @ g).movedim(-1, axis)


def contract_index(t, axis, vec):
    """Contract Lorentz index `axis` of batched `t` (N, 4, ...) with batched `vec` (N, 4) via the metric."""
    moved = t.movedim(axis, -1)
    vec_lower = lower(vec, 1).to(moved.dtype)
    return (moved * vec_lower.view(vec_lower.shape[0], *([1] * (moved.ndim - 2)), 4)).sum(-1)


def trace_pair(t, a, b):
    """Metric trace of `t` over Lorentz indices `a` and `b` (positions include the batch dimension)."""
    return lower(t, a).diagonal(dim1=a, dim2=b).sum(-1)


def assert_close(actual, expected, rtol=1e-10):
    """Scale-aware closeness: max |actual - expected| <= rtol * max |expected|.

    Using the overall scale instead of an element-wise tolerance keeps near-zero elements of
    large tensors from failing on round-off.
    """
    actual = torch.as_tensor(actual)
    expected = torch.as_tensor(expected)
    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    scale = expected.abs().max().item() if expected.numel() else 0.0
    scale = scale if scale > 0 else 1.0
    diff = (actual - expected).abs().max().item() if expected.numel() else 0.0
    assert diff <= rtol * scale, f"max abs diff {diff:.3e} > rtol {rtol:.1e} * scale {scale:.3e}"


def from_cpp(flat, rank):
    """Convert qft++ flat tensor storage (..., 4**rank) to qftppy layout (..., 4, ..., 4).

    qft++ stores element (mu, nu, rho, ...) at mu + 4*nu + 16*rho + ..., i.e. the first index
    varies fastest, so a C-order reshape gives the indices reversed.
    """
    t = torch.as_tensor(flat)
    lead = t.shape[:-1]
    t = t.reshape(*lead, *([4] * rank))
    n_lead = len(lead)
    return t.permute(*range(n_lead), *reversed(range(n_lead, n_lead + rank)))

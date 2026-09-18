import torch

from .matrices import dirac_gamma, pauli_sigma


class DiracSpinorNative:
    """
    Vectorized Dirac Spinors for spin-1/2 particles.
    Supports batching over momenta.
    """

    def __init__(self, qft):
        self.qft = qft
        self.device = qft.device
        self.dtype = torch.complex64 if qft.dtype == torch.float32 else torch.complex128
        self.gamma = dirac_gamma(self.device, self.dtype)
        self.sigma = pauli_sigma(self.device, self.dtype)

    def u(self, p4, mass, mz):
        """
        Calculates spin-1/2 particle spinor u(p, mz) for a batch.
        p4: (N, 4)
        mz: +0.5 or -0.5
        Returns: (N, 4, 1) complex spinors
        """
        N = p4.shape[0]
        E = p4[:, 0]
        norm = torch.sqrt(E + mass).unsqueeze(-1).unsqueeze(-1)
        epm = (E + mass).view(N, 1, 1)

        px, py, pz = p4[:, 1], p4[:, 2], p4[:, 3]
        sigP = self.sigma[0] * px.view(N, 1, 1) + self.sigma[1] * py.view(N, 1, 1) + self.sigma[2] * pz.view(N, 1, 1)

        chi = torch.zeros((N, 2, 1), device=self.device, dtype=self.dtype)
        if mz > 0:
            chi[:, 0, 0] = 1.0
        else:
            chi[:, 1, 0] = 1.0

        res = torch.zeros((N, 4, 1), device=self.device, dtype=self.dtype)
        res[:, 0:2, :] = chi * norm
        res[:, 2:4, :] = (sigP @ chi) * (norm / epm)
        return res

    def bar(self, spinor):
        """Dirac adjoint bar(psi) = spinor^dagger * gamma^0"""
        dag = spinor.conj().transpose(-1, -2)
        return dag @ self.gamma[0]

    def projector(self, p4, mass):
        """Spin-1/2 projector (p_slash + m) / 2m"""
        N = p4.shape[0]
        p_slash = (
            p4[:, 0].view(N, 1, 1) * self.gamma[0]
            - p4[:, 1].view(N, 1, 1) * self.gamma[1]
            - p4[:, 2].view(N, 1, 1) * self.gamma[2]
            - p4[:, 3].view(N, 1, 1) * self.gamma[3]
        )
        eye = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0)
        return (p_slash + eye * mass) / (2.0 * mass)


class DiracAntiSpinorNative(DiracSpinorNative):
    """
    Vectorized Dirac Anti-Spinors for spin-1/2 anti-particles.
    """

    def v(self, p4, mass, mz):
        """
        Calculates spin-1/2 anti-particle spinor v(p, mz) for a batch.
        p4: (N, 4)
        mz: +0.5 or -0.5
        Returns: (N, 4, 1) complex spinors
        """
        N = p4.shape[0]
        E = p4[:, 0]
        norm = torch.sqrt(E + mass).unsqueeze(-1).unsqueeze(-1)
        epm = (E + mass).view(N, 1, 1)

        px, py, pz = p4[:, 1], p4[:, 2], p4[:, 3]
        sigP = self.sigma[0] * px.view(N, 1, 1) + self.sigma[1] * py.view(N, 1, 1) + self.sigma[2] * pz.view(N, 1, 1)

        # chi for anti-particle: chi(1/2) = [0, 1]^T, chi(-1/2) = [1, 0]^T
        chi = torch.zeros((N, 2, 1), device=self.device, dtype=self.dtype)
        if mz > 0:
            chi[:, 1, 0] = 1.0
        else:
            chi[:, 0, 0] = 1.0

        res = torch.zeros((N, 4, 1), device=self.device, dtype=self.dtype)
        res[:, 0:2, :] = (sigP @ chi) * (norm / epm)
        res[:, 2:4, :] = chi * norm
        return res

    def projector(self, p4, mass):
        """Anti-particle projector (p_slash - m) / 2m"""
        N = p4.shape[0]
        p_slash = (
            p4[:, 0].view(N, 1, 1) * self.gamma[0]
            - p4[:, 1].view(N, 1, 1) * self.gamma[1]
            - p4[:, 2].view(N, 1, 1) * self.gamma[2]
            - p4[:, 3].view(N, 1, 1) * self.gamma[3]
        )
        eye = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0)
        return (p_slash - eye * mass) / (2.0 * mass)

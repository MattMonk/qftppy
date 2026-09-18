import math

import torch

from .utils import clebsch


class PolVectorNative:
    """
    Vectorized Polarization Tensors for integer spin particles.
    Supports batching over momenta.
    """

    def __init__(self, qft):
        self.qft = qft
        self.device = qft.device
        self.dtype = torch.complex64 if qft.dtype == torch.float32 else torch.complex128

    def _boost_vectors(self, pols, p4):
        """Boosts a batch of tensors by the velocity of p4"""
        N = p4.shape[0]
        bx, by, bz = p4[:, 1] / p4[:, 0], p4[:, 2] / p4[:, 0], p4[:, 3] / p4[:, 0]

        # We can use the QFTNative.orbital_tensor's logic for boosting or
        # implement a general Lorentz transformation.
        # For simplicity, we'll use the 4-vector boost logic.

        beta2 = bx * bx + by * by + bz * bz
        gamma = 1.0 / torch.sqrt(1.0 - beta2)

        # Lorentz transformation matrix Lambda^mu_nu
        Lambda = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(N, 1, 1)
        Lambda[:, 0, 0] = gamma
        Lambda[:, 0, 1] = -gamma * bx
        Lambda[:, 0, 2] = -gamma * by
        Lambda[:, 0, 3] = -gamma * bz
        Lambda[:, 1, 0] = -gamma * bx
        Lambda[:, 2, 0] = -gamma * by
        Lambda[:, 3, 0] = -gamma * bz

        # (gamma-1)/beta^2 * bi * bj
        mask = beta2 > 1e-9
        factor = (gamma[mask] - 1.0) / beta2[mask]
        Lambda[mask, 1, 1] += factor * bx[mask] * bx[mask]
        Lambda[mask, 1, 2] += factor * bx[mask] * by[mask]
        Lambda[mask, 1, 3] += factor * bx[mask] * bz[mask]
        Lambda[mask, 2, 1] += factor * by[mask] * bx[mask]
        Lambda[mask, 2, 2] += factor * by[mask] * by[mask]
        Lambda[mask, 2, 3] += factor * by[mask] * bz[mask]
        Lambda[mask, 3, 1] += factor * bz[mask] * bx[mask]
        Lambda[mask, 3, 2] += factor * bz[mask] * by[mask]
        Lambda[mask, 3, 3] += factor * bz[mask] * bz[mask]

        # Transform tensor indices
        rank = len(pols.shape) - 2  # (N, mz, mu1, ..., muS)
        res = pols
        for i in range(rank):
            # Move the i-th Lorentz index to position 2 for contraction
            perm = list(range(res.ndim))
            perm[2], perm[2 + i] = perm[2 + i], perm[2]
            res = res.permute(perm)
            # Contract Lambda with the Lorentz index at position 2
            res = torch.einsum("nij,n...j->n...i", Lambda, res)
            # Undo the permutation for the next index
            perm = list(range(res.ndim))
            perm[2], perm[2 + i] = perm[2 + i], perm[2]
            res = res.permute(perm)
        return res

    def epsilon(self, p4, mass, spin, mz):
        """
        Calculates polarization tensor for spin-S and projection mz.
        p4: (N, 4)
        Returns: (N, mu1, ..., muS)
        """
        N = p4.shape[0]
        if spin == 1:
            # Rest frame eps(mz)
            eps_rest = torch.zeros((N, 4), device=self.device, dtype=self.dtype)
            if mz == 1:
                eps_rest[:, 1] = -1.0 / math.sqrt(2.0)
                eps_rest[:, 2] = -1.0j / math.sqrt(2.0)
            elif mz == -1:
                eps_rest[:, 1] = 1.0 / math.sqrt(2.0)
                eps_rest[:, 2] = -1.0j / math.sqrt(2.0)
            elif mz == 0:
                eps_rest[:, 3] = 1.0

            if mass > 0:
                # Boost eps_rest to p4 frame
                # Note: The C++ code uses _BoostPolVectors which is a boost from rest to p4.
                # My _boost_vectors currently does rest to lab if given p4.
                # Actually, standard boost is Lab to Rest. To go Rest to Lab, use -beta.
                # Let's just implement the rank 1 boost here.
                E, px, py, pz = p4[:, 0], p4[:, 1], p4[:, 2], p4[:, 3]
                beta = torch.stack([px, py, pz], dim=1) / E.unsqueeze(-1)
                beta2 = torch.sum(beta**2, dim=1)
                gamma = 1.0 / torch.sqrt(1.0 - beta2)

                # bp = beta . eps_rest_spatial
                bp = beta[:, 0] * eps_rest[:, 1] + beta[:, 1] * eps_rest[:, 2] + beta[:, 2] * eps_rest[:, 3]

                # Start from the rest-frame vector so events at rest (beta^2 ~ 0, boost = identity)
                # keep their polarization; moving events are overwritten below.
                res = eps_rest.clone()
                res[:, 0] = gamma * bp
                mask = beta2 > 1e-9
                fac = (gamma[mask] - 1.0) / beta2[mask]
                res[mask, 1] = eps_rest[mask, 1] + fac * beta[mask, 0] * bp[mask]
                res[mask, 2] = eps_rest[mask, 2] + fac * beta[mask, 1] * bp[mask]
                res[mask, 3] = eps_rest[mask, 3] + fac * beta[mask, 2] * bp[mask]
                return res
            else:
                # Photon case (simplification)
                return eps_rest  # Needs proper helicity rotation for photon

        elif spin > 1:
            # Recursion: eps_S(M) = sum Clebsch * eps_1(m1) % eps_{S-1}(mJ)
            res = None
            J = spin - 1
            for m1 in [-1, 0, 1]:
                for mJ in [x - J for x in range(int(2 * J + 1))]:
                    cg = clebsch(1, m1, J, mJ, spin, mz)
                    if abs(cg) < 1e-9:
                        continue

                    e1 = self.epsilon(p4, mass, 1, m1)
                    eJ = self.epsilon(p4, mass, J, mJ)

                    term = cg * self.qft.outer_product(e1, eJ)
                    if res is None:
                        res = term
                    else:
                        res += term
            return res

    def projector(self, p4, mass, spin):
        """Calculates spin-S projection operator"""
        if spin == 1:
            # P_mu_nu = -g_mu_nu + P_mu*P_nu/m^2
            PP = torch.einsum("ni,nj->nij", p4, p4)
            if mass > 0:
                return -self.qft.g.unsqueeze(0) + PP / (mass * mass)
            else:
                return -self.qft.g.unsqueeze(0)

        # sum epsilon(M) % epsilon*(M)
        res = None
        for i in range(int(2 * spin + 1)):
            mz = i - spin
            eps = self.epsilon(p4, mass, spin, mz)
            term = self.qft.outer_product(eps, eps.conj())
            if res is None:
                res = term
            else:
                res += term
        return res

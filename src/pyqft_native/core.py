import torch
import itertools


class QFTNative:
    """
    Core QFT++ logic implemented in PyTorch.
    Handles metric tensors, Levi-Civita tensors, and basic tensor contractions.
    """

    def __init__(self, device='cpu', dtype=None):
        self.device = device
        if dtype is None:
            self.dtype = torch.float32 if device == 'mps' else torch.float64
        else:
            self.dtype = dtype
        # Minkowski Metric: diag(1, -1, -1, -1)
        self.g = torch.tensor([1, -1, -1, -1], device=device,
                              dtype=self.dtype).diag()

    def metric(self):
        """Returns the Minkowski metric tensor g_{mu,nu}"""
        return self.g.clone()

    def levi_civita(self):
        """Returns the 4D Levi-Civita tensor epsilon_{mu,nu,rho,sigma}"""
        epsilon = torch.zeros((4, 4, 4, 4),
                              device=self.device,
                              dtype=self.dtype)
        for p in itertools.permutations(range(4)):
            inv = 0
            for i in range(4):
                for j in range(i + 1, 4):
                    if p[i] > p[j]:
                        inv += 1
            epsilon[p] = 1.0 if inv % 2 == 0 else -1.0
        return epsilon

    def dot(self, p1, p2):
        """
        Minkowski dot product for batches of 4-vectors.
        p1, p2: (N, 4)
        Returns: (N,)
        """
        return p1[:, 0] * p2[:, 0] - p1[:, 1] * p2[:, 1] - p1[:, 2] * p2[:, 2] - p1[:, 3] * p2[:, 3]

    def outer_product(self, t1, t2):
        """
        Tensor outer product for batches.
        t1: (N, indices1...), t2: (N, indices2...)
        Result: (N, indices1..., indices2...)
        """
        rank1 = len(t1.shape) - 1
        rank2 = len(t2.shape) - 1
        s1 = "".join(chr(ord('a') + i) for i in range(rank1))
        s2 = "".join(chr(ord('a') + rank1 + i) for i in range(rank2))
        return torch.einsum(f'n{s1},n{s2}->n{s1}{s2}', t1, t2)

    def contract(self, t1, t2, n=1):
        """
        Contracts the last n indices of t1 with the first n indices of t2 using Minkowski metric.
        Handles both batched (N, ...) and static (...) tensors.
        """
        # Determine if t1 or t2 are batched
        # If shape[0] is not 4, it's almost certainly a batch dimension N
        is_batched1 = (len(t1.shape) > 0 and t1.shape[0] != 4)
        is_batched2 = (len(t2.shape) > 0 and t2.shape[0] != 4)

        r1 = len(t1.shape) - (1 if is_batched1 else 0)
        r2 = len(t2.shape) - (1 if is_batched2 else 0)

        if n == -1:
            n = min(r1, r2)
        if n == 0:
            return self.outer_product(t1, t2)

        # We'll use a general approach:
        # t1: (N?, left..., i1, ..., in)
        # g: i1j1, i2j2, ..., injn
        # t2: (N?, j1, ..., jn, right...)

        # Build einsum string
        s1_batch = "n" if is_batched1 else ""
        s2_batch = "n" if is_batched2 else ""
        res_batch = "n" if (is_batched1 or is_batched2) else ""

        s_left = "".join(chr(ord('a') + i) for i in range(r1 - n))
        s_i = "".join(chr(ord('A') + i) for i in range(n))
        s_j = "".join(chr(ord('K') + i) for i in range(n))
        s_right = "".join(chr(ord('a') + r1 - n + i) for i in range(r2 - n))

        # metric pairs: i1j1, i2j2, ...
        s_metrics = ",".join(f"{s_i[k]}{s_j[k]}" for k in range(n))

        ein_str = f"{s1_batch}{s_left}{s_i},{s_metrics},{s2_batch}{s_j}{s_right}->{res_batch}{s_left}{s_right}"

        operands = [t1]
        for _ in range(n):
            operands.append(self.g)
        operands.append(t2)

        return torch.einsum(ein_str, *operands)

    def symmetrize(self, T):
        """
        Symmetrizes a batch of tensors T over all indices (except batch dim).
        """
        rank = len(T.shape) - 1
        if rank < 2:
            return T
        indices = list(range(1, rank + 1))
        perms = list(itertools.permutations(indices))
        res = torch.zeros_like(T)
        for p in perms:
            res += T.permute(0, *p)
        return res / len(perms)

    def orbital_tensor(self, pa, pb, rank):
        """
        Calculates Orbital Tensors L(l)_{mu1...mul} for a batch of events.
        Delegates to orbital module.
        """
        from .orbital import compute_orbital_tensor
        return compute_orbital_tensor(self, pa, pb, rank)

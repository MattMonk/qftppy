import torch


def compute_orbital_tensor(qft, pa, pb, rank):
    """
    Calculates Orbital Tensors L(l)_{mu1...mul} for a batch of events.
    pa, pb: Tensors of shape (N, 4)
    qft: Instance of QFTNative with .dot(), .outer_product(), .symmetrize()
    """
    N = pa.shape[0]
    device = qft.device
    dtype = qft.dtype

    if rank == 0:
        return torch.ones((N,), device=device, dtype=dtype)

    P = pa + pb
    p_ab = 0.5 * (pa - pb)
    P2 = qft.dot(P, P).unsqueeze(-1)

    # kperp^mu = p_ab^mu - P^mu (P.p_ab) / P^2 -- direct Gram-Schmidt, stays contravariant
    P_dot_pab = qft.dot(P, p_ab)
    kperp = p_ab - P * (P_dot_pab / P2.squeeze(-1)).unsqueeze(-1)
    kperp_sq = qft.dot(kperp, kperp).unsqueeze(-1).unsqueeze(-1)

    # gbar^{mu,nu} = g^{mu,nu} - P^mu P^nu / P^2 -- contravariant projector for rank>=2 trace terms
    PP = torch.einsum("ni,nj->nij", P, P)  # contravariant P, no lowering
    gbar = qft.g.unsqueeze(0) - PP / P2.unsqueeze(-1)

    if rank == 1:
        return kperp

    if rank == 2:
        # 1.5 * (kperp_mu * kperp_nu - 1/3 * kperp^2 * gbar_{mu,nu})
        kk = torch.einsum("ni,nj->nij", kperp, kperp)
        return 1.5 * (kk - (1.0 / 3.0) * kperp_sq * gbar)

    if rank == 3:
        # 2.5 * (kkk - 1/5 * kperp^2 * 3 * Sym(gbar % kperp))
        kkk = torch.einsum("ni,nj,nk->nijk", kperp, kperp, kperp)
        gk = qft.outer_product(gbar, kperp)
        term2 = (1.0 / 5.0) * kperp_sq.unsqueeze(-1) * (3.0 * qft.symmetrize(gk))
        return 2.5 * (kkk - term2)

    if rank == 4:
        # (35/8) * (kkkk - 1/7 * kperp^2 * 6 * Sym(gbar % kk) + 1/35 * (kperp^2)^2 * 3 * Sym(gbar % gbar))
        kkkk = torch.einsum("ni,nj,nk,nl->nijkl", kperp, kperp, kperp, kperp)
        kk = torch.einsum("ni,nj->nij", kperp, kperp)
        gkk = qft.outer_product(gbar, kk)
        gg = qft.outer_product(gbar, gbar)

        term2 = (1.0 / 7.0) * kperp_sq.unsqueeze(-1).unsqueeze(-1) * (6.0 * qft.symmetrize(gkk))
        term3 = (1.0 / 35.0) * (kperp_sq * kperp_sq).unsqueeze(-1).unsqueeze(-1) * (3.0 * qft.symmetrize(gg))
        return (35.0 / 8.0) * (kkkk - term2 + term3)

    # Recursion for rank > 4, following qft++ OrbitalTensor::SetP4:
    #   z = (2l-1)/l^2 [ sum_i Perm(x % gbar; i, l-1) - 2/(2l-1) sum_{j<i} Perm(Perm(gbar % x; 0, i); 1, j) ]
    #   x = z * kperp
    # where l is the rank being built and "*" contracts through the metric.
    x = compute_orbital_tensor(qft, pa, pb, 4)
    kperp_lower = kperp @ qft.g

    def swap_indices(T, mu, nu):
        dims = list(range(len(T.shape)))
        # Batch dim is 0. Indices start at 1.
        dims[mu + 1], dims[nu + 1] = dims[nu + 1], dims[mu + 1]
        return T.permute(*dims)

    for current_rank in range(5, rank + 1):
        xg = qft.outer_product(x, gbar)  # rank current_rank + 1
        gx = qft.outer_product(gbar, x)

        z = swap_indices(xg, 0, current_rank - 1)
        for i in range(1, current_rank):
            z = z + swap_indices(xg, i, current_rank - 1)
            for j in range(i):
                z = z + (-2.0 / (2.0 * current_rank - 1.0)) * swap_indices(swap_indices(gx, 0, i), 1, j)
        z = z * (2.0 * current_rank - 1.0) / (current_rank * current_rank)

        # x = z * kperp: contract the last index of z with kperp through the metric
        s_z = "".join(chr(ord("a") + i) for i in range(current_rank + 1))
        x = torch.einsum(f"n{s_z},n{s_z[-1]}->n{s_z[:-1]}", z, kperp_lower)

    return x

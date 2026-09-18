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
        return torch.ones((N, ), device=device, dtype=dtype)

    P = pa + pb
    p_ab = 0.5 * (pa - pb)
    P2 = qft.dot(P, P).unsqueeze(-1)

    # kperp^μ = p_ab^μ - P^μ (P·p_ab) / P²  — direct Gram-Schmidt, stays contravariant
    P_dot_pab = qft.dot(P, p_ab)
    kperp = p_ab - P * (P_dot_pab / P2.squeeze(-1)).unsqueeze(-1)
    kperp_sq = qft.dot(kperp, kperp).unsqueeze(-1).unsqueeze(-1)

    # g̃^{μν} = g^{μν} - P^μ P^ν / P²  — contravariant projector for rank≥2 trace terms
    PP = torch.einsum('ni,nj->nij', P, P)  # contravariant P, no lowering
    gbar = qft.g.unsqueeze(0) - PP / P2.unsqueeze(-1)

    if rank == 1:
        return kperp

    if rank == 2:
        # 1.5 * (kperp_mu * kperp_nu - 1/3 * kperp^2 * gbar_{mu,nu})
        kk = torch.einsum('ni,nj->nij', kperp, kperp)
        return 1.5 * (kk - (1.0 / 3.0) * kperp_sq * gbar)

    if rank == 3:
        # 2.5 * (kkk - 1/5 * kperp^2 * 3 * Sym(gbar % kperp))
        kkk = torch.einsum('ni,nj,nk->nijk', kperp, kperp, kperp)
        gk = qft.outer_product(gbar, kperp)
        term2 = (1.0 / 5.0) * kperp_sq.unsqueeze(-1) * (3.0 *
                                                        qft.symmetrize(gk))
        return 2.5 * (kkk - term2)

    if rank == 4:
        # (35/8) * (kkkk - 1/7 * kperp^2 * 6 * Sym(gbar % kk) + 1/35 * (kperp^2)^2 * 3 * Sym(gbar % gbar))
        kkkk = torch.einsum('ni,nj,nk,nl->nijkl', kperp, kperp, kperp, kperp)
        kk = torch.einsum('ni,nj->nij', kperp, kperp)
        gkk = qft.outer_product(gbar, kk)
        gg = qft.outer_product(gbar, gbar)

        term2 = (1.0 / 7.0) * kperp_sq.unsqueeze(-1).unsqueeze(-1) * (
            6.0 * qft.symmetrize(gkk))
        term3 = (1.0 / 35.0) * (kperp_sq * kperp_sq).unsqueeze(-1).unsqueeze(
            -1) * (3.0 * qft.symmetrize(gg))
        return (35.0 / 8.0) * (kkkk - term2 + term3)

    # Generalized recursion for rank > 4
    current_rank = 5
    # Recursively call self (this function) for rank 4
    x = compute_orbital_tensor(qft, pa, pb, 4)

    def swap_indices(T, mu, nu):
        dims = list(range(len(T.shape)))
        # Batch dim is 0. Indices start at 1.
        dims[mu + 1], dims[nu + 1] = dims[nu + 1], dims[mu + 1]
        return T.permute(*dims)

    while current_rank <= rank:
        # x is rank L_prev
        L_prev = current_rank - 1

        # xg = x % gbar
        xg = qft.outer_product(x, gbar)  # Rank L_prev + 2

        # z calculation
        z = swap_indices(xg, 0, L_prev - 1)
        for i in range(1, L_prev):
            z += swap_indices(xg, i, L_prev - 1)
            for j in range(i):
                gx = qft.outer_product(gbar, x)
                # ((gbar%x).Permute(0,i)).Permute(1,j)
                term = swap_indices(swap_indices(gx, 0, i), 1, j)
                z += (-2.0 / (2.0 * L_prev - 1.0)) * term

        z *= (2.0 * L_prev - 1.0) / (L_prev * L_prev)

        # x = z * kperp
        # z: (N, indices..., rho), kperp: (N, rho)
        s_z = "".join(chr(ord('a') + i) for i in range(current_rank + 1))
        s_k = s_z[-1]
        s_res = s_z[:-1]
        x = torch.einsum(f'n{s_z},n{s_k}->n{s_res}', z, kperp)

        current_rank += 1

    return x

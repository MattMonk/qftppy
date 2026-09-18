import math

import torch


def clebsch(j1, m1, j2, m2, J, M):
    """
    Calculates Clebsch-Gordan coefficient (j1, m1; j2, m2 | J, M).
    Ported from QFT++ / Weygand implementation.
    """
    if abs(m1 + m2 - M) > 1e-5:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(M) > J:
        return 0.0

    def dfact(x):
        if x < 0.00001 and x >= 0.0:
            return 1.0
        if x < 0:
            return 0.0
        return x * dfact(x - 1.0)

    # Convert to 2*spin integers
    ij1, im1, ij2, im2, iJ, iM = [int(2 * x) for x in [j1, m1, j2, m2, J, M]]

    sum_val = 0.0
    nu = 0
    while True:
        d3 = (ij1 - ij2 - iM) // 2 + nu
        n2 = (ij1 - im1) // 2 + nu
        if d3 >= 0 and n2 >= 0:
            break
        nu += 1

    while True:
        d1 = (iJ - ij1 + ij2) // 2 - nu
        d2 = (iJ + iM) // 2 - nu
        n1 = (ij2 + iJ + im1) // 2 - nu
        d3 = (ij1 - ij2 - iM) // 2 + nu
        n2 = (ij1 - im1) // 2 + nu

        if d1 < 0 or d2 < 0 or n1 < 0:
            break

        d0 = dfact(nu)
        exp = nu + (ij2 + im2) // 2
        n0 = (-1.0) ** exp

        term = (n0 * dfact(n1) * dfact(n2)) / (d0 * dfact(d1) * dfact(d2) * dfact(d3))
        sum_val += term
        nu += 1

    if sum_val == 0:
        return 0.0

    num = (
        (iJ + 1)
        * dfact((iJ + ij1 - ij2) / 2)
        * dfact((iJ - ij1 + ij2) / 2)
        * dfact((ij1 + ij2 - iJ) / 2)
        * dfact((iJ + iM) / 2)
        * dfact((iJ - iM) / 2)
    )
    den = (
        dfact((ij1 + ij2 + iJ) / 2 + 1)
        * dfact((ij1 - im1) / 2)
        * dfact((ij1 + im1) / 2)
        * dfact((ij2 - im2) / 2)
        * dfact((ij2 + im2) / 2)
    )

    return math.sqrt(num / den) * sum_val


def wigner_d(j, m, n, beta):
    """
    Vectorized Wigner d-function d^j_{m,n}(beta).
    beta: PyTorch tensor of angles (N,)
    """
    J, M, N = [int(2 * x) for x in [j, m, n]]

    if J < 0 or abs(M) > J or abs(N) > J:
        return torch.zeros_like(beta)

    m_p_n = (M + N) // 2
    j_p_m = (J + M) // 2
    j_m_m = (J - M) // 2
    j_p_n = (J + N) // 2
    j_m_n = (J - N) // 2

    kk = math.factorial(j_p_m) * math.factorial(j_m_m) * math.factorial(j_p_n) * math.factorial(j_m_n)
    const_term = ((-1.0) ** j_p_m) * math.sqrt(kk)

    k_low = max(0, m_p_n)
    k_hi = min(j_p_m, j_p_n)

    sum_term = torch.zeros_like(beta)
    for k in range(k_low, k_hi + 1):
        kmn1 = 2 * k - (M + N) // 2
        jmnk = J + (M + N) // 2 - 2 * k
        jmk = (J + M) // 2 - k
        jnk = (J + N) // 2 - k
        kmn2 = k - (M + N) // 2

        facs = math.factorial(k) * math.factorial(jmk) * math.factorial(jnk) * math.factorial(kmn2)

        sum_term += ((-1.0) ** k) * (torch.cos(beta / 2.0) ** kmn1 * torch.sin(beta / 2.0) ** jmnk) / facs

    return const_term * sum_term


def regge_propagator(t, s, a, b, spin, sig, exp_fact=1):
    """
    Vectorized Regge trajectory propagator.
    t, s: PyTorch tensors (N,)
    """
    alpha = a * t + b

    numerator = (s ** (alpha - float(spin))) * math.pi * a
    comp_dtype = torch.complex64 if t.dtype == torch.float32 else torch.complex128
    numerator = numerator.to(comp_dtype)
    numerator *= float(sig) + float(exp_fact) * torch.exp(-1j * math.pi * alpha)

    gamma_arg = alpha + 1.0 - float(spin)
    # Denominator 2 sin(pi z) Gamma(z), evaluated as in qft++: for z < 0 use the reflection
    # formula 2 pi / (|z| Gamma(|z|)) (gammaln only gives |Gamma|, which would lose the sign),
    # and its limit 2 pi at z = 0.
    abs_arg = gamma_arg.abs()
    safe_abs = torch.where(abs_arg > 0, abs_arg, torch.ones_like(abs_arg))
    safe_pos = torch.where(gamma_arg > 0, gamma_arg, torch.ones_like(gamma_arg))
    positive = 2.0 * torch.sin(math.pi * gamma_arg) * torch.exp(torch.special.gammaln(safe_pos))
    negative = 2.0 * math.pi / (safe_abs * torch.exp(torch.special.gammaln(safe_abs)))
    denominator = torch.where(
        gamma_arg > 0, positive, torch.where(gamma_arg < 0, negative, torch.full_like(gamma_arg, 2.0 * math.pi))
    )
    return numerator / denominator

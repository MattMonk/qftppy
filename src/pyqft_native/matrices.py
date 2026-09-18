import torch


def pauli_sigma(device='cpu', dtype=None):
    """
    Returns the Pauli sigma matrices sigma^1, sigma^2, sigma^3.
    Shape: (3, 2, 2)
    """
    if dtype is None:
        dtype = torch.complex64 if device == 'mps' else torch.complex128
    sigma = torch.zeros((3, 2, 2), device=device, dtype=dtype)
    # sigma^1
    sigma[0, 0, 1] = 1.0
    sigma[0, 1, 0] = 1.0
    # sigma^2
    sigma[1, 0, 1] = -1.0j
    sigma[1, 1, 0] = 1.0j
    # sigma^3
    sigma[2, 0, 0] = 1.0
    sigma[2, 1, 1] = -1.0
    return sigma


def dirac_gamma(device='cpu', dtype=None):
    """
    Returns the Dirac gamma matrices gamma^0, gamma^1, gamma^2, gamma^3.
    Shape: (4, 4, 4)
    Representation: Standard Dirac representation (same as qft++).
    """
    if dtype is None:
        dtype = torch.complex64 if device == 'mps' else torch.complex128
    gamma = torch.zeros((4, 4, 4), device=device, dtype=dtype)
    sigma = pauli_sigma(device, dtype)
    eye = torch.eye(2, device=device, dtype=dtype)

    # gamma^0 = [[I, 0], [0, -I]]
    gamma[0, 0:2, 0:2] = eye
    gamma[0, 2:4, 2:4] = -eye

    # gamma^i = [[0, sigma^i], [-sigma^i, 0]]
    for i in range(3):
        gamma[i + 1, 0:2, 2:4] = sigma[i]
        gamma[i + 1, 2:4, 0:2] = -sigma[i]

    return gamma


def dirac_gamma5(device='cpu', dtype=None):
    """
    Returns the gamma^5 matrix.
    gamma^5 = i * gamma^0 * gamma^1 * gamma^2 * gamma^3
    Shape: (4, 4)
    """
    if dtype is None:
        dtype = torch.complex64 if device == 'mps' else torch.complex128
    g5 = torch.zeros((4, 4), device=device, dtype=dtype)
    # Representation used in qft++: [[0, I], [I, 0]]
    eye = torch.eye(2, device=device, dtype=dtype)
    g5[0:2, 2:4] = eye
    g5[2:4, 0:2] = eye
    return g5


def dirac_sigma(device='cpu', dtype=None):
    """
    Returns the Dirac sigma matrices sigma^{mu,nu} = i/2 * [gamma^mu, gamma^nu].
    Shape: (4, 4, 4, 4)
    """
    if dtype is None:
        dtype = torch.complex64 if device == 'mps' else torch.complex128
    gamma = dirac_gamma(device, dtype)
    sigma_munu = torch.zeros((4, 4, 4, 4), device=device, dtype=dtype)
    for mu in range(4):
        for nu in range(4):
            # i/2 * (g_mu * g_nu - g_nu * g_mu)
            sigma_munu[mu, nu] = 0.5j * (gamma[mu] @ gamma[nu] -
                                         gamma[nu] @ gamma[mu])
    return sigma_munu

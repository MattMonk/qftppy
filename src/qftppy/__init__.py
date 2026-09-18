from importlib.metadata import version

from .core import QFTNative
from .matrices import dirac_gamma, dirac_gamma5, dirac_sigma, pauli_sigma
from .orbital import compute_orbital_tensor
from .polarization import PolVectorNative
from .spin import Spin
from .spinors import DiracAntiSpinorNative, DiracSpinorNative
from .utils import clebsch, regge_propagator, wigner_d

__version__ = version("qftppy")

__all__ = [
    "QFTNative",
    "compute_orbital_tensor",
    "Spin",
    "pauli_sigma",
    "dirac_gamma",
    "dirac_gamma5",
    "dirac_sigma",
    "DiracSpinorNative",
    "DiracAntiSpinorNative",
    "clebsch",
    "wigner_d",
    "regge_propagator",
    "PolVectorNative",
]

import pytest
import torch

from qftppy import QFTNative


@pytest.fixture
def qft():
    return QFTNative(dtype=torch.float64)

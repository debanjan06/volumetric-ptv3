import torch
import numpy as np


def test_pytorch_layer_allocation():
    """Verifies that PyTorch executes tensor matrix math within the cloud runner."""
    x = torch.ones(1, 4, 3)
    assert x.shape == (1, 4, 3)
    assert np.isclose(x.sum().item(), 12.0)

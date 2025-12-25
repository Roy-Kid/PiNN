"""
Smoke tests for the PyTorch translation of PiNet building blocks.

These tests only verify that the core blocks run and produce tensors with the
expected shapes. They do not check numerical equivalence with TensorFlow.
"""

import torch

from pinn.networks.pinet_torch import IPLayerTorch, PILayerTorch


def test_pilayer_and_iplayer_shapes_smoke() -> None:
    """PILayerTorch should map (pairs, basis) -> (pairs, d) and IPLayerTorch should map -> (atoms, d)."""
    torch.manual_seed(0)

    n_atoms, n_pairs = 5, 12
    n_prop, n_basis, d = 8, 4, 16

    prop = torch.randn(n_atoms, n_prop)
    ind_2 = torch.randint(0, n_atoms, (n_pairs, 2), dtype=torch.long)
    basis = torch.randn(n_pairs, n_basis)

    pi = PILayerTorch([32, d], n_basis=n_basis, activation="tanh")
    inter = pi(ind_2, prop, basis)
    assert inter.shape == (n_pairs, d)

    ip = IPLayerTorch()
    prop_new = ip(ind_2, prop, inter)
    assert prop_new.shape == (n_atoms, d)

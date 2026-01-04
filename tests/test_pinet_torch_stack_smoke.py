"""
Smoke tests for the PyTorch PiNet depth stack (core loop only).
"""

import torch
from pinn.networks.pinet_torch import PiNetTorchCore


def test_pinet_torch_core_stack_shapes_smoke() -> None:
    """PiNetTorchCore should run a depth loop and return per-atom outputs of shape (n_atoms, out_units)."""
    torch.manual_seed(0)

    n_atoms, n_pairs = 5, 12
    n_prop, n_basis = 8, 4
    out_units = 3

    ind_1 = torch.arange(n_atoms, dtype=torch.long)
    ind_2 = torch.randint(0, n_atoms, (n_pairs, 2), dtype=torch.long)
    prop = torch.randn(n_atoms, n_prop)
    basis = torch.randn(n_pairs, n_basis)

    net = PiNetTorchCore(
        depth=2,
        pp_nodes=[16],
        pi_nodes=[32, 16],
        ii_nodes=[16],
        out_nodes=[16],
        out_units=out_units,
        n_basis=n_basis,
        activation="tanh",
    )
    out = net(ind_1=ind_1, ind_2=ind_2, prop=prop, basis=basis)
    assert out.shape == (n_atoms, out_units)

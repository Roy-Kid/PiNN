# -*- coding: utf-8 -*-
import torch

from pinn import get_model


def _make_sparse_tensors(B: int = 2, N: int = 3) -> dict:
    """Create a minimal PiNN-style sparse batch with B structures and N atoms each."""
    torch.manual_seed(0)
    coord = torch.randn(B * N, 3, dtype=torch.float32)
    elems = torch.ones(B * N, dtype=torch.long)  # hydrogen
    ind_1 = torch.zeros((B * N, 1), dtype=torch.long)
    ind_1[:, 0] = torch.arange(B).repeat_interleave(N)
    return {"coord": coord, "elems": elems, "ind_1": ind_1}


def test_torch_potential_energy_shape_is_per_structure() -> None:
    """Model(tensors) must return one energy per structure: shape (B,)."""
    params = {
        "model_dir": "/tmp/pinn_contract_shape",
        "network": {
            "name": "PiNet",
            "params": {
                "atom_types": [1],
                "rc": 5.0,
                "n_basis": 5,
                "pp_nodes": [8, 8],
                "pi_nodes": [8, 8],
                "ii_nodes": [8, 8],
                "out_nodes": [8, 8],
                "depth": 2,
            },
        },
        "model": {"name": "potential_model", "params": {"e_dress": {1: 0.5}, "e_scale": 5.0, "e_unit": 2.0}},
        "backend": "torch",
    }

    model = get_model(params)
    tensors = _make_sparse_tensors(B=2, N=3)
    E = model(tensors)

    assert isinstance(E, torch.Tensor)
    assert E.shape == (2,), f"Expected per-structure energies (B,), got {tuple(E.shape)}"

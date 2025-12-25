# pinn/torch/model.py
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from pinn.networks.pinet_torch import PiNetTorch


class PiNetPotentialTorch(nn.Module):
    def __init__(self, net, *, e_dress, e_scale, e_unit):
        super().__init__()
        self.net = net
        self.e_dress = {int(k): float(v) for k, v in e_dress.items()}
        self.e_scale = float(e_scale)
        self.e_unit = float(e_unit)

    def forward(self, tensors: dict) -> torch.Tensor:
        """
        Returns:
            energy per structure, shape (n_struct,)
            in ASE energy units
        """
        # 1. atomic energy contributions
        # shape: (n_atoms, 1)
        e_atom = self.net(tensors)

        if e_atom.ndim == 2 and e_atom.shape[1] == 1:
            e_atom = e_atom[:, 0]

        # 2. pool atoms → structures
        ind_1 = tensors["ind_1"][:, 0]
        n_struct = int(ind_1.max()) + 1 if ind_1.numel() else 0

        e_struct = torch.zeros(
            n_struct,
            device=e_atom.device,
            dtype=e_atom.dtype,
        )
        e_struct.index_add_(0, ind_1, e_atom)

        # 3. energy dressing (tensor form, no Python loops over atoms)
        if self.e_dress:
            elems = tensors["elems"]
            dress_atom = torch.zeros_like(e_atom)
            for z, val in self.e_dress.items():
                dress_atom = dress_atom + (elems == z).to(e_atom.dtype) * val
            e_struct = e_struct + torch.zeros_like(e_struct).index_add_(0, ind_1, dress_atom)

        # 4. scaling + units
        e_struct = e_struct * self.e_unit

        return e_struct


def _materialize_lazy(model: torch.nn.Module, *, atom_types) -> None:
    """Run a tiny dummy forward to materialize LazyLinear parameters.

    The calc-reload smoke test saves model.state_dict() immediately after
    pinn.get_model(params), so Lazy modules must be initialized here.
    """
    # Put dummy tensors on the same device as the model (cpu in tests, but be safe).
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    z0 = int(atom_types[0]) if len(atom_types) else 1

    tensors = {
        "coord": torch.zeros((2, 3), dtype=torch.float32, device=device),              # (N,3)
        "elems": torch.tensor([z0, z0], dtype=torch.long, device=device),              # (N,)
        "ind_1": torch.zeros((2, 1), dtype=torch.long, device=device),                 # (N,1)
    }

    model.eval()
    with torch.no_grad():
        _ = model(tensors)

def get_model(params: dict, **kwargs) -> nn.Module:
    """Torch backend model factory.

    Expected schema (mirrors TF tests):
      params["network"]["params"]: PiNet hyperparameters
      params["model"]["params"]:   e_dress, e_scale, e_unit, use_force, etc.
    """
    net_params = params["network"]["params"]
    mparams = params["model"]["params"]

    # Require the keys that define the architecture in tests/YAML.
    required = ["atom_types", "rc"]
    missing = [k for k in required if k not in net_params]
    if missing:
        raise KeyError(f"Missing network params for torch PiNet: {missing}")

    # Optional / nice-to-have defaults are OK.
    act = net_params.get("act", "tanh")

    net = PiNetTorch(
            atom_types=net_params["atom_types"],
            rc=float(net_params["rc"]),
            n_basis=int(net_params.get("n_basis", 5)),
            depth=int(net_params.get("depth", 3)),
            pp_nodes=net_params.get("pp_nodes", [8, 8]),
            pi_nodes=net_params.get("pi_nodes", [8, 8]),
            ii_nodes=net_params.get("ii_nodes", [8, 8]),
            out_nodes=net_params.get("out_nodes", [8, 8]),
            act=net_params.get("act", "tanh"),
            out_units=1,
            out_pool=False,
    )


    model = PiNetPotentialTorch(
        net,
        e_dress=mparams.get("e_dress", {}),
        e_scale=float(mparams.get("e_scale", 1.0)),
        e_unit=float(mparams.get("e_unit", 1.0)),
    )

    _materialize_lazy(model, atom_types=net_params["atom_types"])

    return model

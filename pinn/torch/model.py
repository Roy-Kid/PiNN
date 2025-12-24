# pinn/torch/model.py
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from pinn.networks.pinet_torch import PiNetTorch


class PiNetPotentialTorch(nn.Module):
    """Torch potential wrapper around PiNetTorch.

    PiNetTorch produces per-atom contributions. This wrapper:
      - pools per-atom contributions into per-structure total energies
      - applies optional per-atom energy dressing
      - applies e_scale and e_unit (to match PiNN conventions in tests)
    """

    def __init__(
        self,
        network: PiNetTorch,
        *,
        e_dress: Optional[Dict[int, float]] = None,
        e_scale: float = 1.0,
        e_unit: float = 1.0,
    ) -> None:
        super().__init__()
        self.network = network
        self.e_dress = {int(k): float(v) for k, v in (e_dress or {}).items()}
        self.e_scale = float(e_scale)
        self.e_unit = float(e_unit)

    def forward(self, tensors: dict) -> torch.Tensor:
        """Compute per-structure energies.

        Args:
            tensors: Dict with at least:
              - ind_1: (n_atoms, 1) or (n_atoms, 2) long
              - elems: (n_atoms,) long
              - coord: (n_atoms, 3) float
            Optional:
              - cell: (3,3) or (n_struct,3,3) float

        Returns:
            E: (n_struct,) energies in (e_unit) units, including e_scale and dressing.
        """
        per_atom = self.network(tensors)  # (n_atoms,) when out_units==1 and no pooling

        ind_1 = tensors["ind_1"].long()
        batch = ind_1[:, 0] if ind_1.ndim == 2 else ind_1
        n_struct = int(batch.max().item()) + 1 if batch.numel() else 0

        E = torch.zeros((n_struct,), dtype=per_atom.dtype, device=per_atom.device)
        E.index_add_(0, batch, per_atom)

        if self.e_dress:
            elems = tensors["elems"].long()
            dress_pa = torch.zeros_like(per_atom)
            for Z, val in self.e_dress.items():
                dress_pa = dress_pa + (elems == Z).to(per_atom.dtype) * val
            dress_E = torch.zeros((n_struct,), dtype=per_atom.dtype, device=per_atom.device)
            dress_E.index_add_(0, batch, dress_pa)
            E = E + dress_E

        return E * self.e_scale * self.e_unit


def get_model(params: dict, **kwargs) -> nn.Module:
    """Torch backend model factory.

    Expected schema (mirrors TF tests):
      params["network"]["params"]: PiNet hyperparameters
      params["model"]["params"]:   e_dress, e_scale, e_unit, use_force, etc.
    """
    net_params = params["network"]["params"]
    mparams = params["model"]["params"]

    # Require the keys that define the architecture in tests/YAML.
    required = ["atom_types", "rc", "n_basis", "pp_nodes", "pi_nodes", "ii_nodes", "out_nodes", "depth"]
    missing = [k for k in required if k not in net_params]
    if missing:
        raise KeyError(f"Missing network params for torch PiNet: {missing}")

    # Optional / nice-to-have defaults are OK.
    act = net_params.get("act", "tanh")

    net = PiNetTorch(
        atom_types=net_params["atom_types"],
        rc=float(net_params["rc"]),
        n_basis=int(net_params["n_basis"]),
        pp_nodes=net_params["pp_nodes"],
        pi_nodes=net_params["pi_nodes"],
        ii_nodes=net_params["ii_nodes"],
        out_nodes=net_params["out_nodes"],
        depth=int(net_params["depth"]),
        out_units=1,       # test_pinet_potential assumes scalar energy
        out_pool=False,    # potential wrapper will pool to per-structure energy
        act=act,
    )

    return PiNetPotentialTorch(
        net,
        e_dress=mparams.get("e_dress", {}),
        e_scale=float(mparams.get("e_scale", 1.0)),
        e_unit=float(mparams.get("e_unit", 1.0)),
    )

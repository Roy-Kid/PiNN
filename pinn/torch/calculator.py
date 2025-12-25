# pinn/torch/calculator.py
from __future__ import annotations

import os
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from .model import get_model as build_model


class TorchPiNNCalc(Calculator):
    """ASE calculator for the Torch backend.

    Computes:
      - energy: from model(tensors)
      - forces: -dE/dR via autograd
      - stress: virial-based, ASE 6-vector convention
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, model, *, device="cpu", **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model.to(device)
        self.device = device

    def calculate(self, atoms=None, properties=("energy", "forces", "stress"), system_changes=all_changes):
        """Run torch model and populate ASE results dict.

        Contract:
        - Build PiNN sparse tensors dict (coord/elems/ind_1/(cell)).
        - Call model(tensors) -> energy per structure, shape (B,). Here B=1.
        - Forces from autograd: F = -dE/dR (R is tensors["coord"]).
        - Apply e_dress (in model energy units) then multiply by e_unit.
    """
        super().calculate(atoms, properties, system_changes)

        R_np = atoms.get_positions().astype(np.float32)  # (N,3)
        Z_np = atoms.numbers.astype(np.int64)            # (N,)
        N = R_np.shape[0]

        coord = torch.tensor(R_np, dtype=torch.float32, device=self.device).detach().clone().requires_grad_(True)
        elems = torch.tensor(Z_np, dtype=torch.long, device=self.device)

        tensors = {
            "coord": coord,  # (N,3) requires_grad=True
            "elems": elems,  # (N,)
            "ind_1": torch.zeros((N, 1), dtype=torch.long, device=self.device),  # one structure id = 0
        }

        # If PBC, pass cell (PiNetTorch preprocess will MIC-wrap neighborlist).
        if np.any(atoms.get_pbc()):
            tensors["cell"] = torch.tensor(atoms.cell.array, dtype=torch.float32, device=self.device)

        # Model already returns total energy in ASE units
        E = self.model(tensors)

        if E.ndim == 0:
            E = E.view(1)

        if E.numel() != 1:
            raise ValueError(
                f"Calculator expects one structure energy; got shape {tuple(E.shape)}"
            )

        # Forces via autograd
        dE_dR = torch.autograd.grad(E.sum(), coord, create_graph=False)[0]
        F = -dE_dR

        self.results["energy"] = float(E.item())
        self.results["forces"] = F.detach().cpu().numpy()

        # Simple virial-based stress (same convention you already used)
        if np.any(atoms.get_pbc()):
            V = atoms.get_volume()
            F_np = self.results["forces"]
            virial = np.einsum("ni,nj->ij", R_np, F_np)
            stress_tensor = -virial / V

            self.results["stress"] = np.array(
                [
                    stress_tensor[0, 0],
                    stress_tensor[1, 1],
                    stress_tensor[2, 2],
                    stress_tensor[1, 2],
                    stress_tensor[0, 2],
                    stress_tensor[0, 1],
                ],
                dtype=float,
            )
        else:
            self.results["stress"] = np.zeros(6, dtype=float)


def get_calc(model_spec, **kwargs):
    """Factory used by pinn.get_calc() for backend='torch'."""
    model_dir = model_spec.get("model_dir", None)
    if model_dir is None:
        raise ValueError("Torch calculator requires model_spec['model_dir'].")

    ckpt_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing torch checkpoint: {ckpt_path}")

    device = kwargs.pop("device", "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Prefer rebuilding from checkpoint params (most reliable)
    ckpt_params = ckpt.get("params", None)
    build_params = ckpt_params if isinstance(ckpt_params, dict) else model_spec

    model = build_model(build_params).to(device)

    # Support both formats: full dict checkpoint or raw state_dict
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)

    model.eval()

    return TorchPiNNCalc(model, device=device, **kwargs)

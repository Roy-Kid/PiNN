# pinn/torch/calculator.py
from __future__ import annotations

import os
import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from .model import get_model as build_model


def _find_preprocess(obj):
    """
    Try a bunch of common wrapper attributes to find a .preprocess(tensors) callable.
    Returns callable or None.
    """
    # direct
    if hasattr(obj, "preprocess") and callable(getattr(obj, "preprocess")):
        return obj.preprocess

    # common wrappers
    for attr in ("network", "net", "module", "model"):
        if hasattr(obj, attr):
            sub = getattr(obj, attr)
            if hasattr(sub, "preprocess") and callable(getattr(sub, "preprocess")):
                return sub.preprocess

    return None

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

        need_stress = ("stress" in properties) and np.any(atoms.get_pbc())
        # Only run preprocess if we need stress under PBC
        if need_stress:
            preprocess = _find_preprocess(self.model)
            if preprocess is None:
                raise RuntimeError(
                    "PBC stress requested, but torch model does not expose preprocess(). "
                    "Expose it on the wrapper or ensure model.network.preprocess exists."
                )
            tensors = preprocess(tensors)

        # Model already returns total energy in ASE units
        E = self.model(tensors)

        if E.ndim == 0:
            E = E.view(1)

        if E.numel() != 1:
            raise ValueError(
                f"Calculator expects one structure energy; got shape {tuple(E.shape)}"
            )
        # --- Forces AND TF-faithful stress via autograd ---
        # Important: we need tensors["diff"] and tensors["ind_2"] which are created inside the model's preprocess.
        diff = tensors.get("diff", None)
        if np.any(atoms.get_pbc()):
            if diff is None:
                raise RuntimeError(
                    "PBC stress requires tensors['diff'] (pair displacements). "
                    "Your model/preprocess must populate it."
                )

            # Get gradients wrt coord and diff in one autograd call.
            dE_dR, dE_ddiff = torch.autograd.grad(
                E.sum(),
                [coord, diff],
                create_graph=False,
                retain_graph=False,
            )
        else:
            # Non-PBC: only forces needed
            dE_dR = torch.autograd.grad(E.sum(), coord, create_graph=False)[0]
            dE_ddiff = None

        F = -dE_dR
        self.results["energy"] = float(E.item())
        self.results["forces"] = F.detach().cpu().numpy()

        # --- Stress (TF ground truth): sum_over_pairs(diff ⊗ dE/diff) / det(cell) ---
        if np.any(atoms.get_pbc()):
            cell = tensors["cell"]                     # (3,3) torch tensor
            V = torch.det(cell).abs()                  # det(cell) like TF
            ind_2 = tensors.get("ind_2", None)
            ind_1 = tensors.get("ind_1", None)
            if ind_2 is None or ind_1 is None:
                raise RuntimeError("PBC stress requires tensors['ind_2'] and tensors['ind_1'].")

            # structure id per atom (batch index); in your calc it's all zeros, but keep generic
            batch = (ind_1[:, 0] if ind_1.ndim == 2 else ind_1).long()
            pair_to_batch = batch[ind_2[:, 0].long()]  # (n_pairs,)

            # outer product per pair: (n_pairs,3,3)
            outer = diff.unsqueeze(2) * dE_ddiff.unsqueeze(1)

            # sum per structure
            n_struct = int(batch.max().item()) + 1 if batch.numel() else 1
            stress = torch.zeros((n_struct, 3, 3), dtype=outer.dtype, device=outer.device)
            stress.index_add_(0, pair_to_batch, outer)

            stress = stress / V

            # ASE 6-vector: [xx, yy, zz, yz, xz, xy]
            s = stress[0].reshape(-1)  # row-major: [xx,xy,xz,yx,yy,yz,zx,zy,zz]
            self.results["stress"] = np.array(
                [s[0].item(), s[4].item(), s[8].item(), s[5].item(), s[2].item(), s[1].item()],
                dtype=float,
            )
        else:
            self.results["stress"] = np.zeros(6, dtype=float)
        


def get_calc(model_spec, **kwargs):
    """Factory used by pinn.get_calc() for backend='torch'."""

    # --- Special-case analytic LJ: no checkpoint, just return ASE's LJ calc ---
    if model_spec.get("network", {}).get("name") == "LJ":
        from ase.calculators.lj import LennardJones
        rc = float(model_spec.get("network", {}).get("params", {}).get("rc", 3.0))
        return LennardJones(rc=rc)

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

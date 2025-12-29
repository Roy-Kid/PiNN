import torch
import torch.nn as nn

from pinn.networks.pinet_torch import PiNetTorch


class PiNetPotentialTorch(nn.Module):
    """
    Potential model wrapper: structure -> total energy.

    This mirrors the TF potential model conceptually:
      - run PiNet to get per-atom energies (out_pool=False, out_units=1)
      - sum over atoms per structure to get total energy
    """

    def __init__(self, pinet: PiNetTorch) -> None:
        super().__init__()
        self.pinet = pinet

    def forward(self, tensors: dict) -> torch.Tensor:
        """
        Args:
            tensors: dict with at least ind_1, elems, coord, and optional cell.

        Returns:
            energy: Tensor of shape (n_structures,) with total energies.
        """
        # Per-atom energy contributions
        per_atom = self.pinet({**tensors, "coord": tensors["coord"]})
        if per_atom.ndim != 1:
            per_atom = per_atom.squeeze(-1)

        ind_1 = tensors["ind_1"]
        if ind_1.dtype != torch.long:
            ind_1 = ind_1.long()
        batch = ind_1[:, 0] if ind_1.ndim == 2 else ind_1
        n_struct = int(batch.max().item()) + 1 if batch.numel() else 0

        E = torch.zeros((n_struct,), dtype=per_atom.dtype, device=per_atom.device)
        E.index_add_(0, batch, per_atom)
        return E
    
    def energy_and_forces(model: nn.Module, tensors: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute total energy and forces via autograd.

        Args:
            model: A torch.nn.Module that returns per-structure energies (n_struct,).
            tensors: Input dict containing at least "coord" and "ind_1".

        Returns:
            E: (n_struct,) detached energy tensor.
            F: (n_atoms, 3) detached force tensor, defined as -d(sum(E))/dcoord.
        """
    coord = tensors["coord"].detach().clone().requires_grad_(True)
    tensors2 = {**tensors, "coord": coord}

    E = model(tensors2)          # (n_struct,)
    E_tot = E.sum()              # scalar
    (grad,) = torch.autograd.grad(E_tot, coord, create_graph=False, retain_graph=False)
    F = -grad
    return E.detach(), F.detach()

# pinn/io/torch/preprocess_fns.py
"""
Shared preprocessing functions for Torch backend.

These functions are the single source of truth for:
- building neighbor list (ind_2, shift) using a CellList-style builder
- computing diff/dist from coord, ind_2, cell, shift, ind_1 (supports per-structure cell)
- building prop (one-hot) from elems

Design:
- Used by IO preprocessing (build_dataset / dataloader path)
- Used by model-side PreprocessLayerTorch (compatibility / safety)

Keep this file free of model imports to avoid circular dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, List

import torch
import torch.nn as nn


@torch.no_grad()
def build_nl_celllist(
    *,
    ind_1: torch.Tensor,
    coord: torch.Tensor,
    cell: Optional[torch.Tensor],
    rc: float,
    nl_builder: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    Build a directed neighbor list per structure id using a CellList-style builder.

    Args:
        ind_1: (N,1) or (N,2) long tensor; ind_1[:,0] is structure id.
        coord: (N,3) float tensor
        cell:  None, (3,3), or (B,3,3)
        rc: cutoff (for traceability; nl_builder already configured at rc)
        nl_builder: module with signature nl_builder(coord_b, cell=H or None)
                    returning dict with:
                      - ind_2: (E,2) local indices in [0,n_b)
                      - shift: (E,3) integer lattice shift vectors (optional)

    Returns:
        dict with:
          - ind_2: (E,2) global indices into coord
          - shift: (E,3) integer shifts (zeros if missing)
    """
    if ind_1.dtype != torch.long:
        ind_1 = ind_1.long()
    batch = ind_1[:, 0] if ind_1.ndim == 2 else ind_1
    cell_is_per_struct = (cell is not None and cell.ndim == 3)

    ind2_list: List[torch.Tensor] = []
    shift_list: List[torch.Tensor] = []

    for b in batch.unique(sorted=True).tolist():
        idx = (batch == b).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue

        coord_b = coord[idx]
        if cell is None:
            nl_b = nl_builder(coord_b, cell=None)
        else:
            H = cell[b] if cell_is_per_struct else cell
            nl_b = nl_builder(coord_b, cell=H)

        ind_2_local = nl_b["ind_2"]
        if ind_2_local.numel() == 0:
            continue

        gi = idx[ind_2_local[:, 0]]
        gj = idx[ind_2_local[:, 1]]
        ind2_list.append(torch.stack([gi, gj], dim=1))

        if "shift" in nl_b:
            shift_list.append(nl_b["shift"].to(dtype=torch.long, device=coord.device))
        else:
            shift_list.append(torch.zeros((ind_2_local.shape[0], 3), dtype=torch.long, device=coord.device))

    if len(ind2_list) == 0:
        return {
            "ind_2": coord.new_zeros((0, 2), dtype=torch.long),
            "shift": coord.new_zeros((0, 3), dtype=torch.long),
        }

    return {
        "ind_2": torch.cat(ind2_list, dim=0),
        "shift": torch.cat(shift_list, dim=0),
    }



def compute_diff_dist(
    *,
    coord: torch.Tensor,
    ind_2: torch.Tensor,
    cell: Optional[torch.Tensor],
    shift: Optional[torch.Tensor],
    ind_1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute diff and dist for directed edges.

    Args:
        coord: (N,3)
        ind_2: (E,2) long
        cell: None, (3,3), or (B,3,3)
        shift: None or (E,3) long integer image shifts
        ind_1: (N,1)/(N,2) or (N,) used for structure id mapping if cell is per-structure

    Returns:
        diff: (E,3)
        dist: (E,)
    """
    i = ind_2[:, 0]
    j = ind_2[:, 1]

    # No PBC info
    if cell is None or shift is None:
        diff = coord[j] - coord[i]
        dist = torch.linalg.norm(diff, dim=1)
        return diff, dist

    # PBC global cell
    if cell.ndim == 2:
        H = cell.to(device=coord.device, dtype=coord.dtype)     # (3,3)
        t = shift.to(coord.dtype) @ H                           # (E,3)
        diff = (coord[j] + t) - coord[i]
        dist = torch.linalg.norm(diff, dim=1)
        return diff, dist

    # PBC per-structure cell
    if cell.ndim == 3:
        if ind_1.dtype != torch.long:
            ind_1 = ind_1.long()
        batch = ind_1[:, 0] if ind_1.ndim == 2 else ind_1       # (N,)
        sid = batch[i]                                          # (E,)
        H_pair = cell[sid].to(device=coord.device, dtype=coord.dtype)  # (E,3,3)
        t = torch.einsum("ei,eij->ej", shift.to(coord.dtype), H_pair)  # (E,3)
        diff = (coord[j] + t) - coord[i]
        dist = torch.linalg.norm(diff, dim=1)
        return diff, dist

    raise ValueError(f"Unexpected cell shape {tuple(cell.shape)}")


def atomic_onehot(elems: torch.Tensor, atom_types: Sequence[int]) -> torch.Tensor:
    """
    One-hot encode atomic numbers.

    Args:
        elems: (N,) integer atomic numbers
        atom_types: list of allowed atomic numbers defining channel order

    Returns:
        prop: (N, len(atom_types)) float tensor with 0/1 entries
    """
    if elems.dtype != torch.long:
        elems = elems.long()
    device = elems.device
    types = torch.tensor(list(atom_types), dtype=torch.long, device=device)  # (T,)
    # prop[n,t] = (elems[n] == types[t])
    prop = (elems[:, None] == types[None, :]).to(dtype=torch.float32)
    return prop

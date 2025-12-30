# pinn/io/torch/preprocess.py
"""
Torch IO preprocessing entry point.

This module provides a batch-level preprocessing function that mirrors the
behavior of PreprocessLayerTorch, but lives in IO (so models don't have to
recompute geometry features).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Any

import torch
import torch.nn as nn

from pinn.io.torch.preprocess_fns import build_nl_celllist, compute_diff_dist, atomic_onehot


def preprocess_batch_torch(
    tensors: Dict[str, Any],
    *,
    atom_types: Sequence[int],
    rc: float,
    nl_builder: nn.Module,
) -> Dict[str, Any]:
    """
    Preprocess a *batched* tensor dict for Torch backend.

    Required keys:
      - coord: (N,3) float tensor
      - ind_1: (N,1)/(N,2) long tensor, or (N,) long tensor (structure id)
      - elems: (N,) long tensor (atomic numbers) OR z key can be used by caller

    Optional:
      - cell: None, (3,3), or (B,3,3)
      - ind_2, shift, diff, dist, prop (if present, will not be recomputed)

    Returns:
      dict with prop, ind_2, shift, diff, dist present.
    """
    out = dict(tensors)

    # Normalize elems key (some pipelines use z)
    if "elems" not in out and "z" in out:
        out["elems"] = out["z"]

    if "prop" not in out:
        prop = atomic_onehot(out["elems"], atom_types).to(device=out["coord"].device)
        out["prop"] = prop.to(dtype=out["coord"].dtype)

    if "ind_2" not in out:
        cell = out.get("cell", None)
        nl = build_nl_celllist(
            ind_1=out["ind_1"],
            coord=out["coord"],
            cell=cell,
            rc=rc,
            nl_builder=nl_builder,
        )
        out.update(nl)  # adds ind_2 and shift

    if "diff" not in out or "dist" not in out:
        cell = out.get("cell", None)
        shift = out.get("shift", None)
        diff, dist = compute_diff_dist(
            coord=out["coord"],
            ind_2=out["ind_2"],
            cell=cell,
            shift=shift,
            ind_1=out["ind_1"],
        )
        out["diff"] = diff
        out["dist"] = dist

    return out

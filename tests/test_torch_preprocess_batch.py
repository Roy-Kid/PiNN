# tests/test_torch_preprocess_batch.py
# -*- coding: utf-8 -*-

import numpy as np
import torch

from helpers import *


def _make_batched_trivial(example_np: dict):
    """
    Build a minimal batched dict with ind_1, coord, elems for two structures.
    """
    coord = torch.as_tensor(np.asarray(example_np["coord"]), dtype=torch.float32)
    elems = torch.as_tensor(np.asarray(example_np.get("elems", example_np.get("z"))), dtype=torch.long)

    # Duplicate into 2 structures
    coord2 = torch.cat([coord, coord], dim=0)
    elems2 = torch.cat([elems, elems], dim=0)

    n = coord.shape[0]
    ind_1 = torch.cat([
        torch.zeros((n, 1), dtype=torch.long),
        torch.ones((n, 1), dtype=torch.long),
    ], dim=0)

    return {"coord": coord2, "elems": elems2, "ind_1": ind_1}


def test_io_preprocess_matches_model_preprocess_layer():
    """
    IO preprocess_batch_torch should match PreprocessLayerTorch behavior
    for the keys it produces (prop, ind_2, shift, diff, dist).
    """
    from pinn.io.torch.preprocess import preprocess_batch_torch
    from pinn.networks.pinet_torch import PreprocessLayerTorch  # adjust path if needed

    base = get_trivial_numpy()
    batch = _make_batched_trivial(base)

    atom_types = sorted(set(batch["elems"].tolist()))
    rc = 4.0

    layer = PreprocessLayerTorch(atom_types=atom_types, rc=rc)

    out_layer = layer(dict(batch))
    out_io = preprocess_batch_torch(dict(batch), atom_types=atom_types, rc=rc, nl_builder=layer.nl)

    for k in ("ind_2", "shift", "diff", "dist"):
        assert k in out_layer and k in out_io
        assert torch.allclose(out_layer[k], out_io[k], rtol=1e-6, atol=1e-6)

    assert "prop" in out_layer and "prop" in out_io
    assert out_layer["prop"].shape == out_io["prop"].shape

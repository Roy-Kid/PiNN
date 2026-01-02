# -*- coding: utf-8 -*-
import os
import pytest
import numpy as np

from utils import rotate


def _to_numpy(v):
    # TF tensors → numpy; already-numpy passes through
    return v.numpy() if hasattr(v, "numpy") else v


def _batch_numpy_dict(batch):
    return {k: _to_numpy(v) for k, v in batch.items()}


@pytest.mark.parametrize("backend", ["tf", "torch"])
def test_pinet_energy_invariant(mocked_data, backend):
    os.environ["PINN_BACKEND"] = backend
    theta = 42.0

    for batch in mocked_data:
        b = _batch_numpy_dict(batch)

        if backend == "tf":
            import tensorflow as tf
            from pinn.networks.pinet import PiNet

            model = PiNet(atom_types=[0, 1])
            b1 = {k: tf.convert_to_tensor(v) for k, v in b.items()}
            e1 = model(dict(b1))

            b2 = dict(b1)
            b2["coord"] = rotate(b2["coord"], theta)
            e2 = model(dict(b2))
            tf.debugging.assert_near(e1, e2)

        else:
            import torch
            from pinn.networks.pinet_torch import PiNetTorch

            model = PiNetTorch(
                atom_types=[0, 1],
                rc=5.0,
                n_basis=5,
                depth=3,
                pp_nodes=[8, 8],
                pi_nodes=[8, 8],
                ii_nodes=[8, 8],
                out_nodes=[8, 8],
                out_units=1,
                out_pool=False,
                act="tanh",
            )

            b1 = {
                "coord": torch.tensor(b["coord"], dtype=torch.float32),
                "elems": torch.tensor(b["elems"], dtype=torch.long),
                "ind_1": torch.tensor(b["ind_1"], dtype=torch.long),
            }
            e1 = model(dict(b1))

            b2 = dict(b1)
            b2["coord"] = rotate(b2["coord"], theta)
            e2 = model(dict(b2))

            assert torch.allclose(e1, e2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", ["tf", "torch"])
def test_pinet2_rank3_energy_invariant(mocked_data, backend):
    os.environ["PINN_BACKEND"] = backend
    theta = 42.0

    for batch in mocked_data:
        b = _batch_numpy_dict(batch)

        if backend == "tf":
            import tensorflow as tf
            from pinn.networks.pinet2 import PiNet2

            model = PiNet2(rank=3, atom_types=[0, 1])
            b1 = {k: tf.convert_to_tensor(v) for k, v in b.items()}
            e1 = model(dict(b1))

            b2 = dict(b1)
            b2["coord"] = rotate(b2["coord"], theta)
            e2 = model(dict(b2))
            tf.debugging.assert_near(e1, e2)

        else:
            import torch
            from pinn.networks.pinet2_torch import PiNet2Torch

            model = PiNet2Torch(
                {
                    "rank": 3,
                    "atom_types": [0, 1],
                    "rc": 5.0,
                    "n_basis": 5,
                    "depth": 3,
                    "pp_nodes": [8, 8],
                    "pi_nodes": [8, 8],
                    "ii_nodes": [8, 8],
                    "out_nodes": [8, 8],
                    "act": "tanh",
                    "out_units": 1,
                    "out_pool": False,
                    "torsion_boost": True,
                }
            )

            b1 = {
                "coord": torch.tensor(b["coord"], dtype=torch.float32),
                "elems": torch.tensor(b["elems"], dtype=torch.long),
                "ind_1": torch.tensor(b["ind_1"], dtype=torch.long),
            }
            e1 = model(dict(b1))

            b2 = dict(b1)
            b2["coord"] = rotate(b2["coord"], theta)
            e2 = model(dict(b2))

            assert torch.allclose(e1, e2, rtol=1e-5, atol=1e-6)

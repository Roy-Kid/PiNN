# -*- coding: utf-8 -*-
import os
import pytest
import numpy as np

from utils import rotate


@pytest.mark.parametrize("backend", ["tf", "torch"])
def test_simple_dotlayer(backend):
    os.environ["PINN_BACKEND"] = backend
    n_atoms, n_dims, n_channels = 10, 3, 5
    theta = 42.0

    if backend == "tf":
        import tensorflow as tf
        from pinn.networks.pinet2 import DotLayer

        prop = tf.random.uniform((n_atoms, n_dims, n_channels))
        dot = DotLayer(weighted=False)

        tf.debugging.assert_near(dot(rotate(prop, theta)), dot(prop))

    else:
        import torch
        from pinn.networks.pinet2_torch import DotLayerTorch

        prop = torch.rand((n_atoms, n_dims, n_channels), dtype=torch.float32)
        dot = DotLayerTorch()

        a = dot(rotate(prop, theta))
        b = dot(prop)
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", ["tf", "torch"])
def test_simple_scalelayer(backend):
    os.environ["PINN_BACKEND"] = backend
    n_atoms, n_dims, n_channels = 10, 3, 5

    if backend == "tf":
        import tensorflow as tf
        from pinn.networks.pinet2 import ScaleLayer

        prop = tf.random.uniform((n_atoms, n_dims, n_channels))
        s = tf.random.uniform((n_atoms, n_channels))
        layer = ScaleLayer()
        out = layer([prop, s])

        # Simple shape sanity
        assert out.shape == prop.shape

    else:
        import torch
        from pinn.networks.pinet2_torch import ScaleLayerTorch

        prop = torch.rand((n_atoms, n_dims, n_channels), dtype=torch.float32)
        s = torch.rand((n_atoms, n_channels), dtype=torch.float32)
        layer = ScaleLayerTorch()
        out = layer(prop, s)

        assert out.shape == prop.shape


@pytest.mark.parametrize("backend", ["tf", "torch"])
def test_simple_pixlayer(backend):
    os.environ["PINN_BACKEND"] = backend
    n_atoms, n_pairs, n_dims, n_channels = 10, 40, 3, 5

    ind_2 = np.stack(
        [np.random.randint(0, n_atoms, size=n_pairs), np.random.randint(0, n_atoms, size=n_pairs)],
        axis=1,
    ).astype(np.int64)

    if backend == "tf":
        import tensorflow as tf
        from pinn.networks.pinet2 import PIXLayer

        prop = tf.random.uniform((n_atoms, n_dims, n_channels))
        layer = PIXLayer(weighted=False)
        out = layer([ind_2, prop])
        assert out.shape == (n_pairs, n_dims, n_channels)

    else:
        import torch
        from pinn.networks.pinet2_torch import PIXLayerTorch

        prop = torch.rand((n_atoms, n_dims, n_channels), dtype=torch.float32)
        layer = PIXLayerTorch()
        out = layer(torch.tensor(ind_2, dtype=torch.long), prop)
        assert out.shape == (n_pairs, n_dims, n_channels)


@pytest.mark.parametrize("backend", ["tf", "torch"])
def test_simple_dotlayer_rank2(backend):
    """
    The rank-2 weighted/extended behavior is TF-only in the original tests.
    Torch translation currently targets non-weighted rank=3 (P3) only.
    """
    if backend == "torch":
        pytest.skip("Torch backend: rank2 dotlayer tests are TF-only.")
    import tensorflow as tf
    from pinn.networks.pinet2 import DotLayer

    n_atoms, n_channels = 10, 5
    theta = 42.0

    prop = tf.random.uniform((n_atoms, 3, 3, n_channels))
    dot = DotLayer(weighted=False)
    tf.debugging.assert_near(dot(rotate(prop, theta)), dot(prop))
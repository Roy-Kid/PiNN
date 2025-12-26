# tests/utils.py
import numpy as np

def _rot_matrix_np(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0, c, -s],
         [0.0, s,  c]],
        dtype=np.float32,
    )

def rotate(x, theta):
    """
    Rotate a tensor in xyz by angle theta around x-axis.
    Supports both tf.Tensor and torch.Tensor (and numpy arrays).
    """
    # Torch
    try:
        import torch
        if isinstance(x, torch.Tensor):
            rot = torch.tensor(_rot_matrix_np(theta), dtype=x.dtype, device=x.device)
            ndim = x.ndim - 2
            if ndim == 0:
                return torch.einsum("ix,xy->iy", x, rot)
            elif ndim == 1:
                return torch.einsum("ixc,xy->iyc", x, rot)
            elif ndim == 2:
                return torch.einsum("xw,ixyc,yz->iwzc", rot, x, rot)
            else:
                raise ValueError(f"Unsupported ndim={ndim} for rotate()")
    except Exception:
        pass

    # TensorFlow
    try:
        import tensorflow as tf
        if isinstance(x, tf.Tensor):
            rot = tf.constant(_rot_matrix_np(theta), dtype=x.dtype)
            ndim = x.ndim - 2
            if ndim == 0:
                return tf.einsum("ix,xy->iy", x, rot)
            elif ndim == 1:
                return tf.einsum("ixc,xy->iyc", x, rot)
            elif ndim == 2:
                return tf.einsum("xw,ixyc,yz->iwzc", rot, x, rot)
            else:
                raise ValueError(f"Unsupported ndim={ndim} for rotate()")
    except Exception:
        pass

    # Numpy fallback
    x = np.asarray(x)
    rot = _rot_matrix_np(theta)
    ndim = x.ndim - 2
    if ndim == 0:
        return np.einsum("ix,xy->iy", x, rot)
    elif ndim == 1:
        return np.einsum("ixc,xy->iyc", x, rot)
    elif ndim == 2:
        return np.einsum("xw,ixyc,yz->iwzc", rot, x, rot)
    raise ValueError(f"Unsupported ndim={ndim} for rotate()")

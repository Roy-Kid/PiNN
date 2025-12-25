# -*- coding: utf-8 -*-
#
__version__ = '2.0.0'

from pinn.networks import get as get_network
from pinn.models import get as _get_model_tf
from pinn import report
from .backend import get_backend

def get_calc(model_spec, **kwargs):
    """Return an ASE calculator for a trained PiNN model.

    This function is backend-dispatched (TensorFlow or PyTorch).

    Parameters
    ----------
    model_spec : dict or model object
        Either a model specification dictionary (must include at least
        ``model_dir`` and optionally ``backend``), or an already-constructed
        backend model object (e.g., a TensorFlow Estimator).

        Backend selection follows:
        1) ``model_spec["backend"]`` if present,
        2) environment variable ``PINN_BACKEND``,
        3) default ``"tf"``.

    **kwargs
        Passed to the ASE calculator constructor.

    Notes
    -----
    For the PyTorch backend, the calculator typically loads model weights
    from ``model_dir`` rather than using a TensorFlow Estimator.
    """
    backend = get_backend(model_spec if isinstance(model_spec, dict) else None)
    
    if backend == "torch":
        from .torch.calculator import get_calc as torch_get_calc
        return torch_get_calc(model_spec, **kwargs)

    import tensorflow as tf
    from pinn.calculator import PiNN_calc
    if isinstance(model_spec, tf.estimator.Estimator):
        model = model_spec
    else:
        model = get_model(model_spec)
    return  PiNN_calc(model, **kwargs)

def get_model(params, **kwargs):
    """Return a backend-specific model object.

    For backend="tf", returns a TensorFlow Estimator (existing behavior).
    For backend="torch", returns a torch model instance.
    """
    backend = get_backend(params)
    if backend == "torch":
        from .torch.model import get_model as torch_get_model
        return torch_get_model(params, **kwargs)
    return _get_model_tf(params, **kwargs)

def get_available_networks():
    print("Available networks:")
    print("  - PiNet")
    print("  - PiNet2")
    print("  - BPNN")
    print("  - LJ")

def get_available_models():
    print("Available models:")
    print("  - potential_model")
    print("  - dipole_model")
    print("  - AC_dipole_model")
    print("  - AD_dipole_model")
    print("  - BC_R_dipole_model")
    print("  - AD_OS_dipole_model")
    print("  - AC_AD_dipole_model")
    print("  - AC_BC_R_dipole_model")
    print("  - AD_BC_R_dipole_model")

def train_and_evaluate(model, params, **kwargs):
    backend = get_backend(params)

    if backend == "torch":
        from .torch.runtime import train_and_evaluate as torch_train_and_evaluate
        return torch_train_and_evaluate(model=model, params=params, **kwargs)

    raise AttributeError(
        "pinn.train_and_evaluate is not available for backend='tf' in this test path. "
        "TF uses tf.estimator.train_and_evaluate directly."
    )

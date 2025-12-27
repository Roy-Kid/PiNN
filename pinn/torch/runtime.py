
import os
import numpy as np
import torch
from typing import Dict, Iterator
from pinn.torch.optim import build_optimizer_from_params, apply_grad_clipping

from dataclasses import dataclass

@dataclass(frozen=True)
class PotentialLossConfig:
    """Loss + reporting configuration for potential_model training."""
    e_unit: float = 1.0
    e_scale: float = 1.0
    e_loss_multiplier: float = 1.0
    f_loss_multiplier: float = 1.0
    use_force: bool = True
    use_e_per_atom: bool = False
    log_e_per_atom: bool = False  # affects metrics/reporting only

def _get_potential_loss_config(params: dict) -> PotentialLossConfig:
    """Parse potential_model training knobs from params dict with safe defaults."""
    mp = params.get("model", {}).get("params", {}) or {}
    return PotentialLossConfig(
        e_unit=float(mp.get("e_unit", 1.0)),
        e_scale=float(mp.get("e_scale", 1.0)),
        e_loss_multiplier=float(mp.get("e_loss_multiplier", 1.0)),
        f_loss_multiplier=float(mp.get("f_loss_multiplier", 1.0)),
        use_force=bool(mp.get("use_force", True)),
        use_e_per_atom=bool(mp.get("use_e_per_atom", False)),
        log_e_per_atom=bool(mp.get("log_e_per_atom", False)),
    )


def _make_sparse_batch(
    batch: Dict[str, np.ndarray],
    *,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Convert a dense (B,N,...) batch to PiNN sparse dict tensors.

    Returns a dict with:
      coord: (B*N,3) requires_grad=True
      elems: (B*N,)
      ind_1: (B*N,1) structure ids
      _E_true: (B,)
      _F_true: (B*N,3)
    """
    R = torch.tensor(batch["coord"], dtype=torch.float32, device=device)  # (B,N,3)
    Z = torch.tensor(batch["elems"], dtype=torch.long, device=device)     # (B,N)
    E_true = torch.tensor(batch["e_data"], dtype=torch.float32, device=device)  # (B,)
    F_true = torch.tensor(batch["f_data"], dtype=torch.float32, device=device)  # (B,N,3)

    B, N, _ = R.shape

    coord = R.reshape(B * N, 3).detach().clone().requires_grad_(True)
    elems = Z.reshape(B * N)

    ind_1 = torch.zeros((B * N, 1), dtype=torch.long, device=device)
    ind_1[:, 0] = torch.arange(B, device=device).repeat_interleave(N)

    return {
        "coord": coord,
        "elems": elems,
        "ind_1": ind_1,
        "_E_true": E_true,
        "_F_true": F_true.reshape(B * N, 3),
    }


def iter_batches(
    data: Dict[str, np.ndarray],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int = 0,
    repeat: bool = True,
) -> Iterator[Dict[str, np.ndarray]]:
    """Yield minibatches from the LJ dataset dict.

    This is a torch-friendly replacement for the TF input pipeline used in
    tests/test_potential.py. It yields numpy arrays; the training loop will
    convert them to torch tensors.

    Parameters
    ----------
    data
        Dict with keys: 'coord', 'elems', 'e_data', 'f_data'.
    batch_size
        Number of configurations per batch.
    shuffle
        Whether to shuffle indices each epoch.
    seed
        RNG seed for deterministic shuffling.
    repeat
        If True, loop forever; if False, yield one epoch only.

    Yields
    ------
    batch : dict
        Same keys as `data`, but batched on the first axis.
    """
    n = data["coord"].shape[0]
    rng = np.random.default_rng(seed)

    while True:
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)

        for start in range(0, n, batch_size):
            sel = idx[start:start + batch_size]
            if sel.size == 0:
                continue
            yield {
                "coord": data["coord"][sel],
                "elems": data["elems"][sel],
                "e_data": data["e_data"][sel],
                "f_data": data["f_data"][sel],
            }

        if not repeat:
            break



def train_and_evaluate(
    *,
    model,
    params,
    data,
    max_steps: int,
    eval_steps: int,
    batch_size_train: int,
    batch_size_eval: int,
    shuffle_buffer: int = 0,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    **kwargs,
):
    """Train and evaluate a torch PiNN model on the LJ toy dataset.

    This implements the torch backend counterpart of the TF estimator training
    used in tests/test_potential.py. It consumes the raw `data` dict produced
    by `_get_lj_data()`.

    Metric convention (must match the test)
    --------------------------------------
    The test checks:

        results["METRICS/E_RMSE"] / e_scale  ≈  RMSE( E_pred/e_unit - e_data )
        results["METRICS/F_RMSE"] / e_scale  ≈  RMSE( F_pred/e_unit - f_data )

    Therefore we return:

        METRICS/E_RMSE = e_scale * RMSE( E_pred/e_unit - e_data )
        METRICS/F_RMSE = e_scale * RMSE( F_pred/e_unit - f_data )
        
        If use_force is False, METRICS/F_RMSE is reported as 0.0 (Torch behavior).

    Parameters
    ----------
    model
        Torch model is called as model(tensors_dict) where tensors follow PiNN sparse conventions.
    params : dict
        Model spec dict containing model_dir and model.params (e_unit, e_scale, e_dress).
    data : dict
        LJ dataset dict with keys 'coord', 'elems', 'e_data', 'f_data'.
    max_steps : int
        Training steps.
    eval_steps : int
        Number of eval batches to average metrics over.
    batch_size_train : int
        Training batch size.
    batch_size_eval : int
        Eval batch size.
    lr : float Default learning rate used only if params['optimizer'] is absent.
    seed : int
        RNG seed.
    device : str
        Torch device string.

    Returns
    -------
    dict
        Metrics dict with keys 'METRICS/E_RMSE' and 'METRICS/F_RMSE'.
    """
    cfg = _get_potential_loss_config(params)
    e_unit = cfg.e_unit
    e_scale = cfg.e_scale

    model_dir = params.get("model_dir", None)
    if model_dir is None:
        raise ValueError("params['model_dir'] is required for torch training.")
    os.makedirs(model_dir, exist_ok=True)

    # Move model to device (once you have a real torch model)
    if hasattr(model, "to"):
        model = model.to(device)

    
    # Optimizer/scheduler/clipping from params.yml-style optimizer spec (generic for all torch nets)
    optim, scheduler, clip = build_optimizer_from_params(model, params, default_lr=lr)

    train_it = iter_batches(data, batch_size_train, shuffle=True, seed=seed, repeat=True)
    eval_it = iter_batches(data, batch_size_eval, shuffle=False, seed=seed, repeat=True)

    # ---- training loop ----
    if hasattr(model, "train"):
        model.train()

    for step in range(int(max_steps)):
        batch = next(train_it)

        sb = _make_sparse_batch(batch, device=device)

        tensors = {k: v for k, v in sb.items() if not k.startswith("_")}
        E_true = sb["_E_true"]                 # (B,)
        F_true = sb["_F_true"]                 # (B*N,3)
        coord = tensors["coord"]               # (B*N,3) requires_grad=True

    # ---- forward energy ----
        if callable(model):
            E_pred = model(tensors)            # expected (B,)
        else:
            raise TypeError("Torch backend expects a callable torch model.")

        if E_pred.ndim != 1:
            raise ValueError(f"Expected E_pred shape (B,), got {tuple(E_pred.shape)}")

    #---- forces via autograd (only if needed) ----
        if cfg.use_force:
            dE_dR = torch.autograd.grad(
            E_pred.sum(),
            coord,
            create_graph=True,   # force loss needs gradients w.r.t. model params
            retain_graph=True,   # keep graph for energy backward
            )[0]
            F_pred = -dE_dR
        else:
            F_pred = None  # type: ignore[assignment]

    # ---- loss (same convention as test) ----
        # ---- energy loss (optionally per-atom normalized) ----
        if cfg.use_e_per_atom:
            B = int(E_true.shape[0])
            counts = torch.bincount(tensors["ind_1"][:, 0], minlength=B).to(E_true.dtype).clamp_min(1.0)
            E_pred_used = E_pred / counts
            E_true_used = E_true / counts
        else:
            E_pred_used = E_pred
            E_true_used = E_true

        e_err = (E_pred_used / e_unit) - E_true_used
        e_loss = (e_err ** 2).mean()

        # ---- force loss (optional) ----
        if cfg.use_force:
            f_err = (F_pred / e_unit) - F_true
            f_loss = (f_err ** 2).mean()
        else:
            f_loss = torch.zeros((), dtype=e_loss.dtype, device=e_loss.device)

        loss = cfg.e_loss_multiplier * e_loss + cfg.f_loss_multiplier * f_loss

        if optim is not None:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            apply_grad_clipping(model, clip)
            optim.step()
            if scheduler is not None:
                scheduler.step()

    # ---- evaluation loop (compute RMSE) ----
    if hasattr(model, "eval"):
        model.eval()

    # We need gradients to compute forces, so we cannot wrap eval in torch.no_grad().
    e_sq_sum = 0.0
    e_count = 0
    f_sq_sum = 0.0
    f_count = 0

    for _ in range(int(eval_steps)):
        batch = next(eval_it)

        sb = _make_sparse_batch(batch, device=device)

        tensors = {k: v for k, v in sb.items() if not k.startswith("_")}
        E_true = sb["_E_true"]
        F_true = sb["_F_true"]
        coord = tensors["coord"]

        E_pred = model(tensors)

        # energy error for metrics
        if cfg.log_e_per_atom:
            B = int(E_true.shape[0])
            counts = torch.bincount(tensors["ind_1"][:, 0], minlength=B).to(E_true.dtype).clamp_min(1.0)
            e_err = ((E_pred / counts) / e_unit) - (E_true / counts)
        else:
            e_err = (E_pred / e_unit) - E_true

        e_sq_sum += float((e_err ** 2).sum().item())
        e_count += int(e_err.numel())

        # forces only if requested
        if cfg.use_force:
            dE_dR = torch.autograd.grad(E_pred.sum(), coord, create_graph=False)[0]
            F_pred = -dE_dR
            f_err = (F_pred / e_unit) - F_true
            f_sq_sum += float((f_err ** 2).sum().item())
            f_count += int(f_err.numel())

    e_rmse = float(np.sqrt(e_sq_sum / max(e_count, 1)))
    if cfg.use_force:
        f_rmse = float(np.sqrt(f_sq_sum / max(f_count, 1)))
    else:
        f_rmse = 0.0

    # Save checkpoint
    if hasattr(model, "state_dict"):
        to_save = model
        # If later you ever wrap with nn.DataParallel / DDP
        if hasattr(model, "module"):
            to_save = model.module

        ckpt = {
            "model_state_dict": to_save.state_dict(),
            "params": params,
            # Optional metadata (harmless, helps debugging)
            "pytorch_version": torch.__version__,
        }
        torch.save(ckpt, os.path.join(model_dir, "model.pt"))

    return {
        "METRICS/E_RMSE": e_scale * e_rmse,
        "METRICS/F_RMSE": e_scale * f_rmse,
    }





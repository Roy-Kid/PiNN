
import os
import numpy as np
import torch
from typing import Dict, Iterator
from pinn.torch.optim import build_optimizer_from_params, apply_grad_clipping
from pinn.torch.input_pipeline import TorchDataOptions, make_torch_dataloader_from_yml
from pinn.utils import init_params
import yaml
from torch.utils.tensorboard import SummaryWriter

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

def _flatten_forces(F: torch.Tensor) -> torch.Tensor:
    # Accept (B,N,3) or (BN,3) and return (BN,3)
    if F.ndim == 3 and F.shape[-1] == 3:
        return F.reshape(-1, 3)
    if F.ndim == 2 and F.shape[-1] == 3:
        return F
    raise ValueError(f"Unexpected force shape: {tuple(F.shape)}")


def _inject_e_dress_into_torch_model(model, params: dict) -> None:
    mp = params.get("model", {}).get("params", {}) or {}
    dress = mp.get("e_dress", None)
    if not dress:
        return

    # YAML may load keys as int already, but be strict.
    dress = {int(k): float(v) for k, v in dress.items()}

    # Handle common wrappers: model, model.model, model.network, etc.
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "network", None),
        getattr(model, "net", None),
    ]
    for obj in candidates:
        if obj is None:
            continue
        if hasattr(obj, "e_dress"):
            obj.e_dress = dress

def _structure_iter_from_torch_batches(batch_iter, max_structures=1000):
    """
    Yield per-structure dicts compatible with pinn.utils.init_params.

    This strips batching, gradients, devices, and sparse flattening.
    """
    seen = 0
    for batch in batch_iter:
        # YAML pipeline: coord is (B*N,3), ind_1 maps atoms -> structure
        coord = batch["coord"].detach().cpu()
        elems = batch["elems"].detach().cpu()
        e_data = batch["e_data"].detach().cpu()
        ind_1 = batch["ind_1"][:, 0].detach().cpu()

        B = int(e_data.shape[0])

        for b in range(B):
            mask = ind_1 == b
            yield {
                "coord": coord[mask].numpy(),
                "elems": elems[mask].numpy(),
                "e_data": float(e_data[b]),
            }
            seen += 1
            if seen >= max_structures:
                return

def _torch_structure_iter_to_tf_dataset(struct_iter):
    """
    Adapt a Torch/Python structure iterator to a tf.data.Dataset
    compatible with pinn.utils.init_params.
    """
    import tensorflow as tf

    def gen():
        for s in struct_iter:
            yield {
                "coord": s["coord"],
                "elems": s["elems"],
                "e_data": s["e_data"],
            }

    return tf.data.Dataset.from_generator(
        gen,
        output_signature={
            "coord": tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            "elems": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "e_data": tf.TensorSpec(shape=(), dtype=tf.float32),
        },
    )

def _get_nl_builder_from_model(model):
    """Return the neighbor-list builder module used by the Torch network.

    We keep this in one place because different wrappers may expose the network
    differently (model.network, model.net, model.model, etc.).

    Raises:
        AttributeError if no builder can be found.
    """
    # Common layouts: model.network, model.net, model.model
    candidates = [
        getattr(model, "network", None),
        getattr(model, "net", None),
        getattr(model, "model", None),
        model,
    ]
    for obj in candidates:
        if obj is None:
            continue
        # Common preprocess holders: preprocess, preprocess_layer, pre, pp
        for pre_name in ("preprocess", "preprocess_layer", "pre", "pp"):
            pre = getattr(obj, pre_name, None)
            if pre is not None and hasattr(pre, "nl"):
                return pre.nl
        # Sometimes NL is attached directly
        if hasattr(obj, "nl"):
            return obj.nl

    raise AttributeError(
        "Could not find neighbor-list builder on model. "
        "Expected something like model.network.preprocess.nl"
    )

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



def _repeat_yml_loader(yml_path, *, dataset_role, opts, nl_builder):
    """
    Yield batches from a YAML-backed torch dataloader forever.

    This matches TF's dataset.repeat() behavior and prevents StopIteration
    when max_steps exceeds a single pass over the cached dataset.
    """
    while True:
        loader = make_torch_dataloader_from_yml(
            yml_path, dataset_role=dataset_role, opts=opts, nl_builder=nl_builder
        )
        for batch in loader:
            yield batch


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

    # Torch input sources:
    # - existing tests: data is a numpy dict
    # - CLI path: train_yml/eval_yml point to dataset YAML
    train_yml = kwargs.pop("train_yml", None)
    eval_yml = kwargs.pop("eval_yml", None)

    cache = bool(kwargs.pop("cache", True))
    cache_ram = bool(kwargs.pop("cache_ram", True))
    preprocess_flag = bool(kwargs.pop("preprocess", True))
    scratch_dir = kwargs.pop("scratch_dir", None)
    log_every = int(kwargs.pop("log_every", 100))


    cfg = _get_potential_loss_config(params)
    e_unit = cfg.e_unit
    e_scale = cfg.e_scale

    model_dir = params.get("model_dir", None)
    if model_dir is None:
        raise ValueError("params['model_dir'] is required for torch training.")
    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=model_dir)  # or model_dir/"logs"
    try: 

        # Move model to device (once you have a real torch model)
        if hasattr(model, "to"):
            model = model.to(device)

        # Optimizer/scheduler/clipping from params.yml-style optimizer spec (generic for all torch nets)
        optim, scheduler, clip = build_optimizer_from_params(model, params, default_lr=lr)

        # ---- Build data iterators (numpy dict path OR YAML/CLI path) ----
        net_params = params.get("network", {}).get("params", {}) or {}
        atom_types = net_params.get("atom_types", None)
        rc = float(net_params.get("rc", 0.0))

        if train_yml is not None or eval_yml is not None:
            if train_yml is None or eval_yml is None:
                raise ValueError("Torch runtime requires both train_yml and eval_yml when using YAML datasets.")
            if atom_types is None:
                raise ValueError("params['network']['params']['atom_types'] is required for Torch YAML pipeline.")
            if rc <= 0.0:
                raise ValueError("params['network']['params']['rc'] must be > 0 for Torch YAML pipeline.")

            nl_builder = _get_nl_builder_from_model(model)

            # Cache directory: prefer explicit kwarg, fall back to model_dir
            if scratch_dir is None:
                scratch_dir = model_dir  # safe default

            train_opts = TorchDataOptions(
                batch_size=int(batch_size_train),
                shuffle_buffer=int(shuffle_buffer),
                atom_types=list(atom_types),
                rc=float(rc),
                scratch_dir=scratch_dir,
                cache=True,
                cache_ram=True,
                device=device,
                preprocess=preprocess_flag,
            )
            eval_opts = TorchDataOptions(
                batch_size=int(batch_size_eval),
                shuffle_buffer=0,
                atom_types=list(atom_types),
                rc=float(rc),
                scratch_dir=scratch_dir,
                cache=True,
                cache_ram=True,
                device=device,
                preprocess=preprocess_flag,
            )

            train_it = _repeat_yml_loader(train_yml, dataset_role="train", opts=train_opts, nl_builder=nl_builder)
        
            use_yml_pipeline = True
        else:
            # Existing unit-test path: LJ toy numpy dict
            train_it = iter_batches(data, batch_size_train, shuffle=True, seed=seed, repeat=True)
            eval_it = iter_batches(data, batch_size_eval, shuffle=False, seed=seed, repeat=True)
            use_yml_pipeline = False
        
        # ---- atomic dressing (Torch parity with TF) ----
        model_params = params.get("model", {}).get("params", {}) or {}
        if params.get("model", {}).get("name") == "potential_model" and "e_dress" not in model_params:
            # init_params is TF-only: always feed it a tf.data.Dataset
            try:
                if use_yml_pipeline:
                    from pinn.io import load_tfrecord  # -> tf.data.Dataset
                    tf_train_ds = load_tfrecord(train_yml)
                else:
                    from pinn.io import load_numpy     # -> tf.data.Dataset
                    tf_train_ds = load_numpy(data)

                init_params(params, tf_train_ds)  # TF contract satisfied
                _inject_e_dress_into_torch_model(model, params)
                print("Torch model e_dress active:", getattr(model, "e_dress", None))

            except Exception as e:
                raise RuntimeError(
                    "Failed to initialize e_dress via TF init_params(). "
                    "This requires TensorFlow and a valid TF dataset input."
                ) from e

            # Persist params.yml
            params_path = os.path.join(model_dir, "params.yml")
            with open(params_path, "w") as f:
                yaml.safe_dump(params, f)

        # ---- training loop ----
        if hasattr(model, "train"):
            model.train()

        for step in range(int(max_steps)):
            batch = next(train_it)

            if use_yml_pipeline:
                # batch is already a torch sparse+preprocessed dict
                tensors = {k: v for k, v in batch.items() if k not in ("e_data", "f_data")}
                if "e_data" not in batch or "f_data" not in batch:
                    raise KeyError("YAML Torch pipeline batch must include 'e_data' and 'f_data' for potential_model tests.")
                E_true = batch["e_data"]
                F_true = batch["f_data"]
                F_true = _flatten_forces(F_true)
                coord = tensors["coord"]
                if not coord.requires_grad or not coord.is_leaf:
                    coord = coord.detach().clone().requires_grad_(True)
                    tensors["coord"] = coord
            else:
                # LJ numpy dict path
                sb = _make_sparse_batch(batch, device=device)
                tensors = {k: v for k, v in sb.items() if not k.startswith("_")}
                E_true = sb["_E_true"]
                F_true = sb["_F_true"]
                F_true = _flatten_forces(F_true)
                coord = tensors["coord"]
                if not coord.requires_grad or not coord.is_leaf:
                    coord = coord.detach().clone().requires_grad_(True)
                    tensors["coord"] = coord

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
                assert F_pred.shape == F_true.shape, (F_pred.shape, F_true.shape)
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
                if (step % log_every) == 0:
                    # ---- metric errors: must match eval convention ----
                    if cfg.log_e_per_atom:
                        B = int(E_true.shape[0])
                        counts = (
                            torch.bincount(tensors["ind_1"][:, 0], minlength=B)
                            .to(E_true.dtype)
                            .clamp_min(1.0)
                        )
                        # Energy metric is per-atom
                        e_err_metric = ((E_pred / counts) / e_unit) - (E_true / counts)
                    else:
                        # Energy metric is per-structure
                        e_err_metric = (E_pred / e_unit) - E_true

                    train_e_rmse = torch.sqrt((e_err_metric ** 2).mean()).item()

                    if cfg.use_force:
                        # Force metric is always in force label space (no per-atom normalization)
                        f_err_metric = (F_pred / e_unit) - F_true
                        train_f_rmse = torch.sqrt((f_err_metric ** 2).mean()).item()
                    else:
                        train_f_rmse = 0.0

                    lr_now = optim.param_groups[0]["lr"] if optim is not None else float("nan")
                    writer.add_scalar("train/loss", float(loss.item()), step)
                    writer.add_scalar("train/E_RMSE", train_e_rmse, step)
                    writer.add_scalar("train/F_RMSE", train_f_rmse, step)
                    writer.add_scalar("train/lr", float(lr_now), step)
                    writer.flush()
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

        if use_yml_pipeline:
            # Best practice (minimal change): build a fresh eval loader here and iterate safely.
            eval_loader = make_torch_dataloader_from_yml(
                eval_yml, dataset_role="eval", opts=eval_opts, nl_builder=nl_builder
            )
            eval_iter = iter(eval_loader)
        else:
            # Existing unit-test path: repeat=True iterator
            eval_iter = eval_it

        for _ in range(int(eval_steps)):
            try:
                batch = next(eval_iter)
            except StopIteration:
                # YAML eval set exhausted before eval_steps; stop cleanly.
                break

            if use_yml_pipeline:
                # batch is already torch sparse(+maybe preprocessed)
                if "e_data" not in batch or "f_data" not in batch:
                    raise KeyError(
                        "YAML Torch pipeline batch must include 'e_data' and 'f_data' for potential_model tests."
                    )

                # IMPORTANT: do not pass labels into the model input dict
                tensors = {k: v for k, v in batch.items() if k not in ("e_data", "f_data")}
                E_true = batch["e_data"]
                F_true = batch["f_data"]
                F_true = _flatten_forces(F_true)
                coord = tensors["coord"]
                if not coord.requires_grad or not coord.is_leaf:
                    coord = coord.detach().clone().requires_grad_(True)
                    tensors["coord"] = coord
            else:
                # LJ numpy dict path
                sb = _make_sparse_batch(batch, device=device)
                tensors = {k: v for k, v in sb.items() if not k.startswith("_")}
                E_true = sb["_E_true"]
                F_true = sb["_F_true"]
                F_true = _flatten_forces(F_true)
                coord = tensors["coord"]
                if not coord.requires_grad or not coord.is_leaf:
                    coord = coord.detach().clone().requires_grad_(True)
                    tensors["coord"] = coord

            E_pred = model(tensors)

            # Accept (B,1) or (B,)
            if E_pred.ndim == 2 and E_pred.shape[1] == 1:
                E_pred = E_pred[:, 0]
            elif E_pred.ndim != 1:
                raise ValueError(f"Expected E_pred shape (B,) or (B,1), got {tuple(E_pred.shape)}")

            # energy error for metrics
            if cfg.log_e_per_atom:
                B = int(E_true.shape[0])
                counts = (
                    torch.bincount(tensors["ind_1"][:, 0], minlength=B)
                    .to(E_true.dtype)
                    .clamp_min(1.0)
                )
                e_err = ((E_pred / counts) / e_unit) - (E_true / counts)
            else:
                e_err = (E_pred / e_unit) - E_true

            e_sq_sum += float((e_err ** 2).sum().item())
            e_count += int(e_err.numel())

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
        
        writer.add_scalar("eval/E_RMSE", e_rmse, max_steps)
        writer.add_scalar("eval/F_RMSE", f_rmse, max_steps)
        writer.flush()

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
    finally:
        writer.close()    

    return {
        "METRICS/E_RMSE": e_scale * e_rmse,
        "METRICS/F_RMSE": e_scale * f_rmse,
    }




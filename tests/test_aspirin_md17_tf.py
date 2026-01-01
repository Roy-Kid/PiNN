# -*- coding: utf-8 -*-
"""
TF regression test: notebook-params PiNet must beat the zero baseline on rMD17 aspirin.

External requirement:
- aspirin-1000.npz available locally (path via env ASPIRIN_NPZ or repo root)

Everything else is self-contained and matches the Colab notebook params dict.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from ase import Atoms

import pinn
from pinn import get_model, get_network
from pinn.utils import init_params
from pinn.io import load_numpy, sparse_batch
from pinn.calculator import PiNN_calc


def _find_aspirin_npz() -> Path:
    """Locate aspirin-1000.npz via env or by searching upward from tests/."""
    env = os.environ.get("ASPIRIN_NPZ", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p

    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        cand = p / "aspirin-1000.npz"
        if cand.is_file():
            return cand

    pytest.skip("aspirin-1000.npz not found. Set ASPIRIN_NPZ=/path/to/aspirin-1000.npz")


def _zero_baseline_mae(eval_raw: list[dict]) -> tuple[float, float]:
    """
    Zero predictor baseline (E_pred=0, F_pred=0), in dataset label units.

    Returns:
      energy_mae, force_mae_per_component
    """
    e = np.asarray([ex["e_data"] for ex in eval_raw], dtype=float).reshape(-1)
    f = np.asarray([ex["f_data"] for ex in eval_raw], dtype=float)  # (M,N,3) typically
    return float(np.mean(np.abs(e))), float(np.mean(np.abs(f)))  # force MAE is per-component


def _model_mae_via_calc(model, eval_raw: list[dict]) -> tuple[float, float]:
    """
    Evaluate MAE using ASE calculator path (same pathway as MD usage).

    Returns:
      energy_mae, force_mae_per_component
    """
    calc = PiNN_calc(model)

    e_err = []
    f_abs_sum = 0.0
    f_comp = 0

    for ex in eval_raw:
        atoms = Atoms(ex["elems"], ex["coord"])
        atoms.calc = calc

        e_pred = float(atoms.get_potential_energy())
        f_pred = np.asarray(atoms.get_forces(), dtype=float)

        e_true = float(ex["e_data"])
        f_true = np.asarray(ex["f_data"], dtype=float)

        e_err.append(abs(e_pred - e_true))
        f_abs_sum += np.abs(f_pred - f_true).sum()
        f_comp += f_pred.size  # N*3

    return float(np.mean(e_err)), float(f_abs_sum / max(f_comp, 1))


def test_aspirin_rmd17_tf_notebook_params_beats_zero_baseline(tmp_path, monkeypatch):
    """
    Train briefly and assert eval MAEs beat the zero baseline.

    Notes:
    - Uses the exact params dict you pasted from the Colab notebook.
    - No YAML reading/writing.
    - Estimator requires dataset creation inside input_fn (graph ownership).
    """
    monkeypatch.setenv("PINN_BACKEND", "tf")

    npz_path = _find_aspirin_npz()

    # ---- Load and split data deterministically: first 800 train, last 200 eval ----
    ds_all = load_numpy(np.load(npz_path))
    raw = list(ds_all.as_numpy_iterator())
    assert len(raw) >= 1000, f"Expected >=1000 frames, got {len(raw)}"

    train_raw = raw[:800]
    eval_raw = raw[800:1000]

    # ---- Params: EXACTLY as your notebook block (only add model_dir for pytest) ----
    params = {
        "model": {
            "name": "potential_model",
            "params": {
                "e_loss_multiplier": 1.0,
                "f_loss_multiplier": 10.0,
                "log_e_per_atom": True,
                "use_e_per_atom": True,
                "use_force": True,
            },
        },
        "network": {
            "name": "PiNet2",
            "params": {
                "atom_types": [1, 6, 7, 8],
                "basis_type": "gaussian",
                "depth": 5,
                "n_basis": 10,
                "pi_nodes": [16],
                "ii_nodes": [16, 16],
                "pp_nodes": [16, 16],
                "out_nodes": [16],
                "rank": 3,
                "rc": 4.5,
            },
        },
        "optimizer": {
            "class_name": "Adam",
            "config": {
                "global_clipnorm": 0.01,
                "learning_rate": {
                    "class_name": "ExponentialDecay",
                    "config": {
                        "decay_rate": 0.994,
                        "decay_steps": 100000,
                        "initial_learning_rate": 5.0e-05,
                    },
                },
            },
        },
        # pytest-only: isolate artifacts
        "model_dir": str(tmp_path / "pinet_aspirin_tf_notebook_params"),
    }

    print("tmp_path =", tmp_path)
    print("model_dir =", params["model_dir"])

    # ---- On-disk cache (pytest-isolated) ----
    cache_root = tmp_path / "tfdata_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    train_cache = str(cache_root / "train.cache")
    eval_cache  = str(cache_root / "eval.cache")

    # ---- Baseline (label units) ----
    e0, f0 = _zero_baseline_mae(eval_raw)

    # ---- Prepare numpy arrays (safe to capture in closures) ----
    train_coord = np.asarray([r["coord"] for r in train_raw], dtype=np.float32)
    train_elems = np.asarray([r["elems"] for r in train_raw], dtype=np.int32)
    train_e_data = np.asarray([r["e_data"] for r in train_raw], dtype=np.float32)
    train_f_data = np.asarray([r["f_data"] for r in train_raw], dtype=np.float32)

    eval_coord = np.asarray([r["coord"] for r in eval_raw], dtype=np.float32)
    eval_elems = np.asarray([r["elems"] for r in eval_raw], dtype=np.int32)
    eval_e_data = np.asarray([r["e_data"] for r in eval_raw], dtype=np.float32)
    eval_f_data = np.asarray([r["f_data"] for r in eval_raw], dtype=np.float32)

    # ---- init_params as in notebook (needs a dataset it can iterate) ----
    train_ds_init = load_numpy(
        {
            "coord": train_coord,
            "elems": train_elems,
            "e_data": train_e_data,
            "f_data": train_f_data,
        }
    )
    init_params(params, train_ds_init)

    # ---- Estimator input_fns: dataset must be built inside input_fn ----
    def train_input_fn():
        """Estimator input_fn: dataset pipeline built inside the current graph."""
        ds = tf.data.Dataset.from_tensor_slices(
            {
                "coord": train_coord,
                "elems": train_elems,
                "e_data": train_e_data,
                "f_data": train_f_data,
            }
        )
        ds = ds.apply(sparse_batch(1))

        def pre_fn(tensors):
            network = get_network(params["network"])
            return network.preprocess(tensors)

        ds = ds.map(pre_fn)

        # --- caching (Colab-like) ---
        ds = ds.cache(train_cache)

        ds = ds.shuffle(1000).repeat()

        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def eval_input_fn():
        """Estimator input_fn: eval dataset pipeline inside the current graph."""
        ds = tf.data.Dataset.from_tensor_slices(
            {
                "coord": eval_coord,
                "elems": eval_elems,
                "e_data": eval_e_data,
                "f_data": eval_f_data,
            }
        )
        ds = ds.apply(sparse_batch(1))

        def pre_fn(tensors):
            network = get_network(params["network"])
            return network.preprocess(tensors)

        ds = ds.map(pre_fn)

        # --- caching (Colab-like) ---
        ds = ds.cache(eval_cache)

        return ds

    # ---- Train (short run; configurable) ----
    max_steps = int(os.environ.get("ASPIRIN_TF_MAX_STEPS", "1000"))
    eval_steps = int(os.environ.get("ASPIRIN_TF_EVAL_STEPS", "200"))

    config = tf.estimator.RunConfig(
        keep_checkpoint_max=1,
        save_checkpoints_steps=200,
        log_step_count_steps=200,
        save_summary_steps=200,
    )
    model = get_model(params, config=config)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    # ---- Evaluate using calculator path ----
    e_mae, f_mae = _model_mae_via_calc(model, eval_raw)

    # Convert kcal/mol -> meV (per molecule) using 1 kcal/mol = 43.3641153088 meV
    KCALMOL_TO_MEV = 43.3641153088

    energy_mae_meV = e_mae * KCALMOL_TO_MEV
    force_mae_meV_per_A = f_mae * KCALMOL_TO_MEV

    print(f"Energy MAE: {energy_mae_meV:.2f} meV")
    print(f"Force  MAE: {force_mae_meV_per_A:.2f} meV/Å")

    # ---- Assert: beats baseline ----
    # With the small LR in the notebook params, improvement can be modest at low steps.
    assert e_mae < 0.70 * e0, f"Energy MAE {e_mae} did not beat zero baseline {e0} by 30%"
    assert f_mae < 0.70 * f0, f"Force MAE {f_mae} did not beat zero baseline {f0} by 30%"
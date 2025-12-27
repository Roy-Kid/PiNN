# -*- coding: utf-8 -*-

import tempfile, os, pytest
import tensorflow as tf
import numpy as np
import pinn
from shutil import rmtree


@pytest.mark.forked
@pytest.mark.parametrize("backend", ["tf", "torch"])
def test_potential_model(backend, monkeypatch, tmp_path):
    """Train+eval a tiny potential model on both backends using the same LJ toy data."""
    monkeypatch.setenv("PINN_BACKEND", backend)

    # Reuse the same dataset generator as tests/test_potential.py
    from test_potential import _get_lj_data  #

    data = _get_lj_data()
    model_dir = str(tmp_path / f"pinn_test_{backend}")

    params = {
        "model_dir": model_dir,
        "network": {
            "name": "PiNet",   # keep PiNet here (matches your original intention)
            "params": {
                "ii_nodes": [8, 8],
                "pi_nodes": [8, 8],
                "pp_nodes": [8, 8],
                "out_nodes": [8, 8],
                "rc": 3.0,
                "atom_types": [1],
            },
        },
        "model": {
            "name": "potential_model",
            "params": {"use_force": True},
        },
    }

    model = pinn.get_model(params)

    if backend == "tf":
        tf = pytest.importorskip("tensorflow")
        from pinn.io import load_numpy, sparse_batch

        def train():
            ds = load_numpy(data)  # IMPORTANT: construct inside input_fn (graph context)
            return ds.repeat().shuffle(500).apply(sparse_batch(50))

        def test():
            ds = load_numpy(data)  # IMPORTANT: construct inside input_fn (graph context)
            return ds.repeat().apply(sparse_batch(10))

        train_spec = tf.estimator.TrainSpec(input_fn=train, max_steps=200)
        eval_spec = tf.estimator.EvalSpec(input_fn=test, steps=10)
        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    else:
        # Torch backend path: use torch runtime (expects numpy dict like _get_lj_data()).
        pinn.train_and_evaluate(
            model=model,
            params=params,
            data=data,
            max_steps=200,
            eval_steps=10,
            batch_size_train=50,
            batch_size_eval=10,
            shuffle_buffer=500,
        )


@pytest.mark.forked
def test_derivitives():
    """Test the calcualted derivitives: forces and stress
    with a LJ calculator against ASE's implementation
    """
    from ase.calculators.lj import LennardJones
    from ase.collections import g2
    from ase.build import bulk
    import numpy as np

    params = {
        "model_dir": "/tmp/pinn_test/lj",
        "network": {"name": "LJ", "params": {"rc": 3}},
        "model": {"name": "potential_model", "params": {}},
    }
    pi_lj = pinn.get_calc(params)
    test_set = [bulk("Cu").repeat([3, 3, 3]), bulk("Mg"), g2["H2O"]]
    np.random.seed(0)
    for atoms in test_set:
        pos = atoms.get_positions()
        atoms.set_positions(pos + np.random.uniform(0, 0.2, pos.shape))
        atoms.set_calculator(pi_lj)
        f_pinn, e_pinn = atoms.get_forces(), atoms.get_potential_energy()
        atoms.set_calculator(LennardJones())
        f_ase, e_ase = atoms.get_forces(), atoms.get_potential_energy()
        assert np.allclose(f_pinn, f_ase, rtol=1e-2)
        assert np.allclose(e_pinn, e_ase, rtol=1e-2)
        assert np.abs(e_pinn - e_ase) < 1e-3
        if np.any(atoms.pbc):
            atoms.set_calculator(pi_lj)
            s_pinn = atoms.get_stress()
            atoms.set_calculator(LennardJones())
            s_ase = atoms.get_stress()
            assert np.allclose(s_pinn, s_ase, rtol=1e-2)


@pytest.mark.forked
@pytest.mark.parametrize("backend", ["tf", "torch"])
def test_clist_nl(backend, monkeypatch):
    """Cell list neighbor test: compare with ASE implementation (TF and Torch)."""
    monkeypatch.setenv("PINN_BACKEND", backend)

    from ase.build import bulk
    from ase.neighborlist import neighbor_list
    import numpy as np

    rc = 10.0
    to_test = [bulk("Cu"), bulk("Mg"), bulk("Fe")]

    if backend == "tf":
        tf = pytest.importorskip("tensorflow")
        from pinn.layers import CellListNL

        ind, coord, cell = [], [], []
        for i, a in enumerate(to_test):
            ind.append([[i]] * len(a))
            coord.append(a.positions)
            cell.append(a.cell.array)

        with tf.Graph().as_default():
            tensors = {
                "ind_1": tf.constant(np.concatenate(ind, axis=0), tf.int32),
                "coord": tf.constant(np.concatenate(coord, axis=0), tf.float32),
                "cell": tf.constant(np.stack(cell, axis=0), tf.float32),
            }
            nl = CellListNL(rc=rc)(tensors)
            with tf.compat.v1.Session() as sess:
                dist_pinn = sess.run(nl["dist"])

    else:
        import torch
        from pinn.networks.pinet_torch import CellListNLPyTorch

        def _shifted_dists(
            coord: torch.Tensor,
            ind_2: torch.Tensor,
            shift: torch.Tensor,
            cell: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute pair distances using explicit periodic image shifts.

            coord: (N,3)
            ind_2: (M,2) with (i,j)
            shift: (M,3) integer shifts (sx,sy,sz) such that
                displacement = (r_j + shift @ cell) - r_i
            cell:  (3,3) lattice vectors (ASE convention: rows are vectors)
            """
            i = ind_2[:, 0]
            j = ind_2[:, 1]
            # translation vectors in Cartesian: (M,3) = shift.float() @ cell
            t = shift.to(coord.dtype) @ cell
            d = (coord[j] + t) - coord[i]
            return torch.linalg.norm(d, dim=-1)

        cl = CellListNLPyTorch(rc=rc)
        dists = []

        for a in to_test:
            coord = torch.tensor(a.positions, dtype=torch.float32)
            cell = torch.tensor(a.cell.array, dtype=torch.float32)
            nl = cl(coord=coord, cell=cell)
            ind_2 = nl["ind_2"]
            shift = nl.get("shift", None)

            if ind_2.numel() == 0:
                continue

            if shift is None:
                # fallback: non-PBC style (shouldn't happen here since we pass cell)
                d = coord[ind_2[:, 1]] - coord[ind_2[:, 0]]
                dist = torch.linalg.norm(d, dim=-1)
            else:
                dist = _shifted_dists(coord, ind_2, shift, cell)
            dists.append(dist.cpu().numpy())

        dist_pinn = np.concatenate(dists, axis=0) if dists else np.array([], dtype=float)

    # ASE reference distances
    dist_ase = []
    for a in to_test:
        dist_ase.append(neighbor_list("d", a, rc))
    dist_ase = np.concatenate(dist_ase, axis=0)

    assert np.allclose(np.sort(dist_ase), np.sort(dist_pinn), rtol=1e-2)

@pytest.mark.forked
@pytest.mark.parametrize("backend", ["tf", "torch"])
def test_input_yml(backend, monkeypatch):
    """Ensure params-dict with optimizer block can build a model on both backends."""
    monkeypatch.setenv("PINN_BACKEND", backend)
    if backend == "tf":
        pytest.importorskip("tensorflow")

    from pinn import get_model

    params = {
        "model_dir": "/tmp",
        "model": {
            "name": "potential_model",
            "params": {
                "use_force": True,
                "e_loss_multiplier": 1.0,
                "f_loss_multiplier": 10.0,
                "use_e_per_atom": False,
                "log_e_per_atom": True,
                "e_scale": 1.0,
                "e_unit": 1.0,
            },
        },
        "network": {
            "name": "PiNet",
            "params": {
                "atom_types": [1],
                "rc": 3.0,
                "ii_nodes": [8, 8],
                "pi_nodes": [8, 8],
                "pp_nodes": [8, 8],
                "out_nodes": [8, 8],
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
                        "decay_steps": 10000,
                        "initial_learning_rate": 1.0e-4,
                    },
                },
            },
        },
    }

    assert get_model(params)

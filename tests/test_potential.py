# -*- coding: utf-8 -*-
"""Numerical tests for virials and energy conservation of potentials"""
import tempfile
import os
import pytest
import numpy as np
import tensorflow as tf
from pinn.io import load_numpy, sparse_batch
from shutil import rmtree
from ase import Atoms

#pytest -k "test_pinet_potential and torch"

@pytest.mark.forked
@pytest.mark.parametrize("backend", ["tf", "torch"])
def test_pinet_potential(backend, tmp_path, monkeypatch):
    # Select backend for pinn.get_model / pinn.get_calc factories
    monkeypatch.setenv("PINN_BACKEND", backend)

    # Separate model_dir per backend so checkpoints don't collide
    testpath = tmp_path / backend

    network_params = {
        'ii_nodes': [8, 8],
        'pi_nodes': [8, 8],
        'pp_nodes': [8, 8],
        'out_nodes': [8, 8],
        'depth': 3,
        'rc': 5.,
        'n_basis': 5,
        'atom_types': [1],
    }

    params = {
        'model_dir': str(testpath),
        'network': {'name': 'PiNet', 'params': network_params},
        'model': {
            'name': 'potential_model',
            'params': {
                'use_force': True,
                'e_dress': {1: 0.5},
                'e_scale': 5.0,
                'e_unit': 2.0,
            },
        },
        # Optional but nice: also carry backend in params
        'backend': backend,
    }

    _potential_tests(params)

@pytest.mark.forked
@pytest.mark.parametrize("backend", ["tf", "torch"])
def test_pinet2_p3_potential(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PINN_BACKEND", backend)
    # Separate model_dir per backend so checkpoints don't collide
    testpath = tmp_path / backend

    network_params = {
        "ii_nodes": [8, 8],
        "pi_nodes": [8, 8],
        "pp_nodes": [8, 8],
        "out_nodes": [8, 8],
        "depth": 3,
        "rc": 5.0,
        "n_basis": 5,
        "atom_types": [1],
        "rank": 3,
    }
    params = {
        "model_dir": str(testpath),
        "network": {"name": "PiNet2", "params": network_params},
        "model": {
            "name": "potential_model",
            "params": {"use_force": True, "e_dress": {1: 0.5}, "e_scale": 5.0, "e_unit": 2.0},
        },
    }
    _potential_tests(params)

@pytest.mark.forked
@pytest.mark.parametrize("virial_mode", ["dist", "diff", "cell", "fd"])
def test_pinet2_p3_potential_virial_modes(tmp_path, monkeypatch, virial_mode):
    monkeypatch.setenv("PINN_BACKEND", "torch")
    testpath = tmp_path / f"torch_virial_{virial_mode}"

    network_params = {
        "ii_nodes": [8, 8],
        "pi_nodes": [8, 8],
        "pp_nodes": [8, 8],
        "out_nodes": [8, 8],
        "depth": 3,
        "rc": 5.0,
        "n_basis": 5,
        "atom_types": [1],
        "rank": 3,
    }
    params = {
        "model_dir": str(testpath),
        "network": {"name": "PiNet2", "params": network_params},
        "model": {
            "name": "potential_model",
            "params": {
                "use_force": True,
                "e_dress": {1: 0.5},
                "e_scale": 5.0,
                "e_unit": 2.0,
                "virial_mode": virial_mode,   # torch-only knob
            },
        },
    }

    _potential_tests(params)    

@pytest.mark.forked
@pytest.mark.parametrize("virial_mode", ["dist", "diff", "cell", "fd"])
def test_pinet2_p3_potential_torsion_boost(tmp_path, monkeypatch, virial_mode):
    """TB-enabled PiNet2: run the same potential tests under all torch virial modes."""
    monkeypatch.setenv("PINN_BACKEND", "torch")

    # Separate model dirs so checkpoints don't collide across modes
    testpath = tmp_path / f"torch_torsion_boost_virial_{virial_mode}"

    network_params = {
        "ii_nodes": [8, 8],
        "pi_nodes": [8, 8],
        "pp_nodes": [8, 8],
        "out_nodes": [8, 8],
        "depth": 3,
        "rc": 5.0,
        "n_basis": 5,
        "atom_types": [1],
        "rank": 3,
        "torsion_boost": True,
        # Torch-only: choose stress/virial implementation
        "virial_mode": virial_mode,
    }

    params = {
        "model_dir": str(testpath),
        "network": {"name": "PiNet2", "params": network_params},
        "model": {
            "name": "potential_model",
            "params": {
                "use_force": True,
                "e_dress": {1: 0.5},
                "e_scale": 5.0,
                "e_unit": 2.0,
            },
        },
    }

    _potential_tests(params)

@pytest.mark.forked
def test_pinet2_p5_potential():
    testpath = tempfile.mkdtemp()
    network_params = {
        'ii_nodes': [8, 8],
        'pi_nodes': [8, 8],
        'pp_nodes': [8, 8],
        'out_nodes': [8, 8],
        'depth': 3,
        'rc': 5.,
        'n_basis': 5,
        'atom_types': [1],
        'rank': 5
    }
    params = {
        'model_dir': testpath,
        'network': {
            'name': 'PiNet2',
            'params': network_params},
        'model': {
            'name': 'potential_model',
            'params': {
                'use_force': True,
                'e_dress': {1: 0.5},
                'e_scale': 5.0,
                'e_unit': 2.0}}}
    _potential_tests(params)
    rmtree(testpath)

@pytest.mark.forked
def test_bpnn_potential():
    testpath = tempfile.mkdtemp()
    network_params = {
        'sf_spec': [
            {'type': 'G2', 'i': 1, 'j': 1, 'eta': [
                0.1, 0.1, 0.1], 'Rs': [1., 2., 3.]},
            {'type': 'G3', 'i': 1, 'j': 1, 'k': 1,
             'eta': [0.1, 0.1, 0.1, 0.1], 'lambd': [1., 1., -1., -1.], 'zeta':[1., 1., 4., 4.]},
            {'type': 'G4', 'i': 1, 'j': 1, 'k': 1,
             'eta': [0.1, 0.1, 0.1, 0.1], 'lambd': [1., 1., -1., -1.], 'zeta':[1., 1., 4., 4.]}
        ],
        'nn_spec': {1: [8, 8]},
        'rc': 5.,
    }
    params = {
        'model_dir': testpath,
        'network': {
            'name': 'BPNN',
            'params': network_params},
        'model': {
            'name': 'potential_model',
            'params': {
                'use_force': True,
                'e_dress': {1: 0.5},
                'e_scale': 5.0,
                'e_unit': 2.0}}}
    _potential_tests(params)
    rmtree(testpath)


@pytest.mark.forked
def test_bpnn_potential_pre_cond():
    from pinn.networks.bpnn import BPNN
    testpath = tempfile.mkdtemp()
    network_params = {
        'sf_spec': [
            {'type': 'G2', 'i': 1, 'j': 1, 'eta': [
                0.1, 0.1, 0.1], 'Rs': [1., 2., 3.]},
            {'type': 'G3', 'i': 1, 'j': 1, 'k': 1,
             'eta': [0.1, 0.1, 0.1, 0.1], 'lambd': [1., 1., -1., -1.], 'zeta':[1., 1., 4., 4.]},
            {'type': 'G4', 'i': 1, 'j': 1, 'k': 1,
             'eta': [0.1, 0.1, 0.1, 0.1], 'lambd': [1., 1., -1., -1.], 'zeta':[1., 1., 4., 4.]}
        ],
        'nn_spec': {1: [8, 8]},
        'rc': 5.
    }
    bpnn =  BPNN(**network_params)
    dataset = load_numpy(_get_lj_data())\
        .apply(sparse_batch(10)).map(bpnn.preprocess)

    batches = [tensors for tensors in dataset]
    fp_range = []
    for i in range(len(network_params['sf_spec'])):
        fp_max = max([b[f'fp_{i}'].numpy().max() for b in batches])
        fp_min = max([b[f'fp_{i}'].numpy().min() for b in batches])
        fp_range.append([float(fp_min), float(fp_max)])

    network_params['fp_scale'] = True
    network_params['fp_range'] = fp_range
    params = {
        'model_dir': testpath,
        'network': {
            'name': 'BPNN',
            'params': network_params},
        'model': {
            'name': 'potential_model',
            'params': {
                'use_force': True,
                'e_dress': {1: 0.5},
                'e_scale': 5.0,
                'e_unit': 2.0}}}
    _potential_tests(params)
    rmtree(testpath)


def _get_lj_data():
    from ase.calculators.lj import LennardJones

    atoms = Atoms('H3', positions=[[0, 0, 0], [0, 1, 0], [1, 1, 0]])
    atoms.set_calculator(LennardJones(rc=5.0))
    coord, elems, e_data, f_data = [], [], [], []
    for x_a in np.linspace(-5, 0, 1000):
        atoms.positions[0, 0] = x_a
        coord.append(atoms.positions.copy())
        elems.append(atoms.numbers)
        e_data.append(atoms.get_potential_energy())
        f_data.append(atoms.get_forces())

    data = {
        'coord': np.array(coord),
        'elems': np.array(elems),
        'e_data': np.array(e_data),
        'f_data': np.array(f_data)
    }
    return data


def _potential_tests(params):
    # Series of tasks that a potential should pass
    import os
    import pinn

    

    data = _get_lj_data()

    def train(): return load_numpy(data).repeat().shuffle(
        500).apply(sparse_batch(50))

    def test(): return load_numpy(data).apply(sparse_batch(10))


    backend = os.environ.get("PINN_BACKEND", "tf").lower()
    print("PINN_BACKEND seen by _potential_tests:", backend)
    model = pinn.get_model(params)


    if backend == "tf":
        train_spec = tf.estimator.TrainSpec(input_fn=train, max_steps=1e3)
        eval_spec = tf.estimator.EvalSpec(input_fn=test, steps=100)
        results, _ = tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    elif backend == "torch":
        # Torch path: you will implement this function + signature
        # Minimal inputs: raw data dict + step counts + batch sizes used in the TF pipeline
        results = pinn.train_and_evaluate(
            model=model,
            params=params,
            data=data,
            max_steps=1000,
            eval_steps=100,
            batch_size_train=50,
            batch_size_eval=10,
            shuffle_buffer=500,
        )
    else:
        raise ValueError(f"Unknown PINN_BACKEND={backend!r}")

    # The calculator should be accessable with model_dir
    atoms = Atoms('H3', positions=[[0, 0, 0], [0, 1, 0], [1, 1, 0]])
    calc = pinn.get_calc(params, properties=['energy', 'forces', 'stress'])

    # Test energy dress and scaling
    # Make sure we have the correct error reports
    e_pred, f_pred = [], []
    for coord in data['coord']:
        atoms.positions = coord
        calc.calculate(atoms)
        e_pred.append(calc.get_potential_energy())
        f_pred.append(calc.get_forces())

    f_pred = np.array(f_pred)
    e_pred = np.array(e_pred)

    assert np.allclose(results['METRICS/F_RMSE']/params['model']['params']['e_scale'],
                        np.sqrt(np.mean((f_pred/params['model']['params']['e_unit']
                                         - data['f_data'])**2)), rtol=1e-2)
    assert np.allclose(results['METRICS/E_RMSE']/params['model']['params']['e_scale'],
                       np.sqrt(np.mean((e_pred/params['model']['params']['e_unit']
                                        - data['e_data'])**2)), rtol=1e-2)

    # Test energy conservation
    e_pred, f_pred = [], []
    # Keep the neighbor graph constant (avoid crossing rc where pairs appear/disappear).
    # For H3 with rc=5, x in [-3.8, -3.0] keeps both relevant pairs inside the cutoff.
    #x_a_range = np.linspace(-3.8, -3.0, 500)
    x_a_range = np.linspace(-6, -3, 500)
    for x_a in x_a_range:
        atoms.positions[0, 0] = x_a
        calc.calculate(atoms)
        e_pred.append(calc.get_potential_energy())
        f_pred.append(calc.get_forces())
    e_pred = np.array(e_pred)
    f_pred = np.array(f_pred)

    # --- Localize energy-force inconsistency along the path ---
    x = x_a_range
    E = e_pred
    Fx = f_pred[:, 0, 0]  # force on atom 0 along x

    # cumulative trapezoid integral: I[k] = ∫_{x0}^{xk} Fx dx
    dx = np.diff(x)
    I = np.zeros_like(E, dtype=float)
    I[1:] = np.cumsum(0.5 * (Fx[:-1] + Fx[1:]) * dx)

    dE = (E - E[0]).astype(float)
    minus_I = -I

    res = dE - minus_I  # should be ~0 everywhere if consistent
    dres = np.diff(res)
    print("[Energy-conservation debug] max |Δres| per step:", float(np.max(np.abs(dres))))

    # A robust scale to judge relative size (avoid divide-by-near-zero)
    scale = max(1e-12, float(np.max(np.abs(dE))), float(np.max(np.abs(minus_I))))

    # Find worst point
    k = int(np.argmax(np.abs(res)))
    print("\n[Energy-conservation debug]")
    print("worst k:", k, "x:", x[k])
    print("E[k]-E[0]:", dE[k], " -∫F dx:", minus_I[k], " residual:", res[k])
    print("max |res|:", float(np.max(np.abs(res))), " relative:", float(np.max(np.abs(res)) / scale))

    # Optionally: also print a small neighborhood around the worst point
    k0 = max(0, k-3)
    k1 = min(len(x), k+4)
    print("Neighborhood (k, x, E, Fx, res):")
    for kk in range(k0, k1):
        print(kk, x[kk], E[kk], Fx[kk], res[kk])
    print()


    de = e_pred[-1] - e_pred[0]
    int_f = np.trapz(f_pred[:, 0, 0], x=x_a_range)
    print("[Energy-conservation summary] de:", de, " -int_f:", -int_f, " diff:", de - (-int_f))
    assert np.allclose(de, -int_f, rtol=1e-2)

    # Test virial pressure
    e_pred, p_pred = [], []
    l_range = np.linspace(3, 3.5, 500)
    atoms.positions[0, 0] = 0
    atoms.set_cell([3, 3, 3])
    atoms.set_pbc(True)
    for l in l_range:
        atoms.set_cell([l, l, l], scale_atoms=True)
        calc.calculate(atoms)
        e_pred.append(calc.get_potential_energy())
        p_pred.append(np.sum(calc.get_stress()[:3])/3)

    de = e_pred[-1] - e_pred[0]
    int_p = np.trapz(p_pred, x=l_range**3)
    assert np.allclose(de, int_p, rtol=1e-2)

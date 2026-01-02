import numpy as np
import pinn
import pytest
from ase import Atoms

@pytest.mark.forked
def test_torch_calc_reload_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("PINN_BACKEND", "torch")

    params = {
        "model_dir": str(tmp_path),
        "network": {"name": "PiNet", "params": {"atom_types": [1], "rc": 5.0, "n_basis": 5, "depth": 2,
                                               "pp_nodes": [8,8], "pi_nodes": [8,8], "ii_nodes": [8,8], "out_nodes": [8,8]}},
        "model": {"name": "potential_model", "params": {"e_dress": {1: 0.5}, "e_scale": 5.0, "e_unit": 2.0, "use_force": True}},
        "backend": "torch",
    }

    model = pinn.get_model(params)

    # Save a checkpoint the same way runtime does
    import torch, os
    os.makedirs(params["model_dir"], exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "params": params}, os.path.join(params["model_dir"], "model.pt"))

    atoms = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]])
    calc = pinn.get_calc(params, properties=["energy", "forces", "stress"])
    atoms.calc = calc

    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    assert np.isfinite(e)
    assert np.all(np.isfinite(f))

# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import tempfile
from shutil import rmtree

import pytest

from helpers import *


@pytest.mark.forked
def test_cli_train_torch_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("PINN_BACKEND", "torch")

    from pinn.io import write_tfrecord

    tmp = tempfile.mkdtemp(prefix="pinn_cli_torch_smoke")
    try:
        train_yml = f"{tmp}/train.yml"
        eval_yml = f"{tmp}/eval.yml"

        ds = get_trivial_runner_ds().repeat(10)
        write_tfrecord(train_yml, ds)
        write_tfrecord(eval_yml, ds)

        model_dir = str(tmp_path / "model")

        # Minimal params yaml written to disk (CLI consumes file)
        params_yml = f"{tmp}/params.yml"
        with open(params_yml, "w", encoding="utf-8") as f:
            f.write(f"""
model_dir: {model_dir}
network:
  name: PiNet
  params:
    ii_nodes: [8, 8]
    pi_nodes: [8, 8]
    pp_nodes: [8, 8]
    out_nodes: [8, 8]
    rc: 3.0
    atom_types: [1]
model:
  name: potential_model
  params:
    use_force: true
""")

        # Call CLI train
        cmd = [
            sys.executable, "-m", "pinn.cli",
            "train",
            "-d", model_dir,
            "-t", train_yml,     # train-ds
            "-e", eval_yml,      # eval-ds
            "-b", "2",           # batch size (train)
            "--eval-bs", "2",    # batch size (eval); optional but makes intent explicit
            "--train-steps", "2",
            "--eval-steps", "1",
            "--shuffle-buffer", "4",
            params_yml,          # <-- positional 'params' at the end
        ]
        env = dict(os.environ)
        env["PINN_BACKEND"] = "torch"
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert res.returncode == 0, res.stderr

        # Check checkpoint exists
        assert os.path.exists(os.path.join(model_dir, "model.pt"))
    finally:
        rmtree(tmp, ignore_errors=True)

# -*- coding: utf-8 -*-
"""Tests for backend-neutral dataset builder (spec + build)."""

import tempfile
from shutil import rmtree

import numpy as np
import pytest

from helpers import *


def test_build_dataset_torch_from_tfrecord_yml():
    """
    Build Torch iterable from a legacy TFRecord YAML.

    This tests:
    - spec parsing (kind=tfrecord)
    - torch build path (iterates tf.data.Dataset via as_numpy_iterator)
    - optional disk caching (exhaust once to finalize cache, then read again)
    """
    from pinn.io import write_tfrecord
    from pinn.io.build_dataset import build_dataset, BuildOptions

    tmp = tempfile.mkdtemp(prefix="pinn_build_ds_test")
    try:
        yml_path = f"{tmp}/ds.yml"
        ds = get_trivial_runner_ds().repeat(3)
        write_tfrecord(yml_path, ds)

        # 1) Build Torch iterable with disk cache
        opt = BuildOptions(backend="torch", cache=True, scratch_dir=tmp, cache_ram=False)
        it = build_dataset(yml_path, options=opt, dataset_role="train")

        ex_list = list(it)  # must exhaust once to finalize meta.yml
        assert len(ex_list) > 0

        # 2) Build again (should hit disk cache)
        it2 = build_dataset(yml_path, options=opt, dataset_role="train")
        ex_list2 = list(it2)
        assert len(ex_list2) == len(ex_list)

        # Compare first example numerically
        a = ex_list[0]
        b = ex_list2[0]
        assert set(a.keys()) == set(b.keys())
        for k in a.keys():
            assert np.allclose(np.asarray(a[k]), np.asarray(b[k]))
    finally:
        rmtree(tmp, ignore_errors=True)

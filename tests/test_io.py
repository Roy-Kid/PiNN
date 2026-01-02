# -*- coding: utf-8 -*-
"""Some simple test that dataset can be loaded"""

import os
import pytest, tempfile
import numpy as np
from shutil import rmtree

from helpers import *


def test_numpy():
    dataset = get_trivial_numpy_ds()
    data = get_trivial_numpy()
    iterator = iter(dataset)
    out = next(iterator)
    for k in out.keys():
        assert np.allclose(out[k], data[k])
    with pytest.raises(StopIteration):
        _ = next(iterator)


def test_qm9():
    dataset = get_trivial_qm9_ds()
    data = get_trivial_numpy()
    iterator = iter(dataset)
    out = next(iterator)
    for k in out.keys():
        assert np.allclose(out[k], data[k])
    with pytest.raises(StopIteration):
        _ = next(iterator)


def test_runner():
    dataset = get_trivial_runner_ds()
    bohr2ang = 0.5291772109
    data = get_trivial_numpy()
    data["coord"] *= bohr2ang
    data["cell"] *= bohr2ang
    iterator = iter(dataset)
    out = next(iterator)
    for k in data.keys():  # RuNNer has many labels, we do not test all of them
        assert np.allclose(out[k], data[k])
    with pytest.raises(StopIteration):
        _ = next(iterator)


def test_split():
    # Test that dataset is splitted according to the given ratio
    from pinn.io import load_numpy

    data = get_trivial_numpy()
    data = {k: np.stack([[v]] * 10, axis=0) for k, v in data.items()}
    dataset = load_numpy(data, splits={"train": 8, "test": 2})
    train = iter(dataset["train"])
    test = iter(dataset["test"])

    for _ in range(8):
        _ = next(train)
    with pytest.raises(StopIteration):
        _ = next(train)

    for _ in range(2):
        _ = next(test)
    with pytest.raises(StopIteration):
        _ = next(test)


def test_write():
    import os
    from pinn.io import load_tfrecord, write_tfrecord, sparse_batch

    tmp = tempfile.mkdtemp(prefix="pinn_test")
    try:
        ds = get_trivial_runner_ds().repeat(20)
        write_tfrecord("{}/test.yml".format(tmp), ds)
        ds_tfr = load_tfrecord("{}/test.yml".format(tmp))

        ds_batch = ds.apply(sparse_batch(20))
        write_tfrecord("{}/test_batch.yml".format(tmp), ds_batch)
        ds_batch_tfr = load_tfrecord("{}/test_batch.yml".format(tmp))

        label = next(iter(ds))
        out = next(iter(ds_tfr))
        for k in out.keys():
            assert np.allclose(label[k], out[k])

        label = next(iter(ds_batch))
        out = next(iter(ds_batch_tfr))
        for k in out.keys():
            assert np.allclose(label[k], out[k])
    finally:
        rmtree(tmp, ignore_errors=True)


def test_torch_npz_cache_roundtrip():
    """
    Torch-backend cache test (NPZ shards) to preserve caching feature
    without relying on TFRecord.

    This does not replace TFRecord caching; it complements it.
    """
    from pinn.io.iter_utils import iter_numpy_examples
    from pinn.io.cache import CacheConfig, CachedExamples, fingerprint_spec

    tmp = tempfile.mkdtemp(prefix="pinn_torch_cache_test")
    try:
        ds = get_trivial_runner_ds().repeat(3)

        spec = {"loader": "trivial_runner", "repeat": 3}
        key = fingerprint_spec(spec)

        # Pass 1: build disk cache (must consume to completion to finalize meta.yml)
        cached1 = CachedExamples(
            iter_numpy_examples(ds),
            CacheConfig(enabled=True, ram=False, scratch_dir=tmp, key=key),
            meta_spec=spec,
        )

        # Exhaust once to force cache finalization (writes meta.yml at end)
        ex_list = list(cached1)
        assert len(ex_list) > 0
        ex0 = ex_list[0]

        cache_dir = os.path.join(tmp, key)
        assert os.path.exists(os.path.join(cache_dir, "meta.yml"))
        assert os.path.exists(os.path.join(cache_dir, "ex_00000000.npz"))

        # Pass 2: read from disk cache even if the source is empty
        cached2 = CachedExamples(
            iter_numpy_examples([]),
            CacheConfig(enabled=True, ram=False, scratch_dir=tmp, key=key),
            meta_spec=spec,
        )
        ex0b = next(iter(cached2))

        assert set(ex0b.keys()) == set(ex0.keys())
        for k in ex0.keys():
            v1 = ex0[k]
            v2 = ex0b[k]
            if isinstance(v1, (np.ndarray, np.generic)) or isinstance(v2, (np.ndarray, np.generic)):
                assert np.allclose(np.asarray(v1), np.asarray(v2))
            elif isinstance(v1, (int, float, bool)) and isinstance(v2, (int, float, bool)):
                assert v1 == v2
    finally:
        rmtree(tmp, ignore_errors=True)

# -*- coding: utf-8 -*-
"""Training-precision YAML parameter"""
import os
import tempfile

import pytest
import tensorflow as tf
import yaml
from tensorflow.python.lib.io.file_io import FileIO

from pinn.models.base import apply_precision


@pytest.mark.forked
def test_unknown_precision_raises():
    with pytest.raises(ValueError, match='Unknown precision'):
        apply_precision('fp8')


@pytest.mark.forked
def test_fp32_is_the_default_policy():
    apply_precision('fp32')
    assert tf.keras.mixed_precision.global_policy().name == 'float32'
    apply_precision(None)
    assert tf.keras.mixed_precision.global_policy().name == 'float32'


@pytest.mark.parametrize('name,policy', [
    ('fp32', 'float32'),
    ('float32', 'float32'),
    ('fp16', 'mixed_float16'),
    ('bf16', 'mixed_bfloat16'),
])
@pytest.mark.forked
def test_apply_precision_sets_policy(name, policy):
    apply_precision(name)
    assert tf.keras.mixed_precision.global_policy().name == policy
    apply_precision('fp32')


@pytest.mark.forked
def test_default_params_record_fp32():
    import pinn
    testpath = tempfile.mkdtemp()
    params = {
        'model_dir': testpath,
        'network': {
            'name': 'PiNet',
            'params': {
                'ii_nodes': [4, 4],
                'pi_nodes': [4, 4],
                'pp_nodes': [4, 4],
                'out_nodes': [4],
                'depth': 2,
                'rc': 4.0,
                'n_basis': 4,
                'atom_types': [1]}},
        'model': {
            'name': 'potential_model',
            'params': {'use_force': False}}}
    pinn.get_model(params)
    with FileIO(os.path.join(testpath, 'params.yml'), 'r') as f:
        saved = yaml.load(f, Loader=yaml.Loader)
    assert saved.get('precision', 'fp32') == 'fp32'
    apply_precision('fp32')


@pytest.mark.forked
def test_fp16_precision_trains():
    import numpy as np
    import pinn
    from pinn.io import load_numpy, sparse_batch

    testpath = tempfile.mkdtemp()
    n = 8
    data = {
        'coord': np.random.randn(n, 3, 3).astype(np.float32),
        'elems': np.ones((n, 3), dtype=np.int32),
        'e_data': np.random.randn(n).astype(np.float32),
    }
    params = {
        'model_dir': testpath,
        'precision': 'fp16',
        'network': {
            'name': 'PiNet',
            'params': {
                'ii_nodes': [4, 4],
                'pi_nodes': [4, 4],
                'pp_nodes': [4, 4],
                'out_nodes': [4],
                'depth': 2,
                'rc': 4.0,
                'n_basis': 4,
                'atom_types': [1]}},
        'model': {
            'name': 'potential_model',
            'params': {'use_force': False}}}

    def train():
        return load_numpy(data).repeat().shuffle(n).apply(sparse_batch(4))

    def test():
        return load_numpy(data).apply(sparse_batch(4))

    model = pinn.get_model(params)
    train_spec = tf.estimator.TrainSpec(input_fn=train, max_steps=2)
    eval_spec = tf.estimator.EvalSpec(input_fn=test, steps=1)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    with FileIO(os.path.join(testpath, 'params.yml'), 'r') as f:
        saved = yaml.load(f, Loader=yaml.Loader)
    assert saved['precision'] == 'fp16'
    apply_precision('fp32')

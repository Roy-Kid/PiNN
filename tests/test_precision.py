# -*- coding: utf-8 -*-
"""settings.train_dtype (YAML) and calculator default_dtype (constructor)."""
import os
import tempfile

import pytest
import tensorflow as tf
import yaml
from tensorflow.python.lib.io.file_io import FileIO

from pinn.calculator import PiNN_calc
from pinn.models.base import (
    apply_train_dtype, tf_dtype_from_name, train_dtype_from_params,
)


def test_tf_dtype_from_name():
    assert tf_dtype_from_name('float32') == tf.float32
    assert tf_dtype_from_name('fp64') == tf.float64
    assert tf_dtype_from_name('') == tf.float32
    with pytest.raises(ValueError, match='Unknown dtype'):
        tf_dtype_from_name('float16')


def test_train_dtype_from_params():
    assert train_dtype_from_params({}) == 'float32'
    assert train_dtype_from_params({'settings': None}) == 'float32'
    assert train_dtype_from_params(
        {'settings': {'train_dtype': 'float64'}}) == 'float64'


def test_calc_default_dtype_is_constructor_only():
    class _Model:
        params = {'settings': {'train_dtype': 'float64'}}

    calc = PiNN_calc(model=_Model(), default_dtype='')
    assert calc._dtype_name() == 'float64'
    assert calc._tf_dtype() == tf.float64
    calc_override = PiNN_calc(model=_Model(), default_dtype='float32')
    assert calc_override._tf_dtype() == tf.float32
    calc_plain = PiNN_calc(model=type('M', (), {})())
    assert calc_plain._tf_dtype() == tf.float32


@pytest.mark.forked
def test_unknown_train_dtype_raises():
    with pytest.raises(ValueError, match='Unknown dtype'):
        apply_train_dtype('fp16')


@pytest.mark.forked
def test_float32_is_the_default():
    apply_train_dtype('float32')
    assert tf.keras.backend.floatx() == 'float32'
    apply_train_dtype(None)
    assert tf.keras.backend.floatx() == 'float32'


@pytest.mark.parametrize('name,keras', [
    ('float32', 'float32'),
    ('fp32', 'float32'),
    ('float64', 'float64'),
    ('fp64', 'float64'),
])
@pytest.mark.forked
def test_apply_train_dtype(name, keras):
    apply_train_dtype(name)
    assert tf.keras.backend.floatx() == keras
    apply_train_dtype('float32')


@pytest.mark.forked
def test_default_params_record_train_dtype():
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
    assert saved.get('settings', {}).get('train_dtype', 'float32') == 'float32'
    assert 'default_dtype' not in saved.get('settings', {})
    apply_train_dtype('float32')


@pytest.mark.forked
def test_float64_trains():
    import numpy as np
    import pinn
    from pinn.io import load_numpy, sparse_batch

    testpath = tempfile.mkdtemp()
    n = 8
    data = {
        'coord': np.random.randn(n, 3, 3).astype(np.float64),
        'elems': np.ones((n, 3), dtype=np.int32),
        'e_data': np.random.randn(n).astype(np.float64),
    }
    params = {
        'model_dir': testpath,
        'settings': {'train_dtype': 'float64'},
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
    assert saved['settings']['train_dtype'] == 'float64'
    apply_train_dtype('float32')

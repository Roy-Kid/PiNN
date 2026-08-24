import tensorflow as tf
from pinn.optimizers.ekf import EKF, default_ekf
from pinn.optimizers.gekf import gEKF

default_adam = {
    'class_name': 'Adam',
    'config': {
        'learning_rate': {
            'class_name': 'ExponentialDecay',
            'config':{
                'initial_learning_rate': 3e-4,
                'decay_steps': 10000,
                'decay_rate': 0.994}},
        'clipnorm': 0.01}}

def get(optimizer):
    if isinstance(optimizer, EKF) or isinstance(optimizer, gEKF):
        return optimizer
    if isinstance(optimizer, dict):
        if optimizer['class_name']=='EKF':
            return EKF(**optimizer['config'])
        if optimizer['class_name']=='gEKF':
            return gEKF(**optimizer['config'])
    # PiNN trains in estimator graph mode (tf.gradients + manually assigned
    # .iterations + apply_gradients, see pinn.models.base.get_train_op). TF>=2.11
    # made tf.keras.optimizers.get() return the new-style Keras optimizer, which
    # does not support that v1 pattern. Request the legacy optimizer wherever the
    # API exists (TF 2.11-2.15); on older TF (<2.11) there is only one optimizer.
    import inspect
    if 'use_legacy_optimizer' in inspect.signature(
            tf.keras.optimizers.deserialize).parameters:
        if isinstance(optimizer, str):
            optimizer = {'class_name': optimizer, 'config': {}}
        if isinstance(optimizer, dict):
            return tf.keras.optimizers.deserialize(
                optimizer, use_legacy_optimizer=True)
    return tf.keras.optimizers.get(optimizer)

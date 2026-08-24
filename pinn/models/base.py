# -*- coding: utf-8 -*-
"""Basic functions for PiNN models"""
import tensorflow as tf
from pinn.utils import pi_named

# Public YAML names -> Keras mixed-precision policy.
# fp16/bf16 keep float32 variables and compute in reduced precision.
_PRECISION_POLICIES = {
    'fp32': 'float32',
    'float32': 'float32',
    'fp16': 'mixed_float16',
    'float16': 'mixed_float16',
    'bf16': 'mixed_bfloat16',
    'bfloat16': 'mixed_bfloat16',
}
_FP16_LOSS_SCALE = 128.0


def apply_precision(precision='fp32'):
    """Set the Keras mixed-precision policy used during training.

    Args:
        precision (str): ``fp32`` (default), ``fp16``, or ``bf16``.
            Aliases ``float32`` / ``float16`` / ``bfloat16`` are accepted.

    Returns:
        str: the Keras policy name that was applied.
    """
    if precision is None:
        precision = 'fp32'
    key = str(precision).lower()
    if key not in _PRECISION_POLICIES:
        allowed = ', '.join(sorted(set(_PRECISION_POLICIES)))
        raise ValueError(
            f'Unknown precision {precision!r}. Expected one of: {allowed}.')
    policy = _PRECISION_POLICIES[key]
    tf.keras.mixed_precision.set_global_policy(policy)
    return policy


def export_model(model_fn):
    # default parameters for all models
    from pinn.optimizers import default_adam
    default_params = {'optimizer': default_adam, 'precision': 'fp32'}
    def pinn_model(params, **kwargs):
        model_dir = params['model_dir']
        params_tmp = default_params.copy()
        params_tmp.update(params)
        params = params_tmp
        apply_precision(params.get('precision', 'fp32'))
        def model_fn_with_precision(features, labels, mode, params):
            apply_precision(params.get('precision', 'fp32'))
            return model_fn(features, labels, mode, params)
        model = tf.estimator.Estimator(
            model_fn=model_fn_with_precision, params=params,
            model_dir=model_dir, **kwargs)
        return model
    return pinn_model

class MetricsCollector():
    def __init__(self, mode):
        self.mode = mode
        self.LOSS = []
        self.ERROR = []
        self.METRICS = {}

    def add_error(self, tag, data, pred, mask=None, weight=1.0,
                  use_error=True, log_error=True, log_hist=True):
        """Add the error

        Args:
            tag (str): name of the error.
            data (tensor): data label tensor.
            pred (tensor): prediction tensor.
            mask (tensor): default to None (no mask, not implemented yet).
            weight (tensor): default to 1.0.
            mode: ModeKeys.TRAIN or ModeKeys.EVAL.
            return_error (bool): return error vector (for usage with Kalman Filter).
            log_loss (bool): log the error and loss function.
            log_hist (bool): add data and predicition histogram to log.
            log_mae (bool): add the mean absolute error to log.
            log_rmse (bool): add the root mean squared error to log.
        """
        error = data - pred
        weight = tf.cast(weight, data.dtype)
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            if log_hist:
                tf.compat.v1.summary.histogram(f'{tag}_DATA', data)
                tf.compat.v1.summary.histogram(f'{tag}_PRED', pred)
                tf.compat.v1.summary.histogram(f'{tag}_ERROR', error)
            if log_error:
                mae = tf.reduce_mean(tf.abs(error))
                rmse = tf.sqrt(tf.reduce_mean(error**2))
                tf.compat.v1.summary.scalar(f'{tag}_MAE', mae)
                tf.compat.v1.summary.scalar(f'{tag}_RMSE', rmse)
            if mask is not None:
                error = tf.boolean_mask(error, mask)
            if use_error:
                loss = tf.reduce_mean(error**2 * weight)
                tf.compat.v1.summary.scalar(f'{tag}_LOSS', loss)
                self.ERROR.append(error*tf.math.sqrt(weight))
                self.LOSS.append(loss)
        if self.mode == tf.estimator.ModeKeys.EVAL:
            if log_error:
                self.METRICS[f'METRICS/{tag}_MAE'] = tf.compat.v1.metrics.mean_absolute_error(data, pred)
                self.METRICS[f'METRICS/{tag}_RMSE'] = tf.compat.v1.metrics.root_mean_squared_error(data, pred)
            if mask is not None:
                error = tf.boolean_mask(error, mask)
            if use_error:
                loss = tf.reduce_mean(error**2 * weight)
                self.METRICS[f'METRICS/{tag}_LOSS'] = tf.compat.v1.metrics.mean(loss)
                self.LOSS.append(loss)


@pi_named('TRAIN_OP')
def get_train_op(optimizer, metrics, tvars, separate_errors=False):
    """
    Args:
        optimizer: a PiNN optimizer config.
        params: optimizer parameters.
        loss: scalar loss function.
        error: a list of error vectors (reserved for EKF).
        network: a PiNN network instance.
        sperate_errors (bool): separately update elements in the metrics
    """
    from pinn.optimizers import get, EKF, gEKF
    import numpy as np

    optimizer = get(optimizer)
    optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()
    nvars = np.sum([np.prod(var.shape) for var in tvars])
    compute_dtype = tf.keras.mixed_precision.global_policy().compute_dtype
    print(f'{nvars} trainable vaiables, training with {tvars[0].dtype.name} '
          f'variables / {compute_dtype} compute.')

    if not (isinstance(optimizer, EKF) or isinstance(optimizer, gEKF)):
        loss_list =  metrics.LOSS
        if separate_errors:
            selection = tf.random.uniform([], maxval= len(loss_list), dtype=tf.int32)
            loss = tf.stack(loss_list)[selection]
        else:
            loss = tf.reduce_sum(loss_list)
        # mixed_float16 needs a loss scale so fp16 gradients do not underflow.
        scale = _FP16_LOSS_SCALE if str(compute_dtype) == 'float16' else 1.0
        if scale != 1.0:
            loss = loss * scale
        grads = tf.gradients(loss, tvars)
        if scale != 1.0:
            grads = [g if g is None else g / scale for g in grads]
        return optimizer.apply_gradients(zip(grads, tvars))
    else:
        error_list =  metrics.ERROR
        # EKF error vectors are scaled
        if isinstance(optimizer, EKF):
            error = tf.concat([tf.reshape(e, [-1])/tf.math.sqrt(tf.cast(tf.size(e), e.dtype))
                               for e in error_list], 0)
        # gEKF should handle this automatically
        if isinstance(optimizer, gEKF):
            error = tf.concat([tf.reshape(e, [-1]) for e in error_list], 0)
        if separate_errors:
            selection = tf.random.uniform([], maxval= len(error_list), dtype=tf.int32)
            mask = tf.concat([tf.fill([tf.size(e)], tf.equal(selection,i))
                              for i,e in enumerate(error_list)], 0)
            error = tf.boolean_mask(error, mask)
        return optimizer.get_train_op(error, tvars)

# Models in PiNN

For regression problems for an atomic or molecular property, it's often
sufficient to use an `network`. However, atomic machine learning tasks it's
often desired to train on properties derived from the atomic predictions, such
as forces, stress tensors for dipole moments. `pinn.models` are created for
defining these tasks in a `network`-agnostic way.

Two models are implemented in PiNN at this point, their respective options can
be found in the "Implemented models" section.

## Configuration

Models implemented in PiNN used a serialized format for their parameters. The
parameter file specifies the network architecture, hyperparameters and training
algorithm. A typical parameter file include the following sections:

```yaml
model_dir: pinet_potential
model:
  name: potential_model
  params:
    use_force: true
network:
  name: PiNet
  params:
    atom_types: [1, 6, 7, 8, 9]
optimizer:
  class_name: EKF
  config:
    learning_rate: 0.03
settings:
  train_dtype: float32
```

Among those, the `optimizer` section follows the format of a Keras optimizer.
The `model` and `network` sections specify the name and parameters of initialize
a PiNN model and network, respectively. A model can be initialized by a
parameter file or a corresponding nested python dictionary.

## Settings: `train_dtype`

`settings` is training-time configuration. The only dtype key in the YAML is
`train_dtype` (`float32` or `float64`; default `float32`). It is **not** the
ASE calculator's `default_dtype` (that argument exists only on
`PiNN_calc` / `get_calc`, see [Potential](potential.md#ase-calculator)).

When a model is built, PiNN calls `tf.keras.backend.set_floatx` and
`tf.keras.mixed_precision.set_global_policy` with this value. That is the
Keras **default float type** for the process, so it decides the dtype of
new floating tensors Keras creates, not integer index tensors.

### What it does change

| Piece | Effect of `train_dtype` |
|-------|-------------------------|
| Network weights | `Dense` (and other Keras) kernels and biases are created in this dtype. A float64 run stores float64 checkpoints; a float32 run stores float32. |
| Keras compute | Matmuls, biases, activations inside those layers run in this dtype. |
| Derived float tensors | Features that follow `coord.dtype` (atomic embeddings in PiNet/PiNet2, distances in the neighbor list, cell displacements) inherit the dtype of the coordinate tensor. |
| Numpy datasets | `load_numpy` types every non-integer array as `keras.backend.floatx()`, so `coord`, `e_data`, `f_data`, … enter the pipeline in `train_dtype`. |
| ANI loader | Coordinate and energy specs use `floatx()` as well. |
| Loss and gradients | Energy/force/stress residuals and `tf.gradients` match the prediction/variable dtype. Adam-style optimizer slots follow the variables. |
| `params.yml` | The chosen `train_dtype` is written next to the rest of the model spec so later loads (and an empty calculator `default_dtype`) can see it. |

### What it does **not** change

- **Integer tensors stay `int32`:** `elems`, `ind_1`, `ind_2`, and other sparse indices.
- **Already-written TFRecords:** `pinn convert` serializes whatever dtypes the dataset had *at convert time* into the `.tfr` / `.yml` spec. Loading a float32 record in a float64 training job does **not** rewrite the file; TensorFlow will cast at the Keras layer boundary. For a true float64 pipeline, convert (or `load_numpy`) after `train_dtype` is set, or recast in the input function.
- **EKF / gEKF inversion:** those optimizers still invert in `inv_dtype` (default float64) regardless of `train_dtype`.
- **ASE `Atoms`:** positions and cells remain NumPy arrays (typically float64) until they are copied into a TensorFlow tensor.
- **ASE MD state:** always float64; see inference below.

### Inference and the ASE calculator

**ASE MD is always float64.** Positions, cell, momenta, and the integrator
stay NumPy float64. PiNN does not switch the MD state to float32.

Training stores **every float parameter** in `train_dtype` (float32 or
float64). Inference follows MACE: convert the **model** to the requested
width, then run the whole network in that width.

`default_dtype` is a constructor argument only (not YAML):

```python
get_calc(model_dir)                            # same dtype as training
get_calc(model_dir, default_dtype='float32')   # like MACE model.float()
```

Empty / omitted → `settings.train_dtype`. If it **matches** training, the
checkpoint is restored as usual. If it **differs**, PiNN builds the predict
graph at `default_dtype` and loads the checkpoint with a one-time
`tf.cast` into those variables (TensorFlow variables cannot change dtype
in place, so this is the equivalent of PyTorch `model.to(dtype)` /
`model.float()` / `model.double()`). After that:

- Weights, embeddings, distances, `Dense` matmuls (`tf.linalg.matmul`)
  and bias-adds (`tf.nn.bias_add`) are all `default_dtype`.
- Predictor inputs `coord` and `cell` are created in the same dtype.
  `elems` / `ind_1` stay `int32`.
- There is no per-layer ping-pong cast.

**Train float64, infer `default_dtype='float32'`:** checkpoint kernels
are cast to float32 **once at load**; GEMMs run in float32. ASE still
hands over float64 positions; they are copied into a float32 tensor at
the calculator boundary, then everything in TF stays float32 until
forces are written back. ASE continues the MD step in float64.

**Train float32, infer `default_dtype='float64'`:** weights are cast up
once; the network runs in float64. That is extra precision at inference,
not extra precision that was in the checkpoint.

To train in float32 end-to-end, set `settings.train_dtype: float32`. Do
not expect `default_dtype` to replace that for training.

## Training

The model maybe created by calling the corresponding model function, and a
parameter dictionary mirroring the parameter file:

```Python
import yaml
from pinn.models.potential import potential_model
with open('params.yml') as f:
    params = yaml.load(f, Loader=yaml.Loader)
model = potential_model(params)
```

PiNN provides a shortcut `pinn.get_model` to create an implemented model from a
parameter dictionary or parameter file.

```Python
model = pinn.get_model('params.yml')
```

`pinn.get_model` automatically saves a copy `params.yml` file in the model
directory. When such a file exist, the model can be loaded with its directory as
well.

```Python
model = pinn.get_model('pinet_potential')
```

The PiNN model is a TensorFlow estimator, to train the model in a python script:

```Python
filelist = glob('{DATASET_PATH}/QM9/dsgdb9nsd/*.xyz')
dataset = lambda: load_qm9(filelist, splits={'train':8, 'test':2})
train = lambda: dataset()['train'].repeat().shuffle(1000).apply(sparse_batch(100))
test = lambda: dataset()['test'].repeat().apply(sparse_batch(100))
train_spec = tf.estimator.TrainSpec(input_fn=train, max_steps=1000)
eval_spec = tf.estimator.EvalSpec(input_fn=test, steps=100)
tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
```

## ASE interface
PiNN provides a ``PiNN_calc`` class to interface models with ASE. A calculator
can be created from a model as simple as:

```Python
calc = pinn.get_calc('pinet_potential')
calc.calculate(atoms)
```

The implemented properties of the calculator depend on the model. For example:
the potential model implements energy, forces and stress (with PBC) calculations
and the dipole model implements partial charge and dipole calculations.


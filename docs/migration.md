# Migrating to PiNN 2.x

Since version 2.x, a modularized design, **PiNet2**, has been implemented for equivariant atomistic potential training. **PiNet2** is compatible with **PiNet1**, and you can use the `rank` parameter to specify the desired feature order. To use PiNet, you can either call `pinet` or use `pinet2(rank=3)`—both are functionally equivalent. However, trained models are not interchangeable between the two, meaning you will need to retrain your model if switching versions.

A workflow using [Nextflow](https://www.nextflow.io/docs/latest/index.html) is also integrated, enabling model training on clusters via SLURM or other resource management systems. Examples can be found in the `nextflow.config` file and the [notebook](./notebooks/More_on_training.ipynb).

## Upgrading the TensorFlow backend to 2.15

PiNN's supported TensorFlow window has moved up to **TensorFlow 2.15**
(Python 3.9–3.11, `numpy<2`). This is the last release that still ships
`tf.estimator` and the legacy Keras optimizers — both removed in TF 2.16 — while
also supporting recent GPUs (NVIDIA Hopper / `sm_90`, e.g. GH200). PiNN's
training loop is built on `tf.estimator`, so 2.15 is the ceiling until the
estimator path is rewritten.

### Installing

```sh
pip install -e '.[cpu]'   # tensorflow-cpu >=2.15,<2.16
pip install -e '.[gpu]'   # tensorflow     >=2.15,<2.16  (x86_64 CUDA wheel)
```

!!! note "aarch64 GPUs (e.g. GH200)"
    There is **no GPU TensorFlow wheel for aarch64** on PyPI or conda-forge
    (both are CPU-only). The only aarch64 GPU build of TF 2.15 is the NVIDIA NGC
    container `nvcr.io/nvidia/tensorflow:24.03-tf2-py3`; `Dockerfile.gpu` /
    `Singularity.gpu` are based on it. On x86_64 the normal `tensorflow` CUDA
    wheel works.

### What changed under the hood

Most changes are internal; existing parameter files and trained 2.x models keep
working. The notable points:

- **Optimizers.** TF ≥2.11 returns the new-style Keras optimizer from
  `tf.keras.optimizers.get`, which does not support the
  `tf.gradients` + `apply_gradients` graph-mode pattern PiNN uses inside the
  estimator. `pinn.optimizers.get` now requests the *legacy* optimizer
  (`deserialize(..., use_legacy_optimizer=True)`) automatically. No change is
  needed in your input files; `Adam`, `SGD`, etc. behave as before.
- **ASE calculator.** TF ≥2.15 prefetches `tf.data` pipelines more
  aggressively. The predict `input_fn` now disables autotune/prefetch so the
  cached predictor always reads the freshly-updated atoms (otherwise a
  `calculate()` could return the *previous* step's energy/forces).
- **h5py 3.** TF 2.15 ships h5py ≥3, which removed `Dataset.value`. The ANI-1
  loader now uses `dataset[()]`.
- **numpy ≥1.24.** The removed aliases `np.int` / `np.float` are gone from the
  code (use the Python built-ins or sized dtypes like `np.int32`).
- **tfrecord spec.** `write_tfrecord` now reads `dataset.element_spec` (public)
  instead of the private `DatasetSpec._serialize()`.

### Behavioural note: RNG determinism

`tf.random.set_seed(...)` no longer makes two independently-constructed Keras
models share weights the way it did on older TF. If you relied on that for
reproducibility, set the seed and then **copy weights explicitly**
(`model_b.set_weights(model_a.get_weights())`), or seed the layer initializers
directly.

# Migrating to PiNN 1.x (TF2)

Since version 1.x, PiNN switched to TensorFlow 2 as a backend, this introduces
changes to the API. This document provides information for the changes and
guides for migration.

## New features

**CLI**:
PiNN 1.x introduces a new entry point `pinn` as the command line interface. The
trainer module will be replaced with the `pinn train` sub-command. The CLI also
exposes utilities like dataset conversion for easier usage.

**Parameter file**: 
in PiNN 1.0 the parameter file will serve as a comprehensive input for PiNN
models, the structure of the parameter file is changed, see the documentation
for more information.

**Extended Kalman filter**:
an experimental extended Kalman filter (EKF) optimizer is implemented.


## Notes for developers

- Documentation is now built with mkdocs.
- Documentation is moved to Github pages.
- Continuous integration is moved to Github Actions.
- The Docker Hub repo is now [tec@uu/pinn](https://hub.docker.com/r/tecatuu/pinn).

**Datasets**: dataset loaders should be most compatible with PiNN 0.x. With the
TF2 update, dataset may be inspected interactively with eager execution.
Splitting option is simplified (see below), and splitting of `load_tfrecord`
becomes possible.

**Networks**: following the guideline of TF2, networks in PiNN 1.x are new Keras
models and layers becomes Keras layers. This means the PiNN networks can be used
to perform some simple prediction tasks. Note that PiNN models are still
implemented as TensorFlow estimators since they provide a better control over
the training and prediction behavior. Like the design of PiNN 0.x, the models
interpret the predictions of PiNN networks as physical quantities and interface
them to atomic simulation packages.

**Models**:
new helper function `export_model` and class `MetricsCollector` are implemented to
simplify the implementation of models, see the source of [dipole
model](https://github.com/Teoroo-CMC/PiNN/blob/master/pinn/models/dipole.py) for an
example.

## Breaking changes
- Models trained in PiNN 0.x will not be usable in PiNN 1.x.
- Model parameters need to be adapted to the new parameter format.
- For dataset loaders `load_*`:
    + the `split` argument is renamed to `splits`;
    + splitting is disabled by default;
    + nested splits like `{'train':1, 'test':[1,2,3]}` is not supported anymore.
- `format_dict` is renamed as `ds_spec` to be consistent with
  [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/data/DatasetSpec).

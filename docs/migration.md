# Migrating to PiNN 1.x (TF2)

Since version 1.x, PiNN switched to TensorFlow 2 as a backend, this introduces
changes to the API. This document provides information for the changes and
guides for migration.

## New features

**CLI**:
PiNN 1.x introduces a new entrypoint `pinn` as the command line interface. The
trainer module will be replaced with the `pinn train` sub-command. The CLI also
exposes utilities like dataset conversion for easier usage.

**Parameter file**:
in PiNN 1.0 the parameter file will serve as a canonial input for PiNN models, 
the structure, see the documentation for more information.

**Extended Kalman filter**:
a experimental extended Kalman filter (EKF) optimizer is implemented. 

## Notes for developers

- Documentation is now built with mkdocs
- Documentation is moved to Github pages
- Continuous integration is moved to Github Actions
- Models are now exported with a model exporter

**Datasets**:
the dataset loaders should be most compatible with PiNN 0.x, with the major
difference being the dataset may be inspected interactively with eager execution
of TF2.

**Networks**:
following the guidline of TF2, networks in PiNN 1.x are new Keras models and
layers becomes Keras layers. This means the PiNN networks are be used to perform
some simple prediction tasks. Note that PiNN models are still implemented as
Tensorflow estimators since they provide a better control over the training and
prediction behavior. Like the desgin of PiNN 0.x, the models interpret the
predictions of PiNN networks as physical quantities and interface them to atomic
simulation packages.

**Models**:
new helper function `export_mode` and class `MetricsCollector` is implemented to
simplify the implementation of models, see the source of [dipole
model](https://github.com/yqshao/PiNN/blob/TF2/pinn/models/dipole.py) for an
example.

## Breaking changes
- Models trained in PiNN 0.x will not be usable in PiNN 1.x
- Model parameters needs to be updated to the new parameter format
- Dataset loaders `load_*` will not split the datasets by default
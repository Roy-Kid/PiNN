# Apptainer/Singularity CPU image (x86_64) for PiNN — mirrors Dockerfile.
# Build:
#   cd <repo> && apptainer build /path/on/allowed/fs/pinn-cpu.sif Singularity
Bootstrap: docker
From: tensorflow/tensorflow:2.15.0

%files
    pinn /opt/src/pinn/pinn
    setup.py /opt/src/pinn/setup.py
    requirements-dev.txt /opt/src/pinn/requirements-dev.txt
    requirements-doc.txt /opt/src/pinn/requirements-doc.txt
    requirements-extra.txt /opt/src/pinn/requirements-extra.txt

%post
    apt-get update && apt-get install -y --no-install-recommends locales && \
        locale-gen en_US.UTF-8 && apt-get clean && rm -rf /var/lib/apt/lists/*
    pip install --no-cache-dir --upgrade pip
    # PiNN + `requests` (dataset-prep processes download from figshare/etc.).
    # h5py for the loaders that use it. Base image already has TensorFlow 2.15.
    pip install --no-cache-dir /opt/src/pinn requests h5py

%runscript
    exec pinn "$@"

%labels
    pinn.tensorflow 2.15.0
    pinn.base tensorflow/tensorflow:2.15.0

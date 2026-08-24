# CPU image for PiNN (x86_64) — TensorFlow 2.15, the ceiling that still ships
# tf.estimator (removed in TF 2.16) and Keras 2, both of which PiNN relies on.
#
# Build:
#   docker build -t <repo>/pinn:<tag>-cpu .
# Run:
#   docker run --rm <repo>/pinn:<tag>-cpu --help     # ENTRYPOINT is `pinn`
# For GPU on aarch64 (GH200) use Dockerfile.gpu (NGC-based) instead.
FROM tensorflow/tensorflow:2.15.0

RUN apt-get update && apt-get install -y --no-install-recommends locales && \
    locale-gen en_US.UTF-8 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Base image already provides tensorflow (py3.11); install PiNN + `requests`
# (dataset-prep downloads) + h5py. dev/doc/extra are omitted from the runtime
# image (extras like pymatgen/nglview are for local dev, not the pipeline).
COPY . /opt/src/pinn
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /opt/src/pinn requests h5py

ENTRYPOINT ["pinn"]

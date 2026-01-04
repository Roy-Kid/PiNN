# pinn/io/build_dataset.py
"""
Backend-neutral dataset builder for PiNN.

This module provides a single entry point that:
- reads train.yml / eval.yml (legacy TFRecord YAML or future loader YAML)
- builds either a tf.data.Dataset (TF backend) OR a Python-iterable stream (Torch backend)
- optionally applies caching for Torch via pinn.io.cache (NPZ disk cache + RAM cache)

We intentionally do NOT implement torch preprocessing/batching here yet.
That is Step 5 (preprocess) and Step 6 (CLI dispatch/training runtime integration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Union

from pinn.io.spec import DatasetSpec, parse_dataset_spec
from pinn.io.iter_utils import iter_numpy_examples

# Torch-side cache currently lives at pinn.io.cache (you already have this)
from pinn.io.cache import CacheConfig, CachedExamples, fingerprint_spec


@dataclass(frozen=True)
class BuildOptions:
    """
    Options used when building datasets.

    Attributes:
        backend: "tf" or "torch"
        cache: whether to enable caching (Torch: RAM and/or disk)
        scratch_dir: optional directory for persistent caching (Torch)
        cache_ram: whether to keep RAM cache (Torch)
    """
    backend: str
    cache: bool = True
    scratch_dir: Optional[str] = None
    cache_ram: bool = True


def _build_from_tfrecord(spec: DatasetSpec, *, backend: str) -> Any:
    """
    Build dataset from legacy TFRecord YAML.

    Args:
        spec: DatasetSpec(kind="tfrecord")
        backend: "tf" or "torch"

    Returns:
        - TF backend: tf.data.Dataset
        - Torch backend: Iterable[Dict[str, Any]] (NumPy-friendly examples)
    """
    from pinn.io import load_tfrecord  # TFRecord remains supported

    ds = load_tfrecord(spec.yml_path)
    if backend == "tf":
        return ds

    if backend == "torch":
        # Torch consumes Python iteration over numpy-friendly dicts
        return iter_numpy_examples(ds)

    raise ValueError(f"Unknown backend: {backend}")


def _build_from_loader(spec: DatasetSpec, *, backend: str) -> Any:
    """
    Build dataset from loader-based YAML.

    This is forward-compatible scaffolding. We'll wire more loaders later.

    Loader YAML format:
        loader: "<function name in pinn.io>"
        loader_kwargs: {...}

    Returns:
        - TF backend: whatever the loader returns (typically tf.data.Dataset)
        - Torch backend: iterable over numpy-friendly example dicts
    """
    import pinn.io as io_mod

    assert spec.loader is not None
    fn = getattr(io_mod, spec.loader, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Unknown loader '{spec.loader}' referenced by {spec.yml_path}")

    kwargs = spec.loader_kwargs or {}
    ds = fn(**kwargs)

    if backend == "tf":
        return ds

    if backend == "torch":
        return iter_numpy_examples(ds)

    raise ValueError(f"Unknown backend: {backend}")


def build_dataset(
    yml_path: str,
    *,
    options: BuildOptions,
    dataset_role: str,
) -> Any:
    """
    Build a dataset for the given backend from a dataset YAML file.

    Args:
        yml_path: path to dataset YAML (train.yml or eval.yml)
        options: BuildOptions controlling backend and caching behavior
        dataset_role: a short label used for cache fingerprinting, e.g. "train" or "eval"

    Returns:
        TF backend: tf.data.Dataset
        Torch backend: Iterable[Dict[str, Any]] (NumPy-friendly), possibly cached
    """
    backend = options.backend.lower().strip()
    if backend not in ("tf", "torch"):
        raise ValueError("BuildOptions.backend must be 'tf' or 'torch'.")

    spec = parse_dataset_spec(yml_path)

    if spec.kind == "tfrecord":
        built = _build_from_tfrecord(spec, backend=backend)
    elif spec.kind == "loader":
        built = _build_from_loader(spec, backend=backend)
    else:
        raise ValueError(f"Unknown dataset spec kind: {spec.kind}")

    # Only Torch uses our NPZ cache (TF backend uses its own caching mechanisms)
    if backend == "torch" and options.cache:
        # Cache key should include at least: role + yml path + spec content
        cache_spec: Dict[str, Any] = {
            "role": dataset_role,
            "yml_path": spec.yml_path,
            "kind": spec.kind,
            "meta": spec.meta or {},
        }
        key = fingerprint_spec(cache_spec)

        cfg = CacheConfig(
            enabled=True,
            ram=bool(options.cache_ram),
            scratch_dir=options.scratch_dir,
            key=key if options.scratch_dir else None,
        )

        # If scratch_dir is not set, disk caching is disabled, RAM cache can still work.
        if options.scratch_dir is None:
            cfg = CacheConfig(enabled=True, ram=bool(options.cache_ram), scratch_dir=None, key=None)

        built = CachedExamples(built, cfg, meta_spec=cache_spec)

    return built

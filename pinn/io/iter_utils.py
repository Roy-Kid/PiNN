# pinn/io/iter_utils.py
"""
Utilities for iterating over PiNN dataset sources in a backend-agnostic way.

- This module provides a single conversion point:
    "whatever the loader returns" -> Iterator[Dict[str, np.ndarray]]
- Keep TensorFlow optional: only imported if the object looks like tf.data.Dataset.
- Output values should be NumPy-compatible (np.ndarray / scalars), not torch.Tensors.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Mapping

import numpy as np


def iter_examples(source: Any) -> Iterator[Dict[str, Any]]:
    """
    Convert a variety of dataset-like objects to an iterator of example dicts.

    Supported inputs:
    - Iterator[dict] or Iterable[dict]
    - tf.data.Dataset yielding dict-like elements (via as_numpy_iterator)
    - Any object implementing __iter__ that yields dict-like elements

    Returns:
        Iterator over per-example dictionaries. Values are typically numpy arrays,
        but may include Python scalars.

    Raises:
        TypeError: if the source is not iterable or does not yield mapping-like objects.
    """
    # Fast path: if it exposes as_numpy_iterator (typical tf.data.Dataset)
    as_np_iter = getattr(source, "as_numpy_iterator", None)
    if callable(as_np_iter):
        for ex in as_np_iter():
            if not isinstance(ex, Mapping):
                raise TypeError(
                    "tf.data.Dataset yielded a non-mapping element of type "
                    f"{type(ex)}; expected dict-like."
                )
            yield dict(ex)
        return

    # Generic Python iterable path
    if not hasattr(source, "__iter__"):
        raise TypeError(f"source is not iterable: type={type(source)}")

    for ex in source:
        if not isinstance(ex, Mapping):
            raise TypeError(
                f"Iterable yielded a non-mapping element of type {type(ex)}; expected dict-like."
            )
        yield dict(ex)


def to_numpy_example(example: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Best-effort conversion of an example dict to NumPy-friendly values.

    This function does NOT force everything to np.ndarray (because some fields
    may be strings or nested metadata), but it normalizes common numeric types.

    Args:
        example: dict-like mapping.

    Returns:
        Plain dict with numpy/scalar values.
    """
    out: Dict[str, Any] = {}
    for k, v in example.items():
        # Already numpy scalar/array
        if isinstance(v, (np.ndarray, np.generic)):
            out[k] = v
            continue

        # Python numeric scalars
        if isinstance(v, (int, float, bool)):
            out[k] = v
            continue

        # Lists/tuples of numbers -> np.array
        if isinstance(v, (list, tuple)):
            try:
                out[k] = np.asarray(v)
            except Exception:
                out[k] = v
            continue

        # Fallback: keep as-is (e.g., strings, None)
        out[k] = v
    return out


def iter_numpy_examples(source: Any) -> Iterator[Dict[str, Any]]:
    """
    Convenience wrapper: iter_examples + to_numpy_example.

    Args:
        source: dataset-like object (tf.data.Dataset, iterable of dict, generator, etc.)

    Yields:
        Example dict with NumPy-friendly values.
    """
    for ex in iter_examples(source):
        yield to_numpy_example(ex)

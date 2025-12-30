# pinn/io/spec.py
"""
Dataset spec parsing and normalization for PiNN.

Goals:
- Keep backward compatibility with legacy TFRecord YAMLs written by pinn.io.write_tfrecord.
- Provide a forward-compatible spec format that can describe raw loaders too.
- Normalize both forms into a small number of "kinds" so the builder can dispatch cleanly.

Supported kinds:

1) kind="tfrecord"
   - Any YAML that looks like a TFRecord dataset descriptor written by write_tfrecord.
   - We treat the YAML path itself as the authoritative descriptor and pass it to load_tfrecord.

2) kind="loader"
   - A YAML that contains:
       loader: "<name>"
       loader_kwargs: { ... }   (optional)
   - Later we will wire these to pinn.io.<loader>(**kwargs).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    """
    Normalized dataset specification.

    Attributes:
        kind: "tfrecord" or "loader".
        yml_path: path to the YAML file (always recorded for traceability).
        loader: loader function name if kind == "loader".
        loader_kwargs: kwargs dict if kind == "loader".
        meta: raw YAML content for debugging/traceability.
    """
    kind: str
    yml_path: str
    loader: Optional[str] = None
    loader_kwargs: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None


def read_yaml(path: str) -> Dict[str, Any]:
    """
    Read a YAML file into a dict.

    Args:
        path: path to .yml/.yaml

    Returns:
        dict (possibly empty)

    Raises:
        FileNotFoundError: if path doesn't exist.
        yaml.YAMLError: for invalid YAML.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return dict(obj or {})


def infer_kind(raw: Mapping[str, Any]) -> str:
    """
    Infer dataset kind from YAML content.

    Heuristic:
    - If explicit 'loader' key exists -> loader kind
    - Otherwise assume TFRecord YAML produced by write_tfrecord -> tfrecord kind

    This keeps legacy YAMLs working without requiring changes.
    """
    if "loader" in raw:
        return "loader"
    return "tfrecord"


def parse_dataset_spec(yml_path: str) -> DatasetSpec:
    """
    Parse and normalize a dataset YAML into a DatasetSpec.

    Args:
        yml_path: path to YAML file.

    Returns:
        DatasetSpec
    """
    raw = read_yaml(yml_path)
    kind = infer_kind(raw)

    if kind == "loader":
        loader = raw.get("loader")
        if not isinstance(loader, str) or not loader:
            raise ValueError(f"Invalid loader spec in {yml_path}: 'loader' must be a non-empty string.")
        loader_kwargs = raw.get("loader_kwargs", {})
        if loader_kwargs is None:
            loader_kwargs = {}
        if not isinstance(loader_kwargs, dict):
            raise ValueError(f"Invalid loader_kwargs in {yml_path}: must be a mapping/dict.")
        return DatasetSpec(
            kind="loader",
            yml_path=str(yml_path),
            loader=loader,
            loader_kwargs=dict(loader_kwargs),
            meta=dict(raw),
        )

    # Legacy TFRecord YAML: treat path as the descriptor
    return DatasetSpec(kind="tfrecord", yml_path=str(yml_path), meta=dict(raw))

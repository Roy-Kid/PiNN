# pinn/io/torch/cache.py
"""
Torch-backend caching for PiNN example streams.

This module provides a Torch-friendly cache that does not depend on TFRecord.

Cache modes:
- RAM cache (per process)
- Disk cache under scratch_dir (NPZ shards + meta.yml)

This is intended to preserve the "caching speeds up training" feature
for Torch backend with the same CLI flags (--cache, --scratch-dir).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence
import hashlib
import json

import numpy as np
import yaml


def fingerprint_spec(spec: Mapping[str, Any]) -> str:
    """Stable short hash for a JSON-serializable cache spec."""
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _ensure_dir(p: Path) -> None:
    """Create directory p if needed."""
    p.mkdir(parents=True, exist_ok=True)


def _npz_path(cache_dir: Path, idx: int) -> Path:
    """Path for per-example npz file."""
    return cache_dir / f"ex_{idx:08d}.npz"


def _meta_path(cache_dir: Path) -> Path:
    """Path for metadata YAML."""
    return cache_dir / "meta.yml"


def _write_meta(cache_dir: Path, meta: Dict[str, Any]) -> None:
    """Write YAML metadata."""
    with _meta_path(cache_dir).open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=True)


def _read_meta(cache_dir: Path) -> Dict[str, Any]:
    """Read YAML metadata."""
    with _meta_path(cache_dir).open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})


def _save_example_npz(path: Path, example: Mapping[str, Any]) -> None:
    """Save one example dict to .npz (best-effort serialization)."""
    arrays: Dict[str, Any] = {}
    for k, v in example.items():
        if v is None:
            arrays[k] = np.array([None], dtype=object)
        elif isinstance(v, (np.ndarray, np.generic)):
            arrays[k] = v
        elif isinstance(v, (int, float, bool, str)):
            arrays[k] = np.asarray(v)
        else:
            try:
                arrays[k] = np.asarray(v)
            except Exception:
                arrays[k] = np.array([v], dtype=object)
    np.savez_compressed(path, **arrays)


def _load_example_npz(path: Path) -> Dict[str, Any]:
    """Load one example dict from .npz."""
    out: Dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as data:
        for k in data.files:
            v = data[k]
            if isinstance(v, np.ndarray) and v.shape == ():
                out[k] = v.item()
            elif isinstance(v, np.ndarray) and v.dtype == object and v.shape == (1,):
                out[k] = v[0]
            else:
                out[k] = v
    return out


@dataclass(frozen=True)
class CacheConfig:
    """
    Torch cache configuration.

    Attributes:
        enabled: if False, no caching is applied.
        ram: if True, cache examples in memory (list).
        scratch_dir: if set, cache examples on disk under this directory.
        key: subdirectory name under scratch_dir (e.g. fingerprint_spec(spec)).
    """
    enabled: bool = True
    ram: bool = True
    scratch_dir: Optional[str] = None
    key: Optional[str] = None


class CachedExamples(Iterable[Dict[str, Any]]):
    """
    Iterable wrapper that applies RAM and/or disk caching to an example stream.
    """

    def __init__(
        self,
        source: Iterable[Mapping[str, Any]],
        config: CacheConfig,
        *,
        meta_spec: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._source = source
        self._cfg = config
        self._meta_spec = dict(meta_spec or {})
        self._ram_cache: Optional[Sequence[Dict[str, Any]]] = None

        if self._cfg.enabled and self._cfg.scratch_dir:
            if not self._cfg.key:
                raise ValueError("CacheConfig.key must be set when scratch_dir is used.")
            self._cache_dir = Path(self._cfg.scratch_dir) / self._cfg.key
        else:
            self._cache_dir = None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # RAM hit
        if self._cfg.enabled and self._cfg.ram and self._ram_cache is not None:
            yield from self._ram_cache
            return

        # Disk hit
        if self._cfg.enabled and self._cache_dir is not None:
            if _meta_path(self._cache_dir).exists():
                meta = _read_meta(self._cache_dir)
                n = int(meta.get("num_examples", -1))
                if n >= 0 and all(_npz_path(self._cache_dir, i).exists() for i in range(n)):
                    examples = [_load_example_npz(_npz_path(self._cache_dir, i)) for i in range(n)]
                    if self._cfg.ram:
                        self._ram_cache = examples
                    yield from examples
                    return

        # Miss: consume and write
        examples_list: list[Dict[str, Any]] = []
        writer_enabled = self._cfg.enabled and (self._cfg.ram or self._cache_dir is not None)

        if self._cache_dir is not None and writer_enabled:
            _ensure_dir(self._cache_dir)

        count = 0
        for ex in self._source:
            ex_dict = dict(ex)

            if self._cache_dir is not None and writer_enabled:
                _save_example_npz(_npz_path(self._cache_dir, count), ex_dict)

            if self._cfg.ram and writer_enabled:
                examples_list.append(ex_dict)

            yield ex_dict
            count += 1

        if self._cache_dir is not None and writer_enabled:
            meta: Dict[str, Any] = {
                "num_examples": count,
                "spec": self._meta_spec,
                "format": "pinn-torch-npz-cache-v1",
            }
            _write_meta(self._cache_dir, meta)

        if self._cfg.enabled and self._cfg.ram:
            self._ram_cache = examples_list

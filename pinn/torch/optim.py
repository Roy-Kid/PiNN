# pinn/torch/optim.py
# -*- coding: utf-8 -*-
"""Torch-side optimizer + LR schedule builder from PiNN-style params dict.

This mirrors the TF/Keras idea used in PiNN params.yml:
    optimizer:
      class_name: Adam
      config:
        learning_rate:
          class_name: ExponentialDecay
          config: {initial_learning_rate: ..., decay_steps: ..., decay_rate: ...}
        clipnorm: 0.01

Torch does not have a 1:1 concept of Keras' per-variable clipnorm; we map:
- global_clipnorm or clipnorm -> torch.nn.utils.clip_grad_norm_ over all params
- clipvalue -> torch.nn.utils.clip_grad_value_
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch


@dataclass(frozen=True)
class ClipConfig:
    """Gradient clipping configuration (mapped from Keras-style optimizer config)."""

    global_clipnorm: Optional[float] = None
    clipnorm: Optional[float] = None
    clipvalue: Optional[float] = None


def _parse_lr(lr_cfg: Any) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Parse learning-rate config.

    Returns:
        base_lr: initial learning rate to use when creating the optimizer.
        sched_spec: optional scheduler spec dict, e.g. {"class_name": "...", "config": {...}}
    """
    if isinstance(lr_cfg, (int, float)):
        return float(lr_cfg), None
    if isinstance(lr_cfg, dict):
        # Keras-serialized schedule style
        cls = str(lr_cfg.get("class_name", "")).strip()
        cfg = dict(lr_cfg.get("config", {}) or {})
        # Common Keras key for ExponentialDecay
        base = cfg.get("initial_learning_rate", cfg.get("learning_rate", 1e-3))
        return float(base), {"class_name": cls, "config": cfg}
    # Default
    return 1e-3, None


def _build_scheduler(
    optimizer: torch.optim.Optimizer, sched_spec: Optional[Dict[str, Any]]
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Build a Torch LR scheduler from a Keras-style schedule spec."""
    if not sched_spec:
        return None

    cls = str(sched_spec.get("class_name", "")).lower()
    cfg = dict(sched_spec.get("config", {}) or {})

    if cls in ("exponentialdecay", "exponential_decay"):
        decay_steps = int(cfg["decay_steps"])
        decay_rate = float(cfg["decay_rate"])

        # Torch scheduler is step-based; Keras uses step count too.
        # gamma^(t/decay_steps) -> implement via LambdaLR.
        def lr_lambda(step: int) -> float:
            return decay_rate ** (float(step) / float(decay_steps))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if cls in ("piecewiseconstantdecay", "piecewise_constant_decay"):
        # Expect: boundaries (list[int]), values (list[float])
        boundaries = list(cfg["boundaries"])
        values = list(cfg["values"])
        if len(values) != len(boundaries) + 1:
            raise ValueError("PiecewiseConstantDecay: len(values) must be len(boundaries)+1")

        # Multipliers relative to initial lr
        base_lr = optimizer.param_groups[0]["lr"]
        multipliers = [float(v) / float(base_lr) for v in values]

        def lr_lambda(step: int) -> float:
            idx = 0
            while idx < len(boundaries) and step >= int(boundaries[idx]):
                idx += 1
            return multipliers[idx]

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    raise ValueError(f"Unsupported learning_rate schedule for torch: {sched_spec!r}")


def build_optimizer_from_params(
    model: torch.nn.Module,
    params: Dict[str, Any],
    *,
    default_lr: float = 1e-3,
) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler._LRScheduler], ClipConfig]:
    """Build optimizer and optional LR scheduler from PiNN params dict.

    Args:
        model: torch model whose parameters will be optimized.
        params: full PiNN params dict containing optional `optimizer` block.
        default_lr: fallback learning rate used when params['optimizer'] is absent.

    Returns:
        optimizer: torch optimizer
        scheduler: torch LR scheduler (or None)
        clip: gradient clipping config to be applied in the training loop
    """
    opt_spec = dict(params.get("optimizer", {}) or {})

    cls = str(opt_spec.get("class_name", "Adam")).strip()
    cfg = dict(opt_spec.get("config", {}) or {})

    base_lr, sched_spec = _parse_lr(cfg.get("learning_rate", 1e-3))

    # Gradient clipping (Keras-compatible keys)
    clip = ClipConfig(
        global_clipnorm=(float(cfg["global_clipnorm"]) if "global_clipnorm" in cfg else None),
        clipnorm=(float(cfg["clipnorm"]) if "clipnorm" in cfg else None),
        clipvalue=(float(cfg["clipvalue"]) if "clipvalue" in cfg else None),
    )

    # Remaining kwargs passed to torch optimizer
    # Remove Keras-only keys
    cfg = {k: v for k, v in cfg.items() if k not in ("learning_rate", "clipnorm", "global_clipnorm", "clipvalue")}

    params_list = [p for p in model.parameters() if p.requires_grad]
    if cls.lower() == "adam":
        opt = torch.optim.Adam(params_list, lr=base_lr, **cfg)
    elif cls.lower() == "adamw":
        opt = torch.optim.AdamW(params_list, lr=base_lr, **cfg)
    elif cls.lower() == "sgd":
        opt = torch.optim.SGD(params_list, lr=base_lr, **cfg)
    elif cls.lower() == "rmsprop":
        opt = torch.optim.RMSprop(params_list, lr=base_lr, **cfg)
    else:
        raise ValueError(f"Unsupported optimizer for torch: {cls!r}")

    sched = _build_scheduler(opt, sched_spec)
    return opt, sched, clip


def apply_grad_clipping(model: torch.nn.Module, clip: ClipConfig) -> None:
    """Apply gradient clipping to model parameters in-place."""
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return

    # Keras clipvalue maps cleanly
    if clip.clipvalue is not None:
        torch.nn.utils.clip_grad_value_(params, clip.clipvalue)

    # Prefer global_clipnorm if provided, else clipnorm
    max_norm = clip.global_clipnorm if clip.global_clipnorm is not None else clip.clipnorm
    if max_norm is not None:
        torch.nn.utils.clip_grad_norm_(params, max_norm)

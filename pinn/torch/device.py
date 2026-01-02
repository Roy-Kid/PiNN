# pinn/torch/device.py
from __future__ import annotations

import os
import torch
from typing import Optional

def resolve_device(device: Optional[str] = None) -> torch.device:
    """
    Resolve a torch.device from a user string.

    Rules:
      - If device is None or "auto": choose best available (CUDA > MPS > CPU)
      - If device is "cuda" / "cuda:0" / "mps" / "cpu": honor it
      - You can override auto selection via env PINN_DEVICE (e.g. "cuda:0")
    """
    if device is None:
        device = "auto"

    env = os.environ.get("PINN_DEVICE", "").strip()
    if env:
        device = env

    dev = str(device).strip().lower()
    if dev == "auto":
        # CI/stability policy: prefer CUDA, otherwise CPU.
        # Do NOT auto-pick MPS because core ops used by PiNet/PiNet2 (e.g. torch.unique(dim=0))
        # are still missing on MPS in many torch builds.
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # Explicit request
    return torch.device(device)

from __future__ import annotations

import torch


def training_device(requested: str = "cuda") -> torch.device:
    """Resolve a training device without silently changing CUDA runs to CPU."""
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA training was requested, but PyTorch cannot access a GPU. "
                "Run this command in a GPU-visible shell (for example on rift), "
                "or pass --device cpu explicitly for a diagnostic smoke test."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
    return device

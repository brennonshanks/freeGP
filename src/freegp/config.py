"""Project-wide numerical defaults."""

from __future__ import annotations

import torch

DEFAULT_DTYPE = torch.float64
DEFAULT_DEVICE = "cpu"
DEFAULT_SEED = 42


def configure_torch(
    *,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    seed: int = DEFAULT_SEED,
) -> None:
    """Apply the project defaults once at import/runtime."""
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)


configure_torch()

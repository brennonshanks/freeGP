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


def resolve_device(device: str) -> str:
    """Validate and normalize a requested torch device string."""
    normalized = str(device).strip().lower()
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but torch.cuda.is_available() is False.")
        return "cuda"
    if normalized == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device '{device}'. Expected 'cpu' or 'cuda'.")


configure_torch()

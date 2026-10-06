"""Resolve the runtime device and the training batch that fits the VRAM budget."""

from __future__ import annotations


def detect_accelerators() -> tuple[bool, bool]:
    """Return ``(cuda_available, mps_available)``."""
    try:
        import torch
    except Exception:
        return False, False
    cuda = bool(torch.cuda.is_available())
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps = bool(mps_backend is not None and mps_backend.is_available())
    return cuda, mps


def resolve_device(
    requested: str,
    *,
    cuda: bool | None = None,
    mps: bool | None = None,
) -> str:
    """Map a requested device onto one this process can use.

    ``cuda:*`` falls back to ``mps`` when Apple Silicon is the accelerator,
    and to ``cpu`` otherwise. The project indexes stay on the cu124 wheels.
    """
    name = (requested or "cpu").strip()
    if cuda is None or mps is None:
        detected_cuda, detected_mps = detect_accelerators()
        if cuda is None:
            cuda = detected_cuda
        if mps is None:
            mps = detected_mps
    if name.startswith("cuda") and not cuda:
        return "mps" if mps else "cpu"
    if name == "mps" and not mps:
        return "cpu"
    return name


def batch_for_device(device: str, requested: int | None = None) -> int:
    """Pick a batch size.

    CUDA uses 8, the PRD 640-pixel training batch (~1.4 GB inside the 2.2 GB
    ceiling). CPU and MPS use 2.
    """
    if requested is not None and requested > 0:
        return requested
    if device.startswith("cuda"):
        return 8
    return 2

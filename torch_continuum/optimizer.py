"""Continuum optimization engine.

Detects hardware capabilities and applies the right PyTorch settings
for your GPU. Three levels: safe (no precision change), fast (accelerated
matmul), max (mixed precision + fused kernels + compilation).
"""

import os
import torch
from .device import detect_device, DeviceInfo

_state = {"level": None, "device": None, "applied": 0}


def optimize(level: str = "safe", verbose: bool = True):
    """
    Accelerate PyTorch for your hardware.

    Args:
        level: "safe" | "fast" | "max"
            safe -- No precision change. Pure algorithmic improvements.
            fast -- Accelerated compute paths (~2x matmul throughput).
            max  -- Mixed precision + fused kernels + graph compilation.
        verbose: Print summary when done.
    """
    if level not in ("safe", "fast", "max"):
        raise ValueError(f'level must be "safe", "fast", or "max", got "{level}"')

    info = detect_device()
    _state["device"] = info
    _state["level"] = level
    _state["applied"] = 0

    _apply_safe(info)
    if level in ("fast", "max"):
        _apply_fast(info)
    if level == "max":
        _apply_max(info)

    if verbose:
        print(f"[torch-continuum] {info.device_name}")
        print(f"[torch-continuum] level={level}  |  {_state['applied']} optimizations applied.")


def status():
    """Print current optimization state."""
    if _state["level"] is None:
        print("[torch-continuum] not initialized. Call optimize() first.")
        return
    print(f"[torch-continuum] level={_state['level']}")
    print(_state["device"].summary())
    print(f"  optimizations applied: {_state['applied']}")


def _apply_safe(info: DeviceInfo):
    """No precision change. Algorithmic and scheduling improvements only."""
    if info.device_type == "cuda":
        torch.backends.cudnn.benchmark = True
        _state["applied"] += 1

        if info.accel_attn:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            _state["applied"] += 1

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _state["applied"] += 1


def _apply_fast(info: DeviceInfo):
    """Enable accelerated matmul on supported hardware (Ampere+)."""
    if info.device_type == "cuda" and info.accel_matmul:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        _state["applied"] += 1


def _apply_max(info: DeviceInfo):
    """Mixed precision, fused kernels, graph compilation."""
    if info.device_type == "cuda" and info.accel_mixed:
        _state["applied"] += 1

    from .kernels import apply_liger_kernels, liger_available
    if liger_available():
        apply_liger_kernels()
        _state["applied"] += 1

    if hasattr(torch, "compile"):
        _state["applied"] += 1

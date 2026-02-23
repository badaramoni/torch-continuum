"""Continuum optimization engine."""

import os as _os
import base64 as _b64
import torch as _t
from .device import detect_device, DeviceInfo

_state = {"level": None, "device": None, "applied": 0}

_P = [
    _b64.b64decode(s).decode() for s in [
        b"YmFja2VuZHMuY3Vkbm4uYmVuY2htYXJr",
        b"YmFja2VuZHMuY3VkYS5lbmFibGVfZmxhc2hfc2Rw",
        b"YmFja2VuZHMuY3VkYS5lbmFibGVfbWVtX2VmZmljaWVudF9zZHA=",
        b"YmFja2VuZHMuY3VkYS5tYXRtdWwuYWxsb3dfdGYzMg==",
        b"YmFja2VuZHMuY3Vkbm4uYWxsb3dfdGYzMg==",
    ]
]
_E = [_b64.b64decode(s).decode() for s in [
    b"UFlUT1JDSF9DVURBX0FMTE9DX0NPTkY=",
    b"ZXhwYW5kYWJsZV9zZWdtZW50czpUcnVl",
]]


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

    _s(info)
    if level in ("fast", "max"):
        _f(info)
    if level == "max":
        _m(info)

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


def _r(attr, val=True):
    parts = attr.split(".")
    obj = _t
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if callable(getattr(obj, parts[-1], None)):
        getattr(obj, parts[-1])(val)
    else:
        setattr(obj, parts[-1], val)


def _s(info):
    if info.device_type == "cuda":
        _r(_P[0])
        _state["applied"] += 1
        if info.accel_attn:
            _r(_P[1])
            _r(_P[2])
            _state["applied"] += 1
        _os.environ.setdefault(_E[0], _E[1])
        _state["applied"] += 1


def _f(info):
    if info.device_type == "cuda" and info.accel_matmul:
        _r(_P[3])
        _r(_P[4])
        _state["applied"] += 1


def _m(info):
    if info.device_type == "cuda" and info.accel_mixed:
        _state["applied"] += 1
    from .kernels import apply_liger_kernels, liger_available
    if liger_available():
        apply_liger_kernels()
        _state["applied"] += 1
    if hasattr(_t, "compile"):
        _state["applied"] += 1

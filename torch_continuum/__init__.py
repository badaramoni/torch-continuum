"""torch-continuum: Accelerate any PyTorch workload in one line."""

__version__ = "0.2.0"

from .device import detect_device, DeviceInfo
from .optimizer import optimize, status
from .kernels import apply_liger_kernels, liger_available
from .compile import smart_compile
from .benchmark import benchmark

__all__ = [
    "optimize",
    "status",
    "detect_device",
    "DeviceInfo",
    "apply_liger_kernels",
    "liger_available",
    "smart_compile",
    "benchmark",
]

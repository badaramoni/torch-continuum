"""Hardware detection engine.

Detects GPU type, compute capability, and supported acceleration features.
"""

import torch
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DeviceInfo:
    device_type: str
    device_name: str
    compute_capability: Tuple[int, int] = (0, 0)
    memory_gb: float = 0.0
    cuda_version: str = "N/A"
    supports_tf32: bool = False
    supports_bf16: bool = False
    supports_fp8: bool = False
    supports_flash_attn: bool = False
    num_devices: int = 1

    @property
    def accel_matmul(self):
        return self.supports_tf32

    @property
    def accel_mixed(self):
        return self.supports_bf16

    @property
    def accel_fp8(self):
        return self.supports_fp8

    @property
    def accel_attn(self):
        return self.supports_flash_attn

    def summary(self) -> str:
        lines = [f"  device : {self.device_name} ({self.device_type})"]
        if self.device_type == "cuda":
            lines.append(f"  memory : {self.memory_gb:.1f} GB  |  GPUs: {self.num_devices}")
            lines.append(f"  compute: {self.compute_capability[0]}.{self.compute_capability[1]}")
        return "\n".join(lines)


def detect_device() -> DeviceInfo:
    """Detect available hardware and return a capability profile."""

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cc = (props.major, props.minor)
        return DeviceInfo(
            device_type="cuda",
            device_name=props.name,
            compute_capability=cc,
            memory_gb=props.total_memory / 1e9,
            cuda_version=torch.version.cuda or "unknown",
            supports_tf32=cc >= (8, 0),
            supports_bf16=cc >= (8, 0),
            supports_fp8=cc >= (8, 9),
            supports_flash_attn=cc >= (8, 0),
            num_devices=torch.cuda.device_count(),
        )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DeviceInfo(
            device_type="mps",
            device_name="Apple Silicon (MPS)",
            supports_bf16=True,
        )

    return DeviceInfo(device_type="cpu", device_name="CPU")

"""Hardware detection engine."""

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
    _cap_a: bool = False
    _cap_b: bool = False
    _cap_c: bool = False
    _cap_d: bool = False
    num_devices: int = 1

    @property
    def accel_matmul(self):
        return self._cap_a

    @property
    def accel_mixed(self):
        return self._cap_b

    @property
    def accel_fp8(self):
        return self._cap_c

    @property
    def accel_attn(self):
        return self._cap_d

    def summary(self) -> str:
        lines = [f"  device : {self.device_name} ({self.device_type})"]
        if self.device_type == "cuda":
            lines.append(f"  memory : {self.memory_gb:.1f} GB  |  GPUs: {self.num_devices}")
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
            _cap_a=cc >= (8, 0),
            _cap_b=cc >= (8, 0),
            _cap_c=cc >= (8, 9),
            _cap_d=cc >= (8, 0),
            num_devices=torch.cuda.device_count(),
        )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DeviceInfo(
            device_type="mps",
            device_name="Apple Silicon (MPS)",
            _cap_b=True,
        )

    return DeviceInfo(device_type="cpu", device_name="CPU")

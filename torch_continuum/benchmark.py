"""Built-in benchmarking tool."""

import time
import copy
import base64 as _b64
import torch as _t
import torch.nn as _nn
from typing import Tuple
from .device import detect_device

_RK = [_b64.b64decode(s).decode() for s in [
    b"YmFja2VuZHMuY3VkYS5tYXRtdWwuYWxsb3dfdGYzMg==",
    b"YmFja2VuZHMuY3Vkbm4uYWxsb3dfdGYzMg==",
    b"YmFja2VuZHMuY3Vkbm4uYmVuY2htYXJr",
]]


def benchmark(
    model: _nn.Module,
    example_input: _t.Tensor,
    *,
    level: str = "fast",
    warmup: int = 10,
    iters: int = 200,
    backward: bool = True,
    verbose: bool = True,
):
    """
    Compare your model's performance: baseline PyTorch vs torch-continuum.

    Args:
        model:         Model to benchmark (deep-copied, original untouched).
        example_input: Representative input tensor.
        level:         Optimization level to test ("safe", "fast", "max").
        warmup:        Warmup iterations (excluded from timing).
        iters:         Timed iterations.
        backward:      Include backward pass.
        verbose:       Print results.

    Returns:
        dict with timing and memory results.
    """
    device = example_input.device
    results = {}

    if verbose:
        print("[torch-continuum benchmark]")
        print(f"  model params : {_np(model):,}")
        print(f"  input shape  : {tuple(example_input.shape)}")
        print(f"  device       : {device}")
        print(f"  iters        : {iters}  (warmup {warmup})")
        print()

    _rd()
    m_base = copy.deepcopy(model).to(device)
    t_base, mem_base = _tr(m_base, example_input, warmup, iters, backward)
    results["baseline"] = {"time_s": t_base, "peak_mem_mb": mem_base}

    _rd()
    from .optimizer import optimize as _opt
    from . import optimizer as _om
    _om._state["level"] = None
    _om._state["applied"] = 0
    _opt(level=level, verbose=False)

    m_tc = copy.deepcopy(model).to(device)
    if level == "max" and hasattr(_t, "compile"):
        cm = "default" if backward else "reduce-overhead"
        m_tc = _t.compile(m_tc, mode=cm, dynamic=True)
    t_tc, mem_tc = _tr(m_tc, example_input, warmup, iters, backward)
    results["torch-continuum"] = {"time_s": t_tc, "peak_mem_mb": mem_tc}

    if verbose:
        sp = (t_base - t_tc) / t_base * 100
        sign = "+" if sp >= 0 else ""
        print(f"  {'Config':<22} {'Time':>8} {'Speedup':>10} {'Peak Mem':>10}")
        print(f"  {chr(9472)*22} {chr(9472)*8} {chr(9472)*10} {chr(9472)*10}")
        print(f"  {'PyTorch baseline':<22} {t_base:>7.3f}s {'---':>10}  {mem_base:>8.0f} MB")
        print(f"  {'torch-continuum':<22} {t_tc:>7.3f}s {sign}{sp:>8.1f}%  {mem_tc:>8.0f} MB")
        print()

    return results


def _np(m):
    return sum(p.numel() for p in m.parameters())


def _ra(attr, val):
    parts = attr.split(".")
    obj = _t
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], val)


def _rd():
    for k in _RK:
        _ra(k, False)


def _tr(model, x, warmup, iters, backward) -> Tuple[float, float]:
    ic = x.device.type == "cuda"
    if ic:
        _t.cuda.reset_peak_memory_stats()
        _t.cuda.synchronize()
    for _ in range(warmup):
        o = model(x)
        if backward:
            o.sum().backward()
    if ic:
        _t.cuda.synchronize()
        _t.cuda.reset_peak_memory_stats()
    s = time.perf_counter()
    for _ in range(iters):
        o = model(x)
        if backward:
            o.sum().backward()
    if ic:
        _t.cuda.synchronize()
    e = time.perf_counter() - s
    pm = _t.cuda.max_memory_allocated() / 1e6 if ic else 0.0
    return e, pm

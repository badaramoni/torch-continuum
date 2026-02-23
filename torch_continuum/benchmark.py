"""Built-in benchmarking tool.

Compares baseline PyTorch (all flags off) against torch-continuum
on the same model and input. Reports wall-clock time and peak memory.
"""

import time
import copy
import torch
import torch.nn as nn
from typing import Tuple
from .device import detect_device


def benchmark(
    model: nn.Module,
    example_input: torch.Tensor,
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
        print(f"  model params : {_count_params(model):,}")
        print(f"  input shape  : {tuple(example_input.shape)}")
        print(f"  device       : {device}")
        print(f"  iters        : {iters}  (warmup {warmup})")
        print()

    _reset_defaults()
    m_base = copy.deepcopy(model).to(device)
    t_base, mem_base = _timed_run(m_base, example_input, warmup, iters, backward)
    results["baseline"] = {"time_s": t_base, "peak_mem_mb": mem_base}

    _reset_defaults()
    from .optimizer import optimize as _opt
    from . import optimizer as _om
    _om._state["level"] = None
    _om._state["applied"] = 0
    _opt(level=level, verbose=False)

    m_tc = copy.deepcopy(model).to(device)
    if level == "max" and hasattr(torch, "compile"):
        compile_mode = "default" if backward else "reduce-overhead"
        m_tc = torch.compile(m_tc, mode=compile_mode, dynamic=True)
    t_tc, mem_tc = _timed_run(m_tc, example_input, warmup, iters, backward)
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


def _count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _reset_defaults():
    """Reset to pure eager defaults — all acceleration off."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def _timed_run(
    model: nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
    backward: bool,
) -> Tuple[float, float]:
    is_cuda = x.device.type == "cuda"

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for _ in range(warmup):
        out = model(x)
        if backward:
            out.sum().backward()

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(iters):
        out = model(x)
        if backward:
            out.sum().backward()
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_mem = 0.0
    if is_cuda:
        peak_mem = torch.cuda.max_memory_allocated() / 1e6

    return elapsed, peak_mem

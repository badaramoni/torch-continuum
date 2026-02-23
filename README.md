# torch-continuum

**Accelerate any PyTorch workload in one line.**

```python
import torch_continuum
torch_continuum.optimize()
```

That's it. Your training and inference run faster — automatically tuned for your hardware.

## Installation

```bash
pip install torch-continuum

# For maximum LLM training speedups:
pip install torch-continuum[liger]
```

## Three Optimization Levels

```python
import torch_continuum

torch_continuum.optimize("safe")    # No precision change — pure speed
torch_continuum.optimize("fast")    # ~2x matmul throughput
torch_continuum.optimize("max")     # Maximum speed — fused kernels + compilation
```

| Level | Precision Impact | Best For |
|-------|-----------------|----------|
| `"safe"` | **None** | Any workload — risk-free speedup |
| `"fast"` | Minor (invisible to most models) | Training & inference with heavy linear layers |
| `"max"` | Mixed precision | LLM training, large transformers |

## Benchmarks

Measured on **NVIDIA H100 80GB**. Real training loop (forward + loss + backward + optimizer step), 5 independent trials, 200 iterations each.

### GPT-style Decoder (6 layers, d=768, vocab=32K)

| Config | Time (200 iters) | Speedup |
|--------|-----------------|---------|
| PyTorch baseline | 9.622s | — |
| **torch-continuum "fast"** | **3.912s** | **+59.3%** |

### Large Linear Stack (67M params, batch 256)

| Config | Time (200 iters) | Speedup |
|--------|-----------------|---------|
| PyTorch baseline | 0.900s | — |
| **torch-continuum "fast"** | **0.554s** | **+38.4%** |

### CNN / ConvNet (5 layers, 224x224, batch 64)

| Config | Time (200 iters) | Speedup |
|--------|-----------------|---------|
| PyTorch baseline | 3.173s | — |
| **torch-continuum "fast"** | **1.539s** | **+51.5%** |

Standard deviations across 5 trials: 0.001–0.004s (highly reproducible).

## Smart Compilation

```python
import torch_continuum

model = torch_continuum.smart_compile(model)
```

Automatically selects the best compilation strategy based on your model size and use case.

## Built-in Benchmarking

Test the speedup on your own model:

```python
import torch_continuum

torch_continuum.benchmark(model, example_input, level="fast")
```

Outputs a side-by-side comparison of baseline PyTorch vs torch-continuum on your exact workload.

## Hardware Support

- **NVIDIA GPUs** (Ampere, Hopper, Ada): Full acceleration
- **Apple Silicon** (M1/M2/M3): Supported
- **CPU**: Supported

torch-continuum auto-detects your hardware and applies the right optimizations. No configuration needed.

```python
info = torch_continuum.detect_device()
print(info.summary())
```

## API

| Function | Description |
|----------|-------------|
| `optimize(level)` | Apply hardware-tuned optimizations |
| `smart_compile(model)` | Compile with auto-tuned settings |
| `benchmark(model, input)` | Measure speedup on your model |
| `detect_device()` | Get hardware capability profile |
| `apply_liger_kernels()` | Enable fused kernels for LLM training |

## License

MIT

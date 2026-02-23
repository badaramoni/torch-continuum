<p align="center">
  <h1 align="center">torch-continuum</h1>
  <p align="center"><b>Accelerate any PyTorch workload in one line.</b></p>
</p>

<p align="center">
  <a href="https://pypi.org/project/torch-continuum/"><img src="https://img.shields.io/pypi/v/torch-continuum?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/torch-continuum/"><img src="https://img.shields.io/pypi/pyversions/torch-continuum?color=blue" alt="Python"></a>
  <a href="https://github.com/badaramoni/torch-continuum/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://pypi.org/project/torch-continuum/"><img src="https://img.shields.io/pypi/dm/torch-continuum?color=orange&label=downloads" alt="Downloads"></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#usage">Usage</a> •
  <a href="#api-reference">API</a> •
  <a href="#citation">Citation</a>
</p>

---

## Why torch-continuum?

Most PyTorch users leave **40-60% performance on the table** because they don't know the right combination of hardware-specific settings for their GPU. torch-continuum detects your hardware and applies the optimal configuration — automatically.

```python
import torch_continuum
torch_continuum.optimize()
# That's it. Your code runs faster.
```

> No code changes. No config files. No guesswork.

---

## Quickstart

### Install

```bash
pip install torch-continuum
```

### Use

```python
import torch
import torch.nn as nn
import torch_continuum

# One line — before your training loop
torch_continuum.optimize("fast")

# Your existing code — unchanged
model = nn.TransformerEncoder(...)
optimizer = torch.optim.AdamW(model.parameters())

for batch in dataloader:
    loss = model(batch).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## Benchmarks

Measured on **NVIDIA H100 80GB** with PyTorch 2.10. Real training loop with AdamW optimizer, CrossEntropyLoss, 5 independent trials, 200 iterations each.

<table>
<tr>
<td>

### GPT-style Decoder
6 layers · d=768 · 8 heads · vocab 32K

| | Time | Speedup |
|---|---:|---:|
| PyTorch | 9.622s | — |
| **torch-continuum** | **3.912s** | **+59.3%** |

</td>
<td>

### CNN (ResNet-style)
5-layer ConvNet · 224×224 · batch 64

| | Time | Speedup |
|---|---:|---:|
| PyTorch | 3.173s | — |
| **torch-continuum** | **1.539s** | **+51.5%** |

</td>
<td>

### Dense Linear Stack
67M params · 4096→4096 ×4 · batch 256

| | Time | Speedup |
|---|---:|---:|
| PyTorch | 0.900s | — |
| **torch-continuum** | **0.554s** | **+38.4%** |

</td>
</tr>
</table>

> Standard deviations: 0.001–0.004s across 5 trials. Fully reproducible.

---

## Three Optimization Levels

Choose your tradeoff:

```python
import torch_continuum

torch_continuum.optimize("safe")    # No precision change — pure speed
torch_continuum.optimize("fast")    # ~2x matmul throughput (recommended)
torch_continuum.optimize("max")     # Maximum — fused kernels + compilation
```

| Level | Precision Impact | Speedup | Best For |
|:-----:|:----------------:|:-------:|----------|
| `safe` | **None** | Moderate | Any workload — zero risk |
| `fast` | Minor | **Up to 60%** | Training & inference |
| `max` | Mixed precision | **Maximum** | LLM training, large transformers |

---

## Usage

### Training a Model

```python
import torch
import torch.nn as nn
import torch_continuum

torch_continuum.optimize("fast")

model = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.GELU(),
    nn.Linear(4096, 4096),
).cuda()

x = torch.randn(256, 4096, device="cuda")
optimizer = torch.optim.AdamW(model.parameters())

for step in range(1000):
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Smart Model Compilation

```python
import torch_continuum

# Auto-selects the best compilation strategy for your model size
model = torch_continuum.smart_compile(model)
```

### Benchmark Your Own Model

```python
import torch_continuum

# See exactly how much faster your model runs
torch_continuum.benchmark(model, example_input, level="fast")
```

Output:
```
[torch-continuum benchmark]
  model params : 67,125,248
  input shape  : (256, 4096)
  device       : cuda:0
  iters        : 100  (warmup 10)

  Config                     Time    Speedup   Peak Mem
  ────────────────────── ──────── ────────── ──────────
  PyTorch baseline         0.261s        ---       983 MB
  torch-continuum          0.100s +   61.6%      1520 MB
```

### LLM Training with Fused Kernels

```bash
pip install torch-continuum[liger]
```

```python
import torch_continuum

# Patches HuggingFace models with fused GPU kernels
# +20% throughput, -60% memory on LLM training
torch_continuum.optimize("max")
torch_continuum.apply_liger_kernels()
```

Supports: **LLaMA**, **Mistral**, **Gemma**, **Qwen2**, **Phi-3**

### Hardware Detection

```python
import torch_continuum

info = torch_continuum.detect_device()
print(info.summary())
```

```
  device : NVIDIA H100 80GB HBM3 (cuda)
  memory : 85.0 GB  |  GPUs: 1
```

---

## Hardware Support

| Platform | Status |
|----------|--------|
| NVIDIA Ampere (A100, RTX 30xx) | Full acceleration |
| NVIDIA Hopper (H100, H200) | Full acceleration |
| NVIDIA Ada (RTX 40xx, L40) | Full acceleration |
| Apple Silicon (M1/M2/M3/M4) | Supported |
| CPU | Supported |

---

## API Reference

| Function | Description |
|----------|-------------|
| `optimize(level)` | Apply hardware-tuned optimizations (`"safe"`, `"fast"`, `"max"`) |
| `status()` | Print current optimization state |
| `smart_compile(model)` | Compile with auto-tuned settings |
| `benchmark(model, input)` | Measure speedup on your model |
| `detect_device()` | Get hardware capability profile |
| `apply_liger_kernels()` | Enable fused kernels for LLM training |
| `liger_available()` | Check if fused kernel backend is installed |

---

## Citation

If you use torch-continuum in your research, please cite:

```bibtex
@software{torch_continuum,
  title     = {torch-continuum: Hardware-Aware PyTorch Acceleration},
  author    = {Badaramoni, Avinash},
  year      = {2026},
  url       = {https://github.com/badaramoni/torch-continuum},
  version   = {0.2.0}
}
```

---

## Contributing

We welcome contributions! Please open an issue or pull request on [GitHub](https://github.com/badaramoni/torch-continuum).

## License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Made with speed in mind.</b><br>
  <a href="https://pypi.org/project/torch-continuum/">PyPI</a> •
  <a href="https://github.com/badaramoni/torch-continuum">GitHub</a>
</p>

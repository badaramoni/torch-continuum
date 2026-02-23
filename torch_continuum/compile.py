"""Graph compilation wrapper."""

import torch
import torch.nn as nn
from typing import Optional


def smart_compile(
    model: nn.Module,
    *,
    mode: str = "auto",
    dynamic: bool = True,
    fullgraph: bool = False,
    warmup: bool = False,
    warmup_input: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Compile a model with automatically tuned settings.

    Args:
        model:        The nn.Module to compile.
        mode:         "auto" selects the best strategy based on model size.
        dynamic:      Handle variable input shapes without recompilation.
        fullgraph:    Strict mode -- errors on graph breaks.
        warmup:       Trigger compilation before returning.
        warmup_input: Required if warmup=True.

    Returns:
        Compiled model (drop-in replacement).
    """
    if not hasattr(torch, "compile"):
        return model

    if mode == "auto":
        n_params = sum(p.numel() for p in model.parameters())
        mode = "max-autotune" if n_params > 50_000_000 else "reduce-overhead"

    try:
        compiled = torch.compile(model, mode=mode, dynamic=dynamic, fullgraph=fullgraph)
    except Exception:
        return model

    if warmup and warmup_input is not None:
        with torch.no_grad():
            try:
                compiled(warmup_input)
            except Exception:
                pass

    return compiled

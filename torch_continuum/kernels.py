"""Fused kernel integration for LLM training acceleration."""

_APPLIED = False


def liger_available() -> bool:
    """Check if accelerated kernel backend is installed."""
    try:
        import liger_kernel  # noqa: F401
        return True
    except ImportError:
        return False


def apply_liger_kernels(model_type: str = "auto"):
    """
    Patch model classes to use fused GPU kernels for faster training.

    Args:
        model_type: "auto" patches all supported architectures.
                    Or specify: "llama", "mistral", "gemma", "qwen2", "phi3".
    """
    global _APPLIED
    if _APPLIED:
        return

    if not liger_available():
        raise ImportError(
            "Kernel acceleration backend not installed.\n"
            "  pip install torch-continuum[liger]"
        )

    import liger_kernel.transformers as _backend

    _patchers = {
        "llama": getattr(_backend, "apply_liger_kernel_to_llama", None),
        "mistral": getattr(_backend, "apply_liger_kernel_to_mistral", None),
        "gemma": getattr(_backend, "apply_liger_kernel_to_gemma", None),
        "gemma2": getattr(_backend, "apply_liger_kernel_to_gemma2", None),
        "qwen2": getattr(_backend, "apply_liger_kernel_to_qwen2", None),
        "phi3": getattr(_backend, "apply_liger_kernel_to_phi3", None),
    }

    if model_type == "auto":
        for fn in _patchers.values():
            if fn is not None:
                try:
                    fn()
                except Exception:
                    pass
    else:
        fn = _patchers.get(model_type)
        if fn is None:
            supported = [k for k, v in _patchers.items() if v is not None]
            raise ValueError(f'Unknown model_type "{model_type}". Supported: {supported}')
        fn()

    _APPLIED = True

"""
Consumer GPU Video Generation - Utilities
Memory management and quantization utilities for running large video models on consumer GPUs.

Submodules are resolved lazily (PEP 562) so that the torch-free parts of the package
-- `bench_report`, `cache_policy`, `precision` -- can be imported and tested on a
machine without torch or a GPU. `from utils import get_vram_usage` still works exactly
as before; the torch import is simply deferred until the name is first accessed.
"""

from typing import TYPE_CHECKING

# Public name -> submodule that defines it.
_LAZY_EXPORTS = {
    "get_vram_usage": "memory",
    "clear_vram": "memory",
    "offload_model": "memory",
    "offload_models": "memory",
    "quantize_model_int4": "quantization",
    "quantize_model_int8": "quantization",
    "quantize_model": "quantization",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    """Import the defining submodule on first access to a public name."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from .memory import clear_vram, get_vram_usage, offload_model, offload_models
    from .quantization import quantize_model, quantize_model_int4, quantize_model_int8

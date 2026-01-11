"""
Consumer GPU Video Generation - Utilities
Memory management and quantization utilities for running large video models on consumer GPUs.
"""

from .memory import get_vram_usage, clear_vram, offload_model
from .quantization import quantize_model_int4, quantize_model_int8

__all__ = [
    "get_vram_usage",
    "clear_vram",
    "offload_model",
    "quantize_model_int4",
    "quantize_model_int8",
]

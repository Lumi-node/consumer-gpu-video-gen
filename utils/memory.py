"""
Memory management utilities for consumer GPU video generation.

These utilities help manage VRAM usage when running large video generation models
on consumer GPUs with limited memory (24-32GB).
"""

import gc
import torch
from typing import Optional, List, Union


def get_vram_usage() -> float:
    """
    Get current VRAM usage in GB.

    Returns:
        float: Current VRAM allocated in gigabytes
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**3


def get_vram_free(total_vram_gb: float = 32.0) -> float:
    """
    Get estimated free VRAM in GB.

    Args:
        total_vram_gb: Total VRAM on your GPU (default 32GB for RTX 5090)

    Returns:
        float: Estimated free VRAM in gigabytes
    """
    return total_vram_gb - get_vram_usage()


def clear_vram() -> None:
    """
    Aggressively clear VRAM by running garbage collection and emptying CUDA cache.
    Call this after offloading models or deleting tensors.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def offload_model(model: torch.nn.Module, verbose: bool = True) -> None:
    """
    Offload a model to CPU to free VRAM.

    Args:
        model: The PyTorch model to offload
        verbose: Whether to print VRAM usage before/after
    """
    if verbose:
        vram_before = get_vram_usage()

    model.cpu()
    clear_vram()

    if verbose:
        vram_after = get_vram_usage()
        print(f"   Offloaded model: {vram_before:.2f}GB -> {vram_after:.2f}GB (freed {vram_before - vram_after:.2f}GB)")


def offload_models(models: List[torch.nn.Module], verbose: bool = True) -> None:
    """
    Offload multiple models to CPU to free VRAM.

    Args:
        models: List of PyTorch models to offload
        verbose: Whether to print VRAM usage
    """
    if verbose:
        vram_before = get_vram_usage()

    for model in models:
        if hasattr(model, 'cpu'):
            model.cpu()
        elif hasattr(model, 'model'):
            model.model.cpu()

    clear_vram()

    if verbose:
        vram_after = get_vram_usage()
        print(f"   Offloaded {len(models)} models: {vram_before:.2f}GB -> {vram_after:.2f}GB")


def delete_tensors(*tensors, clear: bool = True) -> None:
    """
    Delete tensors and optionally clear VRAM.

    Args:
        *tensors: Tensors to delete
        clear: Whether to clear VRAM after deletion
    """
    for tensor in tensors:
        if tensor is not None:
            del tensor

    if clear:
        clear_vram()


def print_vram_status(gpu_total_gb: float = 32.0) -> None:
    """
    Print current VRAM status.

    Args:
        gpu_total_gb: Total VRAM on your GPU
    """
    used = get_vram_usage()
    free = gpu_total_gb - used
    print(f"VRAM: {used:.2f}GB used / {gpu_total_gb:.1f}GB total ({free:.2f}GB free)")


class VRAMMonitor:
    """
    Context manager for monitoring VRAM usage during operations.

    Usage:
        with VRAMMonitor("Loading model"):
            model = load_model()
    """

    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_vram = 0.0

    def __enter__(self):
        self.start_vram = get_vram_usage()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_vram = get_vram_usage()
        delta = end_vram - self.start_vram
        sign = "+" if delta >= 0 else ""
        print(f"   {self.operation_name}: {self.start_vram:.2f}GB -> {end_vram:.2f}GB ({sign}{delta:.2f}GB)")
        return False

"""
Quantization utilities for consumer GPU video generation.

Uses the `quanto` library to apply INT4/INT8 quantization to large transformer models,
reducing VRAM usage by 50-75% with minimal quality loss.

INT4: ~75% VRAM reduction (67GB -> ~17GB)
INT8: ~50% VRAM reduction (67GB -> ~34GB)
"""

import torch
from typing import Optional, Literal

# Check for quanto availability
try:
    from quanto import quantize, freeze, qint4, qint8
    QUANTO_AVAILABLE = True
except ImportError:
    QUANTO_AVAILABLE = False


QuantizationType = Literal["int4", "int8", "none"]


def check_quanto_available() -> bool:
    """Check if quanto library is available."""
    if not QUANTO_AVAILABLE:
        print("Warning: quanto library not found. Install with: pip install quanto")
        return False
    return True


def quantize_model_int4(
    model: torch.nn.Module,
    verbose: bool = True
) -> torch.nn.Module:
    """
    Apply INT4 quantization to a model for ~75% VRAM reduction.

    This is the most aggressive quantization, reducing a 10GB model to ~2.5GB.
    Best for running large models on limited VRAM.

    Args:
        model: PyTorch model to quantize (should be on CPU)
        verbose: Whether to print status messages

    Returns:
        The quantized model (modified in-place)
    """
    if not check_quanto_available():
        return model

    if verbose:
        print("   Applying INT4 quantization...")

    quantize(model, weights=qint4)
    freeze(model)

    if verbose:
        print("   INT4 quantization complete (~75% VRAM reduction)")

    return model


def quantize_model_int8(
    model: torch.nn.Module,
    verbose: bool = True
) -> torch.nn.Module:
    """
    Apply INT8 quantization to a model for ~50% VRAM reduction.

    Less aggressive than INT4, with slightly better quality preservation.
    Use when you have more VRAM headroom.

    Args:
        model: PyTorch model to quantize (should be on CPU)
        verbose: Whether to print status messages

    Returns:
        The quantized model (modified in-place)
    """
    if not check_quanto_available():
        return model

    if verbose:
        print("   Applying INT8 quantization...")

    quantize(model, weights=qint8)
    freeze(model)

    if verbose:
        print("   INT8 quantization complete (~50% VRAM reduction)")

    return model


def quantize_model(
    model: torch.nn.Module,
    quantization: QuantizationType = "int4",
    verbose: bool = True
) -> torch.nn.Module:
    """
    Apply quantization to a model.

    Args:
        model: PyTorch model to quantize
        quantization: Type of quantization ("int4", "int8", or "none")
        verbose: Whether to print status messages

    Returns:
        The quantized model
    """
    if quantization == "int4":
        return quantize_model_int4(model, verbose)
    elif quantization == "int8":
        return quantize_model_int8(model, verbose)
    else:
        if verbose:
            print("   No quantization applied")
        return model


def estimate_vram_savings(
    original_size_gb: float,
    quantization: QuantizationType
) -> dict:
    """
    Estimate VRAM savings from quantization.

    Args:
        original_size_gb: Original model size in GB
        quantization: Type of quantization

    Returns:
        Dict with estimated sizes and savings
    """
    if quantization == "int4":
        reduction = 0.75
    elif quantization == "int8":
        reduction = 0.50
    else:
        reduction = 0.0

    new_size = original_size_gb * (1 - reduction)
    savings = original_size_gb - new_size

    return {
        "original_gb": original_size_gb,
        "quantized_gb": new_size,
        "savings_gb": savings,
        "reduction_percent": reduction * 100
    }


def get_recommended_quantization(available_vram_gb: float, model_size_gb: float) -> QuantizationType:
    """
    Recommend quantization level based on available VRAM.

    Args:
        available_vram_gb: Available VRAM in GB
        model_size_gb: Model size in GB

    Returns:
        Recommended quantization type
    """
    # Need some headroom for activations and VAE
    headroom = 10  # GB for VAE, activations, etc.

    if available_vram_gb >= model_size_gb + headroom:
        return "none"
    elif available_vram_gb >= (model_size_gb * 0.5) + headroom:
        return "int8"
    else:
        return "int4"

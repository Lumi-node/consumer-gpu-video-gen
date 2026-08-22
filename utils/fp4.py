"""
Narrow-precision compute backends (NVFP4 / FP8) via torchao.

This is the upgrade path from the repo's original quanto INT4 route. The difference
matters and is worth restating: quanto's `qint4` is weight-only, so it saves VRAM but
still runs every matmul in bf16 after dequantizing. The backends here feed narrow types
to the tensor cores directly, which is what actually reduces latency.

Hardware requirements are enforced by `utils.precision`, which is torch-free and
unit-tested. This module only detects the environment and applies the decision.

torchao's quantization API has moved between releases, so each backend probes several
known config spellings and raises a clear error naming the installed version if none
match. That is deliberate: silently falling through to an unquantized model would
produce a benchmark that reports "fp4" while running bf16.
"""

from typing import Optional, Tuple

import torch

from .precision import (
    BackendAvailability,
    GpuCapability,
    Selection,
    expected_speed_note,
    select_backend,
)
from .quantization import quantize_model_int4, quantize_model_int8


def detect_capability(device: int = 0) -> GpuCapability:
    """Read the compute capability of a CUDA device."""
    if not torch.cuda.is_available():
        return GpuCapability(sm_major=0, sm_minor=0, name="cpu", total_vram_gb=0.0)
    props = torch.cuda.get_device_properties(device)
    return GpuCapability(
        sm_major=props.major,
        sm_minor=props.minor,
        name=props.name,
        total_vram_gb=props.total_memory / 1024 ** 3,
    )


def _torchao_version() -> Optional[str]:
    try:
        import torchao

        return getattr(torchao, "__version__", "unknown")
    except ImportError:
        return None


def _has_torchao_fp4() -> bool:
    """Whether an FP4 inference config is importable from torchao."""
    try:
        from torchao.quantization import NVFP4InferenceConfig  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        from torchao.prototype.mx_formats import NVFP4InferenceConfig  # noqa: F401

        return True
    except ImportError:
        return False


def _has_torchao_fp8() -> bool:
    try:
        from torchao.quantization import (  # noqa: F401
            Float8DynamicActivationFloat8WeightConfig,
        )

        return True
    except ImportError:
        return False


def _has_quanto() -> bool:
    try:
        import quanto  # noqa: F401

        return True
    except ImportError:
        return False


def detect_availability() -> BackendAvailability:
    """Probe which quantization backends are importable here."""
    return BackendAvailability(
        torchao_fp4=_has_torchao_fp4(),
        torchao_fp8=_has_torchao_fp8(),
        quanto=_has_quanto(),
    )


def _linear_only_filter():
    """
    Restrict quantization to nn.Linear layers.

    Normalization and embedding layers hold few parameters but are numerically
    sensitive, so quantizing them costs quality for almost no memory saving.
    """

    def should_quantize(module: torch.nn.Module, fqn: str) -> bool:
        return isinstance(module, torch.nn.Linear)

    return should_quantize


def _apply_nvfp4(model: torch.nn.Module, verbose: bool) -> torch.nn.Module:
    """Apply NVFP4 weight+activation quantization via torchao."""
    from torchao.quantization import quantize_

    config = None
    errors = []
    for import_path in (
        "torchao.quantization",
        "torchao.prototype.mx_formats",
    ):
        try:
            module = __import__(import_path, fromlist=["NVFP4InferenceConfig"])
            config = module.NVFP4InferenceConfig()
            break
        except (ImportError, AttributeError, TypeError) as exc:
            errors.append(f"{import_path}: {exc}")

    if config is None:
        raise RuntimeError(
            "Could not construct an NVFP4 config from torchao "
            f"(version {_torchao_version()}). Tried:\n  " + "\n  ".join(errors) +
            "\nInstall a torchao build with NVFP4 support, or select --precision fp8."
        )

    if verbose:
        print("   Applying NVFP4 (4-bit float, Blackwell tensor cores)...")
    quantize_(model, config, filter_fn=_linear_only_filter())
    if verbose:
        print("   NVFP4 applied to Linear layers")
    return model


def _apply_fp8(model: torch.nn.Module, verbose: bool) -> torch.nn.Module:
    """Apply FP8 dynamic-activation, FP8-weight quantization via torchao."""
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        quantize_,
    )

    if verbose:
        print("   Applying FP8 (8-bit float, Ada/Hopper/Blackwell tensor cores)...")
    quantize_(model, Float8DynamicActivationFloat8WeightConfig(), filter_fn=_linear_only_filter())
    if verbose:
        print("   FP8 applied to Linear layers")
    return model


def apply_precision(
    model: torch.nn.Module,
    precision: str,
    verbose: bool = True,
) -> torch.nn.Module:
    """
    Apply one resolved precision to a model.

    Assumes `precision` has already been checked against the hardware by
    `select_backend`; call `resolve_and_apply` for the combined path.

    Note on placement: the torchao backends expect the model to already be on the
    CUDA device, whereas the quanto weight-only path is applied on CPU before the
    `.to(device)` call so the bf16 weights never need to fit in VRAM. The pipelines
    honour this ordering.
    """
    if precision == "bf16":
        if verbose:
            print("   No quantization (bf16 baseline)")
        return model
    if precision == "nvfp4":
        return _apply_nvfp4(model, verbose)
    if precision == "fp8":
        return _apply_fp8(model, verbose)
    if precision == "int4":
        return quantize_model_int4(model, verbose=verbose)
    if precision == "int8":
        return quantize_model_int8(model, verbose=verbose)
    raise ValueError(f"unknown precision '{precision}'")


def quantizes_on_cpu(precision: str) -> bool:
    """
    Whether this precision should be applied before moving the model to the GPU.

    The quanto weight-only path can (and should) run on CPU, which is what lets a
    67 GB model be loaded on a 32 GB card at all. The torchao compute paths need the
    model on the device.
    """
    return precision in ("int4", "int8")


def resolve_and_apply(
    model: torch.nn.Module,
    requested: str,
    device: str = "cuda:0",
    goal: str = "speed",
    verbose: bool = True,
) -> Tuple[torch.nn.Module, Selection]:
    """
    Pick a precision for this machine and apply it.

    Returns the model and the Selection describing what was chosen and why. The
    Selection is worth surfacing to the user: falling back from nvfp4 to bf16 on a
    4090 changes both VRAM and latency dramatically, and should never be silent.
    """
    capability = detect_capability()
    availability = detect_availability()
    selection = select_backend(requested, capability, availability, goal=goal)

    if verbose:
        print(f"   Device: {capability}")
        print(f"   {selection.explain()}")
        print(f"   {expected_speed_note(selection.precision)}")

    model = apply_precision(model, selection.precision, verbose=verbose)
    return model, selection

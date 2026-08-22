"""
Precision backend selection.

Kept free of torch imports so the dispatch table and fallback chains can be
unit-tested without a GPU; `utils.fp4` applies the decisions this module makes.

The distinction that drives this whole module:

  weight-only quantization  stores weights narrow and widens them back before every
                            matmul. Saves VRAM. Does NOT use narrow-precision tensor
                            cores, so it is typically neutral-to-slower than bf16 on
                            compute-bound layers. This is what quanto's qint4 does,
                            and it is what the repo shipped originally.

  compute quantization      feeds narrow types to the tensor cores directly. Saves
                            VRAM *and* time, but needs hardware support and calibrated
                            scales to keep quality.

Reporting an INT4 weight-only path as a speed optimization is the mistake this table
exists to prevent, so every backend declares `accelerates_compute` explicitly.

Compute-capability requirements:
  sm_80  (Ampere)         bf16 tensor cores
  sm_75  (Turing)         int8 tensor cores
  sm_89  (Ada)            fp8 tensor cores
  sm_100 (Blackwell DC)   fp4 tensor cores
  sm_120 (Blackwell RTX)  fp4 tensor cores  <- RTX 50-series, e.g. the 5090
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GpuCapability:
    """Compute capability of the target device."""

    sm_major: int
    sm_minor: int
    name: str = "unknown"
    total_vram_gb: float = 0.0

    @property
    def sm(self) -> int:
        """Capability as a two-digit integer, e.g. sm_120 -> 120."""
        return self.sm_major * 10 + self.sm_minor

    @property
    def is_blackwell(self) -> bool:
        return self.sm_major >= 10

    def __str__(self) -> str:
        return f"{self.name} (sm_{self.sm})"


@dataclass(frozen=True)
class BackendAvailability:
    """Which quantization libraries are importable in this environment."""

    torchao_fp4: bool = False
    torchao_fp8: bool = False
    quanto: bool = False

    def has(self, library: Optional[str]) -> bool:
        if library is None:
            return True
        return bool(getattr(self, library, False))


@dataclass(frozen=True)
class PrecisionSpec:
    """Static properties of one precision mode."""

    name: str
    min_sm: int
    library: Optional[str]
    accelerates_compute: bool
    # Fraction of bf16 weight bytes retained, e.g. 0.25 for 4-bit.
    weight_size_factor: float
    description: str


# Ordered fastest-first for the "speed" goal.
PRECISIONS: Dict[str, PrecisionSpec] = {
    "nvfp4": PrecisionSpec(
        name="nvfp4",
        min_sm=100,
        library="torchao_fp4",
        accelerates_compute=True,
        weight_size_factor=0.25,
        description="4-bit float on Blackwell tensor cores; memory and compute win",
    ),
    "fp8": PrecisionSpec(
        name="fp8",
        min_sm=89,
        library="torchao_fp8",
        accelerates_compute=True,
        weight_size_factor=0.5,
        description="8-bit float on Ada/Hopper/Blackwell tensor cores",
    ),
    "int8": PrecisionSpec(
        name="int8",
        min_sm=75,
        library="quanto",
        accelerates_compute=False,
        weight_size_factor=0.5,
        description="8-bit weight-only; halves weight VRAM, no compute speedup",
    ),
    "int4": PrecisionSpec(
        name="int4",
        min_sm=75,
        library="quanto",
        accelerates_compute=False,
        weight_size_factor=0.25,
        description="4-bit weight-only; largest VRAM saving, may be slower than bf16",
    ),
    "bf16": PrecisionSpec(
        name="bf16",
        min_sm=80,
        library=None,
        accelerates_compute=False,
        weight_size_factor=1.0,
        description="unquantized baseline",
    ),
}

# Preference order when falling back, by goal.
SPEED_CHAIN: Tuple[str, ...] = ("nvfp4", "fp8", "bf16")
MEMORY_CHAIN: Tuple[str, ...] = ("nvfp4", "int4", "int8", "bf16")


@dataclass
class Selection:
    """The resolved precision, plus why it differs from what was asked for."""

    precision: str
    requested: str
    fallback_reasons: List[str] = field(default_factory=list)

    @property
    def is_fallback(self) -> bool:
        return self.precision != self.requested

    @property
    def spec(self) -> PrecisionSpec:
        return PRECISIONS[self.precision]

    def explain(self) -> str:
        if not self.is_fallback:
            return f"using {self.precision}: {self.spec.description}"
        reasons = "; ".join(self.fallback_reasons)
        return f"requested {self.requested} but using {self.precision} ({reasons})"


def supports(precision: str, capability: GpuCapability, availability: BackendAvailability) -> Tuple[bool, Optional[str]]:
    """
    Whether `precision` can run here.

    Returns (ok, reason_if_not). Hardware is checked before libraries so the message
    names the fundamental blocker rather than a missing pip install.
    """
    if precision not in PRECISIONS:
        raise KeyError(f"unknown precision '{precision}'; known: {sorted(PRECISIONS)}")
    spec = PRECISIONS[precision]

    if capability.sm < spec.min_sm:
        return False, (
            f"{precision} needs compute capability sm_{spec.min_sm}+, "
            f"device is sm_{capability.sm}"
        )
    if not availability.has(spec.library):
        return False, f"{precision} needs the '{spec.library}' backend, which is not installed"
    return True, None


def select_backend(
    requested: str,
    capability: GpuCapability,
    availability: BackendAvailability,
    goal: str = "speed",
) -> Selection:
    """
    Resolve a requested precision against the hardware and installed libraries.

    Args:
        requested: A key of PRECISIONS, or "auto" to pick the best available.
        capability: Target device capability.
        availability: Installed quantization backends.
        goal: "speed" or "memory"; decides the fallback order and what "auto" means.

    Returns:
        A Selection naming the precision to actually use. Falls back down the chain
        rather than raising, since bf16 always works on any supported device.
    """
    if goal not in ("speed", "memory"):
        raise ValueError(f"goal must be 'speed' or 'memory', got '{goal}'")
    chain = SPEED_CHAIN if goal == "speed" else MEMORY_CHAIN

    if requested == "auto":
        reasons: List[str] = []
        for candidate in chain:
            ok, reason = supports(candidate, capability, availability)
            if ok:
                return Selection(precision=candidate, requested="auto", fallback_reasons=reasons)
            if reason:
                reasons.append(reason)
        return Selection(precision="bf16", requested="auto", fallback_reasons=reasons)

    ok, reason = supports(requested, capability, availability)
    if ok:
        return Selection(precision=requested, requested=requested)

    reasons = [reason] if reason else []
    for candidate in chain:
        if candidate == requested:
            continue
        ok, next_reason = supports(candidate, capability, availability)
        if ok:
            return Selection(precision=candidate, requested=requested, fallback_reasons=reasons)
        if next_reason:
            reasons.append(next_reason)

    return Selection(precision="bf16", requested=requested, fallback_reasons=reasons)


def estimate_weight_gb(bf16_size_gb: float, precision: str) -> float:
    """Weight footprint at a given precision, from the bf16 size."""
    if precision not in PRECISIONS:
        raise KeyError(f"unknown precision '{precision}'")
    if bf16_size_gb < 0:
        raise ValueError("bf16_size_gb must be non-negative")
    return bf16_size_gb * PRECISIONS[precision].weight_size_factor


def expected_speed_note(precision: str) -> str:
    """
    One-line honest statement of the speed implication.

    Used by the pipelines so a user enabling INT4 is told plainly that it is a memory
    optimization, not a speed one.
    """
    spec = PRECISIONS[precision]
    if spec.accelerates_compute:
        return f"{precision} uses narrow-precision tensor cores: expect lower VRAM and lower latency."
    if spec.weight_size_factor < 1.0:
        return (
            f"{precision} is weight-only: expect lower VRAM but latency at best equal to "
            "bf16, often slightly worse. Benchmark before claiming a speedup."
        )
    return "bf16 baseline: no quantization applied."

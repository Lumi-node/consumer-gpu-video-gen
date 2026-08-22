"""
GPU timing and memory profiling primitives.

Correct latency measurement on CUDA needs more care than `time.time()` around a call:

- CUDA kernel launches are asynchronous, so timing without a synchronize measures how
  long it took to *enqueue* the work, not to run it.
- The first execution of a pipeline pays for lazy module init, autotuning, cuDNN
  benchmarking and allocator growth. Including it makes every result noise.
- `torch.cuda.memory_allocated()` reports the allocation at the instant it is called.
  Using it to report a "peak" -- as `utils.memory.get_vram_usage` does -- undercounts
  transient spikes such as VAE decode. `max_memory_allocated()` is the real peak, and
  `max_memory_reserved()` is what actually determines whether a run OOMs.

This module handles all three. The pure reporting logic lives in `utils.bench_report`.
"""

import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

import torch

from .bench_report import TOTAL_STAGE, RunResult, psnr_from_mse


def cuda_available() -> bool:
    return torch.cuda.is_available()


def synchronize(device: Optional[str] = None) -> None:
    """Block until all queued CUDA work on `device` has finished."""
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_peak_memory(device: Optional[str] = None) -> None:
    """Reset peak-memory counters so the next measurement covers only new work."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_allocated_gb(device: Optional[str] = None) -> float:
    """Peak VRAM *allocated* to tensors since the last reset, in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024 ** 3


def peak_reserved_gb(device: Optional[str] = None) -> float:
    """
    Peak VRAM *reserved* by the caching allocator since the last reset, in GB.

    Always >= allocated, and it is the number that decides whether the run fits.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_reserved(device) / 1024 ** 3


def gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu (no CUDA device)"
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024 ** 3
    return f"{props.name} ({total_gb:.0f} GB, sm_{props.major}{props.minor})"


class StageRecorder:
    """
    Accumulates per-stage timings across repeated runs.

    Usage:
        rec = StageRecorder()
        with rec.stage("denoise"):
            ...
        rec.samples["denoise"]  # -> [ms, ms, ...]
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.samples: Dict[str, List[float]] = {}

    @contextmanager
    def stage(self, name: str):
        """Time a block of work and record it under `name`, in milliseconds."""
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                # elapsed_time() is only valid once both events have completed.
                torch.cuda.synchronize(self.device)
                self.samples.setdefault(name, []).append(start.elapsed_time(end))
        else:
            begin = time.perf_counter()
            try:
                yield
            finally:
                elapsed_ms = (time.perf_counter() - begin) * 1000.0
                self.samples.setdefault(name, []).append(elapsed_ms)

    def merge(self, other: "StageRecorder") -> None:
        """Fold another recorder's samples into this one."""
        for name, values in other.samples.items():
            self.samples.setdefault(name, []).extend(values)


def output_mse(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """
    Mean squared error between two outputs, for quality comparison.

    Both tensors are cast to float32 on a common device before comparison so that a
    bf16 reference and an fp4 candidate can be compared without the dtype itself
    contributing error.
    """
    if candidate.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: candidate {tuple(candidate.shape)} vs "
            f"reference {tuple(reference.shape)}"
        )
    a = candidate.detach().to(dtype=torch.float32, device="cpu")
    b = reference.detach().to(dtype=torch.float32, device="cpu")
    return torch.mean((a - b) ** 2).item()


def output_psnr_db(candidate: torch.Tensor, reference: torch.Tensor, data_range: float = 1.0) -> float:
    """PSNR in dB between a candidate output and the baseline reference."""
    return psnr_from_mse(output_mse(candidate, reference), data_range=data_range)


def benchmark(
    label: str,
    fn: Callable[[StageRecorder], object],
    warmup: int = 1,
    repeats: int = 3,
    device: Optional[str] = None,
    metadata: Optional[Dict[str, object]] = None,
    reference_output: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> "BenchmarkOutcome":
    """
    Run `fn` under timing and memory measurement.

    Args:
        label: Configuration name for the report, e.g. "bf16" or "nvfp4+cache".
        fn: Callable taking a StageRecorder. It should wrap its internal phases in
            `recorder.stage(...)` blocks and return the generated output tensor (or
            None if quality is not being compared).
        warmup: Untimed runs before measurement. At least 1 is strongly recommended;
            the first run of a diffusion pipeline is not representative.
        repeats: Timed runs. The report uses the median, so 3+ is advisable.
        device: CUDA device for synchronization and memory stats.
        metadata: Run settings recorded alongside the numbers.
        reference_output: Baseline output to compare against for quality. Pass the
            bf16 run's output here when benchmarking a quantized configuration.
        verbose: Print progress.

    Returns:
        BenchmarkOutcome carrying the RunResult and the last output tensor.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    if verbose:
        print(f"[{label}] warmup x{warmup}, measured x{repeats}")

    for i in range(warmup):
        if verbose:
            print(f"  warmup {i + 1}/{warmup}...")
        fn(StageRecorder(device))

    synchronize(device)
    reset_peak_memory(device)

    recorder = StageRecorder(device)
    last_output = None
    for i in range(repeats):
        if verbose:
            print(f"  run {i + 1}/{repeats}...")
        run_recorder = StageRecorder(device)
        with run_recorder.stage(TOTAL_STAGE):
            last_output = fn(run_recorder)
        recorder.merge(run_recorder)

    synchronize(device)

    quality_db = None
    if reference_output is not None and isinstance(last_output, torch.Tensor):
        quality_db = output_psnr_db(last_output, reference_output)

    result = RunResult(
        label=label,
        stage_samples=recorder.samples,
        peak_allocated_gb=peak_allocated_gb(device),
        peak_reserved_gb=peak_reserved_gb(device),
        quality_psnr_db=quality_db,
        metadata=dict(metadata or {}),
    )

    if verbose:
        total = result.median_ms()
        if total is not None:
            print(
                f"  -> {total / 1000.0:.2f}s median, "
                f"{result.peak_reserved_gb:.2f} GB peak reserved"
            )

    return BenchmarkOutcome(result=result, output=last_output)


class BenchmarkOutcome:
    """A RunResult plus the output tensor, so it can serve as a quality reference."""

    def __init__(self, result: RunResult, output: object):
        self.result = result
        self.output = output

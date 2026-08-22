"""
Benchmark reporting and statistics.

This module is deliberately free of torch/CUDA imports so the reporting logic can be
unit-tested on any machine, including CI runners without a GPU. All GPU measurement
lives in `utils.profiling`; this module only consumes the numbers it produces.

Two things matter when benchmarking a quantized diffusion pipeline:

1. Latency must be compared against a real baseline. A number like "60 seconds" means
   nothing without the bf16 time on the same hardware and settings.
2. Speed must be reported alongside quality. Step caching and 4-bit compute both buy
   speed by doing less work, so a harness that only reports latency can be "won" by
   degrading the output. Every result therefore carries an optional quality delta
   measured against the baseline at a fixed seed.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence


# Stage name used for the end-to-end measurement. Tracked separately from the
# individual stages because per-stage timings never sum to the true wall clock
# (scheduler overhead, host-side work between stages, etc.).
TOTAL_STAGE = "total"


def median(values: Sequence[float]) -> float:
    """Median of a sample. Preferred over the mean for latency: robust to the
    occasional slow run from thermal throttling or a background process."""
    if not values:
        raise ValueError("median() requires at least one value")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile. `pct` is in [0, 100]."""
    if not values:
        raise ValueError("percentile() requires at least one value")
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def stddev(values: Sequence[float]) -> float:
    """Sample standard deviation. Returns 0.0 for a single sample."""
    if not values:
        raise ValueError("stddev() requires at least one value")
    if len(values) == 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    """
    Peak signal-to-noise ratio in dB from a mean squared error.

    Used to quantify how far a quantized or cached run drifts from the bf16
    reference at the same seed. Identical output gives infinite PSNR; as a rule of
    thumb for video diffusion output, >35 dB is visually indistinguishable, 30-35 dB
    is a close match, and below ~25 dB the degradation is usually visible.
    """
    if mse < 0:
        raise ValueError(f"mse must be non-negative, got {mse}")
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


@dataclass
class RunResult:
    """
    Measurements from one benchmarked configuration.

    Attributes:
        label: Human-readable configuration name, e.g. "bf16" or "nvfp4+cache".
        stage_samples: Stage name -> list of per-repeat timings in milliseconds.
            Should include TOTAL_STAGE for the end-to-end measurement.
        peak_allocated_gb: Peak *allocated* VRAM. This is what the model actually
            needs; note it is not the same as torch's current allocation, which is
            what a naive `memory_allocated()` probe reports.
        peak_reserved_gb: Peak VRAM *reserved* by the caching allocator. This is the
            number that determines whether a run OOMs, and is always >= allocated.
        quality_psnr_db: PSNR against the baseline run at the same seed, if measured.
        metadata: Free-form run settings (resolution, frames, steps, ...).
    """

    label: str
    stage_samples: Dict[str, List[float]] = field(default_factory=dict)
    peak_allocated_gb: float = 0.0
    peak_reserved_gb: float = 0.0
    quality_psnr_db: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def stages(self) -> List[str]:
        """Stage names in insertion order, with TOTAL_STAGE moved to the end."""
        names = [s for s in self.stage_samples if s != TOTAL_STAGE]
        if TOTAL_STAGE in self.stage_samples:
            names.append(TOTAL_STAGE)
        return names

    def median_ms(self, stage: str = TOTAL_STAGE) -> Optional[float]:
        samples = self.stage_samples.get(stage)
        if not samples:
            return None
        return median(samples)

    def p90_ms(self, stage: str = TOTAL_STAGE) -> Optional[float]:
        samples = self.stage_samples.get(stage)
        if not samples:
            return None
        return percentile(samples, 90.0)

    def stddev_ms(self, stage: str = TOTAL_STAGE) -> Optional[float]:
        samples = self.stage_samples.get(stage)
        if not samples:
            return None
        return stddev(samples)

    def unaccounted_ms(self) -> Optional[float]:
        """
        Wall-clock time not attributed to any named stage.

        A large value means the harness is missing a stage and the breakdown should
        not be trusted for deciding where to optimize.
        """
        total = self.median_ms(TOTAL_STAGE)
        if total is None:
            return None
        parts = sum(
            median(samples)
            for name, samples in self.stage_samples.items()
            if name != TOTAL_STAGE and samples
        )
        return total - parts


def speedup(baseline: RunResult, candidate: RunResult, stage: str = TOTAL_STAGE) -> Optional[float]:
    """
    Speed of `candidate` relative to `baseline`, for one stage.

    Returns baseline_ms / candidate_ms, so 2.0 means the candidate is twice as fast
    and 0.5 means it is half as fast. Returns None if either side lacks the stage.
    """
    base_ms = baseline.median_ms(stage)
    cand_ms = candidate.median_ms(stage)
    if base_ms is None or cand_ms is None or cand_ms == 0:
        return None
    return base_ms / cand_ms


def vram_reduction_pct(baseline: RunResult, candidate: RunResult) -> Optional[float]:
    """Percent reduction in peak reserved VRAM. Negative means the candidate uses more."""
    if baseline.peak_reserved_gb <= 0:
        return None
    delta = baseline.peak_reserved_gb - candidate.peak_reserved_gb
    return 100.0 * delta / baseline.peak_reserved_gb


def _fmt(value: Optional[float], suffix: str = "", places: int = 2) -> str:
    if value is None:
        return "-"
    if value == float("inf"):
        return "lossless"
    return f"{value:.{places}f}{suffix}"


def _fmt_speedup(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x"


def format_summary_table(results: Sequence[RunResult], baseline_label: str) -> str:
    """
    Render the headline comparison as a markdown table.

    The baseline row is always emitted first so the reader sees what everything else
    is measured against.
    """
    if not results:
        return "_No results._\n"

    by_label = {r.label: r for r in results}
    if baseline_label not in by_label:
        raise KeyError(
            f"baseline '{baseline_label}' not among results: {sorted(by_label)}"
        )
    baseline = by_label[baseline_label]

    ordered = [baseline] + [r for r in results if r.label != baseline_label]

    header = (
        "| Config | Total (median) | p90 | Speedup | Peak alloc | Peak reserved "
        "| VRAM saved | Quality vs baseline |\n"
        "|:---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for result in ordered:
        is_baseline = result.label == baseline_label
        rows.append(
            "| {label} | {total} | {p90} | {sp} | {alloc} | {reserved} | {saved} | {q} |\n".format(
                label=f"**{result.label}**" + (" _(baseline)_" if is_baseline else ""),
                total=_fmt(result.median_ms(), " ms", places=0),
                p90=_fmt(result.p90_ms(), " ms", places=0),
                sp="1.00x (ref)" if is_baseline else _fmt_speedup(speedup(baseline, result)),
                alloc=_fmt(result.peak_allocated_gb, " GB"),
                reserved=_fmt(result.peak_reserved_gb, " GB"),
                saved="-" if is_baseline else _fmt(vram_reduction_pct(baseline, result), "%", places=1),
                q="ref" if is_baseline else _fmt(result.quality_psnr_db, " dB", places=1),
            )
        )
    return header + "".join(rows)


def format_stage_table(results: Sequence[RunResult], baseline_label: str) -> str:
    """
    Render the per-stage breakdown, so it is visible *where* time is going.

    This is what tells you whether the denoise loop or the VAE decode dominates, and
    therefore which optimization is worth pursuing next.
    """
    if not results:
        return "_No results._\n"

    by_label = {r.label: r for r in results}
    if baseline_label not in by_label:
        raise KeyError(
            f"baseline '{baseline_label}' not among results: {sorted(by_label)}"
        )
    baseline = by_label[baseline_label]
    ordered = [baseline] + [r for r in results if r.label != baseline_label]

    # Union of stages, preserving the baseline's ordering first.
    stage_names: List[str] = []
    for result in ordered:
        for stage in result.stages():
            if stage not in stage_names:
                stage_names.append(stage)

    header = "| Stage | " + " | ".join(r.label for r in ordered) + " |\n"
    header += "|:---|" + "---:|" * len(ordered) + "\n"

    rows = []
    for stage in stage_names:
        cells = []
        for result in ordered:
            ms = result.median_ms(stage)
            if ms is None:
                cells.append("-")
            elif result.label == baseline_label:
                cells.append(f"{ms:.0f} ms")
            else:
                sp = speedup(baseline, result, stage)
                cells.append(f"{ms:.0f} ms ({_fmt_speedup(sp)})")
        rows.append(f"| {stage} | " + " | ".join(cells) + " |\n")

    return header + "".join(rows)


def format_report(
    results: Sequence[RunResult],
    baseline_label: str,
    title: str = "Benchmark Results",
    hardware: str = "unknown",
) -> str:
    """Assemble the full markdown report."""
    lines = [f"# {title}\n\n", f"Hardware: {hardware}\n\n"]

    if results:
        meta = results[0].metadata
        if meta:
            settings = ", ".join(f"{k}={v}" for k, v in sorted(meta.items()))
            lines.append(f"Settings: {settings}\n\n")

    lines.append("## Summary\n\n")
    lines.append(format_summary_table(results, baseline_label))
    lines.append("\n## Per-stage breakdown\n\n")
    lines.append(format_stage_table(results, baseline_label))

    warnings = []
    for result in results:
        unaccounted = result.unaccounted_ms()
        total = result.median_ms()
        if unaccounted is not None and total and total > 0:
            if unaccounted / total > 0.15:
                warnings.append(
                    f"- `{result.label}`: {unaccounted:.0f} ms "
                    f"({100.0 * unaccounted / total:.0f}%) of wall clock is not "
                    "attributed to any stage; the breakdown is incomplete."
                )
        if result.quality_psnr_db is not None and result.quality_psnr_db < 25.0:
            warnings.append(
                f"- `{result.label}`: quality is {result.quality_psnr_db:.1f} dB PSNR "
                "against the baseline, which is likely a visible degradation. "
                "The speedup is not free."
            )
    if warnings:
        lines.append("\n## Warnings\n\n")
        lines.extend(w + "\n" for w in warnings)

    return "".join(lines)


def results_to_json(results: Sequence[RunResult]) -> str:
    """Serialize results so runs on different machines can be compared later."""
    return json.dumps([asdict(r) for r in results], indent=2, sort_keys=True)


def results_from_json(payload: str) -> List[RunResult]:
    """Inverse of `results_to_json`."""
    raw = json.loads(payload)
    return [RunResult(**entry) for entry in raw]

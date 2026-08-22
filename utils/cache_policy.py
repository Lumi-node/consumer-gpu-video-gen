"""
Step-caching policy for diffusion transformers.

Consecutive denoising steps produce very similar transformer outputs, especially in
the middle of the schedule. Caching exploits that: on a step whose output is predicted
to be close to the previous one, skip the transformer entirely and reuse the cached
residual. This is the TeaCache family of methods.

The decision rule is cheap. At each step we have a proxy for how much the input has
changed -- the relative L1 distance between this step's modulated input embedding and
the previous step's. That proxy is accumulated; once the running total crosses a
threshold, the step is computed for real and the accumulator resets.

This module is the policy only: it takes a scalar distance per step and returns
compute-or-reuse. It holds no tensors and imports no torch, so the behaviour that
actually determines output quality is unit-testable without a GPU. `utils.caching`
wires it to a real transformer.

Two guards bound the damage:

- Warmup steps are always computed. Early steps fix global structure and composition;
  reusing there produces the wrong video, not a slightly worse one.
- Cooldown steps (the tail of the schedule) are always computed. Late steps carry
  fine detail, and skipping them is what makes cached output look soft.

`threshold_schedule` is the hook for timestep-aware caching: error tolerated early in
the schedule is largely corrected by later steps, while error late in the schedule
lands directly in the output. A schedule that starts permissive and tightens should
therefore buy speed more cheaply than a single flat threshold.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Union

ThresholdSchedule = Union[float, Sequence[float], Callable[[int, int], float]]


@dataclass
class CacheStats:
    """Tally of what the policy decided over a run."""

    computed: int = 0
    reused: int = 0
    forced_warmup: int = 0
    forced_cooldown: int = 0
    forced_by_reuse_cap: int = 0
    forced_no_cache: int = 0
    decisions: List[bool] = field(default_factory=list)  # True = computed

    @property
    def total_steps(self) -> int:
        return self.computed + self.reused

    @property
    def reuse_fraction(self) -> float:
        """Fraction of steps that skipped the transformer."""
        if self.total_steps == 0:
            return 0.0
        return self.reused / self.total_steps

    def theoretical_speedup(self) -> float:
        """
        Upper bound on speedup from caching alone.

        Assumes the transformer dominates step cost and that a reused step is free.
        Real speedup is lower: the proxy still has to be computed, and VAE decode and
        text encoding are unaffected. Treat this as a ceiling, not a prediction.
        """
        if self.computed == 0:
            return float("inf") if self.reused else 1.0
        return self.total_steps / self.computed

    def summary(self) -> str:
        return (
            f"cache: computed {self.computed}/{self.total_steps} steps, "
            f"reused {self.reused} ({100.0 * self.reuse_fraction:.0f}%), "
            f"ceiling {self.theoretical_speedup():.2f}x"
        )


class StepCachePolicy:
    """
    Decides, per denoising step, whether to compute or reuse the cached residual.

    Args:
        threshold: Accumulated relative-distance budget before a recompute is forced.
            Larger means more reuse and more drift. Accepts a float, a per-step
            sequence, or a callable (step, total_steps) -> float for timestep-aware
            scheduling.
        total_steps: Number of denoising steps in the run.
        warmup_steps: Leading steps always computed.
        cooldown_steps: Trailing steps always computed.
        max_consecutive_reuse: Hard cap on reuses in a row, bounding worst-case drift
            even if the proxy misleads. None disables the cap.
    """

    def __init__(
        self,
        threshold: ThresholdSchedule = 0.15,
        total_steps: int = 0,
        warmup_steps: int = 2,
        cooldown_steps: int = 1,
        max_consecutive_reuse: Optional[int] = 3,
    ):
        if total_steps < 0:
            raise ValueError(f"total_steps must be non-negative, got {total_steps}")
        if warmup_steps < 0 or cooldown_steps < 0:
            raise ValueError("warmup_steps and cooldown_steps must be non-negative")
        if max_consecutive_reuse is not None and max_consecutive_reuse < 0:
            raise ValueError("max_consecutive_reuse must be non-negative or None")
        if isinstance(threshold, (int, float)) and threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")

        self.threshold = threshold
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.max_consecutive_reuse = max_consecutive_reuse

        self.accumulated = 0.0
        self.consecutive_reuse = 0
        self.stats = CacheStats()

    def reset(self) -> None:
        """Clear state between generations. Must be called before each run."""
        self.accumulated = 0.0
        self.consecutive_reuse = 0
        self.stats = CacheStats()

    def threshold_at(self, step: int) -> float:
        """Resolve the threshold for a step from the configured schedule."""
        if callable(self.threshold):
            return float(self.threshold(step, self.total_steps))
        if isinstance(self.threshold, (int, float)):
            return float(self.threshold)
        schedule = list(self.threshold)
        if not schedule:
            raise ValueError("threshold schedule is empty")
        # Clamp rather than wrap, so a short schedule degrades predictably.
        index = min(step, len(schedule) - 1)
        return float(schedule[index])

    def _is_warmup(self, step: int) -> bool:
        return step < self.warmup_steps

    def _is_cooldown(self, step: int) -> bool:
        if self.cooldown_steps == 0 or self.total_steps <= 0:
            return False
        return step >= self.total_steps - self.cooldown_steps

    def should_compute(self, step: int, distance: float) -> bool:
        """
        Decide whether step `step` must run the transformer.

        Args:
            step: Zero-based index of the denoising step.
            distance: Non-negative relative change proxy for this step versus the
                previous one. Ignored on forced steps.

        Returns:
            True to compute, False to reuse the cached residual.
        """
        if distance < 0:
            raise ValueError(f"distance must be non-negative, got {distance}")

        forced_warmup = self._is_warmup(step)
        forced_cooldown = self._is_cooldown(step)

        if forced_warmup or forced_cooldown:
            if forced_warmup:
                self.stats.forced_warmup += 1
            else:
                self.stats.forced_cooldown += 1
            return self._record(True)

        capped = (
            self.max_consecutive_reuse is not None
            and self.consecutive_reuse >= self.max_consecutive_reuse
        )
        if capped:
            self.stats.forced_by_reuse_cap += 1
            return self._record(True)

        self.accumulated += distance
        if self.accumulated >= self.threshold_at(step):
            return self._record(True)
        return self._record(False)

    def force_compute(self) -> bool:
        """
        Record an unconditional compute, for cases the distance proxy cannot judge.

        Used when there is no usable cache yet (the first step) or when the input
        shape changed, so a distance is undefined rather than merely large.
        """
        self.stats.forced_no_cache += 1
        return self._record(True)

    def _record(self, computed: bool) -> bool:
        """Update counters and accumulator for a decision, then return it."""
        self.stats.decisions.append(computed)
        if computed:
            self.stats.computed += 1
            self.accumulated = 0.0
            self.consecutive_reuse = 0
        else:
            self.stats.reused += 1
            self.consecutive_reuse += 1
        return computed


def linear_threshold_schedule(
    start: float,
    end: float,
) -> Callable[[int, int], float]:
    """
    Timestep-aware schedule interpolating from `start` to `end` across the run.

    Use start > end to be permissive early (where later steps still correct the error)
    and strict late (where error lands in the output unmodified).
    """
    if start < 0 or end < 0:
        raise ValueError("thresholds must be non-negative")

    def schedule(step: int, total_steps: int) -> float:
        if total_steps <= 1:
            return start
        position = min(max(step / (total_steps - 1), 0.0), 1.0)
        return start + (end - start) * position

    return schedule


def constant_threshold_schedule(value: float) -> Callable[[int, int], float]:
    """Flat threshold, matching stock TeaCache behaviour. The control condition."""
    if value < 0:
        raise ValueError("threshold must be non-negative")

    def schedule(step: int, total_steps: int) -> float:
        return value

    return schedule

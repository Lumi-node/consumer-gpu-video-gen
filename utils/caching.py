"""
Step caching for diffusion transformers.

Wraps a transformer's `forward` so that, on steps the policy judges redundant, the
cached residual is reused instead of running the network. The decision logic lives in
`utils.cache_policy` (torch-free and unit-tested); this module supplies the tensors.

What gets cached is the *residual* (output - input), not the output. The input changes
every step, so reusing a raw output would freeze the latent; reusing the residual
applies the same learned update to a moving input, which is what makes the
approximation hold. This is the TeaCache formulation.

The change proxy is the relative L1 distance between this step's input hidden states
and the previous step's:

    distance = mean(|h_t - h_{t-1}|) / mean(|h_{t-1}|)

It is a few elementwise ops on a tensor already in VRAM, so it is negligible next to a
transformer forward, and it needs no model-specific calibration. Methods like TeaCache
improve on it with a per-model polynomial fitted offline; `distance_transform` is the
hook for that once calibration data exists.
"""

from typing import Any, Callable, Optional

import torch

from .cache_policy import StepCachePolicy


def relative_l1_distance(current: torch.Tensor, previous: torch.Tensor) -> float:
    """
    Relative L1 distance between two tensors, as a float.

    Returns 0.0 when the previous tensor is all zeros, which would otherwise divide
    by zero; a zero distance is the safe answer because it biases toward reuse only
    in the degenerate case where there is nothing to compare.
    """
    if current.shape != previous.shape:
        # A shape change means the comparison is meaningless -- force a recompute by
        # reporting an effectively infinite distance.
        return float("inf")
    denominator = previous.abs().mean()
    if denominator.item() == 0.0:
        return 0.0
    return ((current - previous).abs().mean() / denominator).item()


# How a transformer packages its output tensor, recorded on the first real forward so
# that reused steps can hand the pipeline back the same type it expects.
_KIND_TENSOR = "tensor"
_KIND_TUPLE = "tuple"
_KIND_SAMPLE = "sample"


def _split_output(output: Any):
    """
    Normalize a transformer return value to (tensor, kind, cls).

    Handles a bare tensor, a tuple whose first element is the tensor, and diffusers
    ModelOutput objects exposing `.sample`.
    """
    if isinstance(output, torch.Tensor):
        return output, _KIND_TENSOR, type(output)
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], _KIND_TUPLE, type(output)
    sample = getattr(output, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample, _KIND_SAMPLE, type(output)
    raise TypeError(
        f"cannot locate the output tensor in a {type(output).__name__}; "
        "step caching does not know how to cache this transformer's return type"
    )


def _rebuild_output(tensor: torch.Tensor, kind: str, cls: type):
    """
    Repackage a reused tensor into the container the transformer normally returns.

    A fresh container is constructed rather than mutating the one seen earlier: the
    pipeline may still hold a reference to that object, and writing into it would
    corrupt an earlier step's result.
    """
    if kind == _KIND_TENSOR:
        return tensor
    if kind == _KIND_TUPLE:
        return (tensor,)
    if kind == _KIND_SAMPLE:
        try:
            return cls(sample=tensor)
        except TypeError:
            # Some output classes take the sample positionally.
            return cls(tensor)
    raise ValueError(f"unknown output kind '{kind}'")


class StepCache:
    """
    Installs step caching onto a transformer module.

    Args:
        transformer: The denoising transformer to wrap.
        policy: Decision policy. Its `total_steps` should already account for
            `calls_per_step`.
        hidden_states_arg: Name of the keyword argument carrying the latent input.
            Positional call sites are handled by falling back to the first positional
            tensor argument.
        calls_per_step: How many times the pipeline invokes the transformer per
            denoising step. Pipelines that batch conditional and unconditional
            together use 1; pipelines that make two separate calls use 2. Only
            affects how warmup/cooldown windows line up with real steps.
        distance_transform: Optional recalibration of the raw proxy, for fitting a
            model-specific mapping from input change to output change.
        verbose: Print a summary when uninstalled.

    Usage:
        cache = StepCache(pipe.transformer, policy)
        cache.install()
        try:
            frames = pipe(...)
        finally:
            cache.uninstall()
    """

    def __init__(
        self,
        transformer: torch.nn.Module,
        policy: StepCachePolicy,
        hidden_states_arg: str = "hidden_states",
        calls_per_step: int = 1,
        distance_transform: Optional[Callable[[float], float]] = None,
        verbose: bool = True,
    ):
        if calls_per_step < 1:
            raise ValueError(f"calls_per_step must be >= 1, got {calls_per_step}")
        self.transformer = transformer
        self.policy = policy
        self.hidden_states_arg = hidden_states_arg
        self.calls_per_step = calls_per_step
        self.distance_transform = distance_transform
        self.verbose = verbose

        self._original_forward: Optional[Callable] = None
        self._previous_input: Optional[torch.Tensor] = None
        self._cached_residual: Optional[torch.Tensor] = None
        self._output_kind: str = _KIND_TENSOR
        self._output_cls: type = torch.Tensor
        self._call_index = 0

    def _extract_hidden_states(self, args, kwargs) -> Optional[torch.Tensor]:
        value = kwargs.get(self.hidden_states_arg)
        if isinstance(value, torch.Tensor):
            return value
        for arg in args:
            if isinstance(arg, torch.Tensor):
                return arg
        return None

    def install(self) -> "StepCache":
        """Replace the transformer's forward with the caching wrapper."""
        if self._original_forward is not None:
            raise RuntimeError("StepCache is already installed")

        self.policy.reset()
        self._previous_input = None
        self._cached_residual = None
        self._call_index = 0
        self._original_forward = self.transformer.forward

        def cached_forward(*args, **kwargs):
            hidden_states = self._extract_hidden_states(args, kwargs)
            step = self._call_index
            self._call_index += 1

            # Without a locatable input tensor the residual trick cannot be applied,
            # so fall through to the real forward rather than guessing.
            if hidden_states is None:
                return self._original_forward(*args, **kwargs)

            # No usable cache yet, or the shape changed, means the distance proxy is
            # undefined rather than merely large -- compute unconditionally.
            if self._previous_input is None or self._cached_residual is None:
                distance = float("inf")
            else:
                distance = relative_l1_distance(hidden_states, self._previous_input)
                if self.distance_transform is not None and distance != float("inf"):
                    distance = self.distance_transform(distance)

            if distance == float("inf"):
                should_compute = self.policy.force_compute()
            else:
                should_compute = self.policy.should_compute(step, distance)

            self._previous_input = hidden_states.detach().clone()

            if should_compute:
                output = self._original_forward(*args, **kwargs)
                tensor, kind, cls = _split_output(output)
                if tensor.shape == hidden_states.shape:
                    self._cached_residual = (tensor - hidden_states).detach()
                    self._output_kind = kind
                    self._output_cls = cls
                else:
                    # The transformer changes shape (e.g. patch packing), so an
                    # input-space residual is not meaningful. Disable reuse rather
                    # than produce a wrongly-shaped latent.
                    self._cached_residual = None
                return output

            # Reuse: apply the cached update to the current input.
            approximate = hidden_states + self._cached_residual
            return _rebuild_output(approximate, self._output_kind, self._output_cls)

        self.transformer.forward = cached_forward
        return self

    def uninstall(self) -> None:
        """Restore the original forward and release cached tensors."""
        if self._original_forward is None:
            return
        self.transformer.forward = self._original_forward
        self._original_forward = None
        self._previous_input = None
        self._cached_residual = None
        if self.verbose:
            print(f"   {self.policy.stats.summary()}")

    def __enter__(self) -> "StepCache":
        return self.install()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.uninstall()
        return False


def cfg_note(guidance_scale: float) -> str:
    """
    Describe the cost of the chosen guidance scale.

    Classifier-free guidance runs the transformer on a doubled batch (conditional and
    unconditional), so disabling it is close to a flat 2x on the denoise loop. The
    catch is that a model trained to rely on CFG produces washed-out, poorly-prompted
    output at scale 1.0; the 2x is only free on a guidance-distilled checkpoint.
    """
    if guidance_scale > 1.0:
        return (
            f"CFG enabled (scale {guidance_scale}): the transformer runs on a doubled "
            "batch every step. A guidance-distilled checkpoint at scale 1.0 would be "
            "close to 2x faster."
        )
    return (
        "CFG disabled (scale <= 1.0): single batch per step, roughly 2x cheaper. "
        "Verify quality -- this is only free on a guidance-distilled checkpoint."
    )

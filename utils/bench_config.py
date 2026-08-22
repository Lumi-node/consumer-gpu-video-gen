"""
Benchmark configuration specs.

A run is described by a short string such as `nvfp4+cache` or `int4+cache+nocfg`, so
a whole sweep fits on one command line:

    --configs bf16 int4 nvfp4 nvfp4+cache nvfp4+cache+nocfg

Torch-free so the parsing is unit-tested without a GPU.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

# Modifier tokens that may follow the precision.
_MOD_CACHE = "cache"
_MOD_TIMESTEP_AWARE = "tscache"
_MOD_NO_CFG = "nocfg"
_MODIFIERS = {_MOD_CACHE, _MOD_TIMESTEP_AWARE, _MOD_NO_CFG}


@dataclass(frozen=True)
class ConfigSpec:
    """One benchmarked configuration."""

    precision: str
    cache: bool = False
    timestep_aware: bool = False
    use_cfg: bool = True
    label: str = ""

    def resolved_label(self) -> str:
        return self.label or self.precision

    def describe(self) -> str:
        parts = [self.precision]
        if self.timestep_aware:
            parts.append("timestep-aware cache")
        elif self.cache:
            parts.append("flat cache")
        if not self.use_cfg:
            parts.append("CFG disabled")
        return ", ".join(parts)


def parse_config_spec(spec: str) -> ConfigSpec:
    """
    Parse a spec string like "nvfp4+cache+nocfg" into a ConfigSpec.

    The first token is the precision; the rest are modifiers:
        cache    flat-threshold step caching
        tscache  timestep-aware step caching (implies cache)
        nocfg    guidance_scale = 1.0, skipping the unconditional pass
    """
    if not spec or not spec.strip():
        raise ValueError("config spec must be a non-empty string")

    tokens = [t.strip().lower() for t in spec.split("+") if t.strip()]
    if not tokens:
        raise ValueError(f"config spec '{spec}' has no tokens")

    precision = tokens[0]
    modifiers = tokens[1:]

    unknown = [m for m in modifiers if m not in _MODIFIERS]
    if unknown:
        raise ValueError(
            f"unknown modifier(s) {unknown} in '{spec}'; "
            f"valid modifiers: {sorted(_MODIFIERS)}"
        )

    timestep_aware = _MOD_TIMESTEP_AWARE in modifiers
    return ConfigSpec(
        precision=precision,
        cache=_MOD_CACHE in modifiers or timestep_aware,
        timestep_aware=timestep_aware,
        use_cfg=_MOD_NO_CFG not in modifiers,
        label=spec.strip().lower(),
    )


def parse_config_specs(specs: Sequence[str]) -> List[ConfigSpec]:
    """Parse several specs, rejecting duplicate labels."""
    parsed = [parse_config_spec(s) for s in specs]
    seen = set()
    for config in parsed:
        label = config.resolved_label()
        if label in seen:
            raise ValueError(f"duplicate config '{label}'")
        seen.add(label)
    return parsed


def pick_baseline(configs: Sequence[ConfigSpec], requested: Optional[str] = None) -> str:
    """
    Choose which config every other config is measured against.

    Defaults to bf16 when present, because a speedup claim against a quantized run is
    not a speedup claim about the optimization. Falls back to the first config, which
    at least keeps the report internally consistent.
    """
    if not configs:
        raise ValueError("no configs to choose a baseline from")

    labels = [c.resolved_label() for c in configs]
    if requested is not None:
        if requested not in labels:
            raise ValueError(f"baseline '{requested}' is not among configs {labels}")
        return requested

    for config in configs:
        if config.precision == "bf16" and not config.cache and config.use_cfg:
            return config.resolved_label()
    return labels[0]

#!/usr/bin/env python3
"""
Speed and quality benchmark for the video pipelines.

The repo previously reported generation times with no bf16 baseline, which makes a
number like "60 seconds" impossible to interpret. This driver always measures a
reference configuration and expresses everything else relative to it, in both latency
and output fidelity.

Examples:

    # Is weight-only INT4 actually slower than bf16 on this card?
    python benchmarks/benchmark.py --model wan22 \\
        --checkpoint-dir /models/Wan2.2-TI2V-5B --wan-repo /repos/Wan2.2 \\
        --configs bf16 int4

    # Full sweep on a 5090, including the FP4 compute path and caching.
    python benchmarks/benchmark.py --model wan22 \\
        --checkpoint-dir /models/Wan2.2-TI2V-5B --wan-repo /repos/Wan2.2 \\
        --configs bf16 int4 nvfp4 nvfp4+cache nvfp4+tscache+nocfg \\
        --repeats 3 --output benchmarks/results.md

Every configuration runs the same prompt at the same seed, so the PSNR column measures
only what the optimization changed.

Note on VRAM: bf16 configurations of the larger models will not fit on a 32 GB card.
Benchmark those at a reduced resolution or frame count, or accept that the baseline
column is unavailable and compare quantized configurations against each other.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from utils.bench_config import parse_config_specs, pick_baseline  # noqa: E402
from utils.bench_report import format_report, results_to_json  # noqa: E402
from utils.profiling import benchmark, gpu_name  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark video generation configurations against a baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["wan22", "ltx2"], required=True,
        help="Which pipeline to benchmark.",
    )
    parser.add_argument(
        "--configs", nargs="+", default=["bf16", "int4"],
        help=(
            "Configurations to compare. Each is a precision optionally followed by "
            "'+cache', '+tscache' (timestep-aware) and/or '+nocfg'. "
            "Example: bf16 int4 nvfp4 nvfp4+cache"
        ),
    )
    parser.add_argument("--baseline", default=None, help="Config to measure against (default: bf16 if present).")
    parser.add_argument("--prompt", default="An eagle soaring through clouds at sunset")
    parser.add_argument("--seed", type=int, default=42, help="Fixed so quality comparison is meaningful.")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--warmup", type=int, default=1, help="Untimed runs before measurement.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed runs; the median is reported.")
    parser.add_argument("--cache-threshold", type=float, default=0.15)
    parser.add_argument("--guidance-scale", type=float, default=None, help="Overrides the model default.")
    parser.add_argument("--output", default=None, help="Write the markdown report here.")
    parser.add_argument("--json-output", default=None, help="Write raw results as JSON here.")

    # Model paths
    parser.add_argument("--checkpoint-dir", default=None, help="wan22: checkpoint directory")
    parser.add_argument("--wan-repo", default=None, help="wan22: path to the Wan2.2 repository")
    parser.add_argument("--model-path", default=None, help="ltx2: LTX-2 diffusers directory")
    parser.add_argument("--height", type=int, default=448, help="ltx2 only")
    parser.add_argument("--width", type=int, default=640, help="ltx2 only")
    parser.add_argument("--size", default="landscape", choices=["landscape", "portrait"], help="wan22 only")
    return parser


def _default_guidance(model: str) -> float:
    return 5.0 if model == "wan22" else 3.5


def load_pipeline(args, precision: str):
    """Construct and load a pipeline at the given precision."""
    if args.model == "wan22":
        if not args.checkpoint_dir or not args.wan_repo:
            raise SystemExit("--checkpoint-dir and --wan-repo are required for --model wan22")
        from models.wan22 import Wan22Pipeline

        pipeline = Wan22Pipeline(
            checkpoint_dir=args.checkpoint_dir, wan_repo_path=args.wan_repo
        )
        pipeline.load(precision=precision, verbose=True)
        return pipeline

    if not args.model_path:
        raise SystemExit("--model-path is required for --model ltx2")
    from models.ltx2 import LTX2Pipeline

    pipeline = LTX2Pipeline(model_path=args.model_path)
    pipeline.load(precision=precision, verbose=True)
    return pipeline


def make_run_fn(args, pipeline, config, guidance_scale):
    """
    Build the callable that `benchmark` times.

    The whole generation is timed as one stage. Finer per-stage attribution would need
    hooks inside each pipeline's internals; the report flags when a large share of the
    wall clock is unattributed so the gap is never mistaken for full coverage.
    """

    def run(recorder):
        with recorder.stage("generate"):
            if args.model == "wan22":
                return pipeline.generate(
                    prompt=args.prompt,
                    size=args.size,
                    num_frames=args.frames,
                    num_steps=args.steps,
                    guidance_scale=guidance_scale,
                    seed=args.seed,
                    cache=config.cache,
                    cache_threshold=args.cache_threshold,
                    timestep_aware=config.timestep_aware,
                    verbose=False,
                )
            return pipeline.generate(
                prompt=args.prompt,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                num_steps=args.steps,
                guidance_scale=guidance_scale,
                seed=args.seed,
                cache=config.cache,
                cache_threshold=args.cache_threshold,
                timestep_aware=config.timestep_aware,
                verbose=False,
            )

    return run


def _as_tensor(output):
    """Coerce a pipeline result to a tensor for quality comparison, if possible."""
    if isinstance(output, torch.Tensor):
        return output
    try:
        import numpy as np

        if isinstance(output, np.ndarray):
            return torch.from_numpy(output)
    except ImportError:
        pass
    return None


def main() -> int:
    args = build_parser().parse_args()

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found. Timings will not be meaningful.\n")

    configs = parse_config_specs(args.configs)
    baseline_label = pick_baseline(configs, args.baseline)
    guidance_default = args.guidance_scale or _default_guidance(args.model)

    print(f"Hardware: {gpu_name()}")
    print(f"Baseline: {baseline_label}")
    print(f"Configs:  {', '.join(c.resolved_label() for c in configs)}\n")

    metadata = {
        "model": args.model,
        "steps": args.steps,
        "frames": args.frames,
        "seed": args.seed,
        "repeats": args.repeats,
    }

    # Run the baseline first so its output can serve as the quality reference.
    ordered = sorted(configs, key=lambda c: c.resolved_label() != baseline_label)

    results = []
    reference = None
    for config in ordered:
        label = config.resolved_label()
        print(f"\n{'=' * 60}\n{label}: {config.describe()}\n{'=' * 60}")

        guidance_scale = guidance_default if config.use_cfg else 1.0

        # Distilled checkpoints (e.g. LTX-2.5's distilled transformer) run unguided
        # and are driven by an explicit sigma schedule. Passing num_inference_steps
        # instead hands them a generic linear schedule, which quietly costs quality --
        # and would show up here as a quality regression wrongly attributed to
        # caching or quantization.
        if not config.use_cfg:
            print(
                "  NOTE: CFG is disabled, which usually means a distilled checkpoint. "
                "If this checkpoint expects an explicit sigma schedule, "
                f"--steps {args.steps} will be used instead and quality will suffer. "
                "Check the model card before trusting this row."
            )

        try:
            pipeline = load_pipeline(args, config.precision)
        except Exception as exc:  # noqa: BLE001 - one bad config must not sink the sweep
            print(f"  SKIPPED: could not load pipeline ({type(exc).__name__}: {exc})")
            continue

        try:
            outcome = benchmark(
                label=label,
                fn=make_run_fn(args, pipeline, config, guidance_scale),
                warmup=args.warmup,
                repeats=args.repeats,
                metadata=dict(metadata, config=label, guidance_scale=guidance_scale),
                reference_output=reference,
            )
            results.append(outcome.result)
            if label == baseline_label:
                reference = _as_tensor(outcome.output)
                if reference is None:
                    print("  NOTE: baseline output is not a tensor; quality comparison disabled.")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
        finally:
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not results:
        print("\nNo configurations completed successfully.")
        return 1

    available = {r.label for r in results}
    if baseline_label not in available:
        baseline_label = results[0].label
        print(f"\nNOTE: baseline did not complete; reporting against '{baseline_label}'.")

    report = format_report(
        results, baseline_label, title=f"{args.model} benchmark", hardware=gpu_name()
    )
    print("\n" + report)

    if args.output:
        with open(args.output, "w") as handle:
            handle.write(report)
        print(f"Report written to {args.output}")

    if args.json_output:
        with open(args.json_output, "w") as handle:
            handle.write(results_to_json(results))
        print(f"Raw results written to {args.json_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

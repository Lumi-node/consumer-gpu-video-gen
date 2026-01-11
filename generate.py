#!/usr/bin/env python3
"""
Consumer GPU Video Generation CLI

Generate videos using Wan 2.2 or LTX-2 with INT4 optimization on consumer GPUs.
Reduces VRAM requirements by 50-75%, enabling large models on 24-32GB cards.

Examples:
    # Wan 2.2 (recommended for most users)
    python generate.py --model wan22 --prompt "A cat playing in a garden" \\
        --checkpoint /path/to/Wan2.2-TI2V-5B --wan-repo /path/to/Wan2.2

    # LTX-2 (larger model, requires more VRAM)
    python generate.py --model ltx2 --prompt "An eagle soaring through clouds" \\
        --checkpoint /path/to/LTX-2
"""

import os
import sys
import argparse
from datetime import datetime

# Set memory allocation config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate videos with INT4-optimized models on consumer GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Wan 2.2:
    python generate.py --model wan22 \\
        --prompt "A fluffy cat walking through a sunny garden" \\
        --checkpoint /path/to/Wan2.2-TI2V-5B \\
        --wan-repo /path/to/Wan2.2

  LTX-2:
    python generate.py --model ltx2 \\
        --prompt "A majestic eagle soaring through golden clouds" \\
        --checkpoint /path/to/LTX-2

Supported Models:
  wan22   - Alibaba Wan 2.2 TI2V-5B (~16GB VRAM with INT4)
  ltx2    - Lightricks LTX-2 19B (~22GB VRAM with INT4)
        """
    )

    # Model selection
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["wan22", "ltx2"],
        required=True,
        help="Model to use (wan22 or ltx2)"
    )

    # Paths
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--wan-repo",
        type=str,
        default=None,
        help="Path to Wan2.2 repository (required for wan22 model)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output video path (default: output_<timestamp>.mp4)"
    )

    # Generation parameters
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Text prompt for video generation"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt (what to avoid)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=33,
        help="Number of frames (default: 33, must be 4n+1 for Wan)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Diffusion steps (default: 30)"
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=5.0,
        help="Guidance scale (default: 5.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed (-1 for random)"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="landscape",
        choices=["landscape", "portrait"],
        help="Video orientation (default: landscape)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Output video FPS (default: 24)"
    )

    # Optimization
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        default="int4",
        choices=["int4", "int8", "none"],
        help="Quantization level (default: int4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device (default: cuda:0)"
    )

    # Output control
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    # Validate arguments
    if args.model == "wan22" and args.wan_repo is None:
        print("Error: --wan-repo is required for wan22 model")
        sys.exit(1)

    # Generate output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"output_{args.model}_{timestamp}.mp4"

    # Import torch after setting env vars
    import torch

    if verbose:
        print()
        print("=" * 60)
        print("Consumer GPU Video Generation")
        print("=" * 60)
        print(f"Model: {args.model}")
        print(f"Quantization: {args.quantization}")
        print(f"Device: {args.device}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print()

    # Load and run appropriate model
    if args.model == "wan22":
        from models.wan22 import Wan22Pipeline

        pipeline = Wan22Pipeline(
            checkpoint_dir=args.checkpoint,
            wan_repo_path=args.wan_repo,
            device=args.device,
        )
        pipeline.load(quantization=args.quantization, verbose=verbose)

        video = pipeline.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            size=args.size,
            num_frames=args.frames,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            verbose=verbose,
        )

        pipeline.save_video(video, args.output, fps=args.fps)

    elif args.model == "ltx2":
        from models.ltx2 import LTX2Pipeline

        # Adjust dimensions based on size
        if args.size == "landscape":
            width, height = 640, 448
        else:
            width, height = 448, 640

        pipeline = LTX2Pipeline(
            model_path=args.checkpoint,
            device=args.device,
        )
        pipeline.load(quantization=args.quantization, verbose=verbose)

        frames = pipeline.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt or "blurry, low quality",
            height=height,
            width=width,
            num_frames=args.frames,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            verbose=verbose,
        )

        pipeline.save_video(frames, args.output, fps=args.fps)

    if verbose:
        print()
        print("Done!")
        print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

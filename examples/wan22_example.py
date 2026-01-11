#!/usr/bin/env python3
"""
Wan 2.2 TI2V-5B Example

Generates a video using the optimized Wan 2.2 pipeline with INT4 quantization.
Requires ~16GB VRAM instead of ~25GB.
"""

import os
import sys

# Set memory config before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.wan22 import Wan22Pipeline

# === CONFIGURE THESE PATHS ===
CHECKPOINT_DIR = "/path/to/Wan2.2-TI2V-5B"  # Downloaded model
WAN_REPO_PATH = "/path/to/Wan2.2"            # Cloned Wan repository
OUTPUT_DIR = "./outputs"
# =============================


def validate_paths():
    """Check that paths are configured before running."""
    if "/path/to/" in CHECKPOINT_DIR or "/path/to/" in WAN_REPO_PATH:
        print("ERROR: Please configure the paths at the top of this file:")
        print(f"  CHECKPOINT_DIR = {CHECKPOINT_DIR}")
        print(f"  WAN_REPO_PATH = {WAN_REPO_PATH}")
        print("\nDownload instructions:")
        print("  git clone https://github.com/Wan-Video/Wan2.2")
        print("  huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B")
        sys.exit(1)

    if not os.path.isdir(CHECKPOINT_DIR):
        print(f"ERROR: Checkpoint directory not found: {CHECKPOINT_DIR}")
        sys.exit(1)

    if not os.path.isdir(WAN_REPO_PATH):
        print(f"ERROR: Wan repository not found: {WAN_REPO_PATH}")
        sys.exit(1)


def main():
    validate_paths()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Initialize pipeline
    print("Initializing Wan 2.2 pipeline...")
    pipeline = Wan22Pipeline(
        checkpoint_dir=CHECKPOINT_DIR,
        wan_repo_path=WAN_REPO_PATH,
        device="cuda:0",
    )

    # Load with INT4 quantization
    pipeline.load(quantization="int4", verbose=True)

    # Generate video
    prompt = "A fluffy orange cat walking through a sunny garden, " \
             "cinematic lighting, 4K quality, detailed fur"

    video = pipeline.generate(
        prompt=prompt,
        size="landscape",      # 1280x704
        num_frames=33,         # ~1.4 seconds at 24fps
        num_steps=30,          # Diffusion steps
        guidance_scale=5.0,    # CFG scale
        seed=42,               # For reproducibility
        verbose=True,
    )

    # Save output
    output_path = os.path.join(OUTPUT_DIR, "wan22_example.mp4")
    pipeline.save_video(video, output_path, fps=24)

    print(f"\nDone! Video saved to: {output_path}")


if __name__ == "__main__":
    main()

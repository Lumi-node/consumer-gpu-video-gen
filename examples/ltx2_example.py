#!/usr/bin/env python3
"""
LTX-2 Example

Generates a video using the optimized LTX-2 pipeline with INT4 quantization.
Requires ~22GB VRAM instead of ~67GB.
"""

import os
import sys

# Set memory config before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ltx2 import LTX2Pipeline

# === CONFIGURE THESE PATHS ===
MODEL_PATH = "/path/to/LTX-2"  # Downloaded model (diffusers format)
OUTPUT_DIR = "./outputs"
# =============================


def validate_paths():
    """Check that paths are configured before running."""
    if "/path/to/" in MODEL_PATH:
        print("ERROR: Please configure MODEL_PATH at the top of this file:")
        print(f"  MODEL_PATH = {MODEL_PATH}")
        print("\nDownload the model from Hugging Face:")
        print("  huggingface-cli download Lightricks/LTX-Video --local-dir ./LTX-2")
        sys.exit(1)

    if not os.path.isdir(MODEL_PATH):
        print(f"ERROR: Model directory not found: {MODEL_PATH}")
        sys.exit(1)


def main():
    validate_paths()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Initialize pipeline
    print("Initializing LTX-2 pipeline...")
    pipeline = LTX2Pipeline(
        model_path=MODEL_PATH,
        device="cuda:0",
    )

    # Load with INT4 quantization
    pipeline.load(quantization="int4", verbose=True)

    # Generate video
    prompt = "A majestic eagle soaring through golden sunset clouds, " \
             "cinematic, slow motion, detailed feathers"

    frames = pipeline.generate(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted",
        height=448,            # Reduced for memory
        width=640,             # 16:9 aspect ratio
        num_frames=33,         # ~1.4 seconds at 24fps
        num_steps=25,          # Diffusion steps
        guidance_scale=3.5,    # CFG scale
        seed=42,               # For reproducibility
        verbose=True,
    )

    # Save output
    output_path = os.path.join(OUTPUT_DIR, "ltx2_example.mp4")
    pipeline.save_video(frames, output_path, fps=24)

    print(f"\nDone! Video saved to: {output_path}")


if __name__ == "__main__":
    main()

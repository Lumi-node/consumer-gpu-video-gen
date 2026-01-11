"""
LTX-2 Optimized Pipeline

Runs Lightricks' LTX-2 19B audio-video model with INT4 quantization on consumer GPUs.
Original model requires ~67GB VRAM, optimized version runs in ~22GB.

This is a massive model with:
- Gemma-3 text encoder (~27GB original)
- 19B parameter transformer (~40GB original)
- Video + Audio VAE decoders

INT4 quantization makes it runnable on RTX 4090/5090 (24-32GB VRAM).
"""

import os
import sys
import torch
import gc
from typing import Optional, Literal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.memory import get_vram_usage, clear_vram, print_vram_status
from utils.quantization import quantize_model_int4, QuantizationType


class LTX2Pipeline:
    """
    Optimized LTX-2 pipeline for consumer GPUs.

    Uses INT4 quantization to reduce VRAM from ~67GB to ~22GB,
    making it runnable on RTX 4090, RTX 5090, and similar cards.

    Example:
        pipeline = LTX2Pipeline(model_path="/path/to/LTX-2")
        pipeline.load(quantization="int4")
        frames = pipeline.generate("An eagle soaring through clouds")
        pipeline.save_video(frames, "output.mp4")
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
    ):
        """
        Initialize the LTX-2 pipeline.

        Args:
            model_path: Path to LTX-2 model directory (diffusers format)
            device: CUDA device to use
        """
        self.model_path = model_path
        self.device = device

        # Pipeline components
        self.pipe = None
        self._loaded = False

    def load(
        self,
        quantization: QuantizationType = "int4",
        verbose: bool = True
    ) -> "LTX2Pipeline":
        """
        Load the LTX-2 pipeline with component-wise quantization.

        Args:
            quantization: Quantization type ("int4", "int8", or "none")
            verbose: Whether to print loading progress

        Returns:
            Self for chaining
        """
        from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast
        from diffusers import LTX2VideoTransformer3DModel, FlowMatchEulerDiscreteScheduler
        from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
        from diffusers.models.autoencoders.autoencoder_kl_ltx2_audio import AutoencoderKLLTX2Audio
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
        from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
        from diffusers import LTX2Pipeline as DiffusersLTX2Pipeline

        if verbose:
            print("=" * 60)
            print("Loading LTX-2 19B with INT4 Optimization")
            print("=" * 60)
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Model path: {self.model_path}")
            print()

        # 1. Load and quantize text encoder
        if verbose:
            print("1. Loading & quantizing text encoder (Gemma-3)...")
        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            os.path.join(self.model_path, "text_encoder"),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        if quantization != "none":
            quantize_model_int4(text_encoder, verbose=False)

        text_encoder.to(self.device)
        clear_vram()
        if verbose:
            print(f"   Text encoder VRAM: {get_vram_usage():.2f} GB")

        # Tokenizer
        tokenizer = GemmaTokenizerFast.from_pretrained(
            os.path.join(self.model_path, "tokenizer")
        )

        # 2. Load and quantize transformer
        if verbose:
            print("\n2. Loading & quantizing transformer...")
        transformer = LTX2VideoTransformer3DModel.from_pretrained(
            os.path.join(self.model_path, "transformer"),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        if quantization != "none":
            quantize_model_int4(transformer, verbose=False)

        transformer.to(self.device)
        clear_vram()
        if verbose:
            print(f"   After transformer VRAM: {get_vram_usage():.2f} GB")

        # 3. Load VAE (no quantization needed, relatively small)
        if verbose:
            print("\n3. Loading VAE...")
        vae = AutoencoderKLLTX2Video.from_pretrained(
            os.path.join(self.model_path, "vae"),
            torch_dtype=torch.bfloat16,
        )
        vae.to(self.device)
        clear_vram()
        if verbose:
            print(f"   After VAE VRAM: {get_vram_usage():.2f} GB")

        # 4. Scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            os.path.join(self.model_path, "scheduler")
        )

        # 5. Audio components
        if verbose:
            print("\n4. Loading audio components...")
        audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
            os.path.join(self.model_path, "audio_vae"),
            torch_dtype=torch.bfloat16,
        )
        audio_vae.to(self.device)

        connectors = LTX2TextConnectors.from_pretrained(
            os.path.join(self.model_path, "connectors"),
            torch_dtype=torch.bfloat16,
        )
        connectors.to(self.device)

        vocoder = LTX2Vocoder.from_pretrained(
            os.path.join(self.model_path, "vocoder"),
            torch_dtype=torch.bfloat16,
        )
        vocoder.to(self.device)
        clear_vram()
        if verbose:
            print(f"   After audio components VRAM: {get_vram_usage():.2f} GB")

        # 6. Assemble pipeline
        if verbose:
            print("\n5. Assembling pipeline...")
        self.pipe = DiffusersLTX2Pipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            audio_vae=audio_vae,
            connectors=connectors,
            vocoder=vocoder,
        )

        # Enable optimizations
        if hasattr(self.pipe, 'enable_vae_tiling'):
            self.pipe.enable_vae_tiling()
        if hasattr(self.pipe, 'enable_attention_slicing'):
            self.pipe.enable_attention_slicing("auto")

        self._loaded = True
        clear_vram()

        if verbose:
            print()
            print_vram_status()
            print()

        return self

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, low quality",
        height: int = 448,
        width: int = 640,
        num_frames: int = 33,
        num_steps: int = 25,
        guidance_scale: float = 3.5,
        seed: int = -1,
        verbose: bool = True,
    ):
        """
        Generate a video from a text prompt.

        Args:
            prompt: Text description of the video
            negative_prompt: What to avoid
            height: Video height (default 448 for memory efficiency)
            width: Video width (default 640 for 16:9)
            num_frames: Number of frames to generate
            num_steps: Diffusion steps
            guidance_scale: CFG scale
            seed: Random seed (-1 for random)
            verbose: Print progress

        Returns:
            Generated frames
        """
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded. Call .load() first.")

        if seed < 0:
            import random
            seed = random.randint(0, 2**32 - 1)

        if verbose:
            print(f"Generating: {width}x{height}, {num_frames} frames, {num_steps} steps")
            print(f"Prompt: '{prompt[:80]}{'...' if len(prompt) > 80 else ''}'")

        generator = torch.Generator(self.device).manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        return output.frames[0]

    def save_video(
        self,
        frames,
        output_path: str,
        fps: int = 24
    ) -> str:
        """
        Save generated frames to video file.

        Args:
            frames: Generated frames from generate()
            output_path: Path to save video
            fps: Frames per second

        Returns:
            Path to saved video
        """
        from diffusers.utils import export_to_video
        export_to_video(frames, output_path, fps=fps)

        file_size = os.path.getsize(output_path) / 1024
        print(f"Video saved: {output_path} ({file_size:.1f} KB)")

        return output_path

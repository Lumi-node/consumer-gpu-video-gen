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

from utils.memory import get_vram_usage, clear_vram, offload_model, print_vram_status
from utils.quantization import quantize_model_int4, QuantizationType
from utils.fp4 import apply_precision, detect_availability, detect_capability, quantizes_on_cpu
from utils.precision import expected_speed_note, select_backend
from utils.cache_policy import StepCachePolicy, linear_threshold_schedule
from utils.caching import StepCache, cfg_note


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
        self.selection = None
        self.offload_text_encoder = True
        self._text_encoder_offloaded = False

    def _prepare_component(self, module, precision: str, verbose: bool):
        """
        Quantize a component and place it on the device, in the right order.

        Weight-only backends (quanto int4/int8) are applied while the module is still
        on CPU, which is the whole reason a 67 GB model can be loaded on a 32 GB card:
        the bf16 weights never need to fit in VRAM. The torchao compute backends
        (nvfp4/fp8) need the module on the device first.
        """
        if precision == "bf16":
            module.to(self.device)
        elif quantizes_on_cpu(precision):
            apply_precision(module, precision, verbose=verbose)
            module.to(self.device)
        else:
            module.to(self.device)
            apply_precision(module, precision, verbose=verbose)
        clear_vram()
        return module

    def load(
        self,
        quantization: QuantizationType = "int4",
        precision: str = None,
        goal: str = "memory",
        offload_text_encoder: bool = True,
        verbose: bool = True
    ) -> "LTX2Pipeline":
        """
        Load the LTX-2 pipeline with component-wise quantization.

        Args:
            quantization: Legacy quantization type ("int4", "int8", or "none"). Kept
                for backwards compatibility; `precision` takes priority when given.
            precision: Precision backend: "auto", "nvfp4", "fp8", "int4", "int8" or
                "bf16". "nvfp4" is the only option that reduces both VRAM and latency,
                and it needs a Blackwell card (RTX 50-series). Falls back with an
                explanation on other hardware.
            goal: "memory" or "speed"; decides what "auto" resolves to and the
                fallback order.
            offload_text_encoder: Move the Gemma-3 text encoder back to CPU after
                prompt encoding. It runs once per generation but would otherwise
                occupy VRAM for the entire denoise loop.
            verbose: Whether to print loading progress

        Returns:
            Self for chaining
        """
        # Map the legacy argument onto the precision system when the new one is unset.
        if precision is None:
            precision = "bf16" if quantization == "none" else quantization

        capability = detect_capability()
        availability = detect_availability()
        self.selection = select_backend(precision, capability, availability, goal=goal)
        resolved = self.selection.precision
        self.offload_text_encoder = offload_text_encoder
        from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast
        from diffusers import LTX2VideoTransformer3DModel, FlowMatchEulerDiscreteScheduler
        from diffusers.models.autoencoders.autoencoder_kl_ltx2 import AutoencoderKLLTX2Video
        from diffusers.models.autoencoders.autoencoder_kl_ltx2_audio import AutoencoderKLLTX2Audio
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
        from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
        from diffusers import LTX2Pipeline as DiffusersLTX2Pipeline

        if verbose:
            print("=" * 60)
            print(f"Loading LTX-2 19B ({resolved})")
            print("=" * 60)
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Model path: {self.model_path}")
            print(f"   {self.selection.explain()}")
            print(f"   {expected_speed_note(resolved)}")
            print()

        # 1. Load and quantize text encoder
        if verbose:
            print("1. Loading & quantizing text encoder (Gemma-3)...")
        text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            os.path.join(self.model_path, "text_encoder"),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        self._prepare_component(text_encoder, resolved, verbose=False)
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

        self._prepare_component(transformer, resolved, verbose=False)
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
        cache: bool = False,
        cache_threshold: float = 0.15,
        timestep_aware: bool = False,
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
            guidance_scale: CFG scale. Above 1.0 the transformer runs on a doubled
                batch every step; 1.0 disables that but needs a guidance-distilled
                checkpoint to keep quality.
            seed: Random seed (-1 for random)
            cache: Enable step caching (reuse the transformer residual on steps
                predicted to be redundant).
            cache_threshold: Accumulated-change budget before a recompute is forced.
                Higher means more reuse and more drift.
            timestep_aware: Use a decreasing threshold schedule instead of a flat one,
                permissive early and strict late. Experimental.
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
            print(f"   {cfg_note(guidance_scale)}")

        generator = torch.Generator(self.device).manual_seed(seed)

        # With CFG the pipeline batches conditional and unconditional into one
        # transformer call, so the cache still sees one call per denoising step.
        step_cache = None
        if cache:
            threshold = (
                linear_threshold_schedule(cache_threshold * 2.0, cache_threshold * 0.25)
                if timestep_aware
                else cache_threshold
            )
            policy = StepCachePolicy(
                threshold=threshold,
                total_steps=num_steps,
                warmup_steps=2,
                cooldown_steps=2,
            )
            step_cache = StepCache(self.pipe.transformer, policy, verbose=verbose)
            step_cache.install()

        prompt_kwargs = self._encode_and_maybe_offload(
            prompt, negative_prompt, verbose=verbose
        )

        try:
            output = self.pipe(
                num_frames=num_frames,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **prompt_kwargs,
            )
        finally:
            if step_cache is not None:
                step_cache.uninstall()
            if getattr(self, "_text_encoder_offloaded", False):
                self.reload_text_encoder()
                self._text_encoder_offloaded = False

        return output.frames[0]

    def _encode_and_maybe_offload(self, prompt: str, negative_prompt: str, verbose: bool) -> dict:
        """
        Encode the prompt up front so the text encoder can leave the GPU.

        Returns the keyword arguments to pass to the pipeline: either precomputed
        embeddings (encoder offloaded) or the raw strings (encoder stays resident).

        Diffusers' `encode_prompt` signature varies across versions and pipelines, so
        a mismatch falls back to the original behaviour with a warning rather than
        failing the generation. This path is the one part of the offload work that
        needs validating against a real LTX-2 checkout.
        """
        self._text_encoder_offloaded = False
        raw = {"prompt": prompt, "negative_prompt": negative_prompt}

        if not getattr(self, "offload_text_encoder", False):
            return raw
        if not hasattr(self.pipe, "encode_prompt"):
            if verbose:
                print("   Text encoder offload skipped: pipeline has no encode_prompt()")
            return raw

        try:
            encoded = self.pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=True,
                device=self.device,
            )
        except TypeError as exc:
            if verbose:
                print(f"   Text encoder offload skipped: encode_prompt signature mismatch ({exc})")
            return raw

        if not isinstance(encoded, (tuple, list)) or len(encoded) != 4:
            if verbose:
                print(
                    "   Text encoder offload skipped: unexpected encode_prompt return "
                    f"({type(encoded).__name__})"
                )
            return raw

        prompt_embeds, prompt_mask, negative_embeds, negative_mask = encoded

        vram_before = get_vram_usage()
        self.offload_text_encoder_now(verbose=False)
        self._text_encoder_offloaded = True
        if verbose:
            print(
                f"   Text encoder offloaded before denoise: "
                f"{vram_before:.2f} GB -> {get_vram_usage():.2f} GB"
            )

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_mask,
            "negative_prompt_embeds": negative_embeds,
            "negative_prompt_attention_mask": negative_mask,
        }

    def offload_text_encoder_now(self, verbose: bool = True) -> None:
        """
        Move the Gemma-3 text encoder to CPU.

        The encoder runs once per generation, but the original pipeline left it
        resident on the GPU for the whole denoise loop -- roughly 8 GB at int4 doing
        nothing. Call this after prompt encoding, or use `generate_offloaded`.
        """
        if self.pipe is None or getattr(self.pipe, "text_encoder", None) is None:
            return
        offload_model(self.pipe.text_encoder, verbose=verbose)

    def reload_text_encoder(self) -> None:
        """Move the text encoder back to the device before the next encode."""
        if self.pipe is not None and getattr(self.pipe, "text_encoder", None) is not None:
            self.pipe.text_encoder.to(self.device)
            clear_vram()

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

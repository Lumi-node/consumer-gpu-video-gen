"""
Wan 2.2 TI2V-5B Optimized Pipeline

Runs Alibaba's Wan 2.2 Text-to-Video model with INT4 quantization on consumer GPUs.
Original model requires ~25GB VRAM, optimized version runs in ~16GB peak.

Supported resolutions (TI2V-5B):
- 1280x704 (landscape)
- 704x1280 (portrait)
"""

import os
import sys
import torch
from typing import Optional, Tuple, Literal
from tqdm import tqdm

# Add parent path for utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.memory import get_vram_usage, clear_vram, offload_models, print_vram_status
from utils.quantization import quantize_model_int4, QuantizationType


class Wan22Pipeline:
    """
    Optimized Wan 2.2 TI2V-5B pipeline for consumer GPUs.

    Uses INT4 quantization to reduce VRAM from ~25GB to ~16GB,
    making it runnable on RTX 4090, RTX 5090, and similar cards.

    Example:
        pipeline = Wan22Pipeline(
            checkpoint_dir="/path/to/Wan2.2-TI2V-5B",
            wan_repo_path="/path/to/Wan2.2"
        )
        pipeline.load(quantization="int4")
        video = pipeline.generate("A cat walking in a garden")
        pipeline.save_video(video, "output.mp4")
    """

    # Supported sizes for TI2V-5B
    SUPPORTED_SIZES = {
        "landscape": (1280, 704),
        "portrait": (704, 1280),
    }

    def __init__(
        self,
        checkpoint_dir: str,
        wan_repo_path: str,
        device: str = "cuda:0",
    ):
        """
        Initialize the Wan 2.2 pipeline.

        Args:
            checkpoint_dir: Path to Wan2.2-TI2V-5B checkpoint directory
            wan_repo_path: Path to Wan2.2 repository (contains wan/ module)
            device: CUDA device to use
        """
        self.checkpoint_dir = checkpoint_dir
        self.wan_repo_path = wan_repo_path
        self.device = torch.device(device)

        # Add Wan repo to path
        if wan_repo_path not in sys.path:
            sys.path.insert(0, wan_repo_path)

        # Models (loaded later)
        self.t5 = None
        self.vae = None
        self.dit = None
        self.config = None
        self.scheduler = None

        self._loaded = False

    def load(
        self,
        quantization: QuantizationType = "int4",
        verbose: bool = True
    ) -> "Wan22Pipeline":
        """
        Load all model components with optional quantization.

        Args:
            quantization: Quantization type ("int4", "int8", or "none")
            verbose: Whether to print loading progress

        Returns:
            Self for chaining
        """
        # Import Wan modules
        from wan.configs import WAN_CONFIGS
        from wan.modules.model import WanModel
        from wan.modules.t5 import T5EncoderModel
        from wan.modules.vae2_2 import Wan2_2_VAE
        from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler

        self.config = WAN_CONFIGS["ti2v-5B"]

        if verbose:
            print("=" * 60)
            print("Loading Wan 2.2 TI2V-5B with INT4 Optimization")
            print("=" * 60)
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Checkpoint: {self.checkpoint_dir}")
            print()

        # 1. Load T5 text encoder
        if verbose:
            print("1. Loading T5 text encoder...")
        t5_path = os.path.join(self.checkpoint_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        self.t5 = T5EncoderModel(
            text_len=self.config.text_len,
            dtype=self.config.t5_dtype,
            device=self.device,
            checkpoint_path=t5_path,
            tokenizer_path=os.path.join(self.checkpoint_dir, "google/umt5-xxl"),
            shard_fn=None,
        )
        if verbose:
            print(f"   T5 VRAM: {get_vram_usage():.2f} GB")

        # 2. Load VAE
        if verbose:
            print("\n2. Loading VAE...")
        vae_path = os.path.join(self.checkpoint_dir, "Wan2.2_VAE.pth")
        self.vae = Wan2_2_VAE(
            vae_pth=vae_path,
            device=self.device,
        )
        if verbose:
            print(f"   After VAE VRAM: {get_vram_usage():.2f} GB")

        # 3. Load DiT transformer on CPU
        if verbose:
            print("\n3. Loading DiT transformer...")
        self.dit = WanModel.from_pretrained(self.checkpoint_dir, device="cpu")
        if verbose:
            print("   DiT loaded on CPU")

        # 4. Apply quantization
        if quantization != "none":
            if verbose:
                print(f"\n4. Applying {quantization.upper()} quantization to DiT...")
            quantize_model_int4(self.dit, verbose=False)
            if verbose:
                print(f"   DiT quantized to {quantization.upper()}")

        # 5. Move DiT to GPU
        self.dit = self.dit.to(self.device)
        self.dit.eval()
        clear_vram()
        if verbose:
            print(f"   After DiT on GPU: {get_vram_usage():.2f} GB")

        # Setup scheduler
        self.scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )

        self._loaded = True

        if verbose:
            print()
            print_vram_status()
            print()

        return self

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        size: Literal["landscape", "portrait"] = "landscape",
        num_frames: int = 33,
        num_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = -1,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Generate a video from a text prompt.

        Args:
            prompt: Text description of the video to generate
            negative_prompt: What to avoid (uses default Chinese prompt if None)
            size: "landscape" (1280x704) or "portrait" (704x1280)
            num_frames: Number of frames (must be 4n+1: 17, 21, 25, 29, 33, 49, 81, etc.)
            num_steps: Number of diffusion steps (30 recommended)
            guidance_scale: Classifier-free guidance scale (5.0 recommended)
            seed: Random seed (-1 for random)
            verbose: Whether to print progress

        Returns:
            Video tensor of shape (C, T, H, W)
        """
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded. Call .load() first.")

        from wan.utils.fm_solvers import get_sampling_sigmas, retrieve_timesteps

        # Get dimensions
        width, height = self.SUPPORTED_SIZES[size]
        if negative_prompt is None:
            negative_prompt = self.config.sample_neg_prompt

        if verbose:
            print(f"Generating: {width}x{height}, {num_frames} frames, {num_steps} steps")
            print(f"Prompt: '{prompt[:80]}{'...' if len(prompt) > 80 else ''}'")

        # Setup seed
        if seed < 0:
            import random
            seed = random.randint(0, 2**32 - 1)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # Calculate latent shape
        vae_stride = self.config.vae_stride  # (4, 16, 16)
        target_shape = (
            self.vae.model.z_dim,  # 48
            (num_frames - 1) // vae_stride[0] + 1,
            height // vae_stride[1],
            width // vae_stride[2],
        )

        # Encode text
        with torch.no_grad():
            context = self.t5([prompt], self.device)
            context_null = self.t5([negative_prompt], self.device)

        # Create initial noise
        noise = torch.randn(
            target_shape[0], target_shape[1], target_shape[2], target_shape[3],
            dtype=torch.float32, device=self.device, generator=seed_g
        )

        # Setup timesteps
        sampling_sigmas = get_sampling_sigmas(num_steps, self.config.sample_shift)
        timesteps, _ = retrieve_timesteps(self.scheduler, device=self.device, sigmas=sampling_sigmas)

        # Calculate seq_len
        patch_size = self.config.patch_size
        seq_len = (target_shape[2] * target_shape[3]) // (patch_size[1] * patch_size[2]) * target_shape[1]

        # Create masks
        mask2 = torch.ones(1, 1, target_shape[1], target_shape[2], target_shape[3],
                          device=self.device, dtype=torch.float32)

        # Prepare context args
        arg_c = {'context': context, 'seq_len': seq_len}
        arg_null = {'context': context_null, 'seq_len': seq_len}

        # Denoising loop
        latent = noise
        iterator = tqdm(timesteps, desc="Generating") if verbose else timesteps

        with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            for t in iterator:
                # Prepare timestep
                timestep = torch.tensor([t], device=self.device)
                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                # CFG
                noise_pred_cond = self.dit([latent], t=timestep, **arg_c)[0]
                noise_pred_uncond = self.dit([latent], t=timestep, **arg_null)[0]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # Scheduler step
                latent = self.scheduler.step(
                    noise_pred.unsqueeze(0), t, latent.unsqueeze(0),
                    return_dict=False, generator=seed_g
                )[0].squeeze(0)

        # Offload models before VAE decode
        if verbose:
            print("Offloading models for VAE decode...")
        self.dit.cpu()
        self.t5.model.cpu()
        del noise, context, context_null, arg_c, arg_null, mask2
        torch.cuda.synchronize()
        clear_vram()

        # VAE decode
        if verbose:
            print("Decoding with VAE...")
        with torch.no_grad():
            video = self.vae.decode([latent])

        return video[0]

    def save_video(
        self,
        video: torch.Tensor,
        output_path: str,
        fps: int = 24
    ) -> str:
        """
        Save generated video to file.

        Args:
            video: Video tensor from generate()
            output_path: Path to save video
            fps: Frames per second

        Returns:
            Path to saved video
        """
        from wan.utils.utils import save_video
        save_video(video, output_path, fps=fps)

        file_size = os.path.getsize(output_path) / 1024
        print(f"Video saved: {output_path} ({file_size:.1f} KB)")

        return output_path

    def reload_models(self) -> None:
        """Reload models to GPU after offloading."""
        if self.dit is not None:
            self.dit.to(self.device)
        if self.t5 is not None:
            self.t5.model.to(self.device)
        clear_vram()

# VRAM Usage Benchmarks

Tested on NVIDIA RTX 5090 (32GB VRAM) with CUDA 12.8, PyTorch 2.9.1

## Wan 2.2 TI2V-5B

| Component | Original (BF16) | INT4 Quantized |
|-----------|-----------------|----------------|
| T5 Text Encoder | ~11 GB | ~11 GB (not quantized) |
| VAE | ~3 GB | ~3 GB (not quantized) |
| DiT Transformer | ~11 GB | ~3 GB |
| **Peak During Inference** | ~25 GB | ~16 GB |
| **Peak During VAE Decode** | - | ~15 GB (after offload) |

### Generation Settings
- Resolution: 1280x704 (landscape)
- Frames: 33
- Steps: 30
- Time: ~50 seconds

## LTX-2 19B

| Component | Original (BF16) | INT4 Quantized |
|-----------|-----------------|----------------|
| Gemma-3 Text Encoder | ~27 GB | ~8 GB |
| Transformer | ~40 GB | ~10 GB |
| VAE + Audio | ~5 GB | ~5 GB (not quantized) |
| **Peak During Inference** | ~67 GB | ~22 GB |

### Generation Settings
- Resolution: 640x448 (reduced)
- Frames: 33
- Steps: 25
- Time: ~60 seconds

## Memory Reduction Summary

| Quantization | Memory Reduction | Quality Impact |
|--------------|------------------|----------------|
| INT4 | ~75% | Minimal |
| INT8 | ~50% | Very minimal |
| None | 0% | Baseline |

## Tips for Lower VRAM GPUs

### RTX 4090 (24GB)
- Use INT4 quantization
- Reduce frames to 17-25
- Wan 2.2 works well
- LTX-2 may be tight

### RTX 3090 (24GB)
- Same as RTX 4090
- May need smaller resolution for LTX-2

### RTX 4080 (16GB)
- Wan 2.2 with INT4 is borderline
- Reduce frames to 17
- LTX-2 unlikely to work

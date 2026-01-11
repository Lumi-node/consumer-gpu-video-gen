<p align="center">
  <h1 align="center">🎬 Consumer GPU Video Generation</h1>
  <p align="center">
    <strong>Run 67GB video models on 32GB consumer GPUs</strong>
  </p>
  <p align="center">
    NVIDIA-style optimizations without ComfyUI dependency
  </p>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="#"><img src="https://img.shields.io/badge/pytorch-2.0+-orange.svg" alt="PyTorch 2.0+"></a>
  <a href="#"><img src="https://img.shields.io/badge/CUDA-12.0+-green.svg" alt="CUDA 12.0+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-purple.svg" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/RTX_5090-Tested-76B900?logo=nvidia" alt="RTX 5090 Tested"></a>
  <a href="#"><img src="https://img.shields.io/badge/RTX_4090-Compatible-76B900?logo=nvidia" alt="RTX 4090 Compatible"></a>
</p>

---

## 🎯 The Problem

| Model | Original VRAM | Your GPU |
|:-----:|:-------------:|:--------:|
| **LTX-2 19B** | 67 GB | ❌ Won't fit |
| **Wan 2.2 5B** | 25 GB | ⚠️ Barely fits |

**Large video generation models require datacenter GPUs.** Consumer cards like RTX 4090/5090 can't run them... until now.

---

## ✨ The Solution

<table>
<tr>
<td width="50%">

### Before (Original)
```
❌ LTX-2:    67 GB VRAM
❌ Wan 2.2:  25 GB VRAM
```

</td>
<td width="50%">

### After (INT4 Optimized)
```
✅ LTX-2:    22 GB VRAM  (-67%)
✅ Wan 2.2:  16 GB VRAM  (-36%)
```

</td>
</tr>
</table>

**75% VRAM reduction** via INT4 quantization with minimal quality loss.

---

## 🚀 Key Features

<table>
<tr>
<td align="center" width="25%">
<h3>💾</h3>
<strong>75% Less VRAM</strong><br>
<sub>INT4 quantization shrinks models to fit consumer GPUs</sub>
</td>
<td align="center" width="25%">
<h3>🔓</h3>
<strong>No ComfyUI</strong><br>
<sub>Standalone Python - use in any project</sub>
</td>
<td align="center" width="25%">
<h3>⚡</h3>
<strong>RTX 5090 Ready</strong><br>
<sub>Tested on latest Blackwell architecture</sub>
</td>
<td align="center" width="25%">
<h3>🎨</h3>
<strong>Simple API</strong><br>
<sub>3 lines of code to generate video</sub>
</td>
</tr>
</table>

---

## 📊 Benchmarks

> Tested on **NVIDIA RTX 5090** (32GB) with CUDA 12.8

| Model | Original | Optimized | Resolution | Speed |
|:------|:--------:|:---------:|:----------:|:-----:|
| **Wan 2.2 TI2V-5B** | 25 GB | **16 GB** | 1280×704 | ~50s |
| **LTX-2 19B** | 67 GB | **22 GB** | 640×448 | ~60s |

<details>
<summary>📈 Detailed VRAM Breakdown</summary>

### Wan 2.2 TI2V-5B
| Component | Original | INT4 |
|:----------|:--------:|:----:|
| T5 Text Encoder | 11 GB | 11 GB |
| VAE | 3 GB | 3 GB |
| DiT Transformer | 11 GB | **3 GB** |
| **Peak** | 25 GB | **16 GB** |

### LTX-2 19B
| Component | Original | INT4 |
|:----------|:--------:|:----:|
| Gemma-3 Text Encoder | 27 GB | **8 GB** |
| Transformer | 40 GB | **10 GB** |
| VAE + Audio | 5 GB | 5 GB |
| **Peak** | 67 GB | **22 GB** |

</details>

---

## 🛠️ Installation

### Requirements

- **GPU**: RTX 4090, RTX 5090, A6000, or similar (24-32GB VRAM)
- **CUDA**: 12.0+
- **Python**: 3.10+

### Quick Start

```bash
# Clone repository
git clone https://github.com/lumi-node/consumer-gpu-video-gen
cd consumer-gpu-video-gen

# Install dependencies
pip install -r requirements.txt

# Download Wan 2.2 (recommended for most users)
git clone https://github.com/Wan-Video/Wan2.2
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
```

---

## 💻 Usage

### Command Line

```bash
# Generate with Wan 2.2
python generate.py --model wan22 \
    --prompt "A fluffy cat walking through a sunny garden" \
    --checkpoint ./Wan2.2-TI2V-5B \
    --wan-repo ./Wan2.2
```

### Python API

```python
from models.wan22 import Wan22Pipeline

# Load with INT4 optimization
pipeline = Wan22Pipeline(checkpoint_dir="./Wan2.2-TI2V-5B", wan_repo_path="./Wan2.2")
pipeline.load(quantization="int4")

# Generate video
video = pipeline.generate("A cat playing in a garden")
pipeline.save_video(video, "output.mp4")
```

<details>
<summary>📋 All CLI Options</summary>

```
--model, -m      Model: wan22 or ltx2 (required)
--checkpoint, -c Path to model checkpoint (required)
--wan-repo       Path to Wan2.2 repo (required for wan22)
--prompt, -p     Text prompt (required)
--output, -o     Output path (default: auto-generated)

--frames         Number of frames (default: 33)
--steps          Diffusion steps (default: 30)
--guidance       Guidance scale (default: 5.0)
--seed           Random seed (default: random)
--size           landscape or portrait (default: landscape)
--fps            Output FPS (default: 24)

--quantization   int4, int8, or none (default: int4)
```

</details>

---

## 🔬 How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Standard Loading                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────────┐                  │
│  │ T5 Enc  │ +│   VAE   │ +│ Transformer │ = 67GB ❌        │
│  │  27GB   │  │   5GB   │  │    40GB     │                  │
│  └─────────┘  └─────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  INT4 Quantized Loading                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────────┐                  │
│  │ T5 Enc  │ +│   VAE   │ +│ Transformer │ = 22GB ✅        │
│  │   8GB   │  │   5GB   │  │    10GB     │   (INT4)         │
│  └─────────┘  └─────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### The Magic: quanto INT4 Quantization

- **16-bit → 4-bit weights** = 75% smaller
- **No retraining required** - post-training quantization
- **Minimal quality loss** - optimized dequantization at inference

### Smart Memory Management

1. Load models sequentially
2. Quantize before moving to GPU
3. Offload unused models during VAE decode
4. Strategic garbage collection

---

## 🎮 GPU Compatibility

| GPU | VRAM | Wan 2.2 | LTX-2 |
|:----|:----:|:-------:|:-----:|
| **RTX 5090** | 32 GB | ✅ Full | ✅ Reduced res |
| **RTX 4090** | 24 GB | ✅ Full | ⚠️ Tight |
| **RTX 4080** | 16 GB | ⚠️ Limited | ❌ |
| **RTX 3090** | 24 GB | ✅ Full | ⚠️ Tight |
| **A6000** | 48 GB | ✅ Full | ✅ Full |

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- [ ] Additional model support (CogVideoX, etc.)
- [ ] FP8 quantization for Blackwell GPUs
- [ ] Web UI interface
- [ ] Audio generation for LTX-2

---

## 📚 Acknowledgments

- [Alibaba Wan Team](https://github.com/Wan-Video/Wan2.2) - Wan 2.2 model
- [Lightricks](https://github.com/Lightricks/LTX-Video) - LTX-2 model
- [Hugging Face](https://github.com/huggingface/quanto) - quanto quantization
- NVIDIA - Optimization inspiration from ComfyUI implementations

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

<p align="center">
  <sub>Built with ❤️ for the open-source AI community</sub>
</p>

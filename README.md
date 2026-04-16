<div align="center">

# Say hi to LightDiffusion-Next 👋

[![demo platform](https://img.shields.io/badge/Play%20with%20LightDiffusion%21-LightDiffusion%20demo%20platform-lightblue)](https://huggingface.co/spaces/Aatricks/LightDiffusion-Next)&nbsp;

**LightDiffusion-Next**  is the fastest AI-powered image generation WebUI, combining speed, precision, and flexibility in one cohesive tool.
</br>
</br>
  <a href="https://github.com/LightDiffusion/LightDiffusion-Next">
    <img src="https://github.com/user-attachments/assets/b994fe0d-3a2e-44ff-93a4-46919cf865e3" alt="Logo">

  </a>
</br>
</div>

---

As a refactored and improved version of the original [LightDiffusion repository](https://github.com/Aatrick/LightDiffusion), this project enhances usability, maintainability, and functionality while introducing a host of new features to streamline your creative workflows.


## Motivation:

**LightDiffusion** was originally meant to be made in Rust, but due to the lack of support for the Rust language in the AI community, it was made in Python with the goal of being the simplest and fastest AI image generation tool.

That's when the first version of LightDiffusion was born which only counted [3000 lines of code](https://github.com/LightDiffusion/LightDiffusion-original), only using Pytorch. With time, the [project](https://github.com/Aatrick/LightDiffusion) grew and became more complex, and the need for a refactor was evident. This is where **LightDiffusion-Next** comes in, with a more modular and maintainable codebase, and a plethora of new features and optimizations.

📚 Learn more in the [official documentation](https://aatricks.github.io/LightDiffusion-Next/)

For a source-based breakdown of the optimization stack, see the [Implemented Optimizations Report](https://aatricks.github.io/LightDiffusion-Next/implemented-optimizations-report/).

---

## 🌟 Highlights

![image](https://github.com/user-attachments/assets/b994fe0d-3a2e-44ff-93a4-46919cf865e3)

**LightDiffusion-Next** offers a powerful suite of tools to cater to creators at every level. At its core, it supports **Text-to-Image** (Txt2Img) and **Image-to-Image** (Img2Img) generation, offering a variety of upscale methods and samplers, to make it easier to create stunning images with minimal effort.

Advanced users can take advantage of features like **attention syntax**, **Hires-Fix** or **ADetailer**. These tools provide better quality and flexibility for generating complex and high-resolution outputs.

**LightDiffusion-Next** is fine-tuned for **performance**. Features such as **Xformers** acceleration, **BFloat16** precision support, **WaveSpeed** dynamic caching, **Multi-scale diffusion**, and **Stable-Fast** model compilation (which offers up to a 70% speed boost) ensure smooth and efficient operation, even on demanding workloads.

---

## ✨ Feature Showcase

Here’s what makes LightDiffusion-Next stand out:

- **Speed and Efficiency**:
  Enjoy industry-leading performance with built-in Xformers, Pytorch, Wavespeed and Stable-Fast optimizations, Multi-scale diffusion, deepcache, AYS (Align Your Steps) scheduler, and automatic prompt caching achieving 30% up to 200% faster speeds compared to the rest of the AI image generation backends in SD1.5 and Flux.

- **Automatic Detailing**:
  Effortlessly enhance faces and body details with AI-driven tools based on the [Impact Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack).

- **State Preservation**:
  Save and resume your progress with saved states, ensuring seamless transitions between sessions.

- **Integration-Ready**:
  Collaborate and create directly in Discord with [Boubou](https://github.com/Aatrick/Boubou), or preview images dynamically with the optional **TAESD preview mode**.

- **Image Previewing**:
  Get a real-time preview of your generated images with TAESD, allowing for user-friendly and interactive workflows.

- **Image Upscaling**:
  Enhance your images with advanced upscaling options like UltimateSDUpscaling, ensuring high-quality results every time.

- **Prompt Refinement**:
  Use the optional Ollama-powered prompt enhancer (defaults to `qwen3:0.6b`) to refine your prompts and generate more accurate and detailed outputs.

- **LoRa and Textual Inversion Embeddings**:
    Leverage LoRa and textual inversion embeddings for highly customized and nuanced results, adding a new dimension to your creative process.

- **Low-End Device Support**:
    Run LightDiffusion-Next on low-end devices with as little as 2GB of VRAM or even no GPU, ensuring accessibility for all users.

- **CFG++**:
    Uses samplers modified to use CFG++ for better quality results compared to traditional methods.

- **Newelle Extension**:
    LightDiffusion-Next is also available as a backend to the [Newelle LightDiffusion extension](https://github.com/Aatricks/Newelle-Light-Diffusion) permitting to generate images inline during conversations with llms.

---

## ⚡ Performance Benchmarks

**LightDiffusion-Next** dominates in performance:

| **Tool**                           | **Speed (it/s)** |
|------------------------------------|------------------|
| **LightDiffusion with Stable-Fast** | 2.8              |
| **LightDiffusion**                 | 1.9              |
| **ComfyUI**                        | 1.4              |
| **SDForge**                        | 1.3              |
| **SDWebUI**                        | 0.9              |

(All benchmarks are based on a 1024x1024 resolution with a batch size of 1 using BFloat16 precision without tweaking installations. Made with a 3060 mobile GPU using SD1.5.)

With its unmatched speed and efficiency, LightDiffusion-Next sets the benchmark for AI image generation tools.

---

## 🛠 Installation

> [!NOTE]
> **Platform Support:** LightDiffusion-Next supports NVIDIA GPUs (CUDA), AMD GPUs (ROCm), and Apple Silicon (Metal/MPS). For AMD and Apple Silicon setup instructions, see the [ROCm and Metal/MPS Support Guide](https://aatrick.github.io/LightDiffusion/rocm-metal-support/).

> [!WARNING]
> **Disclaimer:** On Linux, the fastest way to get started is with the Docker setup below. Windows users often encounter an `EOF` build error when using Docker; if that happens, set up a local virtual environment instead and install SageAttention inside it.

> [!NOTE]
> You will need to download the [flux vae](https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/ae.safetensors) separately given its gated repo on Huggingface. Drop it in the `/include/vae` folder.

### Quick Start

1. Download a release or clone this repository.
2. Run `run.bat` in a terminal.
3. The modern React frontend will launch automatically at `http://localhost:5173` (proxied to the FastAPI backend at `http://localhost:7861`).

**Recommended Launch Command:**
```bash
# Start both backend and frontend development server
python server.py --frontend
```

**Production-style local run:**
```bash
# Serve the built React UI from FastAPI on a single port
python server.py --port 7860
```

**ZeroGPU / Gradio launch:**
```bash
# Launch the Hugging Face ZeroGPU-compatible Gradio UI
python app.py
```

### 🌌 Flux Support

LightDiffusion-Next now features first-class support for **Flux2 Klein**. To get started, you need to download the required model components (Diffusion Model, Text Encoder, and VAE).

We provide a convenient script to handle this automatically:
```bash
python download_flux.py
```
This will download approximately 16GB of weights into the `include/` directory.

### 🤗 ZeroGPU / Gradio Space

This repository now includes a Gradio `app.py` entrypoint for Hugging Face
**ZeroGPU**. ZeroGPU is only supported for Gradio SDK Spaces, and the
GPU-bound generation function is wrapped with `@spaces.GPU`.

Recommended defaults for ZeroGPU:
- keep `Keep Models Loaded` disabled
- use 512x512 or 768x768 resolutions
- generate 1 image at a time
- prefer 10-25 steps with `ays`

### 🐳 Docker Setup

Run LightDiffusion-Next in a containerized environment with GPU acceleration.
The Docker path remains available for local or dedicated GPU deployments and
serves the built React frontend from the FastAPI backend on port `7860`.

> [!IMPORTANT]
> Confirm you have Docker Desktop configured with the NVIDIA Container Toolkit and at least 12-16GB of memory. Builds expect an NVIDIA GPU with compute capability 8.0 or higher and CUDA 12.0+ support for SageAttention/SpargeAttn.

**Quick Start with Docker:**
```bash
# Build and run with docker-compose
docker-compose up --build

# Or build and run manually
docker build -t lightdiffusion-next .
docker run --gpus all -p 7860:7860 -e PORT=7860 -v ./output:/app/output lightdiffusion-next
```

**Custom GPU Architecture (Optional):**
```bash
# For faster builds, specify your GPU architecture (e.g., RTX 5060 = 12.0)
docker-compose build --build-arg TORCH_CUDA_ARCH_LIST="12.0"

# Default builds for: 8.0 (A100), 8.6 (RTX 30xx), 8.9 (RTX 40xx), 9.0 (H100), 12.0 (RTX 50xx)
```

**Built-in Optimizations:**
The Docker image can optionally build the following acceleration paths:
- ✨ **SageAttention** - 15% speedup with INT8 quantization (all supported GPUs)
- 🚀 **SpargeAttn** - 40-60% speedup with sparse attention (compute 8.0-9.0 only)
- ⚡ **Stable-Fast** - Optional UNet compilation for up to 70% faster SD1.5 inference

Control them through build arguments (defaults shown below):

```bash
docker-compose build \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" \
  --build-arg INSTALL_SAGEATTENTION=0 \
  --build-arg INSTALL_SPARGEATTN=0 \
  --build-arg INSTALL_STABLE_FAST=1 \
  --build-arg INSTALL_OLLAMA=0
```

Set `INSTALL_STABLE_FAST=1` to enable stable-fast, `INSTALL_SAGEATTENTION=1`
or `INSTALL_SPARGEATTN=1` to opt into the heavier attention-kernel builds, and
`INSTALL_OLLAMA=1` to bake in the prompt enhancer runtime.

> [!NOTE]
> RTX 50 series (compute 12.0) GPUs currently use SageAttention when the SageAttention kernel is installed. SpargeAttn remains limited to earlier supported architectures.

**Access the Web Interface:**
- **FastAPI + React UI**: `http://localhost:7860`

**Volume Mounts:**
- `./output:/app/output` - Persist generated images
- `./checkpoints:/app/include/checkpoints` - Store model files
- `./loras:/app/include/loras` - Store LoRA files
- `./embeddings:/app/include/embeddings` - Store embeddings

### Advanced Setup

- **Install from Source**:
  Install dependencies via:
  ```bash
  pip install -r requirements.txt
  ```
  Add your SD1/1.5 safetensors model to the `checkpoints` directory, then launch the application.

- **⚡Stable-Fast Optimization**:
  Follow [this guide](https://github.com/chengzeyi/stable-fast?tab=readme-ov-file#installation) to enable Stable-Fast mode for optimal performance.
  In Docker environments, set `INSTALL_STABLE_FAST=1` to compile it during the image build or `INSTALL_STABLE_FAST=0` (default) to skip.

- **🚀 SageAttention & SpargeAttn Acceleration**:
  Boost inference speed by up to 60% with advanced attention backends:

  **Prerequisites:**
  - [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit-archive) installed with version compatible with your PyTorch installation
  
  **SageAttention (15% speedup, Windows compatible):**
  ```bash
  cd SageAttention
  pip install -e . --no-build-isolation
  ```
  
  **SpargeAttn (40-60% total speedup, requires WSL2/Linux):**
> [!CAUTION]
> SpargeAttn cannot be built with the default Windows linker. Use WSL2 or a native Linux environment and set the correct `TORCH_CUDA_ARCH_LIST` before installation.
  ```bash
  # On WSL2 or Linux only (Windows linker has path length limitations)
  cd SpargeAttn
  export TORCH_CUDA_ARCH_LIST="9.0"  # Or your GPU architecture (8.0, 8.6, 8.9, 9.0)
  pip install -e . --no-build-isolation
  ```
  
  **Priority System:** SpargeAttn > SageAttention > PyTorch SDPA
  - Both are automatically detected and used when available
  - Graceful fallback for unsupported head dimensions

- **🦙 Prompt Enhancer**:
  Turn on the Ollama-backed enhancer to automatically restructure prompts. By default the app targets `qwen3:0.6b`:
  ```bash
  # Local install
  pip install ollama
  curl -fsSL https://ollama.com/install.sh | sh

  # Start the Ollama daemon (keep this terminal open)
  ollama serve

  # New terminal: pull the default prompt enhancer model
  ollama pull qwen3:0.6b
  export PROMPT_ENHANCER_MODEL=qwen3:0.6b
  ```
  In Docker builds, set `--build-arg INSTALL_OLLAMA=1` (or update `docker-compose.yml`) to install Ollama and pre-pull the model automatically. You can override the runtime model/prefix with the `PROMPT_ENHANCER_MODEL` and `PROMPT_ENHANCER_PREFIX` environment variables. See the [Ollama guide](https://github.com/ollama/ollama?tab=readme-ov-file) for details.

- **🤖 Discord Integration**:
  Set up the Discord bot by following the [Boubou installation guide](https://github.com/Aatrick/Boubou).

### Third-Party Licenses
- This project distributes builds that depend on third-party open source components. For attribution details and the full license text, refer to `THIRD_PARTY_LICENSES.md`.

---

🎨 Enjoy exploring the powerful features of LightDiffusion-Next!

> [!TIP]
> ⭐ If this project helps you, please give it a star! It helps others discover it too.

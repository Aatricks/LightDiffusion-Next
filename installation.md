# Installation & Setup

LightDiffusion-Next can run locally on Windows or Linux, inside Docker, or on cloud GPUs. This page walks you through the supported installation paths and the assets you must download before your first generation.

## Hardware & software requirements

The project is tuned for NVIDIA GPUs and CUDA 12.x drivers, but it also supports AMD GPUs with ROCm and Apple Silicon with Metal Performance Shaders (MPS). See [ROCm and Metal/MPS Support](rocm-metal-support.md) for platform-specific installation instructions.

- **Operating system:** Windows 10/11, Ubuntu 22.04+, macOS 12.3+ (for Apple Silicon), or any distro supported by NVIDIA Container Toolkit.
- **Python:** 3.10.x. The run scripts create a virtual environment automatically.
- **GPU:** 
  - **NVIDIA:** Card with at least compute capability 8.0 (Ampere) for SageAttention/SpargeAttn. RTX 50 series (compute 12.0) runs with SageAttention + Stable-Fast.
  - **AMD:** RDNA 2+ or CDNA architectures with ROCm 5.0+. See [ROCm Support](rocm-metal-support.md#rocm-support-amd-gpus).
  - **Apple Silicon:** M1/M2/M3 series with macOS 12.3+. See [Metal/MPS Support](rocm-metal-support.md#metalmps-support-apple-silicon).
- **VRAM:** 6 GB minimum (12 GB recommended) for SD1.5 workflows. Flux quantized pipelines require 16 GB+ for comfortable batching.
- **Disk space:** ~15 GB for dependencies plus your checkpoints, LoRAs and flux assets.

## Choose an installation path

- [Windows quick start](#windows-quick-start-runbat)
- [Linux or WSL2 manual setup](#linuxwsl2-manual-setup)
- [Containerized deployment](#docker-and-containers)
- [Headless server API](#running-only-the-fastapi-server)

### Windows quick start (`run.bat`)

The root repository ships with a convenience script that handles environment creation, dependency installation via `uv`, GPU detection and launching the Streamlit UI.

1. Install the latest [Python 3.10](https://www.python.org/downloads/release/python-3100/) build and ensure `python` is on your `PATH`.
2. Install the [NVIDIA CUDA 12 runtime driver](https://developer.nvidia.com/cuda-downloads) that matches your GPU.
3. Clone the repository and place your checkpoints in `include/checkpoints` (see [Model assets](#model-assets)).
4. Double-click `run.bat` from a terminal. The script will:

	- Create `.venv` (if it does not exist) and upgrade `pip`.
	- Install `uv` for fast dependency resolution.
	- Detect an NVIDIA GPU via `nvidia-smi` and install the matching PyTorch wheels.
	- Install all requirements and start Streamlit at `http://localhost:8501`.

5. When you are done, close the terminal to stop the UI. The virtual environment is reusable across runs.

> **Tip:** To launch the Gradio UI instead, activate `.venv` and run `python app.py`.

### Linux/WSL2 manual setup

1. Install system dependencies:

	```bash
	sudo apt update && sudo apt install python3.10 python3.10-venv python3-pip build-essential git
	```

2. (Optional) Install the [NVIDIA CUDA 12 toolkit](https://developer.nvidia.com/cuda-toolkit-archive) so SageAttention/SpargeAttn can compile native extensions.
3. Create and activate a virtual environment:

	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	pip install --upgrade pip uv
	```

4. Install PyTorch and core dependencies:

	```bash
	uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision "triton>=2.1.0"
	uv pip install -r requirements.txt
	```

5. Launch the Streamlit UI:

	```bash
	streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501
	```

	Use `python app.py` if you prefer the Gradio interface.

6. Deactivate the environment with `deactivate` when finished.

### Docker and containers

Use Docker when you want an immutable runtime with SageAttention, SpargeAttn and Stable-Fast prebuilt.

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
2. Clone the repository and review `docker-compose.yml`. Adjust:

	- `TORCH_CUDA_ARCH_LIST` if you only target a specific GPU architecture.
	- `INSTALL_STABLE_FAST` and `INSTALL_OLLAMA` build arguments if you want Stable-Fast or the Ollama prompt enhancer baked into the image.
	- Volume mounts for `output/` and the `include/*` directories where you store checkpoints, LoRAs, embeddings and YOLO detectors.

3. Build and start the stack:

	```bash
	docker-compose up --build
	```

	Streamlit is exposed on `http://localhost:8501` by default; Gradio is mapped to port `7860` and can be enabled by setting `UI_FRAMEWORK=gradio`.

4. To rebuild with a different GPU architecture or optional component:

	```bash
	docker-compose build --build-arg TORCH_CUDA_ARCH_LIST="9.0" --build-arg INSTALL_STABLE_FAST=1
	```

### Running only the FastAPI server

If you want to integrate LightDiffusion-Next into automation pipelines or Discord bots, run the backend without launching a UI.

1. Follow any of the setup methods above.
2. Run:

	```bash
	uvicorn server:app --host 0.0.0.0 --port 7861
	```

3. Use the [REST API reference](api.md) to submit generation jobs via `POST /api/generate` and inspect queue health via `GET /api/telemetry`.

## Model assets

LightDiffusion-Next does not bundle model weights. Place your assets into the `include/` tree before you start generating.

- `include/checkpoints/` — SD1.5 style `.safetensors` checkpoints (e.g. Meina V10, DreamShaper). The default pipeline expects a file named `Meina V10 - baked VAE.safetensors` unless you override it.
- `include/vae/ae.safetensors` — Flux VAE (download from [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)). Required for Flux mode.
- `include/loras/` — LoRA adapters loaded from the UI or CLI.
- `include/embeddings/` — Negative prompt embeddings such as `EasyNegative`, `badhandv4`.
- `include/yolos/` — YOLO detectors used by ADetailer (`person_yolov8m-seg.pt`, `face_yolov9c.pt`).
- `include/ESRGAN/` — RealESRGAN models leveraged by UltimateSDUpscale in Img2Img workflows.
- `include/sd1_tokenizer/` — Tokenizer files for SD1.x. The repository already includes the defaults.

Store generated outputs under `output/` (separated into Classic, Flux, Img2Img, HiresFix and ADetailer sub-folders). The folders are created automatically during the first run.

## Optional accelerations

- **Stable-Fast** — 70% faster SD1.5 inference through UNet compilation. Set `INSTALL_STABLE_FAST=1` in Docker or pass `--stable-fast` in the CLI/UI to compile on demand. Compilation adds a one-time warm-up cost.
- **SageAttention** — INT8 attention kernels with 15% speedup and lower VRAM use. Built automatically in Docker images; on bare metal, clone [SageAttention](https://github.com/thu-ml/SageAttention) and run `pip install -e . --no-build-isolation` inside your environment.
- **SpargeAttn** — Sparse attention kernels with 40–60% speedup (compute 8.0–9.0 GPUs only). Build from [SpargeAttn](https://github.com/thu-ml/SpargeAttn) using `TORCH_CUDA_ARCH_LIST="8.9"` or similar.
- **Ollama prompt enhancer** — Install [Ollama](https://ollama.com/) and pull `qwen3:0.6b`. Set `PROMPT_ENHANCER_MODEL=qwen3:0.6b` before launching LightDiffusion-Next to enable the automatic prompt rewrite toggle.

## Verify your installation

1. Start the UI or FastAPI server.
2. Watch the startup logs — the initialization progress bar runs the dependency download routine (`CheckAndDownload`) and loads the default checkpoint.
3. Generate a 512×512 image with the default prompt. The status bar shows timing and the output appears in `output/Classic`.
4. Confirm the telemetry endpoint is reachable:

	```bash
	curl http://localhost:7861/health
	curl http://localhost:7861/api/telemetry
	```

## Updating or rebuilding

- Pull the latest Git changes and rerun `uv pip install -r requirements.txt` in the virtual environment.
- For Docker users, rebuild with `docker-compose build --no-cache` to pick up updates.
- If you upgraded your GPU driver or CUDA toolkit, delete `~/.cache/torch_extensions` to force SageAttention/SpargeAttn to recompile.

You are now ready to explore the [UI guide](usage.md) and start generating.
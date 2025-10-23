# Architecture

LightDiffusion-Next is split into three cooperating layers: UX surfaces, a FastAPI gateway and a modular inference core. Requests move through these layers, picking up metadata and transformations before image tensors ever touch the GPU. This page decomposes the system and highlights the extension points you are most likely to touch.

## Layers in detail

### UX layer (`streamlit_app.py`, `app.py`, `ui/*`)

- Streamlit exposes rich controls, preset management and history in `ui/settings.py` and `ui/history.py`.
- Gradio powers Spaces deployments (`app.py`). It streams previews via generators and mirrors the Streamlit control surface.
- Both UIs instantiate a shared `AppInstance` which holds the pipeline, preview queues and cached settings.

### FastAPI gateway (`server.py`)

- Implements `/api/generate`, `/api/telemetry`, `/api/interrogate` and health probes.
- `GenerationBuffer` batches jobs with compatible shapes, models and LoRA overlays to maximize GPU utilization.
- Telemetry exposes queue lengths, average latency, VRAM usage and cached model fingerprints.
- Server-side logging includes per-request identifiers and request tracebacks in `logs/server.log`.

### Pipeline core (`src/user/pipeline.py`)

This module orchestrates conditioning, diffusion, optional refinements and output serialization.

- **Model resolution** — `src/FileManaging/Loader` locates checkpoints, VAE, CLIP weights and LoRAs. Stable-Fast backends live in `src/StableFast` and can be toggled in settings.
- **Conditioning** — Prompts are tokenized through `src/cond/cond.py`. Negative prompts, style presets and textual inversion embeddings are applied here.
- **Sampling** — `src/sample/sampling.KSampler` coordinates samplers (`ddim`, `dpmpp`, `k-diffusion`, etc.) with CFG++ and Flux schedulers.
- **Enhancements** — Multi-scale diffusion (`multiscale_presets.py`), AutoDetailer (YOLO detection + inpainting), UltimateSDUpscale and AutoHDR run after the base diffusion loop.
- **Outputs** — `src/FileManaging/ImageSaver` writes PNGs, JSON metadata and optionally sends frames to the preview queues.

### Device and cache (`src/Device/ModelCache.py`)

- Maintains reference-counted handles for UNet, VAE, CLIP and Flux components.
- Handles VRAM telemetry and eviction policies so the UI can show “keep loaded” toggles without manual restarts.
- Tracks whether Stable-Fast kernels, SageAttention or SD1.5 attention patches are initialized.

### Asset management (`src/FileManaging/Downloader.py`)

- Validates required checkpoints, VAE files, LoRAs, embeddings, YOLO detectors and Flux components at startup.
- Supports mirrored download hosts and resumable transfers for large files.
- Exposes helper methods used by the UI to fetch missing assets on demand.

### Preview subsystem (`src/user/app_instance.py`)

- Provides `get_latest_previews()` for UI clients, backed by a dedicated thread that consumes preview tensors straight from the pipeline.
- Supports interrupt handling by setting `app_instance.interrupt = True`, which causes the sampler to exit gracefully.

## Request lifecycle

1. **Submission** — A UI or REST client creates a job payload containing prompts, dimensions, sampler settings, seed and post-processing flags.
2. **Queueing & batching** — Jobs are inserted into `GenerationBuffer`. Depending on `LD_BATCH_WAIT_SINGLETONS`, single jobs may wait briefly for compatible companions to maximize GPU throughput.
3. **Model preparation** — The pipeline loads or reuses cached models, applies LoRA deltas, textual inversion embeddings and optional quantization adapters (via `src/Quantize`).
4. **Diffusion** — The sampler executes the denoising loop. Flux mode uses `src/BlackForest/Flux.py` for decoder steps; Stable-Fast kernels speed up SD1.5/SDXL.
5. **Refinement** — Optional stages (HiRes Fix, AutoDetailer, AutoHDR, UltimateSDUpscale) run sequentially per sample.
6. **Persistence** — Final images and metadata are written to `output/<workflow>/`. Streamlit previews receive running frames; REST clients receive base64 PNG payloads plus telemetry.

## Filesystem overview

- `include/checkpoints` — SD checkpoints (1.5, SDXL, Flux, etc.).
- `include/loras`, `include/embeddings` — LoRA adapters and textual inversion concepts.
- `include/clip` — Tokenizer and encoder configs.
- `include/yolos` — Object detectors for AutoDetailer.
- `include/ESRGAN` — Upscaler models for UltimateSDUpscale.
- `output/*` — Organized galleries (Classic, Flux, Img2Img, Upscale, etc.).
- `webui_settings.json` — Persisted Streamlit configuration.

## Extending LightDiffusion-Next

- **New samplers** — Implement in `src/sample/samplers.py` and register with `KSampler`. Add UI and REST switches via `ui/settings.py` and `GenerateRequest`.
- **Additional post-processing** — Follow the pattern in `UltimateSDUpscale` or `AutoHDR` and register the stage near the end of `pipeline()`.
- **Custom model managers** — Plug alternative download logic into `FileManaging/Downloader` or mount volumes in Docker deployments.
- **Observability** — Add metrics/log statements in `GenerationBuffer` or extend `/api/telemetry` to fit orchestrator dashboards.

Armed with this bird’s-eye view, you can dive into the [usage guide](usage.md) for operator workflows or the upcoming [API reference](api.md) for automation hooks.
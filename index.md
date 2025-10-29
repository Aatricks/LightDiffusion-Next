# LightDiffusion-Next

LightDiffusion-Next is a refactored and performance-first Stable Diffusion stack that bundles a modern Streamlit UI, an optional Gradio web app, a batched FastAPI backend and highly tuned inference primitives such as Stable-Fast, SageAttention and WaveSpeed caching.

## Why pick LightDiffusion-Next

LightDiffusion-Next is built to handle day-to-day generation workloads on consumer GPUs while still scaling up to multi-user servers.

- **Fast by default.** Stable-Fast compilation, SageAttention, SpargeAttn and WaveSpeed caching are wired in so you can hit top-tier it/s without manual patching.
- **Multiple front-doors.** Choose between the Streamlit control room, a Gradio web UI (great for Spaces) or the programmable FastAPI queue for integrations.
- **Feature complete.** Txt2Img, Img2Img, Flux pipelines, AutoHDR, TAESD previews, prompt enhancement through Ollama, multi-scale diffusion with presets, LoRA mixing and automatic detailing are all available out of the box.
- **Operations friendly.** Docker images, GPU-aware batched serving, model caching controls and observability endpoints make it easy to deploy and monitor.

## What ships in the box

- 🚀 **Streamlined UI** with live previews, history, presets, interrupt/resume controls and automatic metadata tagging.
- 🧠 **Prompt toolkit** including reusable negative embeddings, multi-concept weighting, prompt enhancement and prompt history.
- 🧩 **Modular pipeline** that routes SD1.5, SDXL-inspired workflows and quantized Flux models through a single code path with per-sample overrides for HiresFix, ADetailer or Img2Img.
- 🛠️ **Production API** powered by FastAPI with smart request coalescing, telemetry endpoints and base64 image responses ready for bots or creative tooling.
- 📦 **Deployment artifacts** such as Dockerfiles, docker-compose, run scripts for Windows, configurable GPU architecture flags and optional Ollama/Stable-Fast builds.

## Quick pathways

- [Installation](installation.md) — pick Docker, Windows batch or manual Python setup.
- [First run & UI tour](usage.md) — learn the Streamlit layout, generation controls and history tools.
- [Workflow playbook](examples.md) — step through Txt2Img, Flux, Img2Img and API recipes.
- [Performance optimizations](optimizations.md) — understand SageAttention, Stable-Fast, WaveSpeed caching and the new AYS scheduler for 2-5x speedup.
- [Align Your Steps](ays-scheduler.md) — learn about AYS scheduler and prompt caching for additional speedup.
- [Prompt Caching](prompt-caching.md) — deep dive into prompt attention caching mechanics and tuning.
- [Performance tuning](quirks.md) — squeeze out extra throughput or reduce VRAM usage.
- [Architecture](architecture.md) — understand how the UI, pipeline and server cooperate.
- [REST & automation](api.md) — integrate Discord bots, automations or other clients.

## Supported environments at a glance

- NVIDIA GPUs with CUDA 12.x drivers. SageAttention and SpargeAttn availability is detected at runtime and depends on installed kernels, drivers and GPU compute capability; some kernels may be disabled on newer CUDA runtimes (for example CUDA 12+). RTX 50xx and newer cards may use SageAttention + Stable-Fast where supported.
- Windows 10/11, Ubuntu 22.04+ and containerized deployments via Docker with NVIDIA Container Toolkit.
- Optional CPU-only mode for experimentation (no Stable-Fast/SageAttention speed-ups).

## Where to head next

- Start with [Installation](installation.md) to get your environment ready.
- Drop into the [Streamlit UI guide](usage.md) for a tour of generation features and presets.
- Explore [Architecture](architecture.md) when you are ready to customize or embed LightDiffusion-Next in larger systems.
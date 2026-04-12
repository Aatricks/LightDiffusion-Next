# FAQ

Q: Where do I put my checkpoints?

A: Put them in `include/checkpoints` (create the folder if missing). The UI and `src/FileManaging/Loader` will detect and list them.

Q: Why is GPU memory insufficient?

A: Try reducing `width`/`height`, turning off `keep models loaded`, or enable quantized Flux/GGUF models. See [Performance & Troubleshooting](quirks.md).

Q: Can I run headless on a server?

A: Yes — use the FastAPI backend with `docker-compose` or run `server.py` directly. Disable Streamlit if you don’t need the web UI.

Q: How do I contribute models or LoRAs?

A: Place LoRA files in `include/loras` and embeddings in `include/embeddings`. See [Contributing](contributing.md) for guidelines.

/// details | Which diffusion models are supported out of the box?
LightDiffusion-Next ships with Stable Diffusion 1.5-friendly defaults and includes helpers for SDXL-inspired checkpoints, Flux (via the `include/Flux` assets) and quantized Stable-Fast backends. Drop your `.safetensors` or `.ckpt` files into `include/checkpoints`, LoRAs into `include/loras`, embeddings into `include/embeddings`, and Flux weights into `include/Flux`. The loader auto-detects formats and will prompt for missing companions (VAE, CLIP) at startup.
///

/// details | What GPU and driver versions do I need?
NVIDIA GPUs with CUDA 12.1+ drivers are recommended. Availability of Stable-Fast, SageAttention and SpargeAttn depends on your installed kernels, drivers and GPU compute capability — the runtime detects and enables compatible backends automatically. For Docker, install the NVIDIA Container Toolkit and verify `nvidia-smi` works inside the container.
///

/// details | Can I run LightDiffusion-Next without a GPU?
Yes, but performance will be limited. Install CPU wheels of PyTorch or rely on the bundled Intel oneAPI runtime (Linux only). Disable Stable-Fast/SageAttention in settings, reduce resolution (≤384×384), lower steps (<20) and turn off AutoDetailer/HiResFix to avoid minute-long renders.
///

/// details | Where do generated images and metadata live?
Outputs are grouped by workflow under `output/`. For example, standard Txt2Img lands in `output/classic`, HiresFix into `output/HiresFix`, Flux into `output/Flux`, Img2Img upscales into `output/Img2Img`, etc. Each PNG embeds prompt metadata; accompanying JSON manifests are saved when enabled in settings.
///

/// details | How do I switch between Streamlit, Gradio and the API?
Use the launch scripts:

- `streamlit run streamlit_app.py` (default UI)
- `python app.py` (Gradio app for Spaces/remote hosting)
- `uvicorn server:app --host 0.0.0.0 --port 7861` (FastAPI)

All three share the same pipeline and config. Streamlit/Gradio speak directly to the pipeline, while the API feeds the batching queue in `server.py`.
///

/// details | How do I enable Stable-Fast or SageAttention?

In Streamlit, toggle **Stable-Fast** under *Performance*. The app will compile kernels the first time and reuse them afterwards (cache in `~/.cache/torch_extensions`). SageAttention is enabled automatically on supported GPUs; you can force-disable it by setting `LD_DISABLE_SAGE_ATTENTION=1` before launching. Docker images already ship with the patched kernels compiled.
///

/// details | What if the app says a model is missing?

The downloader checks `include/` on startup and whenever a feature needs a new asset (YOLO, Flux, TAESD). Provide URLs or Hugging Face tokens when prompted, or pre-populate the folders manually. For offline environments, copy the files into the correct directories and ensure filenames match the expected suffixes (e.g., `anything-v4.5-pruned.safetensors`).
///

/// details | Can I enhance prompts automatically with Ollama?

Yes. Install Ollama locally, download a language model (`ollama run mistral`), then enable **Prompt Enhancer** in the UI or set `enhance_prompt=true` in the REST payload. Set `OLLAMA_BASE_URL` if Ollama is not on `http://localhost:11434`.
///

/// details | How do I reset persistent settings or history?

Delete `webui_settings.json` in the project root to reset saved toggles and defaults. Remove individual history directories under `ui/history/` to clear the UI gallery without touching generated images.
///

/// details | Need more help?

Check the [Troubleshooting guide](quirks.md) or [open an issue](https://github.com/Aatricks/LightDiffusion-Next/issues) with logs, hardware specs and steps to reproduce.
///

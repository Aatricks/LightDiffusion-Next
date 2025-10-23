# Usage

# First Run & UI Tour

This page walks you through launching LightDiffusion-Next, understanding the Streamlit layout, using the optional Gradio UI and triggering a first generation from the command line.

## Launching the Streamlit UI

- **Windows:** run `run.bat` (see [Installation](installation.md)).
- **Linux/macOS/WSL2:** activate your virtual environment and run `streamlit run streamlit_app.py --server.port=8501`.
- **Docker:** start the compose stack and open `http://localhost:8501`.

You will see an initialization progress indicator while checkpoints and auxiliary models are downloaded. Once complete the app switches to a two-tab layout: **🎨 Generate** and **📜 History**.

## Generate tab

The Generate tab is designed as a control surface where the left sidebar contains parameters and the right canvas displays previews and final renders.

### Prompt & base settings

- **Prompt / Negative prompt** — text areas at the top of the sidebar. Negative prompts are optional; the pipeline automatically falls back to a curated default containing `EasyNegative`, `badhandv4`, `lr` and `ng_deepnegative` embeddings.
- **Dimensions** — width/height sliders (64–2048) with automatic aspect handling in the gallery.
- **Images & batch** — request up to 10 images per job, with a separate batch size controlling internal chunking.

### Feature toggles

- **HiRes Fix** — Upscales the latent and runs an extra sampling pass. Generates output in `output/HiresFix`.
- **ADetailer** — Uses SAM + YOLO and Impact Pack prompt heads to redraw faces/bodies. Additional artifacts are saved to `output/Adetailer`.
- **Enhance prompt** — Sends your prompt through the Ollama model specified by `PROMPT_ENHANCER_MODEL` (defaults to `qwen3:0.6b`). The rewritten prompt is shown in the sidebar and in image metadata.
- **Stable-Fast / Prioritize speed** — Enables UNet compilation (after the first warm-up) and switches the sampler to `dpmpp_2m_cfgpp` for faster iterations.
- **Flux mode** — Routes the job through the quantized Flux pipeline (requires the `ae.safetensors` VAE and quantized GGUF weights downloaded via `CheckAndDownloadFlux`).
- **Img2Img mode** — Reveals an image uploader. The selected picture is used as the source latent, optionally combined with UltimateSDUpscale.
- **Keep models in VRAM** — Toggle model caching between jobs to reduce load time at the cost of VRAM retention.
- **Real-time preview** — Streams TAESD previews into a responsive gallery while sampling is still running. Disable it when running headless to save resources.

### Multi-scale diffusion presets

Under the “Multi-Scale Diffusion Settings” accordion you can:

- Choose a preset (`quality`, `performance`, `balanced`, `disabled`).
- Override the scale factor and the number of steps to run at full resolution.
- Enable intermittent full-resolution refinement.

Multi-scale diffusion provides major frame-time savings at high resolutions and is enabled by default.

### Model cache management

- **🔍 Check VRAM Usage** — reports total/used/free VRAM, cached checkpoints and whether the “keep loaded” flag is active.
- **🗑️ Clear Model Cache** — evicts models from VRAM so the next job reloads everything fresh.

### Status & previews

- A status bar at the bottom of the page surfaces timing, generation stage and any warnings.
- When real-time preview is enabled, the canvas shows the six most recent TAESD frames. They disappear automatically when generation completes.

## Keyboard shortcuts & session state

- Most sliders support arrow-key and shift + arrow adjustments.
- The UI remembers your last-used settings inside `webui_settings.json`. Toggle “Verbose mode” in the settings drawer to see more runtime information.
- Seeds are stored in `include/last_seed.txt`. Enable “Reuse seed” to repeat a composition.

## History tab

- Displays every PNG in the `output/**` tree with metadata overlays (timestamp, dimensions, prompt).
- Use “🔄 Refresh History” to rescan the folders, “🗑️ Delete Selected Image” for targeted cleanup or “⚠️ Clear All Images” to wipe everything.
- Selections show exact file paths so you can open them in external editors.

## Using the Gradio UI

Run `python app.py` (or set `UI_FRAMEWORK=gradio` in Docker) to launch the Gradio frontend at `http://localhost:7860`.

- The controls mirror the Streamlit sidebar but the layout is optimized for Hugging Face Spaces.
- Live previews stream directly to the main gallery while jobs run.
- The 📸 Image History tab reads from the same `output/` folders as Streamlit, so both UIs share artifacts and metadata.

## Command-line pipeline

You can invoke the pipeline without any UI for scripted jobs.

```bash
python -m src.user.pipeline "a futuristic city at dusk" 768 512 2 2 --hires-fix --adetailer --stable-fast --reuse-seed
```

- Positional arguments: `prompt width height number batch`.
- Flags mirror the UI toggles (`--img2img`, `--flux`, `--prio-speed`, `--multiscale-preset`, etc.).
- Img2Img uses the prompt as a filesystem path unless you pass `--img2img-image` through the FastAPI server (see [REST & automation](api.md)).

## Streamlit tips

- Click “Retry Initialization” if the download step fails — the app reruns `CheckAndDownload()`.
- Use the sidebar menu → **Rerun** if you change source code while developing custom nodes.
- When running on laptops, disable “Keep models in VRAM” before closing the UI to release GPU memory for other applications.

Ready for real workflows? Jump to the [examples and recipes](examples.md) page.
Ready for real workflows? Jump to the [examples and recipes](examples.md) page.

## Programmatic pipeline usage (Python)

You can import and call the pipeline directly from Python. The function lives at `src.user.pipeline.pipeline` and accepts the same runtime flags as the CLI. The example below shows a minimal, synchronous call that runs the pipeline and handles the returned mapping when running in batched mode.

```python
from src.user.pipeline import pipeline

result = pipeline(
    prompt=["a futuristic city at dusk", "a cyberpunk alley, rainy"],
    w=768,
    h=512,
    number=2,
    batch=2,
    hires_fix=False,
    adetailer=False,
    stable_fast=False,
    reuse_seed=False,
    flux_enabled=False,
)

# When run in batched mode `pipeline` returns a dict with key 'batched_results'
if isinstance(result, dict) and "batched_results" in result:
    for req_id, entries in result["batched_results"].items():
        print(f"Request {req_id} produced {len(entries)} artifacts")
else:
    print("Pipeline completed; check output/ for generated images")
```
```

**Memory management:**

- Use small resolutions (128x128 or 256x256) on constrained devices
- Enable CPU offloading flags to reduce native memory pressure
- Always use `preferSystemDownloader = true` for model downloads
- Monitor with `MemoryMetrics` to avoid OOM

See [StableDiffusionActivity example](examples.md#stablediffusionactivity) for complete implementation with error recovery and adaptive resolution.

### Best Practices

**Threading:**

- Always use background dispatchers (`Dispatchers.IO` or `Dispatchers.Default`) for model loading and inference
- Update UI only via `withContext(Dispatchers.Main)`
- Call `.close()` in `onDestroy()` to free native memory

**Memory optimization:**

- Use quantized models (Q4_K_M, Q5_K_M)
- Reduce `contextSize` (2048-4096 for constrained devices)
- Cap `numThreads` to avoid CPU oversubscription
- Monitor with `MemoryMetrics.snapshot()`

**See also:**

- [Architecture](architecture.md) for system design and flow diagrams
- [Quirks & Troubleshooting](quirks.md) for detailed JNI notes and debugging
- [Examples](examples.md) for complete working code

### API reference

Key methods:

- `SmolLM.load(modelPath: String, params: InferenceParams)` — loads a GGUF model from a path
- `SmolLM.loadFromHuggingFace(...)` — downloads and loads a model from Hugging Face
- `SmolLM.getResponse(query: String): String` — runs blocking generation and returns complete text
- `SmolLM.getResponseAsFlow(query: String): Flow<String>` — runs streaming generation
- `SmolLM.addSystemPrompt(prompt: String)` — adds system prompt to chat history
- `SmolLM.addUserMessage(message: String)` — adds user message to chat history
- `SmolLM.close()` — releases native resources
- `OcrEngine.extractText(image: ImageSource, params: OcrParams): OcrResult` — extracts text from image
- `ImageUnderstanding.process(image: ImageSource, mode: VisionMode, prompt: String?)` — processes image with vision/OCR

Refer to the `llmedge-examples` activities for complete, working code samples.
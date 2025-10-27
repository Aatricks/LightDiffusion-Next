# Quirks & Troubleshooting

This playbook highlights the most common operational quirks you may encounter while running LightDiffusion-Next and the quickest ways to resolve them.

## GPU memory headaches

| Symptom | Likely cause | Quick fixes |
| --- | --- | --- |
| `CUDA out of memory` during base diffusion | Resolution or batch too high | Drop to 512×512 or smaller, decrease batch to 1, disable HiresFix or AutoDetailer, prefer Euler/Karras samplers in **CFG++** mode |
| OOM triggered mid-way through HiRes | VRAM spikes when loading VAE/second UNet | Enable **Keep models loaded** (to avoid reloading) or run HiRes on CPU by toggling *VAE on CPU* in settings |
| Flux runs crash immediately | Missing Flux decoder or running on <16 GB VRAM | Place Flux weights in `include/Flux`, disable Flux or use SD1.5 profile on smaller cards |

Additional tips:

- Enable **VRAM budget** in Streamlit to see live usage (requires `LD_SHOW_VRAM=1`).
- In Docker, pass `--gpus all` and ensure `NVIDIA_VISIBLE_DEVICES` is not empty.
- Clear `~/.cache/torch_extensions` if Stable-Fast kernels were compiled against an older driver and now fail to load.

## Slow first runs or repeated recompilation

- Stable-Fast and SageAttention compile custom kernels on first use. This can take several minutes. Once complete, the compiled artifacts live under `~/.cache/torch_extensions` (host) or `/root/.cache/torch_extensions` (Docker). Mount this directory as a volume for faster cold starts.
- If Streamlit re-compiles every launch, ensure the container or user has write access to the cache directory and that the system clock is correct.
- Set `LD_DISABLE_SAGE_ATTENTION=1` to isolate issues related specifically to SageAttention.

## Downloader complaints about missing assets

- The startup checks look for standard filenames (e.g., `yolov8n.pt`, `taesdxl_decoder.safetensors`). Verify these live under the correct subdirectories in `include/`.
- For offline setups, drop the files manually and create empty `.ok` sentinels (e.g., `include/checkpoints/.downloads-ok`) to skip prompts.
- Hugging Face rate limits manifest as HTTP 429. Provide a token via the prompt, set `HF_TOKEN` in the environment or download manually.

## Streamlit UI quirks

- **Preview stuck on “Waiting for GPU”** – Check FastAPI logs; the batching worker may be paused. Restart the Streamlit session or run `python server.py` to inspect queue telemetry.
- **Settings reset on restart** – Ensure the process can write to `webui_settings.json`. Remove the file to revert to defaults if it becomes corrupted.
- **History thumbnails missing** – Delete the entry under `ui/history/<timestamp>`; the next render will recreate previews.

## Gradio or API automation issues

- `/api/generate` returns 500 with “No images produced”: inspect server logs for `Pipeline import error` or missing models. Ensure `pipeline.py` is importable and the working directory is the repository root.
- Jobs appear stuck: call `/api/telemetry` to inspect `pending_by_signature`. Mixed resolutions or toggles prevent batching; if running single job automation, set `LD_BATCH_WAIT_SINGLETONS=0` to avoid coalescing delays.
- Health checks: `/health` returns `{ "status": "ok" }`. If it fails, the FastAPI app likely crashed—restart and inspect `logs/server.log`.

## Docker-specific notes

- Always build with the provided `Dockerfile` to get SageAttention patches precompiled.
- Forward model assets by mounting `./include` into the container (`-v $(pwd)/include:/app/include`).
- On Windows + WSL2, ensure the WSL distro has the NVIDIA driver bridge (`wsl --status`).

## Logging & diagnostics

- Server logs live under `logs/server.log` with per-request IDs. Tail them during load testing: `tail -f logs/server.log`.
- Enable debug logging by exporting `LD_SERVER_LOGLEVEL=DEBUG` before launching Streamlit/Gradio/uvicorn.
- To inspect queue depth without hitting the API, watch the `GenerationBuffer` logs; each batch prints signature summaries.

## When all else fails

- Clear the `include/last_seed.txt` file if seed reuse behaves unexpectedly.
- Regenerate Stable-Fast kernels by deleting the cache directory and re-running with `stable_fast` enabled.
- Collect the following before opening an issue: GPU model, driver version, operating system, a copy of `logs/server.log`, hardware info from `/api/telemetry`, and reproduction steps.

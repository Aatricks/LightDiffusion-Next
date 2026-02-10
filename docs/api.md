# REST API & Automation (Quick Reference)

LightDiffusion-Next ships with a FastAPI service (`server.py`) that sits in front of the shared pipeline. It batches compatible requests, streams telemetry and exposes health probes so you can plug the system into automation workflows, bots or orchestrators.

## Common endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Lightweight readiness probe. Returns `{ "status": "ok" }` when the server is reachable. |
| `GET` | `/api/telemetry` | Queue and VRAM telemetry: batching stats, pending requests, cache state, uptime. |
| `POST` | `/api/generate` | Submit a generation job. Requests are buffered, batched when signatures match and resolved asynchronously. |

The service listens on port `7861` by default. Launch it with:

```fish
uvicorn server:app --host 0.0.0.0 --port 7861
```

## Payload schema (`/api/generate`)

```json
{
  "prompt": "string",
  "negative_prompt": "string",
  "width": 512,
  "height": 512,
  "num_images": 1,
  "batch_size": 1,
  "scheduler": "ays",
  "sampler": "dpmpp_sde_cfgpp",
  "steps": 20,
  "hires_fix": false,
  "adetailer": false,
  "enhance_prompt": false,
  "img2img_enabled": false,
  "img2img_image": null,
  "stable_fast": false,
  "reuse_seed": false,
  "flux_enabled": false,
  "realistic_model": false,
  "multiscale_enabled": true,
  "multiscale_intermittent": true,
  "multiscale_factor": 0.5,
  "multiscale_fullres_start": 10,
  "multiscale_fullres_end": 8,
  "keep_models_loaded": true,
  "enable_preview": false,
  "guidance_scale": null,
  "seed": null
}
```

Not all fields are required—only `prompt`, `width`, `height` and `num_images` are strictly necessary. Any unknown keys are ignored, making the endpoint forward-compatible with UI features.

### Response format

Successful requests return either:

```json
{ "image": "<base64-png>" }
```

or, if multiple images were requested:

```json
{ "images": ["<base64-png>", "<base64-png>"] }
```

Base64 strings represent PNG files with embedded metadata identical to the Streamlit UI output. Decode and write them to disk.

### Img2Img uploads

When `img2img_enabled` is `true`, `img2img_image` may be provided as any of the following:

- A local file path (e.g., `"tests/test.png"`)
- A data URL (e.g., `"data:image/png;base64,<...>"`)
- A raw Base64-encoded PNG string

The server will decode data URLs and raw Base64 strings and save them to the system temporary directory before processing (default max upload size: 10 MB). Keep payloads under a few megabytes to avoid HTTP timeouts.

## Telemetry shape (`/api/telemetry`)

The telemetry endpoint returns operational stats that help with autoscaling or queue dashboards. Example snippet:

```json
{
  "uptime_seconds": 1234.56,
  "pending_count": 2,
  "pending_by_signature": {
    "(False, 512, 512, True, False, False, True, True, 0.5, 10, 8, False, True, False)": 2
  },
  "pending_preview": [
    {"request_id": "a1b2c3d4", "waiting_s": 0.42, "prompt_preview": "a cinematic robot..."}
  ],
  "max_batch_size": 4,
  "batch_timeout": 0.5,
  "batches_processed": 12,
  "items_processed": 24,
  "requests_processed": 12,
  "avg_processed_wait_s": 0.31,
  "pending_avg_wait_s": 0.12,
  "memory_info": {
    "vram_allocated_mb": 5623,
    "vram_reserved_mb": 6144,
    "system_ram_mb": 12345
  },
  "loaded_models_count": 2,
  "loaded_models": ["SD15 UNet", "SD15 VAE"],
  "pipeline_import_ok": true,
  "pipeline_import_error": null
}
```

Use this data to spot batching mismatches (different signatures cannot be coalesced), monitor VRAM usage or expose metrics to Prometheus/Grafana.

## Queue tuning knobs

The queue accepts a few environment variables that influence behaviour:

| Variable | Default | Effect |
| --- | --- | --- |
| `LD_MAX_BATCH_SIZE` | `4` | Maximum items processed together when signatures match. |
| `LD_BATCH_TIMEOUT` | `0.5` | Seconds to wait before flushing a batch. |
| `LD_BATCH_WAIT_SINGLETONS` | `0` | If `1`, single jobs wait the timeout hoping for companions. Set to `0` to process singletons immediately. |
| `LD_SERVER_LOGLEVEL` | `DEBUG` | Logging verbosity for `logs/server.log`. |

## Deploying behind a reverse proxy

When hosting remotely:

- Front the FastAPI app with Nginx/Caddy and increase client body size if you accept Img2Img uploads.
- Expose `/health` for liveness checks and `/api/telemetry` for readiness/autoscaling gates.
- Mount `./include`, `./output` and `~/.cache/torch_extensions` as volumes so workers share models, outputs and compiled kernels.

## Testing the service quickly

```fish
# Send a simple generation job
curl -X POST http://localhost:7861/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "painted nebula over distant mountains", "width": 512, "height": 512, "num_images": 1}' \
  | jq -r '.image' | base64 -d > nebula.png

# Inspect queue state
curl http://localhost:7861/api/telemetry | jq
```

That’s it! Check the [Troubleshooting guide](quirks.md) if the service reports missing models or the queue appears stalled.

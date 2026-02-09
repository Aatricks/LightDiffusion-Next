from __future__ import annotations

import base64
import glob
import os
import io
from src.AutoEncoders.taesd import decode_latents_to_images

# Ensure we can import pipeline from this repo
import sys
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.Device.ModelCache import get_model_cache
from src.Device.ModelCache import get_model_cache
from src.Core.Models.ModelFactory import list_available_models, list_available_controlnets

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Logging setup
import asyncio
import functools
import logging
import uuid
from logging.handlers import RotatingFileHandler


# Create a module-level logger with rotating file handler and request-id support
class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple utility
        if not hasattr(record, "rid"):
            record.rid = "-"
        return True


def _setup_logger() -> logging.Logger:
    os.makedirs("./logs", exist_ok=True)
    logger = logging.getLogger("lightdiffusion.server")
    if logger.handlers:
        return logger

    level_name = os.getenv("LD_SERVER_LOGLEVEL", "DEBUG").upper()
    try:
        level = getattr(logging, level_name, logging.DEBUG)
    except Exception:  # pragma: no cover
        level = logging.DEBUG
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | rid=%(rid)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        filename=os.path.join("./logs", "server.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_RequestIdFilter())
    logger.addHandler(file_handler)

    # Also log to stderr for interactive runs; avoid duplicate handlers if uvicorn config already propagates
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(_RequestIdFilter())
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


logger = _setup_logger()
logger.debug("server module loaded; cwd=%s", os.getcwd())

# Record server start time for telemetry
SERVER_START_TS = time.time()

try:
    # Import app_instance to control preview behavior during generation
    from src.user import app_instance as _app_instance
    from src.user.pipeline import pipeline
except Exception as e:
    # Defer import error to runtime response for clarity
    pipeline = None  # type: ignore
    _pipeline_import_error = e
    logger.exception("Failed to import pipeline: %s", e)
else:
    _pipeline_import_error = None
    logger.info("Pipeline and app_instance imported successfully")


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 512
    height: int = 512
    num_images: int = 1
    batch_size: int = 1
    scheduler: str = "ays"
    sampler: str = "dpmpp_sde_cfgpp"
    steps: int = 20
    hiresfix: bool = False
    adetailer: bool = False
    enhance_prompt: bool = False
    img2img_mode: bool = False
    img2img_image: Optional[str] = None
    img2img_denoise: float = 0.75  # Denoising strength: 0=keep original, 1=full generation
    stable_fast: bool = False
    reuse_seed: bool = False
    realistic_model: bool = False
    enable_multiscale: bool = False
    multiscale_preset: Optional[str] = "balanced"
    multiscale_intermittent: bool = True
    multiscale_factor: float = 0.5
    multiscale_fullres_start: int = 10
    multiscale_fullres_end: int = 8
    keep_models_loaded: bool = True
    enable_preview: bool = False
    # CFG-free sampling parameters
    cfg_free_enabled: bool = False
    cfg_free_start_percent: float = 70.0
    # Token Merging parameters
    tome_enabled: bool = False
    tome_ratio: float = 0.5
    tome_max_downsample: int = 1
    # Advanced CFG optimization parameters (batched_cfg enabled by default for 8% speedup)
    batched_cfg: bool = True
    dynamic_cfg_rescaling: bool = False
    dynamic_cfg_method: str = "variance"
    dynamic_cfg_percentile: float = 95.0
    dynamic_cfg_target_scale: float = 7.0
    adaptive_noise_enabled: bool = False
    adaptive_noise_method: str = "complexity"
    # Guidance
    cfg_scale: float = 7.0
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None  # If provided >=0 we will reuse it
    
    # Model Selection
    model_path: Optional[str] = None
    refiner_model_path: Optional[str] = None
    refiner_switch_step: Optional[int] = None

    # ControlNet
    controlnet_enabled: bool = False
    controlnet_model: Optional[str] = None
    controlnet_strength: float = 1.0
    controlnet_type: str = "canny"


app = FastAPI(title="LightDiffusion Server", version="1.0.0")


@app.get("/api/controlnets")
async def get_controlnets():
    """List available ControlNet models."""
    try:
        models = list_available_controlnets()
        return {"models": models}
    except Exception as e:
        logger.exception("Failed to list controlnets")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Capture event loop reference and start background worker."""
    global _main_event_loop
    _main_event_loop = asyncio.get_running_loop()
    await _generation_buffer.start()
    logger.info("Server startup complete, event loop captured for preview broadcasting")


# Batching buffer -----------------------------------------------------------
LD_MAX_BATCH_SIZE = int(os.getenv("LD_MAX_BATCH_SIZE", "4"))
LD_BATCH_TIMEOUT = float(os.getenv("LD_BATCH_TIMEOUT", "0.5"))
# If set to true (1/true), the worker will wait the coalescing timeout when
# there is a single candidate in a chosen group; otherwise singletons are
# processed immediately. Default is to process singletons immediately to
# favor throughput and avoid perceived "stuck" behavior.
LD_BATCH_WAIT_SINGLETONS = os.getenv("LD_BATCH_WAIT_SINGLETONS", "0").lower() in ("1", "true", "yes")


class PendingRequest:
    def __init__(self, req: GenerateRequest, request_id: str):
        self.req = req
        self.request_id = request_id
        self.arrival = time.time()
        self.future: asyncio.Future = asyncio.get_running_loop().create_future()


class GenerationBuffer:
    def __init__(self):
        self._pending: List[PendingRequest] = []
        self._lock = asyncio.Lock()
        self._new_request = asyncio.Event()
        
        # Prefetching state
        self._prefetch_lock = asyncio.Lock()
        self._prefetch_task: Optional[asyncio.Task] = None
        self._current_prefetch_path: Optional[str] = None

        # Statistics
        self._items_processed = 0
        self._batches_processed = 0
        self._requests_processed = 0
        self._cumulative_wait_time = 0.0
        self._last_batch_ts = 0.0
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the background worker task."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("GenerationBuffer worker task started")

    async def enqueue(self, pending: PendingRequest) -> dict:
        """Add a request to the queue and wait for completion."""
        async with self._lock:
            self._pending.append(pending)
            self._new_request.set()
        
        # Wait for the worker to process this request
        return await pending.future

    async def _look_ahead_and_prefetch(self, current_batch_signature: tuple):
        """Analyze remaining queue and pre-load the next model if different."""
        from src.user.pipeline import resolve_checkpoint_path
        
        async with self._lock:
            if not self._pending:
                return

            # Find the next group that has a different signature
            next_req = None
            for p in self._pending:
                sig = self._signature_for(p.req)
                if sig != current_batch_signature:
                    next_req = p.req
                    break
            
            if not next_req:
                return

            # Resolve the path for the next model
            target_path = resolve_checkpoint_path(
                realistic_model=next_req.realistic_model
            )
            
        # Perform prefetch outside the queue lock
        async with self._prefetch_lock:
            # Skip if already prefetched or currently prefetching the same path
            if target_path == self._current_prefetch_path:
                return
            
            # Cancel existing prefetch if it's for a different model
            if self._prefetch_task and not self._prefetch_task.done():
                self._prefetch_task.cancel()
                try:
                    await self._prefetch_task
                except asyncio.CancelledError:
                    pass

            self._current_prefetch_path = target_path
            
            async def prefetch_task():
                try:
                    logger.info("Prefetcher: Starting background load of %s", target_path)
                    # Load to CPU RAM using the optimized util
                    sd = await asyncio.to_thread(util.load_torch_file, target_path)
                    # Store in cache
                    get_model_cache().set_prefetched_model(target_path, sd)
                    logger.info("Prefetcher: Successfully pre-loaded %s into RAM", target_path)
                except Exception as e:
                    logger.warning("Prefetcher: Failed to pre-load %s: %s", target_path, e)
                finally:
                    self._current_prefetch_path = None

            self._prefetch_task = asyncio.create_task(prefetch_task())

    def _signature_for(self, req: GenerateRequest) -> tuple:
        # Grouping signature - requests must match these to be batched
        
        # Detect model type to determine if refiner is relevant
        from src.Core.Models.ModelFactory import detect_model_type
        is_sdxl = (detect_model_type(req.model_path) == "SDXL")
        
        return (
            str(req.model_path),  # Model must match
            bool(req.realistic_model),
            int(req.width),
            int(req.height),
            bool(req.stable_fast),
            bool(req.img2img_mode),
            str(req.scheduler),
            str(req.sampler),
            int(req.steps),
            # Treat multiscale options as batch-level — mixing them may
            # change the sampling schedule and therefore cannot be
            # safely combined into a single forward pass.
            bool(req.enable_multiscale),
            bool(req.multiscale_intermittent),
            float(req.multiscale_factor),
            int(req.multiscale_fullres_start),
            int(req.multiscale_fullres_end),
            # VRAM retention flags are also batch level
            bool(req.keep_models_loaded),
            # ControlNet (must match)
            bool(req.controlnet_enabled),
            str(req.controlnet_model),
            float(req.controlnet_strength),
            str(req.controlnet_type),
            # Refiner (must match only if it will actually be used)
            str(req.refiner_model_path) if is_sdxl else "",
            (int(req.refiner_switch_step) if req.refiner_switch_step is not None else -1) if is_sdxl else -1,
            # Note: hires_fix, adetailer, and enable_preview remain intentionally 
            # NOT part of this signature because they are executed per-sample
            # (or as side-effects) after or during a shared forward pass.
        )

    async def _worker(self):
        logger.info("Batching worker started; max_batch=%s timeout=%s", LD_MAX_BATCH_SIZE, LD_BATCH_TIMEOUT)
        while True:
            await self._new_request.wait()
            # Small throttle to coalesce multiple arrivals
            await asyncio.sleep(0)

            async with self._lock:
                if not self._pending:
                    self._new_request.clear()
                    continue
                # Group pending requests by signature
                groups: Dict[tuple, List[PendingRequest]] = {}
                for p in self._pending:
                    sig = self._signature_for(p.req)
                    groups.setdefault(sig, []).append(p)

                # Choose the group with the oldest request
                chosen_sig = None
                oldest_time = float("inf")
                for sig, arr in groups.items():
                    if arr and arr[0].arrival < oldest_time:
                        chosen_sig = sig
                        oldest_time = arr[0].arrival

                if chosen_sig is None:
                    self._new_request.clear()
                    continue

                candidates = groups[chosen_sig]
                # Sort by arrival time (oldest first)
                candidates.sort(key=lambda x: x.arrival)

                # Debug: show group sizes for observability
                try:
                    group_summary = {str(sig): len(arr) for sig, arr in groups.items()}
                    logger.debug("Batch worker: pending groups=%s chosen_sig=%s group_size=%d oldest_arrival=%.3f",
                                 group_summary, str(chosen_sig), len(candidates), candidates[0].arrival if candidates else 0.0)
                except Exception:
                    pass

                # Determine whether to wait for coalescing when there's only a
                # single candidate. This is controlled by LD_BATCH_WAIT_SINGLETONS
                # so operators can toggle the behavior at runtime via env.
                if len(candidates) == 1:
                    age = time.time() - candidates[0].arrival
                    if LD_BATCH_WAIT_SINGLETONS and age < LD_BATCH_TIMEOUT:
                        # Old behavior: wait a bit for more arrivals before
                        # processing a singleton so we can form a larger batch.
                        logger.debug("Singleton group for signature %s is too new (age=%.3fs < timeout=%.3fs). Sleeping to coalesce.", str(chosen_sig), age, LD_BATCH_TIMEOUT)
                        self._new_request.clear()
                        await asyncio.sleep(LD_BATCH_TIMEOUT)
                        continue
                    else:
                        # Eager processing path (default): process singletons
                        # immediately to avoid perceived "stuck" behavior.
                        logger.debug("Processing singleton group for signature %s immediately (age=%.3fs). LD_BATCH_WAIT_SINGLETONS=%s",
                                     str(chosen_sig), age, LD_BATCH_WAIT_SINGLETONS)

                # Pick up to max batch size
                to_process = candidates[:LD_MAX_BATCH_SIZE]
                # Remove selected items from pending list
                for p in to_process:
                    try:
                        self._pending.remove(p)
                    except ValueError:
                        pass
                if not self._pending:
                    self._new_request.clear()

            # Trigger prefetching for the NEXT group while we process this one
            await self._look_ahead_and_prefetch(chosen_sig)

            # Process the selected group outside the lock
            try:
                try:
                    logger.debug("Processing group chosen_sig=%s items=%d request_ids=%s", str(chosen_sig), len(to_process), [p.request_id for p in to_process])
                except Exception:
                    pass
                await self._process_group(to_process)
                # Update lightweight metrics only on success
                try:
                    now_ts = time.time()
                    self._batches_processed += 1
                    self._items_processed += sum(
                        max(1, p.req.num_images) for p in to_process
                    )
                    self._requests_processed += len(to_process)
                    # Update cumulative wait time per-request
                    wait_total = sum(now_ts - p.arrival for p in to_process)
                    self._cumulative_wait_time += wait_total
                    self._last_batch_ts = now_ts
                except Exception:
                    # Metrics must never crash the worker loop
                    logger.exception("Failed updating batch metrics")
            except Exception as e:
                logger.exception("Batch processing failed: %s", e)

    async def _process_group(self, items: List[PendingRequest]):
        # All items share a signature as enforced by the grouping logic.
        if not items:
            return

        # Build flat per-sample prompt list and per-sample info
        prompts: List[str] = []
        per_sample_info: List[dict] = []
        for p in items:
            for k in range(max(1, p.req.num_images)):
                prompts.append(p.req.prompt)
                per_sample_info.append(
                    {
                        "request_id": p.request_id,
                        "filename_prefix": f"LD-REQ-{p.request_id}",
                        "seed": p.req.seed if (p.req.seed is not None and p.req.seed >= 0) else None,
                        "hires_fix": bool(p.req.hiresfix),
                        "adetailer": bool(p.req.adetailer),
                    }
                )

        # Prepare pipeline kwargs based on the shared signature (take from first)
        first_req = items[0].req
        pipeline_kwargs = dict(
            prompt=prompts,
            w=first_req.width,
            h=first_req.height,
            number=len(prompts),
            batch=first_req.batch_size,
            scheduler=first_req.scheduler,
            sampler=first_req.sampler,
            steps=first_req.steps,
            cfg_scale=first_req.cfg_scale if first_req.guidance_scale is None else first_req.guidance_scale,
            enhance_prompt=first_req.enhance_prompt,
            img2img=first_req.img2img_mode,
            img2img_denoise=first_req.img2img_denoise,
            stable_fast=first_req.stable_fast,
            reuse_seed=first_req.reuse_seed,
            autohdr=True,
            realistic_model=first_req.realistic_model,
            model_path=first_req.model_path,
            refiner_model_path=first_req.refiner_model_path,
            refiner_switch_step=first_req.refiner_switch_step,
            negative_prompt=[p.req.negative_prompt or "" for p in items for _ in range(max(1, p.req.num_images))],
            multiscale_preset=first_req.multiscale_preset,
            enable_multiscale=first_req.enable_multiscale,
            multiscale_factor=first_req.multiscale_factor,
            multiscale_fullres_start=first_req.multiscale_fullres_start,
            multiscale_fullres_end=first_req.multiscale_fullres_end,
            multiscale_intermittent_fullres=first_req.multiscale_intermittent,
            img2img_image=first_req.img2img_image,
            per_sample_info=per_sample_info,
            cfg_free_enabled=first_req.cfg_free_enabled,
            cfg_free_start_percent=first_req.cfg_free_start_percent,
            tome_enabled=first_req.tome_enabled,
            tome_ratio=first_req.tome_ratio,
            tome_max_downsample=first_req.tome_max_downsample,
            # Advanced CFG optimizations (batched_cfg always enabled)
            batched_cfg=first_req.batched_cfg,
            dynamic_cfg_rescaling=first_req.dynamic_cfg_rescaling,
            dynamic_cfg_method=first_req.dynamic_cfg_method,
            dynamic_cfg_percentile=first_req.dynamic_cfg_percentile,
            dynamic_cfg_target_scale=first_req.dynamic_cfg_target_scale,
            adaptive_noise_enabled=first_req.adaptive_noise_enabled,
            adaptive_noise_method=first_req.adaptive_noise_method,
            # ControlNet
            controlnet_model=first_req.controlnet_model if first_req.controlnet_enabled else None,
            controlnet_strength=first_req.controlnet_strength,
            controlnet_type=first_req.controlnet_type,
            # Add callback for WebSocket preview broadcasting
            callback=make_server_callback(first_req.steps),
        )

        # Toggle preview state for the duration of the pipeline call
        prev_preview_state = None
        prev_keep_models_loaded = None
        try:
            try:
                prev_preview_state = _app_instance.app.previewer_var.get()
                _app_instance.app.previewer_var.set(bool(first_req.enable_preview))
            except Exception:
                prev_preview_state = None

            # Respect per-group model cache directive: toggle "keep loaded"
            # so the sampling pipeline sees the requested caching behavior.
            try:
                model_cache = get_model_cache()
                prev_keep_models_loaded = model_cache.get_keep_models_loaded()
                model_cache.set_keep_models_loaded(bool(first_req.keep_models_loaded))
            except Exception:
                prev_keep_models_loaded = None

            loop = asyncio.get_running_loop()
            saved_map: Dict[str, List[dict]] = {}
            
            # Run pipeline in thread pool to avoid blocking the event loop
            func = functools.partial(pipeline, **pipeline_kwargs)
            result = await loop.run_in_executor(None, func)

            # Expect pipeline to return a mapping under 'batched_results' when
            # run in batched mode; otherwise fall back to scanning.
            if isinstance(result, dict) and "batched_results" in result:
                saved_map = result["batched_results"]
            else:
                # Fallback: scan filesystem for files created since group start
                files = _find_images_since(time.time() - 60)
                for f in files:
                    # naive grouping by filename prefix (LD-REQ-<rid>)
                    name = os.path.basename(f)
                    for p in items:
                        if f"LD-REQ-{p.request_id}" in name:
                            saved_map.setdefault(p.request_id, []).append({
                                "filename": name,
                                "subfolder": os.path.relpath(os.path.dirname(f), "./output"),
                            })

            # For each pending item, collect its images and set future result
            for p in items:
                imgs = saved_map.get(p.request_id, [])
                # Filter and select the first N images requested
                selected = imgs[: max(1, p.req.num_images)]
                if not selected:
                    p.future.set_exception(HTTPException(status_code=500, detail="No images produced"))
                    continue
                # Build full paths and encode
                b64_list = []
                for entry in selected:
                    if isinstance(entry, list):
                        # Safeguard against nested lists if any processor still returns them
                        entry = entry[0] if entry else {}
                    
                    if not isinstance(entry, dict):
                        continue
                        
                    filename = entry.get("filename", "")
                    path = os.path.join("./output", entry.get("subfolder", ""), filename)
                    try:
                        b64_data = _encode_png_to_base64(path)
                        mime_type = "image/png"
                        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
                            mime_type = "image/jpeg"
                        elif filename.lower().endswith(".webp"):
                            mime_type = "image/webp"
                        
                        b64_list.append(f"data:{mime_type};base64,{b64_data}")
                    except Exception as e:
                        logger.exception("Failed to read image for request %s: %s", p.request_id, e)

                if len(b64_list) == 0:
                    p.future.set_exception(HTTPException(status_code=500, detail="Failed to read generated images"))
                elif len(b64_list) == 1:
                    p.future.set_result({"image": b64_list[0]})
                else:
                    p.future.set_result({"images": b64_list})
        finally:
            try:
                if prev_preview_state is not None:
                    _app_instance.app.previewer_var.set(prev_preview_state)
            except Exception:
                pass
            try:
                # Restore previous model cache keep-loaded setting if we
                # changed it above.
                if prev_keep_models_loaded is not None:
                    try:
                        model_cache = get_model_cache()
                        model_cache.set_keep_models_loaded(bool(prev_keep_models_loaded))
                    except Exception:
                        pass
            except Exception:
                pass


# Instantiate the buffer and start it on startup
_generation_buffer = GenerationBuffer()


@app.on_event("startup")
async def _start_buffer():
    await _generation_buffer.start()



@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/telemetry")
async def telemetry() -> Dict[str, Any]:
    """Return basic server and batching buffer telemetry.

    Fields:
    - uptime_seconds
    - pending_count
    - pending_by_signature (human-readable)
    - pending_preview (list of small pending request summaries)
    - worker_running
    - max_batch_size, batch_timeout
    - batches_processed, items_processed, last_batch_time
    - pipeline_import_ok and pipeline_import_error
    """
    rid = uuid.uuid4().hex[:8]
    log = logging.LoggerAdapter(logger, {"rid": rid})
    log.debug("telemetry requested")

    now = time.time()
    uptime = now - SERVER_START_TS

    # Build a small snapshot of queue state under the buffer lock
    async with _generation_buffer._lock:
        pending_count = len(_generation_buffer._pending)
        # Group pending requests by signature for visibility
        sig_counts: Dict[str, int] = {}
        pending_preview: List[Dict[str, Any]] = []
        for p in _generation_buffer._pending:
            try:
                sig = _generation_buffer._signature_for(p.req)
                sig_key = str(sig)
            except Exception:
                sig_key = "<unknown>"
            sig_counts[sig_key] = sig_counts.get(sig_key, 0) + 1
            # Keep preview small to avoid large payloads
            preview = {
                "request_id": p.request_id,
                "waiting_s": round(now - p.arrival, 3),
                "prompt_preview": (p.req.prompt[:120] + "…") if (p.req.prompt and len(p.req.prompt) > 120) else (p.req.prompt or ""),
            }
            pending_preview.append(preview)

        batches_processed = _generation_buffer._batches_processed
        items_processed = _generation_buffer._items_processed
        last_batch_ts = _generation_buffer._last_batch_ts

    worker_running = (
        _generation_buffer._worker_task is not None
        and (not _generation_buffer._worker_task.done())
    )

    # Compute average wait times
    requests_processed = _generation_buffer._requests_processed
    cumulative_wait = _generation_buffer._cumulative_wait_time
    avg_processed_wait_s = (
        (cumulative_wait / requests_processed) if requests_processed > 0 else None
    )
    # Pending average wait (current queue)
    pending_avg_wait_s = (
        (sum(now - p.arrival for p in _generation_buffer._pending) / pending_count)
        if pending_count > 0
        else 0.0
    )

    # Model cache telemetry (memory and loaded models)
    memory_info_error = None
    try:
        model_cache = get_model_cache()
        memory_info = model_cache.get_memory_info()
        loaded_raw = model_cache.get_cached_sampling_models()
        loaded_models = []
        for m in loaded_raw:
            try:
                name = getattr(m, "name", None) or getattr(m, "__class__", type(m)).__name__
            except Exception:
                name = str(type(m))
            loaded_models.append(name)
        loaded_models_count = len(loaded_models)
    except Exception as e:
        # Don't fail telemetry if model cache query fails. Capture a short
        # error string so callers can display a hint without exposing full
        # stack traces. Device-side CUDA asserts can leave the device in an
        # unusable state and will cause subsequent CUDA queries to fail; we
        # surface a concise message here instead of crashing the endpoint.
        try:
            # Prefer a succinct message
            memory_info_error = str(e)
        except Exception:
            memory_info_error = "unknown"
        logger.exception("Failed to fetch model cache telemetry: %s", memory_info_error)
        memory_info = None
        loaded_models = []
        loaded_models_count = 0

    return {
        "uptime_seconds": round(uptime, 3),
        "server_start_ts": SERVER_START_TS,
        "pending_count": pending_count,
        "pending_by_signature": sig_counts,
        "pending_preview": pending_preview[:20],
        "worker_running": worker_running,
        "max_batch_size": LD_MAX_BATCH_SIZE,
        "batch_timeout": LD_BATCH_TIMEOUT,
    "batches_processed": batches_processed,
    "items_processed": items_processed,
    "requests_processed": requests_processed,
    "last_batch_time": last_batch_ts,
    "avg_processed_wait_s": avg_processed_wait_s,
    "pending_avg_wait_s": pending_avg_wait_s,
    "memory_info": memory_info,
    "loaded_models_count": loaded_models_count,
    "loaded_models": loaded_models,
        "pipeline_import_ok": pipeline is not None,
        "pipeline_import_error": str(_pipeline_import_error) if _pipeline_import_error is not None else None,
    }


def _encode_png_to_base64(path: str) -> str:
    # Retry a few times in case the file is still being finalized on disk
    last_err: Optional[Exception] = None
    for attempt in range(20):  # up to ~2s total
        try:
            with open(path, "rb") as f:
                data = f.read()
                if attempt > 0:
                    logger.debug("Read image after %d retries: %s", attempt, path)
                return base64.b64encode(data).decode("utf-8")
        except Exception as e:
            last_err = e
            time.sleep(0.1)
    # One last attempt or raise detailed error
    try:
        with open(path, "rb") as f:
            logger.debug("Final attempt succeeded reading: %s", path)
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error("Failed to read generated image %s: %s", path, e if e else last_err)
        raise HTTPException(status_code=500, detail=f"Failed to read generated image: {e if e else last_err}")


def _list_existing_images() -> List[str]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join("./output", "**", ext), recursive=True))
    logger.debug("Found %d existing images", len(files))
    return files


def _find_images_since(start_ts: float) -> List[str]:
    """Return images whose mtime is at or after start_ts (with small grace)."""
    grace = 0.25
    files = _list_existing_images()
    recent = [p for p in files if os.path.getmtime(p) >= (start_ts - grace)]
    recent.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    logger.debug("%d images modified since %.3f", len(recent), start_ts)
    return recent

# WebSocket preview endpoint for real-time streaming
_preview_clients: List[WebSocket] = []
_main_event_loop: Optional[asyncio.AbstractEventLoop] = None


def sync_broadcast_preview(
    step: int,
    total_steps: int,
    images: Optional[List[str]] = None,
    message_type: str = "preview"
):
    """Synchronous wrapper to broadcast preview from pipeline thread.
    
    This function can be called from the pipeline callback running in a
    thread pool executor. It schedules the async broadcast on the main
    event loop.
    """

    global _main_event_loop
    
    if not _preview_clients:
        if step % 10 == 0:
            logger.debug("No preview clients connected, skipping broadcast")
        return
    
    if _main_event_loop is None:
        logger.error("Main event loop is None! Cannot broadcast preview.")
        return
    
    try:
        if step % 5 == 0 or step == total_steps - 1:
            logger.info(f"Broadcasting preview step {step}/{total_steps}")
            
        future = asyncio.run_coroutine_threadsafe(
            broadcast_preview(step, total_steps, images, message_type),
            _main_event_loop
        )
        # Wait for broadcast to complete to ensure ordering
        try:
            future.result(timeout=0.5)
        except Exception:
            pass # Don't block generation on slow clients
    except Exception as e:
        logger.error(f"Preview broadcast failed: {e}")
        pass  # Don't let preview errors affect generation


def make_server_callback(total_steps: int):
    """Create a pipeline callback that broadcasts progress via WebSocket.
    
    Args:
        total_steps: Total number of sampling steps
        
    Returns:
        Callback function compatible with pipeline
    """
    def callback(args):
        # Extract step info from args dict
        step = args.get("i", 0)
        curr_total_steps = args.get("total_steps", total_steps)
        
        # Only process images on broadcast steps to save compute
        # Broadcast every 5 steps or last step
        is_broadcast_step = (step % 5 == 0) or (step == curr_total_steps - 1)
        
        images_b64 = None
        if is_broadcast_step:
            try:
                # prefer denoised, fallback to x ONLY if early step
                latents_tensor = args.get("denoised")
                if latents_tensor is None and step < 5:
                    latents_tensor = args.get("x")
                
                if latents_tensor is not None:
                    # Detect flux from shape (Flux has 16 or 32 channels)
                    # This is a heuristic, ideal would be to pass it in args
                    is_flux = (latents_tensor.shape[1] == 16 or latents_tensor.shape[1] == 32)
                    
                    pil_images = decode_latents_to_images(latents_tensor, flux=is_flux)
                    
                    images_b64 = []
                    for img in pil_images:
                        buffered = io.BytesIO()
                        img.save(buffered, format="JPEG", quality=70)
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        images_b64.append(f"data:image/jpeg;base64,{img_str}")
            except Exception as e:
                logger.error(f"Preview generation failed: {e}")
                pass

        # Broadcast progress update with images
        sync_broadcast_preview(step, curr_total_steps, images=images_b64, message_type="preview" if images_b64 else "progress")
    
    return callback


@app.websocket("/ws/preview")
async def websocket_preview(websocket: WebSocket):
    """WebSocket endpoint for real-time preview streaming.
    
    Clients receive JSON messages with:
    - type: "preview" | "progress" | "complete" | "error"
    - step: Current step number
    - total_steps: Total number of steps
    - timestamp: Unix timestamp
    - images: List of base64 encoded preview images (for "preview" type)
    """
    await websocket.accept()
    _preview_clients.append(websocket)
    logger.info("WebSocket client connected to /ws/preview (total: %d)", len(_preview_clients))
    
    try:
        # Keep connection alive and listen for close
        while True:
            try:
                # Wait for any message (ping/pong or close)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Echo back to confirm alive
                await websocket.send_json({"type": "pong", "timestamp": time.time()})
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_json({"type": "ping", "timestamp": time.time()})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("WebSocket connection error: %s", e)
    finally:
        if websocket in _preview_clients:
            _preview_clients.remove(websocket)
        logger.info("WebSocket client disconnected (remaining: %d)", len(_preview_clients))


async def broadcast_preview(
    step: int,
    total_steps: int,
    images: Optional[List[str]] = None,
    message_type: str = "preview"
):
    """Broadcast preview update to all connected WebSocket clients.
    
    Args:
        step: Current step number
        total_steps: Total number of steps
        images: Optional list of base64-encoded images
        message_type: Type of message (preview, progress, complete, error)
    """
    if not _preview_clients:
        return
    
    payload = {
        "type": message_type,
        "step": step,
        "total_steps": total_steps,
        "timestamp": time.time(),
    }
    
    if images:
        payload["images"] = images
    
    # Send to all clients, removing any that fail
    disconnected = []
    for client in _preview_clients:
        try:
            await client.send_json(payload)
        except Exception:
            disconnected.append(client)
    
    for client in disconnected:
        if client in _preview_clients:
            _preview_clients.remove(client)


@app.post("/api/generate")
async def generate(req: GenerateRequest) -> Dict[str, Any]:
    rid = uuid.uuid4().hex[:8]
    log = logging.LoggerAdapter(logger, {"rid": rid})
    log.info("/api/generate called")

    # Validate pipeline import
    global pipeline, _pipeline_import_error
    if pipeline is None:
        log.error("Pipeline import error: %s", _pipeline_import_error)
        raise HTTPException(status_code=500, detail=f"Pipeline import error: {_pipeline_import_error}")

    # Optionally honor requested seed by writing include/last_seed.txt and enabling reuse
    reuse_seed = req.reuse_seed
    if req.seed is not None and req.seed >= 0:
        os.makedirs("./include", exist_ok=True)
        with open(os.path.join("./include", "last_seed.txt"), "w", encoding="utf-8") as f:
            f.write(str(int(req.seed)))
        reuse_seed = True

    # For buffered execution we pass request data into the queue; the
    # background worker will control how the prompt and img2img path are
    # consumed when invoking the pipeline.

    # Log request summary (avoid dumping huge strings)
    def _truncate(s: Optional[str], n: int = 200) -> str:
        if not s:
            return ""
        return s if len(s) <= n else s[:n] + "…"

    log.debug(
        "Request: w=%s h=%s num_images=%s batch=%s scheduler=%s sampler=%s steps=%s hiresfix=%s adetailer=%s enhance=%s img2img=%s stable_fast=%s reuse_seed=%s realistic=%s multiscale=%s intermittent=%s factor=%s fullres=[%s,%s] keep_models_loaded=%s enable_preview=%s prompt='%s' neg='%s' img2img_image_present=%s",
        req.width,
        req.height,
        req.num_images,
        req.batch_size,
        req.scheduler,
        req.sampler,
        req.steps,
        req.hiresfix,
        req.adetailer,
        req.enhance_prompt,
        req.img2img_mode,
        req.stable_fast,
        reuse_seed,
        req.realistic_model,
        req.enable_multiscale,
        req.multiscale_intermittent,
        req.multiscale_factor,
        req.multiscale_fullres_start,
        req.multiscale_fullres_end,
        req.keep_models_loaded,
        req.enable_preview,
        _truncate(req.prompt, 200),
        _truncate(req.negative_prompt or "", 200),
        bool(req.img2img_image),
    )

    # Enqueue the request for batched processing. The background worker will
    # perform the actual pipeline invocation and will restore any preview
    # state toggles after generation completes.
    # Enqueue the request for batched processing. The background worker will
    # perform the actual pipeline invocation and will restore any preview
    # state toggles after generation completes.
    pending = PendingRequest(req, rid)
    result = await _generation_buffer.enqueue(pending)

    # Return the result produced by the background worker (dict with
    # either 'image' or 'images').
    return result

    # Background worker will have returned the final result for this request.


@app.get("/api/models")
async def list_models() -> List[Dict[str, Any]]:
    """List available models with type detection and capabilities."""
    try:
        from src.Core.Models.ModelFactory import list_available_models, detect_model_type, create_model
        
        models = list_available_models(return_mapping=True)
        results = []
        for name, path in models:
            try:
                # We create a temporary instance to get capabilities without full loading
                # detect_model_type is fast
                mtype = detect_model_type(path)
                
                # Get capabilities from the model class
                # ModelFactory.create_model returns an uninitialized instance
                model_instance = create_model(model_path=path, model_type=mtype)
                caps = model_instance.capabilities
                
                # Convert capabilities dataclass to dict
                cap_dict = {
                    "supports_hires_fix": caps.supports_hires_fix,
                    "supports_img2img": caps.supports_img2img,
                    "supports_controlnet": caps.supports_controlnet,
                    "supports_inpainting": caps.supports_inpainting,
                    "supports_stable_fast": caps.supports_stable_fast,
                    "supports_deepcache": caps.supports_deepcache,
                    "supports_tome": caps.supports_tome,
                    "preferred_resolution": caps.preferred_resolution,
                }
                
                results.append({
                    "name": name, 
                    "path": path, 
                    "type": mtype,
                    "capabilities": cap_dict
                })
            except Exception as e:
                logger.warning(f"Failed to detect type/caps for {name}: {e}")
                results.append({
                    "name": name, 
                    "path": path, 
                    "type": "SD15",
                    "capabilities": {}
                })
        return results
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/interrupt")
async def interrupt_generation():
    """Interrupt current generation."""
    # Logic to interrupt generation
    # We need to signal the pipeline to stop
    # The pipeline checks app_instance.app.interrupt_flag
    
    if _app_instance and hasattr(_app_instance, "app") and _app_instance.app:
        _app_instance.app.request_interrupt()
        logger.info("Interrupt requested via API")
        return {"status": "interrupted"}
    else:
        logger.error("Cannot interrupt: app_instance not available")
        raise HTTPException(status_code=503, detail="App instance not available")





# Mount frontend if build exists
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    logger.info(f"Serving frontend from {frontend_dist}")
else:
    logger.warning(f"Frontend build not found at {frontend_dist}. Run 'npm run build' in frontend directory.")


if __name__ == "__main__":
    import uvicorn
    import argparse
    import subprocess
    import signal

    parser = argparse.ArgumentParser(description="LightDiffusion Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7861, help="Port to bind to")
    parser.add_argument("--frontend", action="store_true", help="Launch the frontend development server")
    args = parser.parse_args()

    frontend_proc = None
    if args.frontend:
        frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
        if os.path.exists(frontend_dir):
            logger.info("Launching frontend development server...")
            try:
                # Use shell=True for windows to find npm
                frontend_proc = subprocess.Popen(
                    ["npm", "run", "dev"],
                    cwd=frontend_dir,
                    shell=True
                )
                logger.info("Frontend development server launched")
            except Exception as e:
                logger.error(f"Failed to launch frontend: {e}")
        else:
            logger.warning(f"Frontend directory not found at {frontend_dir}")

    try:
        uvicorn.run("server:app", host=args.host, port=args.port, reload=False, ws="websockets")
    finally:
        if frontend_proc:
            logger.info("Shutting down frontend development server...")
            if sys.platform == "win32":
                # On Windows, we need to kill the process tree because shell=True creates a cmd.exe wrapper
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(frontend_proc.pid)], capture_output=True)
            else:
                frontend_proc.terminate()
            logger.info("Frontend development server shut down")

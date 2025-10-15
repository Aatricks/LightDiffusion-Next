from __future__ import annotations

import base64
import glob
import os
import time
from typing import Any, Dict, List, Optional
from src.Device.ModelCache import get_model_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure we can import pipeline from this repo
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Logging setup
import logging
from logging.handlers import RotatingFileHandler
import uuid
import asyncio
import functools

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
    from src.user.pipeline import pipeline
    # Import app_instance to control preview behavior during generation
    from src.user import app_instance as _app_instance
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
    hires_fix: bool = False
    adetailer: bool = False
    enhance_prompt: bool = False
    img2img_enabled: bool = False
    img2img_image: Optional[str] = None
    stable_fast: bool = False
    reuse_seed: bool = False
    flux_enabled: bool = False
    prio_speed: bool = False
    realistic_model: bool = False
    multiscale_enabled: bool = True
    multiscale_intermittent: bool = True
    multiscale_factor: float = 0.5
    multiscale_fullres_start: int = 10
    multiscale_fullres_end: int = 8
    keep_models_loaded: bool = True
    enable_preview: bool = False
    # Optional extras (may not be used by the current pipeline but accepted)
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None  # If provided >=0 we will reuse it


app = FastAPI(title="LightDiffusion Server", version="1.0.0")


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
        self._worker_task: Optional[asyncio.Task] = None
        # Telemetry counters
        self._batches_processed = 0
        self._items_processed = 0
        self._last_batch_ts: Optional[float] = None
        # Additional telemetry for average queue wait time
        self._requests_processed = 0
        self._cumulative_wait_time = 0.0

    async def start(self):
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def enqueue(self, pending: PendingRequest):
        async with self._lock:
            self._pending.append(pending)
            self._new_request.set()
        return await pending.future

    def _signature_for(self, req: GenerateRequest) -> tuple:
        # Grouping signature - requests must match these to be batched
        return (
            bool(req.realistic_model),
            int(req.width),
            int(req.height),
            bool(req.stable_fast),
            bool(req.flux_enabled),
            bool(req.img2img_enabled),
            # Treat multiscale options as batch-level — mixing them may
            # change the sampling schedule and therefore cannot be
            # safely combined into a single forward pass.
            bool(req.multiscale_enabled),
            bool(req.multiscale_intermittent),
            float(req.multiscale_factor),
            int(req.multiscale_fullres_start),
            int(req.multiscale_fullres_end),
            # Speed/priority and VRAM retention flags are also batch
            # level: they affect model selection and caching.
            bool(req.prio_speed),
            bool(req.keep_models_loaded),
            # Note: hires_fix and adetailer remain intentionally NOT part
            # of this signature because they are executed per-sample
            # after a shared forward pass.
            bool(req.enable_preview),
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
                        "hires_fix": bool(p.req.hires_fix),
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
            enhance_prompt=first_req.enhance_prompt,
            img2img=first_req.img2img_enabled,
            stable_fast=first_req.stable_fast,
            reuse_seed=first_req.reuse_seed,
            flux_enabled=first_req.flux_enabled,
            prio_speed=first_req.prio_speed,
            autohdr=True,
            realistic_model=first_req.realistic_model,
            negative_prompt=[p.req.negative_prompt or "" for p in items for _ in range(max(1, p.req.num_images))],
            multiscale_preset=None,
            enable_multiscale=first_req.multiscale_enabled,
            multiscale_factor=first_req.multiscale_factor,
            multiscale_fullres_start=first_req.multiscale_fullres_start,
            multiscale_fullres_end=first_req.multiscale_fullres_end,
            multiscale_intermittent_fullres=first_req.multiscale_intermittent,
            img2img_image=first_req.img2img_image,
            per_sample_info=per_sample_info,
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
            # Special-case: flux-enabled groups require flux-specific
            # model/clip loaders which the batched code path does not
            # configure. To ensure flux requests continue to work we run
            # them individually through the pipeline and collect their
            # saved artifacts per-request.
            saved_map: Dict[str, List[dict]] = {}
            if first_req.flux_enabled:
                for p in items:
                    start_ts_single = time.time()
                    # Build per-request per-sample info list
                    per_single_info = []
                    for k in range(max(1, p.req.num_images)):
                        per_single_info.append(
                            {
                                "request_id": p.request_id,
                                "filename_prefix": f"LD-REQ-{p.request_id}",
                                "seed": p.req.seed if (p.req.seed is not None and p.req.seed >= 0) else None,
                                "hires_fix": bool(p.req.hires_fix),
                                "adetailer": bool(p.req.adetailer),
                            }
                        )

                    pipeline_kwargs_single = dict(
                        prompt=p.req.prompt,
                        w=first_req.width,
                        h=first_req.height,
                        number=max(1, p.req.num_images),
                        batch=p.req.batch_size,
                        enhance_prompt=p.req.enhance_prompt,
                        img2img=p.req.img2img_enabled,
                        stable_fast=p.req.stable_fast,
                        reuse_seed=p.req.reuse_seed,
                        flux_enabled=True,
                        prio_speed=first_req.prio_speed,
                        autohdr=True,
                        realistic_model=first_req.realistic_model,
                        negative_prompt=p.req.negative_prompt or "",
                        multiscale_preset=None,
                        enable_multiscale=first_req.multiscale_enabled,
                        multiscale_factor=first_req.multiscale_factor,
                        multiscale_fullres_start=first_req.multiscale_fullres_start,
                        multiscale_fullres_end=first_req.multiscale_fullres_end,
                        multiscale_intermittent_fullres=first_req.multiscale_intermittent,
                        img2img_image=p.req.img2img_image,
                        per_sample_info=per_single_info,
                    )

                    try:
                        result_single = await loop.run_in_executor(None, functools.partial(pipeline, **pipeline_kwargs_single))
                    except Exception as e:
                        logger.exception("Per-request pipeline call failed for flux request %s: %s", p.request_id, e)
                        result_single = None

                    if isinstance(result_single, dict) and "batched_results" in result_single:
                        for rid, arr in result_single["batched_results"].items():
                            saved_map.setdefault(rid, []).extend(arr if isinstance(arr, list) else [arr])
                    else:
                        # Fallback: scan filesystem for files created since the
                        # single-call started and group by filename prefix.
                        files = _find_images_since(start_ts_single - 60)
                        for f in files:
                            name = os.path.basename(f)
                            if f"LD-REQ-{p.request_id}" in name:
                                saved_map.setdefault(p.request_id, []).append({
                                    "filename": name,
                                    "subfolder": os.path.relpath(os.path.dirname(f), "./output"),
                                })
            else:
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
                    path = os.path.join("./output", entry.get("subfolder", ""), entry.get("filename", ""))
                    try:
                        b64_list.append(_encode_png_to_base64(path))
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
        "Request: w=%s h=%s num_images=%s batch=%s hires_fix=%s adetailer=%s enhance=%s img2img=%s stable_fast=%s reuse_seed=%s flux=%s prio_speed=%s realistic=%s multiscale=%s intermittent=%s factor=%s fullres=[%s,%s] keep_models_loaded=%s enable_preview=%s prompt='%s' neg='%s' img2img_image_present=%s",
        req.width,
        req.height,
        req.num_images,
        req.batch_size,
        req.hires_fix,
        req.adetailer,
        req.enhance_prompt,
        req.img2img_enabled,
        req.stable_fast,
        reuse_seed,
        req.flux_enabled,
        req.prio_speed,
        req.realistic_model,
        req.multiscale_enabled,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=7861, reload=False)

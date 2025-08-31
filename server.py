from __future__ import annotations

import base64
import glob
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure we can import pipeline from this repo
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from src.user.pipeline import pipeline
    # Import app_instance to control preview behavior during generation
    from src.user import app_instance as _app_instance
except Exception as e:
    # Defer import error to runtime response for clarity
    pipeline = None  # type: ignore
    _pipeline_import_error = e
else:
    _pipeline_import_error = None


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
    multiscale_intermittent: bool = False
    multiscale_factor: float = 0.5
    multiscale_fullres_start: int = 3
    multiscale_fullres_end: int = 8
    keep_models_loaded: bool = True
    enable_preview: bool = False
    # Optional extras (may not be used by the current pipeline but accepted)
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None  # If provided >=0 we will reuse it


app = FastAPI(title="LightDiffusion Server", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _encode_png_to_base64(path: str) -> str:
    # Retry a few times in case the file is still being finalized on disk
    last_err: Optional[Exception] = None
    for _ in range(20):  # up to ~2s total
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            last_err = e
            time.sleep(0.1)
    # One last attempt or raise detailed error
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read generated image: {e if e else last_err}")


def _list_existing_images() -> List[str]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join("./output", "**", ext), recursive=True))
    return files


def _find_images_since(start_ts: float) -> List[str]:
    """Return images whose mtime is at or after start_ts (with small grace)."""
    grace = 0.25
    files = _list_existing_images()
    recent = [p for p in files if os.path.getmtime(p) >= (start_ts - grace)]
    recent.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return recent


@app.post("/api/generate")
def generate(req: GenerateRequest) -> Dict[str, Any]:
    # Validate pipeline import
    global pipeline, _pipeline_import_error
    if pipeline is None:
        raise HTTPException(status_code=500, detail=f"Pipeline import error: {_pipeline_import_error}")

    # Optionally honor requested seed by writing include/last_seed.txt and enabling reuse
    reuse_seed = req.reuse_seed
    if req.seed is not None and req.seed >= 0:
        os.makedirs("./include", exist_ok=True)
        with open(os.path.join("./include", "last_seed.txt"), "w", encoding="utf-8") as f:
            f.write(str(int(req.seed)))
        reuse_seed = True

    # Prepare prompt: if img2img is enabled and an image path is given, the pipeline expects the path in `prompt`
    effective_prompt = req.prompt
    if req.img2img_enabled and req.img2img_image:
        effective_prompt = req.img2img_image

    # Mark the start time (use to detect images modified by this call even if filenames are reused)
    start_time = time.time()

    # Run generation
    prev_preview_state = None
    try:
        # Ensure preview computation is enabled only if explicitly requested
        # Many samplers check app_instance.app.previewer_var.get() before doing TAESD preview work.
        # Default of PreviewerVar is True; we force it based on request for this call.
        try:
            prev_preview_state = _app_instance.app.previewer_var.get()
            _app_instance.app.previewer_var.set(bool(req.enable_preview))
        except Exception:
            prev_preview_state = None  # If app_instance is unavailable, proceed without toggling

        try:
            pipeline(
                prompt=effective_prompt,
                w=req.width,
                h=req.height,
                number=req.num_images,
                batch=req.batch_size,
                hires_fix=req.hires_fix,
                adetailer=req.adetailer,
                enhance_prompt=req.enhance_prompt,
                img2img=req.img2img_enabled,
                stable_fast=req.stable_fast,
                reuse_seed=reuse_seed,
                flux_enabled=req.flux_enabled,
                prio_speed=req.prio_speed,
                autohdr=True,
                realistic_model=req.realistic_model,
                negative_prompt=req.negative_prompt or None,
                multiscale_preset=None,
                enable_multiscale=req.multiscale_enabled,
                multiscale_factor=req.multiscale_factor,
                multiscale_fullres_start=req.multiscale_fullres_start,
                multiscale_fullres_end=req.multiscale_fullres_end,
                multiscale_intermittent_fullres=req.multiscale_intermittent,
            )
        finally:
            # Restore previous preview state if we changed it
            try:
                if prev_preview_state is not None:
                    _app_instance.app.previewer_var.set(prev_preview_state)
            except Exception:
                pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # Find images produced by this call; consider files modified since start_time
    timeout_s = 60.0
    poll_interval = 0.25
    images: List[str] = []
    while time.time() - start_time < timeout_s:
        images = _find_images_since(start_time)
        if images:
            break
        time.sleep(poll_interval)

    if not images:
        # As a last resort, return the latest image(s) even if mtimes couldn't be compared
        latest = _list_existing_images()
        if not latest:
            raise HTTPException(status_code=500, detail="No images generated")
        latest.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        images = latest

    # If multiple requested and found, return list; else return single image
    if req.num_images > 1 and len(images) > 1:
        # Sort again by mtime desc and take the first N
        images.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        selected = images[: req.num_images]
        b64_list = [_encode_png_to_base64(p) for p in selected]
        return {"images": b64_list}
    else:
        b64 = _encode_png_to_base64(images[0])
        return {"image": b64}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=7861, reload=False)

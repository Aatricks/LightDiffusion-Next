"""Generation runner helpers.

This module contains the long-running generation logic so the main
Streamlit entrypoint stays focused on layout. The implementation is a
direct port of the original logic but lives in a separate module.
"""
import os
import glob
import threading
import time
import gc
import streamlit as st
from PIL import Image

from ui.helpers import compute_display_size, render_responsive_image
from ui.history import add_to_history
from src.user.pipeline import pipeline
from src.user import app_instance
from src.Device.ModelCache import (
    set_keep_models_loaded,
    get_memory_info,
    clear_model_cache,
)


def generate_images(settings, status_placeholder, gallery_placeholder, status_bar=None):
    """Generate images with live preview support (moved from streamlit_app).

    The function intentionally mirrors the previous behavior and uses the
    same session-state fields to remain compatible with the existing UI.
    """
    # Pre-generation checks
    if not st.session_state.lightdiffusion_ready:
        status_placeholder.error("❌ LightDiffusion is not initialized yet!")
        return []

    if not settings["prompt"].strip():
        status_placeholder.warning("⚠️ Please enter a prompt!")
        return []

    # Setup generation state
    st.session_state.interrupt_generation = False
    st.session_state.is_generating = True
    
    # Configure prompt cache based on settings
    try:
        from src.Utilities import prompt_cache
        prompt_cache.enable_prompt_cache(settings.get("prompt_cache_enabled", True))
    except Exception:
        pass
    
    try:
        app_instance.app.cleanup_all_previews()
    except Exception:
        pass
    try:
        app_instance.app.clear_interrupt()
    except Exception:
        pass
    set_keep_models_loaded(settings["keep_models_loaded"])

    try:
        if settings.get("num_images", 1) >= 10 and settings.get("keep_models_loaded", True):
            try:
                status_placeholder.info("⚠️ Large job detected — the app will unload models between chunks to reduce VRAM usage.")
            except Exception:
                pass
    except Exception:
        pass

    # Create output directories used for previews
    os.makedirs("./output/preview_display", exist_ok=True)

    pipeline_result = None
    generation_complete = threading.Event()
    generation_error = None

    def run_generation():
        nonlocal pipeline_result, generation_error
        original_keep_models = bool(settings.get("keep_models_loaded", True))
        forced_unload_for_large_job = False

        try:
            total_images = int(settings.get("num_images", 1))
            configured_batch = int(settings.get("batch_size", 1))

            LARGE_JOB_MIN_IMAGES = 10
            if total_images >= LARGE_JOB_MIN_IMAGES and original_keep_models:
                try:
                    set_keep_models_loaded(False)
                    forced_unload_for_large_job = True
                except Exception:
                    forced_unload_for_large_job = False

            if settings.get("multiscale_custom"):
                multiscale_params = {
                    "multiscale_preset": None,
                    "enable_multiscale": True,
                    "multiscale_factor": settings.get("multiscale_factor", 0.5),
                    "multiscale_fullres_start": settings.get("multiscale_fullres_start", 3),
                    "multiscale_fullres_end": settings.get("multiscale_fullres_end", 8),
                    "multiscale_intermittent_fullres": settings.get("multiscale_intermittent_fullres", False),
                }
            else:
                multiscale_params = {"multiscale_preset": settings.get("multiscale_preset", "balanced")}

            images_generated = 0
            last_error = None

            def _attempt_memory_recovery(force_clear=False):
                try:
                    try:
                        app_instance.app.cleanup_all_previews()
                    except Exception:
                        pass

                    gc.collect()

                    try:
                        if force_clear:
                            clear_model_cache()
                        else:
                            try:
                                mem_info = get_memory_info()
                                used = float(mem_info.get('used_vram', 0) or 0)
                                total = float(mem_info.get('total_vram', 0) or 0)
                                if total > 0 and (used / total) >= 0.85:
                                    clear_model_cache()
                            except Exception:
                                pass
                    except Exception:
                        pass

                    try:
                        import torch
                        if getattr(torch, "cuda", None) and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                    time.sleep(0.15)
                except Exception:
                    pass

            while images_generated < total_images and not stop_event.is_set():
                remaining = total_images - images_generated
                target_chunk = min(configured_batch, remaining)
                attempt_chunk = target_chunk

                while attempt_chunk > 0 and not stop_event.is_set():
                    try:
                        result = pipeline(
                            prompt=settings.get("prompt", ""),
                            negative_prompt=settings.get("negative_prompt", ""),
                            w=settings.get("width"),
                            h=settings.get("height"),
                            number=attempt_chunk,
                            scheduler=settings.get("scheduler", "ays"),
                            sampler=settings.get("sampler", "dpmpp_sde_cfgpp"),
                            steps=settings.get("steps", 20),
                            # Honor the configured batch size as an independent
                            # setting. Previously the batch argument was clamped
                            # to the remaining number of images which made the
                            # Batch Size ineffective when requesting fewer
                            # images than the configured batch. Pass the
                            # configured_batch explicitly so the pipeline can
                            # use it for internal grouping.
                            batch=configured_batch,
                            hires_fix=settings.get("hiresfix", False),
                            adetailer=settings.get("adetailer", False),
                            enhance_prompt=settings.get("enhance_prompt", False),
                            img2img=settings.get("img2img_mode", False),
                            stable_fast=settings.get("stable_fast", False),
                            reuse_seed=settings.get("reuse_seed", False),
                            flux_enabled=settings.get("flux_mode", False),
                            autohdr=True,
                            realistic_model=settings.get("realistic_mode", False),
                            img2img_image=settings.get("input_image_path") if settings.get("img2img_mode", False) else None,
                            deepcache_enabled=settings.get("deepcache_enabled", False),
                            deepcache_interval=settings.get("deepcache_interval", 3),
                            deepcache_depth=settings.get("deepcache_depth", 2),
                            deepcache_start_step=settings.get("deepcache_start_step", 0),
                            deepcache_end_step=settings.get("deepcache_end_step", 1000),
                            **multiscale_params,
                        )

                        pipeline_result = True
                        try:
                            del result
                        except Exception:
                            pass

                        images_generated += attempt_chunk
                        _attempt_memory_recovery(force_clear=forced_unload_for_large_job)
                        break
                    except Exception as e:
                        last_error = e
                        msg = str(e).lower() if e is not None else ""
                        is_oom = (
                            "out of memory" in msg
                            or "cuda" in msg and ("out" in msg or "oom" in msg)
                            or isinstance(e, MemoryError)
                        )

                        if not is_oom:
                            raise

                        new_attempt = max(1, attempt_chunk // 2)
                        if new_attempt >= attempt_chunk:
                            new_attempt = 1
                        attempt_chunk = new_attempt
                        _attempt_memory_recovery(force_clear=True)

                        if attempt_chunk == 1:
                            try:
                                clear_model_cache()
                            except Exception:
                                pass
                            gc.collect()
                            try:
                                import torch as _torch
                                if getattr(_torch, "cuda", None) and _torch.cuda.is_available():
                                    _torch.cuda.empty_cache()
                            except Exception:
                                pass

            if last_error is not None and images_generated < total_images:
                raise last_error

        except Exception as e:
            generation_error = e
            if settings.get("verbose_mode"):
                import traceback
                traceback.print_exc()
        finally:
            try:
                if forced_unload_for_large_job and original_keep_models:
                    set_keep_models_loaded(original_keep_models)
            except Exception:
                pass

            generation_complete.set()

    stop_event = threading.Event()

    gen_thread = threading.Thread(target=run_generation, daemon=True)
    gen_thread.start()

    st.session_state.generation_job = {
        "thread": gen_thread,
        "complete_event": generation_complete,
        "stop_event": stop_event,
        "start_time": time.time(),
    }

    display_size = compute_display_size(
        settings["width"],
        settings["height"],
        max_width=800,
        max_height=600,
    )
    st.session_state.display_size = display_size

    UI_SCALE = float(settings.get("ui_scale", 1.0))
    UI_MAX_WIDTH = 1400
    UI_MAX_HEIGHT = 1000
    ui_full_w = min(int(display_size[0] * UI_SCALE), UI_MAX_WIDTH)
    ui_full_h = min(int(display_size[1] * UI_SCALE), UI_MAX_HEIGHT)
    st.session_state.ui_display_size = (ui_full_w, ui_full_h)

    preview_container = gallery_placeholder.empty() if settings["enable_preview"] else None
    last_preview_time = 0

    while not generation_complete.wait(0.2):
        if st.session_state.interrupt_generation:
            try:
                app_instance.app.request_interrupt()
            except Exception:
                pass
            try:
                if status_bar is not None:
                    status_bar.markdown("<div class=\"ld-status-bar auto-hide\">⏹️ Stopping generation...</div>", unsafe_allow_html=True)
                else:
                    status_placeholder.warning("⏹️ Stopping generation...")
            except Exception:
                status_placeholder.warning("⏹️ Stopping generation...")
            break

        if preview_container and settings["enable_preview"]:
            current_time = time.time()
            if current_time - last_preview_time > 0.5:
                previews = []
                try:
                    previews = app_instance.app.get_latest_previews()
                except Exception:
                    previews = []
                if previews:
                    try:
                        recent = previews[-6:]
                        cols_count = min(3, len(recent)) or 1
                        with preview_container.container():
                            cols = st.columns(cols_count)
                            for i, pth in enumerate(recent):
                                try:
                                    with Image.open(pth) as img_prev:
                                        cols_for_preview = cols_count or 1
                                        tile_w = max(64, int(ui_full_w / cols_for_preview))
                                        # Preserve the preview's real aspect ratio to avoid stretching
                                        orig_w, orig_h = img_prev.size if getattr(img_prev, 'size', None) else (tile_w, tile_w)
                                        tile_h = max(64, int(tile_w * (orig_h / (orig_w or 1))))
                                        render_responsive_image(img_prev, (tile_w, tile_h), cols[i % cols_count])
                                except Exception:
                                    pass
                        last_preview_time = current_time
                    except Exception:
                        pass

        elapsed = time.time() - st.session_state.generation_job["start_time"]
        try:
            if status_bar is not None:
                status_bar.markdown(f"<div class=\"ld-status-bar\">🎨 Generating — {elapsed:.1f}s</div>", unsafe_allow_html=True)
            else:
                status_placeholder.info(f"🎨 Generating... ({elapsed:.1f}s)")
        except Exception:
            status_placeholder.info(f"🎨 Generating... ({elapsed:.1f}s)")

    try:
        app_instance.app.cleanup_all_previews()
    except Exception:
        pass

    time.sleep(0.3)

    st.session_state.is_generating = False

    if generation_error:
        short_err = str(generation_error)
        if len(short_err) > 120:
            short_err = short_err[:117] + "..."
        try:
            if status_bar is not None:
                status_bar.markdown(f"<div class=\"ld-status-bar auto-hide\">❌ Generation failed: {short_err}</div>", unsafe_allow_html=True)
            else:
                status_placeholder.error(f"❌ Generation failed: {generation_error}")
        except Exception:
            status_placeholder.error(f"❌ Generation failed: {generation_error}")
        return []

    generated_image_paths = []

    if settings["flux_mode"]:
        primary_dirs = ["./output/Flux"]
    elif settings["img2img_mode"]:
        primary_dirs = ["./output/Img2Img"]
    elif settings["adetailer"]:
        primary_dirs = ["./output/Adetailer", "./output/Classic", "./output/HiresFix"]
    elif settings["hiresfix"]:
        primary_dirs = ["./output/HiresFix"]
    else:
        primary_dirs = ["./output/Classic"]

    all_outputs = []
    for output_dir in primary_dirs:
        if os.path.exists(output_dir):
            files = glob.glob(f"{output_dir}/*.png")
            all_outputs.extend(files)

    if settings["verbose_mode"]:
        status_placeholder.info(f"Searching in: {primary_dirs}, Found {len(all_outputs)} files")

    if all_outputs:
        all_outputs = sorted(all_outputs, key=os.path.getmtime, reverse=True)
        job_start = None
        try:
            job_start = st.session_state.generation_job.get("start_time")
        except Exception:
            job_start = None

        now_time = time.time()
        if job_start:
            job_window = [f for f in all_outputs if os.path.getmtime(f) >= (job_start - 2.0) and os.path.getmtime(f) <= (now_time + 2.0)]
        else:
            job_window = []

        if job_window:
            # The pipeline may produce multiple outputs for a single
            # request when an internal batch is larger than the
            # user-visible `num_images`. Previews already show the
            # per-batch results; include all files produced during the
            # job so the final gallery matches what was previewed.
            generated_image_paths.extend(sorted(job_window, key=os.path.getmtime, reverse=True))
        else:
            # Fallback: when we couldn't find files within the job time
            # window, pick the most recent outputs. Show at least the
            # user's requested number but also at least the configured
            # batch size so batch-generated outputs are visible.
            n_show = max(int(settings.get("num_images", 1)), int(settings.get("batch_size", 1)))
            for f in all_outputs[:n_show]:
                if os.path.exists(f):
                    generated_image_paths.append(f)
                else:
                    if settings["verbose_mode"]:
                        status_placeholder.warning(f"File disappeared before collection: {f}")

    if generated_image_paths:
        st.session_state.generated_image_paths = generated_image_paths

        # Use the actual dimensions of the first generated image to compute
        # the UI display size so we match the file's aspect ratio instead of
        # the user-entered settings. This prevents stretching for img2img
        # and ADetailer outputs which may use different dims.
        try:
            with Image.open(generated_image_paths[0]) as first_img:
                first_w, first_h = first_img.size
        except Exception:
            first_w, first_h = settings.get('width', 512), settings.get('height', 512)

        display_size = compute_display_size(first_w, first_h, max_width=800, max_height=600)
        st.session_state.display_size = display_size

        job_elapsed = time.time() - st.session_state.generation_job.get("start_time", time.time())

        try:
            from PIL.PngImagePlugin import PngInfo
            for path in generated_image_paths:
                try:
                    with Image.open(path) as im:
                        png_meta = getattr(im, "info", {}) or {}

                        steps_val = png_meta.get("steps") or png_meta.get("step")
                        steps = None
                        try:
                            if steps_val is not None:
                                steps = int(float(steps_val))
                        except Exception:
                            steps = None

                        avg_iters = None
                        if steps is not None and job_elapsed > 0:
                            avg_iters = steps / job_elapsed

                        pnginfo = PngInfo()
                        for k, v in (png_meta.items() if isinstance(png_meta, dict) else []):
                            try:
                                pnginfo.add_text(str(k), str(v))
                            except Exception:
                                pass

                        pnginfo.add_text("generation_duration", f"{job_elapsed:.3f}")
                        if avg_iters is not None:
                            pnginfo.add_text("avg_iters_per_s", f"{avg_iters:.3f}")
                        else:
                            pnginfo.add_text("avg_iters_per_s", "unknown")

                        try:
                            im.save(path, pnginfo=pnginfo)
                        except Exception:
                            if settings.get("verbose_mode"):
                                status_placeholder.warning(f"Could not persist metadata to {os.path.basename(path)}")
                except Exception:
                    if settings.get("verbose_mode"):
                        status_placeholder.warning(f"Failed updating metadata for {path}")
        except Exception:
            if settings.get("verbose_mode"):
                status_placeholder.warning("Could not attach generation metadata to images")

        add_to_history(generated_image_paths, settings)

        # Recompute UI display size based on the first generated image so the
        # UI matches the actual file aspect ratio (prevents stretching).
        try:
            new_ui_full_w = min(int(display_size[0] * UI_SCALE), UI_MAX_WIDTH)
            new_ui_full_h = min(int(display_size[1] * UI_SCALE), UI_MAX_HEIGHT)
            st.session_state.ui_display_size = (new_ui_full_w, new_ui_full_h)
            ui_full_w, ui_full_h = new_ui_full_w, new_ui_full_h
        except Exception:
            # If recompute fails just keep previous UI sizes.
            pass

        # Try to render a gallery of generated images; if that fails attempt a
        # simple fallback display showing the first image.
        try:
            with gallery_placeholder.container():
                cols_count = min(3, len(generated_image_paths)) or 1
                cols = st.columns(cols_count)
                for idx, path in enumerate(generated_image_paths):
                    try:
                        with Image.open(path) as img:
                            orig_w, orig_h = img.size
                            if len(generated_image_paths) == 1:
                                # For a single image, show it using the full UI-scaled size
                                tile_w, tile_h = ui_full_w, ui_full_h
                            else:
                                tile_w = max(64, int(ui_full_w / cols_count))
                                # Preserve each image's aspect ratio when calculating height
                                tile_h = max(64, int(tile_w * (orig_h / (orig_w or 1))))

                            try:
                                render_responsive_image(img, (tile_w, tile_h), cols[idx % cols_count])
                                cols[idx % cols_count].caption(f"Image {idx+1}")
                            except Exception:
                                with cols[idx % cols_count]:
                                    st.image(img, caption=f"Image {idx+1}", use_container_width=True)
                    except Exception:
                        if settings.get("verbose_mode"):
                            status_placeholder.warning(f"Error rendering {path}")
        except Exception:
            # Fallback: try rendering the first image directly into the gallery
            try:
                with Image.open(generated_image_paths[0]) as first_img:
                    orig_w, orig_h = first_img.size
                    if len(generated_image_paths) == 1:
                        render_responsive_image(first_img, (ui_full_w, ui_full_h), gallery_placeholder)
                    else:
                        fallback_tile_w = max(64, int(ui_full_w / min(3, len(generated_image_paths))))
                        fallback_tile_h = max(64, int(fallback_tile_w * (orig_h / (orig_w or 1))))
                        render_responsive_image(first_img, (fallback_tile_w, fallback_tile_h), gallery_placeholder)
            except Exception:
                if settings.get("verbose_mode"):
                    status_placeholder.warning("Failed to render gallery or fallback image")

        elapsed = time.time() - st.session_state.generation_job["start_time"]
        try:
            if status_bar is not None:
                status_bar.markdown(f"<div class=\"ld-status-bar auto-hide\">✅ Generated {len(generated_image_paths)} image(s) — {elapsed:.1f}s</div>", unsafe_allow_html=True)
            else:
                status_placeholder.success(f"✅ Generated {len(generated_image_paths)} image(s) in {elapsed:.1f}s")
        except Exception:
            status_placeholder.success(f"✅ Generated {len(generated_image_paths)} image(s) in {elapsed:.1f}s")
        return generated_image_paths
    else:
        if pipeline_result:
            checked_dirs = ", ".join(primary_dirs)
            try:
                if status_bar is not None:
                    status_bar.markdown(f"<div class=\"ld-status-bar auto-hide\">⚠️ Completed but no images found (checked: {checked_dirs})</div>", unsafe_allow_html=True)
                else:
                    status_placeholder.warning(f"⚠️ Generation completed but no images found. Checked: {checked_dirs}")
            except Exception:
                status_placeholder.warning(f"⚠️ Generation completed but no images found. Checked: {checked_dirs}")
        else:
            try:
                if status_bar is not None:
                    status_bar.markdown("<div class=\"ld-status-bar auto-hide\">Generation stopped</div>", unsafe_allow_html=True)
                else:
                    status_placeholder.info("Generation stopped")
            except Exception:
                status_placeholder.info("Generation stopped")
        return []


def stop_generation():
    """Request generation stop (delegated)"""
    st.session_state.interrupt_generation = True
    try:
        app_instance.app.request_interrupt()
    except Exception:
        pass

    try:
        job = st.session_state.get("generation_job")
        if job and isinstance(job, dict):
            ev = job.get("complete_event")
            if ev is not None and hasattr(ev, "set"):
                ev.set()
            stop_ev = job.get("stop_event")
            if stop_ev is not None and hasattr(stop_ev, "set"):
                try:
                    stop_ev.set()
                except Exception:
                    pass
    except Exception:
        pass

    st.session_state.is_generating = False
    try:
        st.session_state.start_generation = False
        job = st.session_state.get("generation_job")
        if job and isinstance(job, dict):
            job["aborted"] = True
            st.session_state.generation_job = job
    except Exception:
        pass

    try:
        st.rerun()
    except Exception:
        pass


def prepare_generation():
    """Prepare state so the UI disables controls before generation begins."""
    st.session_state.start_generation = True
    st.session_state.is_generating = True

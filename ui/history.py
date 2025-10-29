import os
import json
import glob
import time
import re
import streamlit as st
from PIL import Image

HISTORY_FILE = "./webui_history.json"


def sanitize_seed_for_display(seed_value):
    """Return a safe seed string or None if the value looks like a tensor/image dump."""
    if seed_value is None:
        return None
    if isinstance(seed_value, (int, float)):
        return str(int(seed_value))
    if isinstance(seed_value, str):
        s = seed_value.strip()
        # If the value is clearly a dump (tensor, list, multiline or huge)
        # avoid returning the full content to the UI. As a helpful fallback
        # try to extract a numeric token (common when a seed is embedded in
        # a larger string). This keeps the compact, user-friendly seed in
        # the Details view while preserving the full raw metadata in the
        # hidden JSON blob.
        if "tensor(" in s.lower() or "[" in s or "\n" in s or len(s) > 240:
            # Try to salvage a numeric-looking substring (at least 4 digits)
            m = re.search(r"(\d{4,})", s)
            if m:
                return m.group(0)
            return None
        return s
    return None


def load_history():
    """Load generation history from disk, sanitizing any large seed dumps."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                changed = False
                for e in saved:
                    if isinstance(e, dict):
                        if 'seed' in e:
                            sanitized = sanitize_seed_for_display(e.get('seed'))
                            if sanitized != e.get('seed'):
                                e['seed'] = sanitized
                                changed = True
                        png_meta = e.get('png_metadata') or {}
                        # Preserve PNG metadata as-is; however, if the top-
                        # level `seed` is missing or was removed by
                        # sanitization, try to populate a friendly top-level
                        # seed using the PNG's embedded value so the Details
                        # view shows a concise identifier.
                        if not e.get('seed') and isinstance(png_meta, dict) and png_meta.get('seed'):
                            try:
                                e['seed'] = sanitize_seed_for_display(png_meta.get('seed'))
                                changed = True
                            except Exception:
                                pass
                        # Normalize stored width/height values. If they are
                        # strings or missing, try to convert or read from the
                        # image file so the UI can rely on accurate dimensions
                        # for thumbnail sizing.
                        w = e.get('width')
                        h = e.get('height')
                        normalized = False
                        try:
                            if isinstance(w, str) and w.isdigit():
                                e['width'] = int(w)
                                normalized = True
                            if isinstance(h, str) and h.isdigit():
                                e['height'] = int(h)
                                normalized = True
                        except Exception:
                            pass

                        if (e.get('width') is None or e.get('height') is None) and isinstance(e.get('image_path'), str):
                            try:
                                if os.path.exists(e['image_path']):
                                    with Image.open(e['image_path']) as _img:
                                        iw, ih = _img.size
                                        if iw and ih:
                                            e['width'] = iw
                                            e['height'] = ih
                                            normalized = True
                            except Exception:
                                pass

                        if normalized:
                            changed = True
                if changed:
                    try:
                        save_history(saved)
                    except Exception:
                        pass
                return saved
        except Exception as e:
            try:
                st.warning(f"Could not load history: {e}")
            except Exception:
                pass
    return []


def save_history(history):
    """Save generation history to disk."""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        try:
            st.error(f"Could not save history: {e}")
        except Exception:
            pass


def add_to_history(image_paths, settings):
    """Add generated image file paths to the persistent history store."""
    history = load_history()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue

        png_meta = {}
        width = None
        height = None
        try:
            with Image.open(img_path) as _img:
                png_meta = getattr(_img, "info", {}) or {}
                try:
                    width, height = _img.size
                except Exception:
                    width, height = None, None
        except Exception:
            png_meta = {}
            width, height = None, None

        # Keep the PNG metadata raw so it can be inspected in the
        # 'All metadata' view. Compute a sanitized seed string for the
        # top-level display separately so the Details section remains
        # compact and user-friendly.
        seed_meta = sanitize_seed_for_display(png_meta.get('seed'))

        png_prompt = png_meta.get('prompt')
        png_negative = png_meta.get('negative_prompt')

        entry = {
            "timestamp": timestamp,
            "image_path": img_path,
            "prompt": png_prompt if png_prompt not in (None, "") else settings.get("prompt", ""),
            "negative_prompt": png_negative if png_negative not in (None, "") else settings.get("negative_prompt", ""),
            # Record the actual file dimensions — ensures thumbnails match the file
            "width": width,
            "height": height,
            "batch_size": settings.get("batch_size"),
                # Record model type if available from PNG metadata, otherwise infer from path
                'model_type': (png_meta.get('model_type') or ( 'FLUX' if 'Flux' in img_path else None )),
                'model_path': png_meta.get('model_path'),
            "seed": seed_meta,
            "sampler": png_meta.get("sampler"),
            "steps": png_meta.get("steps"),
            "generation_duration": None,
            "avg_iters_per_s": None,
            "cfg": png_meta.get("cfg"),
            "scheduler": png_meta.get("scheduler"),
            "denoise": png_meta.get("denoise"),
            "png_metadata": png_meta,
        }

        try:
            gd = png_meta.get("generation_duration")
            if gd is not None:
                try:
                    entry["generation_duration"] = float(gd)
                except Exception:
                    try:
                        entry["generation_duration"] = float(str(gd).rstrip('s'))
                    except Exception:
                        entry["generation_duration"] = None
        except Exception:
            pass

        try:
            ai = png_meta.get("avg_iters_per_s")
            if ai is not None:
                try:
                    entry["avg_iters_per_s"] = float(ai)
                except Exception:
                    try:
                        entry["avg_iters_per_s"] = float(str(ai).rstrip('s'))
                    except Exception:
                        entry["avg_iters_per_s"] = None
        except Exception:
            pass

        history.insert(0, entry)

    history = history[:100]
    save_history(history)


def clear_history():
    """Remove all history and delete associated image files."""
    history = load_history()
    for entry in history:
        img_path = entry.get("image_path")
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception as e:
                try:
                    st.warning(f"Could not delete {os.path.basename(img_path)}: {e}")
                except Exception:
                    pass
    save_history([])


def scan_output_folders():
    """Scan common output folders and build a history list of found PNGs."""
    output_dirs = [
        "./output/Classic",
        "./output/Flux",
        "./output/HiresFix",
        "./output/Img2Img",
        "./output/Adetailer",
    ]

    all_images = []
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            images = glob.glob(f"{output_dir}/*.png")
            all_images.extend(images)

    all_images = sorted(all_images, key=os.path.getmtime, reverse=True)
    existing_history = load_history()
    existing_paths = {entry['image_path']: entry for entry in existing_history}

    new_history = []
    for img_path in all_images[:100]:
        if img_path in existing_paths:
            new_history.append(existing_paths[img_path])
        else:
            try:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(img_path)))
                with Image.open(img_path) as img:
                    width, height = img.size
                    png_meta = getattr(img, 'info', {}) or {}

                entry = {
                    'timestamp': timestamp,
                    'image_path': img_path,
                    'prompt': png_meta.get('prompt', '(prompt not available)'),
                    'negative_prompt': png_meta.get('negative_prompt', ''),
                    # Preserve the discovered image's actual dimensions so that
                    # the UI can scale thumbnails correctly.
                    'width': width,
                    'height': height,
                    'batch_size': png_meta.get('batch_size'),
                    # Record model type/path when available in PNG metadata
                    'model_type': png_meta.get('model_type') or ('FLUX' if 'Flux' in img_path else None),
                    'model_path': png_meta.get('model_path'),
                    'seed': sanitize_seed_for_display(png_meta.get('seed')),
                    'sampler': png_meta.get('sampler'),
                    'steps': png_meta.get('steps'),
                    'generation_duration': None,
                    'avg_iters_per_s': None,
                    'cfg': png_meta.get('cfg'),
                    'scheduler': png_meta.get('scheduler'),
                    'denoise': png_meta.get('denoise'),
                    'png_metadata': png_meta,
                }

                try:
                    gd = png_meta.get('generation_duration')
                    if gd is not None:
                        try:
                            entry['generation_duration'] = float(gd)
                        except Exception:
                            try:
                                entry['generation_duration'] = float(str(gd).rstrip('s'))
                            except Exception:
                                entry['generation_duration'] = None
                except Exception:
                    pass

                try:
                    ai = png_meta.get('avg_iters_per_s')
                    if ai is not None:
                        try:
                            entry['avg_iters_per_s'] = float(ai)
                        except Exception:
                            try:
                                entry['avg_iters_per_s'] = float(str(ai).rstrip('s'))
                            except Exception:
                                entry['avg_iters_per_s'] = None
                except Exception:
                    pass

                new_history.append(entry)
            except Exception:
                pass

    save_history(new_history)
    return new_history


def delete_history_entry(entry_index):
    history = load_history()
    if 0 <= entry_index < len(history):
        entry = history[entry_index]
        img_path = entry['image_path']
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception as e:
                try:
                    st.error(f"Could not delete image file: {e}")
                except Exception:
                    pass
        history.pop(entry_index)
        save_history(history)
        return True
    return False

import os
import gguf
from src.FileManaging import Loader
from src.Quantize import Quantizer


def list_available_models(return_mapping: bool = False):
    """Return candidate model files from include/checkpoints and include/unet.

    By default returns a list of user-friendly display names (basename).
    If `return_mapping=True` returns a list of (display_name, full_path)
    tuples which is useful for UI code that needs to persist the full
    filesystem path while showing a compact label to the user.
    """
    candidates = []
    valid_ext = (".gguf", ".safetensors", ".pt", ".pth")
    search_dirs = [
        os.path.abspath(os.path.join("./include/", "checkpoints")),
        os.path.abspath(os.path.join("./include/", "unet")),
    ]

    for d in search_dirs:
        try:
            for name in os.listdir(d):
                full = os.path.join(d, name)
                if os.path.isfile(full) and name.lower().endswith(valid_ext):
                    candidates.append((name, full))
        except Exception:
            # ignore missing directories or permissions
            pass
    # (Deliberately do NOT search huggingface cache or include/clip here -
    # user requested discovery limited to ./include/checkpoints and ./include/unet.)

    # Deduplicate by full path and sort by display name
    seen = set()
    unique = []
    for disp, full in sorted(candidates, key=lambda x: x[0].lower()):
        if full not in seen:
            seen.add(full)
            unique.append((disp, full))

    if return_mapping:
        return unique
    # default: return only display names for compact UI labels
    return [d for d, _ in unique]


def detect_model_type(model_path: str) -> str:
    """Detect model type from file extension / gguf header / safetensors keys.

    Returns: one of 'FLUX', 'SDXL', 'SD15' (default)
    """
    if model_path is None:
        return "SD15"
    lp = model_path.lower()
    if lp.endswith(".gguf"):
        try:
            r = gguf.GGUFReader(model_path)
            field = r.get_field("general.architecture")
            if field is not None:
                # The GGUF field may store parts or raw bytes depending on
                # writer. Try a few ways to decode it robustly.
                try:
                    # Preferred: if field exposes a bytes-like 'data'
                    raw = b"".join([bytes([b]) if isinstance(b, int) else b for b in field.data])
                    arch = raw.decode("utf-8", errors="ignore").strip().lower()
                except Exception:
                    try:
                        arch = str(field.parts[field.data[-1]], encoding="utf-8").lower()
                    except Exception:
                        arch = None

                if arch:
                    if "flux" in arch:
                        return "FLUX"
                    if "sdxl" in arch:
                        return "SDXL"
                    if "sd1" in arch or "sd" in arch:
                        return "SD15"
        except Exception:
            # fallback to filename heuristic
            if "flux" in lp:
                return "FLUX"
            if "sdxl" in lp:
                return "SDXL"
            return "SD15"

    # safetensors / pt heuristics
    if lp.endswith(".safetensors") or lp.endswith(".pt") or lp.endswith(".pth"):
        # cheap heuristic: sdxl checkpoints often include 'sdxl' or 'refiner' or have label_emb
        base = os.path.basename(lp)
        if "sdxl" in base.lower() or "refiner" in base.lower() or "hassaku" in base.lower():
            return "SDXL"
        # default to SD1.5
        return "SD15"

    # otherwise default
    return "SD15"


def load_model_for_pipeline(model_path: str = None, flux_dequant_dtype: str = None, flux_patch_dtype: str = None):
    """Load model artifacts appropriate for the detected model type.

    Returns a tuple: (model_type, loader_result)
    loader_result is the existing loader return for that model (varies by type)
    """
    # Determine model type; prefer explicit extension check for GGUF files so
    # we never accidentally attempt to torch.load a gguf file.
    model_type = detect_model_type(model_path)
    try:
        if model_path and str(model_path).lower().endswith(".gguf"):
            model_type = "FLUX"
    except Exception:
        pass

    if model_type == "FLUX":
        # For flux, expect gguf unet in include/unet and gguf clips in include/clip
        unet_name = os.path.basename(model_path) if model_path and os.path.exists(model_path) else "flux1-dev-Q8_0.gguf"
        unet_loader = Quantizer.UnetLoaderGGUF()
        # Pass through dequant/patch dtype if provided
        try:
            return (
                "FLUX",
                unet_loader.load_unet(
                    unet_name=unet_name,
                    dequant_dtype=flux_dequant_dtype,
                    patch_dtype=flux_patch_dtype,
                ),
            )
        except Exception:
            # bubble up the exception to caller
            raise
    else:
        # SD1.5 / SDXL: use CheckpointLoaderSimple which handles safetensors/pt
        ckpt = model_path or "./include/checkpoints/Meina V10 - baked VAE.safetensors"
        loader = Loader.CheckpointLoaderSimple()
        out = loader.load_checkpoint(ckpt_name=ckpt)
        # detect SDXL specifically and return model type
        model_type = detect_model_type(ckpt)
        return (model_type, out)

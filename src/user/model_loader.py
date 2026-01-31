import os
from src.FileManaging import Loader


def list_available_models(return_mapping: bool = False):
    """Return candidate model files from include/checkpoints and include/unet.

    By default returns a list of user-friendly display names (basename).
    If `return_mapping=True` returns a list of (display_name, full_path)
    tuples which is useful for UI code that needs to persist the full
    filesystem path while showing a compact label to the user.
    """
    candidates = []
    valid_ext = (".safetensors", ".pt", ".pth")  # .gguf no longer supported
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
    """Detect model type from file extension / safetensors keys.

    Returns: one of 'SDXL', 'SD15' (default)
    
    Note: GGUF files (.gguf) are no longer supported and will raise ValueError.
    """
    if model_path is None:
        return "SD15"
    lp = model_path.lower()
    
    # GGUF files are no longer supported
    if lp.endswith(".gguf"):
        raise ValueError(
            f"GGUF files are no longer supported: {model_path}. "
            "Please use .safetensors or .pt models instead."
        )

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


def load_model_for_pipeline(model_path: str = None):
    """Load model artifacts appropriate for the detected model type.

    Returns a tuple: (model_type, loader_result)
    loader_result is the existing loader return for that model (varies by type)
    
    Note: GGUF/FLUX support has been removed. Only SD1.5 and SDXL models are supported.
    """
    # Determine model type
    model_type = detect_model_type(model_path)

    # SD1.5 / SDXL: use CheckpointLoaderSimple which handles safetensors/pt
    ckpt = model_path or "./include/checkpoints/Meina V10 - baked VAE.safetensors"
    loader = Loader.CheckpointLoaderSimple()
    out = loader.load_checkpoint(ckpt_name=ckpt)
    # detect SDXL specifically and return model type
    model_type = detect_model_type(ckpt)
    return (model_type, out)

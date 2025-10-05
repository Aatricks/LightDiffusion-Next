import os
from typing import Optional

from src.Utilities import util

try:
    import ollama  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ollama = None  # type: ignore

DEFAULT_PREFIX = (
    "masterpiece, best quality, (extremely detailed CG unity 8k wallpaper, "
    "masterpiece, best quality, ultra-detailed, best shadow), high contrast, "
    "(best illumination), ((cinematic light)), hyper detail, dramatic light, "
    "depth of field"
)


def _resolve_prompt(source_prompt: Optional[str]) -> str:
    try:
        saved_prompt = util.load_parameters_from_file()[0]
    except Exception:
        saved_prompt = ""
    return source_prompt if source_prompt else saved_prompt


def _extract_content(raw: str) -> str:
    if "</think>" in raw:
        raw = raw.split("</think>")[-1]
    return raw.strip().strip('"')


def enhance_prompt(p: Optional[str]) -> str:
    """Enhance a prompt using an Ollama model when available."""

    prompt = _resolve_prompt(p)

    if not prompt or ollama is None:
        if ollama is None:
            print("Prompt enhancer skipped: Ollama client not available.")
        return prompt

    model = os.environ.get("PROMPT_ENHANCER_MODEL", "qwen3:0.6b").strip() or "qwen3:0.6b"
    prefix = os.environ.get("PROMPT_ENHANCER_PREFIX", DEFAULT_PREFIX).strip()

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a prompt composer for text-to-image diffusion models. "
                        "Rewrite the user prompt into a single comma-separated line "
                        "ordered by visual priority. Use short descriptive phrases "
                        "covering image type, mood, lighting, composition, background, "
                        "palette, key objects, artistic style, and subject appearance. "
                        "Wrap especially important clauses in parentheses. Avoid extra "
                        "commentary or labels. User prompt: "
                        f"{prompt}"
                    ),
                }
            ],
        )
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"Prompt enhancer failed ({exc}); returning original prompt.")
        return prompt

    content = response.get("message", {}).get("content", "").strip()
    if not content:
        print("Prompt enhancer returned empty response; using original prompt.")
        return prompt

    enhanced = _extract_content(content)
    if not enhanced:
        return prompt

    if prefix:
        if not prefix.endswith(","):
            prefix = prefix + ","
        return f"{prefix} {enhanced}".strip()

    return enhanced

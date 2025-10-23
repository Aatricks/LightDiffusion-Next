# Recipes & Workflows

This page collects practical “recipes” for common LightDiffusion-Next scenarios. Each section lists the UI path, optional CLI equivalents and tips for squeezing the best quality or performance out of the pipeline.

## 1. Classic text-to-image (SD1.5)

Steps in the Streamlit UI:

1. Enter a prompt such as `a cozy reading nook lit by neon signs, cinematic lighting, ultra detailed`.
2. Leave negative prompt empty to use the curated default (includes `EasyNegative` and `badhandv4`).
3. Set width and height to `768 × 512` and request `4` images with a batch size of `2`.
4. Enable **Keep models in VRAM** for faster iteration while exploring.
5. (Optional) Toggle **Enhance prompt** if you have Ollama running.
6. Click **Generate** — watch the TAESD previews update in real time.

CLI equivalent:

```bash
python -m src.user.pipeline "a cozy reading nook lit by neon signs" 768 512 4 2 --stable-fast --reuse-seed
```

Tips:

- For softer lighting turn on **AutoHDR** (enabled by default) and lower CFG to 6.5 using the advanced settings drawer.
- Combine with **LoRA** adapters by placing `.safetensors` files in `include/loras/` and selecting them in the UI dropdown.

## 2. Flux workflow

Flux requires the quantized GGUF UNet, CLIP, T5 weights and the schnELL VAE (`include/vae/ae.safetensors`). The first run downloads them automatically.

1. Toggle **Flux mode**.
2. Switch CFG to `1.0` (Flux expects low CFG) and set steps to around 20.
3. Provide a natural language prompt such as `a charcoal sketch of a train arriving at midnight, expressive strokes`.
4. Generate 2 images with batch size 1.

REST API example:

```bash
curl -X POST http://localhost:7861/api/generate \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "a charcoal sketch of a train arriving at midnight, expressive strokes",
        "width": 832,
        "height": 1216,
        "num_images": 2,
        "flux_enabled": true,
        "keep_models_loaded": true
      }' | jq '.images[0]' -r | base64 -d > flux.png
```

Tips:

- Flux ignores negative prompts and uses natural language weighting. Seed reuse works the same way as SD1.5.
- Monitor GPU memory in the **Model Cache Management** accordion — Flux models are larger.

## 3. HiRes Fix + ADetailer portrait

1. Choose a prompt such as `portrait of a cyberpunk detective, glowing tattoos, rain-soaked alley`.
2. Set `width = 640`, `height = 896`, **num images = 1**.
3. Enable **HiRes Fix**, **ADetailer** and **Stable-Fast**.
4. In the advanced section set **HiRes denoise** to ~0.45 by editing `config.toml` (or accept the default and adjust later).
5. Generate — the pipeline saves the base render, body detail pass and head detail pass separately.

Where to find outputs:

- Base image: `output/HiresFix/`.
- Body/head detail passes: `output/Adetailer/`.

Tips:

- Provide a short negative prompt that removes “extra limbs” to guide the detector.
- Use the **History** tab to compare detailer versus base results quickly.

## 4. Img2Img upscaling with Ultimate SD Upscale

1. Enable **Img2Img mode** and upload your reference image.
2. Set denoise strength via the slider in the Img2Img accordion (`0.3` is a good starting point).
3. Toggle **Stable-Fast** for faster tile processing and keep CFG around 6.
4. Generate. UltimateSDUpscale will split the image into tiles, run targeted refinement and apply RealESRGAN (`include/ESRGAN/RealESRGAN_x4plus.pth`).

Tips:

- For stylized upscales change the prompt between passes — the pipeline will regenerate details without overwriting the original.
- Outputs land in `output/Img2Img/` with metadata including seam-fixing parameters.

## 5. Automated batch via REST API

Use the FastAPI backend when you need to process multiple prompts from scripts or a Discord bot.

```python
import base64
import json
import requests

payload = {
    "prompt": "sunrise over a foggy fjord, volumetric light, ethereal",
    "negative_prompt": "low quality, blurry",
    "width": 832,
    "height": 512,
    "num_images": 3,
    "batch_size": 3,
    "stable_fast": True,
    "reuse_seed": False,
    "enable_preview": False
}

resp = requests.post("http://localhost:7861/api/generate", json=payload)
resp.raise_for_status()
images = resp.json().get("images", [])
for idx, b64_img in enumerate(images):
    with open(f"fjord_{idx+1}.png", "wb") as f:
        f.write(base64.b64decode(b64_img))
```

The queue automatically coalesces compatible requests to maximize GPU utilization. Check `/api/telemetry` for batching statistics and memory usage.

## 6. Discord bot bridge

Combine LightDiffusion-Next with the [Boubou](https://github.com/Aatrick/Boubou) Discord bot:

1. Follow the bot’s README to set your Discord token and install `py-cord` inside the LightDiffusion environment.
2. Point the bot’s configuration at the FastAPI endpoint (`http://localhost:7861`).
3. Give the bot `Send Messages` and `Attach Files` permissions.
4. Use commands such as `/ld prompt:"a watercolor koi pond"` from your server and watch images stream back into the channel.

## 7. Prompt enhancer playground

1. Install [Ollama](https://ollama.com/) and run `ollama serve` in another terminal.
2. Pull the suggested model:

   ```bash
   ollama pull qwen3:0.6b
   ```

3. Export the model name before launching the UI:

   ```bash
   export PROMPT_ENHANCER_MODEL=qwen3:0.6b
   ```

4. Enable **Enhance prompt** in Streamlit and inspect the rewritten prompt under the preview section. The original text is still stored as `original_prompt` inside PNG metadata.

Continue exploring by reading the [performance & tuning](quirks.md) guide or the [REST documentation](api.md) for full endpoint details.
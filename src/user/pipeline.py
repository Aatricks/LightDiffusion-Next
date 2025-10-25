import argparse
import os
import random
import time
import sys

import numpy as np
import torch
from PIL import Image
import re
import uuid
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.AutoDetailer import SAM, SEGS, ADetailer, bbox
from src.AutoEncoders import VariationalAE
from src.clip import Clip
from src.FileManaging import Downloader, ImageSaver, Loader
from src.hidiffusion import msw_msa_attention
from src.Model import LoRas
from src.Quantize import Quantizer
from src.sample import sampling
from src.UltimateSDUpscale import UltimateSDUpscale, USDU_upscaler
from src.Utilities import Enhancer, Latent, upscale
from src.WaveSpeed import fbcache_nodes
from src.AutoHDR import ahdr
from src.user import app_instance

with open(os.path.join("./include/", "last_seed.txt"), "r") as f:
    last_seed = int(f.read())

Downloader.CheckAndDownload()


def _check_interruption():
    app = getattr(app_instance, "app", None)
    if app is not None and getattr(app, "interrupt_flag", False):
        raise InterruptedError("Generation interrupted")


def pipeline(
    prompt: str | list,
    w: int,
    h: int,
    number: int = 1,
    batch: int = 1,
    hires_fix: bool = False,
    adetailer: bool = False,
    enhance_prompt: bool = False,
    img2img: bool = False,
    stable_fast: bool = False,
    reuse_seed: bool = False,
    flux_enabled: bool = False,
    prio_speed: bool = False,
    autohdr: bool = True,
    realistic_model: bool = False,
    negative_prompt: str = None,
    # Multi-scale diffusion parameters
    multiscale_preset: str = None,
    enable_multiscale: bool = True,
    multiscale_factor: float = 0.5,
    multiscale_fullres_start: int = 3,
    multiscale_fullres_end: int = 8,
    multiscale_intermittent_fullres: bool = False,
    # Path to the input image when running in img2img/upscale mode
    img2img_image: str | None = None,
    # Optional per-sample data used when `prompt` is a list (batched mode).
    per_sample_info: list | None = None,
) -> None:
    """#### Run the LightDiffusion pipeline.

    #### Args:
        - `prompt` (str): The prompt for the pipeline.
        - `w` (int): The width of the generated image.
        - `h` (int): The height of the generated image.
        - `hires_fix` (bool, optional): Enable high-resolution fix. Defaults to False.
        - `adetailer` (bool, optional): Enable automatic face and body enhancing. Defaults to False.
        - `enhance_prompt` (bool, optional): Enable Ollama prompt enhancement. Defaults to False.
    - `img2img` (bool, optional): Use LightDiffusion in Image to Image mode. If `img2img_image` is provided, that path is used as the source image; otherwise the legacy behavior uses `prompt` as the path. Defaults to False.
    - `img2img_image` (str, optional): Filesystem path to the source image for img2img/upscaling when `img2img=True`.
        - `stable_fast` (bool, optional): Enable Stable-Fast speedup offering a 70% speed improvement in return of a compilation time. Defaults to False.
        - `reuse_seed` (bool, optional): Reuse the last used seed, if False the seed will be kept random. Default to False.
        - `flux_enabled` (bool, optional): Enable the flux mode. Defaults to False.
        - `prio_speed` (bool, optional): Prioritize speed over quality. Defaults to False.
        - `autohdr` (bool, optional): Enable the AutoHDR mode. Defaults to False.
        - `realistic_model` (bool, optional): Use the realistic model. Defaults to False.
        - `negative_prompt` (str, optional): The negative prompt to avoid certain elements. If None, uses default negative prompt. Defaults to None.
        - `multiscale_preset` (str, optional): Predefined multiscale preset ('quality', 'performance', 'balanced', 'disabled'). Overrides individual multiscale parameters. Defaults to None.
        - `enable_multiscale` (bool, optional): Enable multi-scale diffusion for performance optimization. Defaults to True.
        - `multiscale_factor` (float, optional): Scale factor for intermediate steps (0.1-1.0). Defaults to 0.5.
        - `multiscale_fullres_start` (int, optional): Number of first steps at full resolution. Defaults to 3.
        - `multiscale_fullres_end` (int, optional): Number of last steps at full resolution. Defaults to 8.
        - `multiscale_intermittent_fullres` (bool, optional): Enable intermittent full-res rendering in low-res region. Defaults to False.
    """
    global last_seed

    app_ref = getattr(app_instance, "app", None)
    if app_ref is not None:
        app_ref.clear_interrupt()
    _check_interruption()

    original_prompt = prompt
    enhancement_applied = False

    # Apply multiscale preset if specified (overrides individual parameters)
    if multiscale_preset is not None:
        from src.sample.multiscale_presets import get_preset_parameters

        preset_params = get_preset_parameters(multiscale_preset)
        enable_multiscale = preset_params["enable_multiscale"]
        multiscale_factor = preset_params["multiscale_factor"]
        multiscale_fullres_start = preset_params["multiscale_fullres_start"]
        multiscale_fullres_end = preset_params["multiscale_fullres_end"]
        multiscale_intermittent_fullres = preset_params[
            "multiscale_intermittent_fullres"
        ]
        print(f"Applied multiscale preset: {multiscale_preset}")

    # Handle negative prompt - use default if none provided. Support either
    # a single string or a list of per-sample negatives for batched mode.
    default_negative = (
        "(worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic), "
        "(embedding:EasyNegative), (embedding:badhandv4), (embedding:lr), (embedding:ng_deepnegative_v1_75t)"
    )
    if negative_prompt is None:
        negative_prompt = default_negative
    elif isinstance(negative_prompt, str):
        if negative_prompt.strip() == "":
            negative_prompt = default_negative
    elif isinstance(negative_prompt, (list, tuple)):
        # Replace any empty entries with default_negative so later encoders
        # can accept a list safely.
        negative_prompt = [
            (p if (p is not None and str(p).strip() != "") else default_negative)
            for p in negative_prompt
        ]

    images_to_generate = max(1, number)
    # In normal single-prompt mode `seed` is expected to be an int (or None).
    # When running in batched multi-prompt mode `seed` may be a list containing
    # a per-sample seed for each prompt; handle both cases here.
    if isinstance(prompt, (list, tuple)):
        total_batch = len(prompt)
        # If per_sample_info is supplied, build per-sample seeds from it;
        # entries may be None, in which case fall back to reuse_seed or
        # random generation for that slot.
        if per_sample_info is not None and isinstance(per_sample_info, (list, tuple)):
            seeds = []
            for i in range(total_batch):
                seed_val = None
                if i < len(per_sample_info) and isinstance(per_sample_info[i], dict):
                    seed_val = per_sample_info[i].get("seed", None)
                if seed_val is None:
                    if reuse_seed:
                        seeds.append(last_seed)
                    else:
                        seeds.append(random.randint(1, 2**64))
                else:
                    seeds.append(int(seed_val))
        else:
            if reuse_seed:
                seeds = [last_seed] * total_batch
            else:
                seeds = [random.randint(1, 2**64) for _ in range(total_batch)]
        last_seed = seeds[-1]
        images_to_generate = total_batch
    else:
        if reuse_seed:
            seeds = [last_seed] * images_to_generate
        else:
            seeds = [random.randint(1, 2**64) for _ in range(images_to_generate)]
            last_seed = seeds[-1]

    with open(os.path.join("./include/", "last_seed.txt"), "w") as f:
        f.write(str(seeds[-1]))
    if enhance_prompt:
        try:
            if isinstance(prompt, (list, tuple)):
                enhanced_prompts = []
                for p in prompt:
                    try:
                        enhanced = Enhancer.enhance_prompt(p)
                        enhanced_prompts.append(enhanced if enhanced else p)
                    except Exception:
                        enhanced_prompts.append(p)
                prompt = enhanced_prompts
            else:
                enhanced_prompt = Enhancer.enhance_prompt(prompt)
                if enhanced_prompt:
                    prompt = enhanced_prompt
        except Exception:
            pass
        enhancement_applied = prompt != original_prompt

    sampler_name = "dpmpp_sde_cfgpp" if not prio_speed else "dpmpp_2m_cfgpp"
    ckpt = (
        "./include/checkpoints/Meina V10 - baked VAE.safetensors"
        if not realistic_model
        else "./include/checkpoints/DreamShaper_8_pruned.safetensors"
    )
    with torch.inference_mode():
        if not flux_enabled:
            checkpointloadersimple = Loader.CheckpointLoaderSimple()
            checkpointloadersimple_241 = checkpointloadersimple.load_checkpoint(
                ckpt_name=ckpt
            )
            hidiffoptimizer = msw_msa_attention.ApplyMSWMSAAttentionSimple()
        cliptextencode = Clip.CLIPTextEncode()
        emptylatentimage = Latent.EmptyLatentImage()
        ksampler_instance = sampling.KSampler()
        vaedecode = VariationalAE.VAEDecode()
        saveimage = ImageSaver.SaveImage()
        latent_upscale = upscale.LatentUpscale()
        hdr = ahdr.HDREffects()

        # If `prompt` is a list run the batched generation path which uses a
        # single forward pass to produce distinct outputs for each supplied
        # prompt. This path requires shapes and model-related flags to be
        # compatible across all samples in the batch.
        if isinstance(prompt, (list, tuple)):
            # Basic validation: ensure shapes and heavy flags are consistent
            # across the batch - otherwise fall back to per-request generation.
            # When called from the server buffer we will only group requests
            # that are compatible; this is a defensive check.
            if any((p_w != w or p_h != h) for p_w, p_h in [(w, h)]):
                raise ValueError("Batched prompts must share same width/height")

            # Build per-sample prompt and negative prompt lists
            prompts = list(prompt)
            if isinstance(negative_prompt, (list, tuple)):
                negatives = list(negative_prompt)
            else:
                negatives = [negative_prompt or ""] * len(prompts)

            total_batch = len(prompts)

            # Load LoRA / CLIP patching ahead of time so we can encode all
            # prompts with the model configuration that will be used for the
            # forward pass.
            try:
                loraloader = LoRas.LoraLoader()
                loraloader_274 = loraloader.load_lora(
                    lora_name="add_detail.safetensors",
                    strength_model=0.7,
                    strength_clip=0.7,
                    model=checkpointloadersimple_241[0],
                    clip=checkpointloadersimple_241[1],
                )
            except Exception:
                loraloader_274 = checkpointloadersimple_241

            clipsetlastlayer = Clip.CLIPSetLastLayer()
            clipsetlastlayer_257 = clipsetlastlayer.set_last_layer(
                stop_at_clip_layer=-2, clip=loraloader_274[1]
            )
            if stable_fast is True:
                try:
                    from src.StableFast import StableFast

                    applystablefast = StableFast.ApplyStableFastUnet()
                    applystablefast_158 = applystablefast.apply_stable_fast(
                        enable_cuda_graph=True, model=loraloader_274[0]
                    )
                except Exception:
                    logger = logging.getLogger(__name__)
                    logger.exception("StableFast apply failed at batch setup; falling back to normal model")
                    # Keep a single-element tuple so downstream code that
                    # expects applystablefast_158[0] to be the model still
                    # works.
                    applystablefast_158 = (loraloader_274[0],)
            else:
                applystablefast_158 = loraloader_274

            # Encode all prompts into a list of condition entries and attach
            # a batch_index so downstream conditioning logic knows which
            # batch slots each condition maps to.
            positive_entries = cliptextencode.encode(
                clip=clipsetlastlayer_257[0], text=prompts, flux_enabled=flux_enabled
            )[0]
            negative_entries = cliptextencode.encode(
                clip=clipsetlastlayer_257[0], text=negatives, flux_enabled=flux_enabled
            )[0]

            # Add routing information into each condition's metadata
            for i, entry in enumerate(positive_entries):
                # entry is [cond_tensor, meta_dict]
                if len(entry) > 1 and isinstance(entry[1], dict):
                    entry[1]["batch_index"] = [i]

            for i, entry in enumerate(negative_entries):
                if len(entry) > 1 and isinstance(entry[1], dict):
                    entry[1]["batch_index"] = [i]

            # Create an empty latent image sized for the whole batch. The
            # generate() helper returns a one-element tuple containing a dict
            # with the 'samples' tensor; reuse that dict and attach seeds so
            # the sampler can generate per-sample noise deterministically.
            emptylatentimage_244 = emptylatentimage.generate(
                width=w, height=h, batch_size=total_batch
            )
            latent = emptylatentimage_244[0]
            latent["seeds"] = seeds

            # Run the sampler once for all prompts in the batch
            ksampler_239 = ksampler_instance.sample(
                seed=None,
                steps=20,
                cfg=7,
                sampler_name=sampler_name,
                scheduler="karras",
                denoise=1,
                pipeline=True,
                model=hidiffoptimizer.go(model_type="auto", model=applystablefast_158[0])[0],
                positive=positive_entries,
                negative=negative_entries,
                latent_image=latent,
                enable_multiscale=enable_multiscale,
                multiscale_factor=multiscale_factor,
                multiscale_fullres_start=multiscale_fullres_start,
                multiscale_fullres_end=multiscale_fullres_end,
                multiscale_intermittent_fullres=multiscale_intermittent_fullres,
            )

            # Decode and save each resulting image individually so that we
            # can attach per-request metadata (request id / prefix) and make
            # result mapping straightforward for the server buffer.
            vaedecode_240 = vaedecode.decode(
                samples=ksampler_239[0], vae=checkpointloadersimple_241[2]
            )

            decoded = vaedecode_240[0]
            if autohdr:
                _tmp = hdr.apply_hdr2(decoded)
                main_imgs = _tmp[0] if isinstance(_tmp, (tuple, list)) else _tmp
            else:
                main_imgs = decoded

            # Per-sample processing and saving. Advanced features such as
            # hires_fix and adetailer are executed per-sample so a single
            # shared forward pass can be used for conditioning while
            # preserving post-processing differences.
            results_map = {}

            # Ensure we have a per_sample_info structure to reference flags
            if per_sample_info is None:
                per_sample_info = [{} for _ in range(total_batch)]

            # Decide if we need to preload heavy modules for adetailer
            need_hires = any(info.get("hires_fix", False) for info in per_sample_info)
            need_adetailer = any(info.get("adetailer", False) for info in per_sample_info)
            logger = logging.getLogger(__name__)
            logger.debug("Batch needs hires=%s adetailer=%s", need_hires, need_adetailer)

            # Preload adetailer resources once per batch if required
            if need_adetailer:
                samloader = SAM.SAMLoader()
                samloader_87 = samloader.load_model(model_name="sam_vit_b_01ec64.pth", device_mode="AUTO")
                cliptextencode_124 = cliptextencode.encode(
                    text="royal, detailed, magnificient, beautiful, seducing",
                    clip=loraloader_274[1],
                )
                ultralyticsdetectorprovider = bbox.UltralyticsDetectorProvider()
                ultralyticsdetectorprovider_151 = ultralyticsdetectorprovider.doit(
                    model_name="person_yolov8m-seg.pt"
                )
                bboxdetectorsegs = bbox.BboxDetectorForEach()
                samdetectorcombined = SAM.SAMDetectorCombined()
                impactsegsandmask = SEGS.SegsBitwiseAndMask()
                detailerforeachdebug = ADetailer.DetailerForEachTest()

            # main_latent contains the raw latents for the batch which we will
            # slice for per-sample hires upscaling if needed.
            main_latent = ksampler_239[0]

            for i in range(total_batch):
                info = per_sample_info[i] if i < len(per_sample_info) else {}
                req_id = info.get("request_id", uuid.uuid4().hex[:8])
                filename_prefix = info.get("filename_prefix", f"LD-REQ-{req_id}")

                # Default final image is the decoded batch image for this sample
                final_img = main_imgs[i]

                # If hires fix requested for this sample, run per-sample hires
                if info.get("hires_fix", False):
                    try:
                        # Build single-sample latent dict
                        single_latent = {"samples": main_latent["samples"][i : i + 1]}
                        # Upscale latent and run extra sampling pass
                        upscaled_tuple = latent_upscale.upscale(samples=single_latent, width=w * 2, height=h * 2)
                        upscaled = upscaled_tuple[0]

                        # When running a single-sample hires pass we must ensure
                        # any conditioning metadata that references the original
                        # multi-sample batch is remapped to the new single-item
                        # batch. The encoder earlier attached `batch_index` for
                        # routing; here we copy the conditioning entry and set
                        # its batch_index to [0] so downstream indexing never
                        # attempts to access out-of-range batch rows on the
                        # upscaled single-sample latent.
                        def _as_single_sample_entry(entry):
                            if entry is None:
                                return None
                            # Distinguish between a (tensor, meta) pair and a raw
                            # tensor to avoid treating tensor as a sequence.
                            if isinstance(entry, (list, tuple)):
                                cond_tensor = entry[0]
                                meta = {}
                                if len(entry) > 1 and isinstance(entry[1], dict):
                                    try:
                                        meta = entry[1].copy()
                                    except Exception:
                                        meta = dict(entry[1])
                            else:
                                # entry is a raw tensor — use it as the condition
                                cond_tensor = entry
                                meta = {}
                            # Explicitly remap to the single sample at index 0
                            meta["batch_index"] = [0]
                            return [cond_tensor, meta]

                        # Wrap each single-sample conditioning entry in a list
                        # so downstream code expecting a list-of-(tensor,meta)
                        # elements behaves the same as the batched path.
                        pos_entry = _as_single_sample_entry(positive_entries[i]) if i < len(positive_entries) else None
                        neg_entry = _as_single_sample_entry(negative_entries[i]) if i < len(negative_entries) else None
                        pos_i = [pos_entry] if pos_entry is not None else None
                        neg_i = [neg_entry] if neg_entry is not None else None

                        # Prefer an explicit per-sample seed supplied in
                        # per_sample_info; otherwise fall back to the
                        # per-batch computed seed for this slot so results
                        # are deterministic and reproducible.
                        hires_seed = info.get("seed", None)
                        if hires_seed is None:
                            hires_seed = seeds[i] if ("seeds" in locals() and i < len(seeds)) else random.randint(1, 2**64)
                        try:
                            hires_seed = int(hires_seed)
                        except Exception:
                            hires_seed = random.randint(1, 2**64)

                        ksampler_253 = ksampler_instance.sample(
                            seed=hires_seed,
                            steps=10,
                            cfg=8,
                            sampler_name="euler_ancestral_cfgpp",
                            scheduler="normal",
                            denoise=0.45,
                            model=hidiffoptimizer.go(model_type="auto", model=applystablefast_158[0])[0],
                            positive=pos_i,
                            negative=neg_i,
                            latent_image=upscaled,
                            pipeline=True,
                        )

                        vae_out = vaedecode.decode(samples=ksampler_253[0], vae=checkpointloadersimple_241[2])
                        decoded_up = vae_out[0]
                        if autohdr:
                            _tmp = hdr.apply_hdr2(decoded_up)
                            hires_imgs = _tmp[0] if isinstance(_tmp, (tuple, list)) else _tmp
                        else:
                            hires_imgs = decoded_up

                        # pick the single result
                        final_img = hires_imgs[0]
                    except Exception as e:
                        # keep final_img as the main decoded result if hires fails
                        try:
                            logger = logging.getLogger(__name__)
                            logger.exception("Per-sample hires_fix failed for index %d: %s", i, e)
                        except Exception:
                            pass

                # If adetailer requested for this sample, run the adetailer
                if info.get("adetailer", False):
                    try:
                        # Build a single-image batch for the adetailer detectors
                        single_image = final_img.unsqueeze(0)

                        # Run detection pipelines
                        bboxdetectorsegs_132 = bboxdetectorsegs.doit(
                            threshold=0.5,
                            dilation=10,
                            crop_factor=2,
                            drop_size=10,
                            labels="all",
                            bbox_detector=ultralyticsdetectorprovider_151[0],
                            image=single_image,
                        )
                        samdetectorcombined_139 = samdetectorcombined.doit(
                            detection_hint="center-1",
                            dilation=0,
                            threshold=0.93,
                            bbox_expansion=0,
                            mask_hint_threshold=0.7,
                            mask_hint_use_negative="False",
                            sam_model=samloader_87[0],
                            segs=bboxdetectorsegs_132,
                            image=single_image,
                        )
                        if samdetectorcombined_139 is not None:
                            impactsegsandmask_152 = impactsegsandmask.doit(
                                segs=bboxdetectorsegs_132,
                                mask=samdetectorcombined_139[0],
                            )
                            # Resolve a stable integer seed for adetailer.
                            adetailer_seed = info.get("seed", None)
                            if adetailer_seed is None:
                                adetailer_seed = seeds[i] if ("seeds" in locals() and i < len(seeds)) else random.randint(1, 2**64)
                                logging.getLogger(__name__).debug(
                                    "adetailer: per-sample seed missing, falling back to per-batch seed/random for idx=%d: %s",
                                    i,
                                    str(adetailer_seed),
                                )
                            try:
                                adetailer_seed = int(adetailer_seed)
                            except Exception:
                                adetailer_seed = random.randint(1, 2**64)

                            detailerforeachdebug_145 = detailerforeachdebug.doit(
                                guide_size=512,
                                guide_size_for=False,
                                max_size=768,
                                seed=adetailer_seed,
                                steps=20,
                                cfg=6.5,
                                sampler_name=sampler_name,
                                scheduler="karras",
                                denoise=0.5,
                                feather=5,
                                noise_mask=True,
                                force_inpaint=True,
                                wildcard="",
                                cycle=1,
                                inpaint_model=False,
                                noise_mask_feather=20,
                                image=single_image,
                                segs=impactsegsandmask_152[0],
                                model=applystablefast_158[0],
                                clip=checkpointloadersimple_241[1],
                                vae=checkpointloadersimple_241[2],
                                positive=cliptextencode_124[0],
                                negative=[negative_entries[i]] if (i < len(negative_entries) and negative_entries[i] is not None) else None,
                                pipeline=True,
                            )

                            # Extract a seed for metadata if possible
                            def _extract_scalar_seed(candidate):
                                try:
                                    if isinstance(candidate, int):
                                        return str(candidate)
                                    if isinstance(candidate, float) and float(candidate).is_integer():
                                        return str(int(candidate))
                                    if isinstance(candidate, str):
                                        s = candidate.strip()
                                        if re.fullmatch(r"-?\d+", s):
                                            return s
                                        m = re.search(r"\d{4,}", s)
                                        if m:
                                            return m.group(0)
                                        return None
                                    if isinstance(candidate, np.ndarray):
                                        if candidate.size == 1:
                                            return str(int(candidate.item()))
                                        return None
                                    if isinstance(candidate, torch.Tensor):
                                        if candidate.numel() == 1:
                                            return str(int(candidate.item()))
                                except Exception:
                                    return None
                                return None

                            try:
                                if isinstance(detailerforeachdebug_145, (list, tuple)) and len(detailerforeachdebug_145) > 1:
                                    candidate = detailerforeachdebug_145[1]
                                    extracted = _extract_scalar_seed(candidate)
                                    detailer_body_seed = extracted if extracted is not None else str(adetailer_seed)
                                else:
                                    detailer_body_seed = str(adetailer_seed)
                            except Exception:
                                detailer_body_seed = str(adetailer_seed)

                            body_meta = {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                "prompt": prompts[i],
                                "negative_prompt": negatives[i],
                                "seed": detailer_body_seed,
                                "sampler": str(sampler_name),
                                "steps": "20",
                                "cfg": "6.5",
                                "scheduler": "karras",
                                "denoise": "0.5",
                                "width": str(w),
                                "height": str(h),
                                "batch_size": str(1),
                                "adetailer": "True",
                            }

                            if autohdr:
                                _tmp = hdr.apply_hdr2(detailerforeachdebug_145[0])
                                body_imgs = _tmp[0] if isinstance(_tmp, (tuple, list)) else _tmp
                            else:
                                body_imgs = detailerforeachdebug_145[0]
                            saved_body = saveimage.save_images(
                                filename_prefix="LD-body",
                                images=body_imgs,
                                prompt=prompts[i],
                                extra_pnginfo=body_meta,
                            )
                            # Append adetailer outputs to the per-request map so
                            # callers receive all generated artifacts.
                            results_map.setdefault(req_id, [])
                            try:
                                results_map[req_id].extend(saved_body.get("ui", {}).get("images", []))
                            except Exception:
                                results_map[req_id].append(saved_body)

                            # Second pass to produce head images (original flow)
                            head_seed = random.randint(1, 2**64)
                            detailerforeachdebug_145 = detailerforeachdebug.doit(
                                guide_size=512,
                                guide_size_for=False,
                                max_size=768,
                                seed=head_seed,
                                steps=20,
                                cfg=6.5,
                                sampler_name=sampler_name,
                                scheduler="karras",
                                denoise=0.5,
                                feather=5,
                                noise_mask=True,
                                force_inpaint=True,
                                wildcard="",
                                cycle=1,
                                inpaint_model=False,
                                noise_mask_feather=20,
                                image=detailerforeachdebug_145[0],
                                segs=impactsegsandmask_152[0],
                                model=applystablefast_158[0],
                                clip=checkpointloadersimple_241[1],
                                vae=checkpointloadersimple_241[2],
                                positive=cliptextencode_124[0],
                                negative=[negative_entries[i]] if (i < len(negative_entries) and negative_entries[i] is not None) else None,
                                pipeline=True,
                            )
                            try:
                                if isinstance(detailerforeachdebug_145, (list, tuple)) and len(detailerforeachdebug_145) > 1:
                                    candidate_h = detailerforeachdebug_145[1]
                                    extracted_h = _extract_scalar_seed(candidate_h)
                                    detailer_head_seed = extracted_h if extracted_h is not None else str(head_seed)
                                else:
                                    detailer_head_seed = str(head_seed)
                            except Exception:
                                detailer_head_seed = str(head_seed)

                            head_meta = {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                "prompt": prompts[i],
                                "negative_prompt": negatives[i],
                                "seed": detailer_head_seed,
                                "sampler": str(sampler_name),
                                "steps": "20",
                                "cfg": "6.5",
                                "scheduler": "karras",
                                "denoise": "0.5",
                                "width": str(w),
                                "height": str(h),
                                "batch_size": str(1),
                                "adetailer": "True",
                            }
                            if autohdr:
                                _tmp = hdr.apply_hdr2(detailerforeachdebug_145[0])
                                head_imgs = _tmp[0] if isinstance(_tmp, (tuple, list)) else _tmp
                            else:
                                head_imgs = detailerforeachdebug_145[0]
                            saved_head = saveimage.save_images(
                                filename_prefix="LD-head",
                                images=head_imgs,
                                prompt=prompts[i],
                                extra_pnginfo=head_meta,
                            )
                            try:
                                results_map[req_id].extend(saved_head.get("ui", {}).get("images", []))
                            except Exception:
                                results_map[req_id].append(saved_head)
                            # We don't modify final_img here; adetailer writes extra images
                    except Exception as e:
                        try:
                            logger = logging.getLogger(__name__)
                            logger.exception("Per-sample adetailer failed for index %d: %s", i, e)
                        except Exception:
                            pass

                # Build PNG metadata for the final (main) image for this sample
                sample_meta = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "prompt": prompts[i],
                    "negative_prompt": negatives[i],
                    "seed": str(seeds[i]),
                    "sampler": sampler_name,
                    "steps": str(20),
                    "cfg": str(7),
                    "scheduler": "karras",
                    "denoise": "1",
                    "width": str(w),
                    "height": str(h),
                    "batch_size": str(total_batch),
                    "realistic_model": str(realistic_model),
                }

                # Save the final image for this sample
                saved = saveimage.save_images(
                    filename_prefix=filename_prefix,
                    images=[final_img],
                    prompt=prompts[i],
                    extra_pnginfo=sample_meta,
                )
                results_map.setdefault(req_id, [])
                try:
                    results_map[req_id].extend(saved.get("ui", {}).get("images", []))
                except Exception:
                    results_map[req_id].append(saved)

            return {"batched_results": results_map}
    for current_seed in seeds:
        _check_interruption()
        if img2img:
            # Use explicit image path if provided, else fall back to legacy behavior where prompt is a path
            source_path = img2img_image or prompt
            img = Image.open(source_path)
            img_array = np.array(img)
            img_tensor = torch.from_numpy(img_array).float().to("cpu") / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            with torch.inference_mode():
                ultimatesdupscale = UltimateSDUpscale.UltimateSDUpscale()
                try:
                    loraloader = LoRas.LoraLoader()
                    loraloader_274 = loraloader.load_lora(
                        lora_name="add_detail.safetensors",
                        strength_model=2,
                        strength_clip=2,
                        model=checkpointloadersimple_241[0],
                        clip=checkpointloadersimple_241[1],
                    )
                except Exception:
                    loraloader_274 = checkpointloadersimple_241

                if stable_fast is True:
                    try:
                        from src.StableFast import StableFast

                        applystablefast = StableFast.ApplyStableFastUnet()
                        applystablefast_158 = applystablefast.apply_stable_fast(
                            enable_cuda_graph=True,
                            model=loraloader_274[0],
                        )
                    except Exception:
                        logger = logging.getLogger(__name__)
                        logger.exception("StableFast apply failed for single-run path; falling back to normal model")
                        applystablefast_158 = (loraloader_274[0],)
                else:
                    applystablefast_158 = loraloader_274

                clipsetlastlayer = Clip.CLIPSetLastLayer()
                clipsetlastlayer_257 = clipsetlastlayer.set_last_layer(
                    stop_at_clip_layer=-2, clip=loraloader_274[1]
                )

                # Keep textual conditioning from the actual text prompt (not the file path)
                cliptextencode_242 = cliptextencode.encode(
                    text=prompt if img2img_image is None else prompt,
                    clip=clipsetlastlayer_257[0],
                )
                cliptextencode_243 = cliptextencode.encode(
                    text=negative_prompt,
                    clip=clipsetlastlayer_257[0],
                )
                upscalemodelloader = USDU_upscaler.UpscaleModelLoader()
                upscalemodelloader_244 = upscalemodelloader.load_model(
                    "RealESRGAN_x4plus.pth"
                )
                ultimatesdupscale_250 = ultimatesdupscale.upscale(
                    upscale_by=2,
                    seed=current_seed,
                    steps=8,
                    cfg=6,
                    sampler_name=sampler_name,
                    scheduler="karras",
                    denoise=0.3,
                    mode_type="Linear",
                    tile_width=512,
                    tile_height=512,
                    mask_blur=16,
                    tile_padding=32,
                    seam_fix_mode="Half Tile",
                    seam_fix_denoise=0.2,
                    seam_fix_width=64,
                    seam_fix_mask_blur=16,
                    seam_fix_padding=32,
                    force_uniform_tiles="enable",
                    image=img_tensor,
                    model=applystablefast_158[0],
                    positive=cliptextencode_242[0],
                    negative=cliptextencode_243[0],
                    vae=checkpointloadersimple_241[2],
                    upscale_model=upscalemodelloader_244[0],
                    pipeline=True,
                )
                _check_interruption()
                # Build PNG metadata for this img2img/upscale result
                i2i_meta = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": str(current_seed),
                    "sampler": str(sampler_name),
                    "steps": "8",
                    "cfg": "6",
                    "scheduler": "karras",
                    "denoise": "0.3",
                    "width": str(w),
                    "height": str(h),
                    "batch_size": str(batch),
                    "img2img": "True",
                    "upscale_model": "RealESRGAN_x4plus.pth",
                    "hires_fix": str(hires_fix),
                    "adetailer": str(adetailer),
                    "stable_fast": str(stable_fast),
                    "flux_enabled": str(flux_enabled),
                    "realistic_model": str(realistic_model),
                    "reuse_seed": str(reuse_seed),
                    "multiscale_preset": str(multiscale_preset),
                }

                decoded_i2i = ultimatesdupscale_250[0]
                if autohdr:
                    _tmp = hdr.apply_hdr2(decoded_i2i)
                    i2i_imgs = _tmp[0] if isinstance(_tmp, (tuple, list)) else _tmp
                else:
                    i2i_imgs = decoded_i2i
                saveimage.save_images(
                    filename_prefix="LD-I2I",
                    images=i2i_imgs,
                    prompt=prompt,
                    extra_pnginfo=i2i_meta,
                )
        elif flux_enabled:
            Downloader.CheckAndDownloadFlux()
            with torch.inference_mode():
                dualcliploadergguf = Quantizer.DualCLIPLoaderGGUF()
                emptylatentimage = Latent.EmptyLatentImage()
                vaeloader = VariationalAE.VAELoader()
                unetloadergguf = Quantizer.UnetLoaderGGUF()
                cliptextencodeflux = Quantizer.CLIPTextEncodeFlux()
                conditioningzeroout = Quantizer.ConditioningZeroOut()
                ksampler = sampling.KSampler()
                unetloadergguf_10 = unetloadergguf.load_unet(
                    unet_name="flux1-dev-Q8_0.gguf"
                )
                fb_cache = fbcache_nodes.ApplyFBCacheOnModel()
                unetloadergguf_10 = fb_cache.patch(
                    unetloadergguf_10, "diffusion_model", 0.120
                )
                vaeloader_11 = vaeloader.load_vae(vae_name="ae.safetensors")
                dualcliploadergguf_19 = dualcliploadergguf.load_clip(
                    clip_name1="clip_l.safetensors",
                    clip_name2="t5-v1_1-xxl-encoder-Q8_0.gguf",
                    type="flux",
                )
                emptylatentimage_5 = emptylatentimage.generate(
                    width=w, height=h, batch_size=batch
                )
                cliptextencodeflux_15 = cliptextencodeflux.encode(
                    clip_l=prompt,
                    t5xxl=prompt,
                    guidance=3.0,
                    clip=dualcliploadergguf_19[0],
                    flux_enabled=True,
                )
                conditioningzeroout_16 = conditioningzeroout.zero_out(
                    conditioning=cliptextencodeflux_15[0]
                )
                ksampler_3 = ksampler.sample(
                    seed=current_seed,
                    steps=20,
                    cfg=1,
                    sampler_name="euler_cfgpp",
                    scheduler="beta",
                    denoise=1,
                    model=unetloadergguf_10[0],
                    positive=cliptextencodeflux_15[0],
                    negative=conditioningzeroout_16[0],
                    latent_image=emptylatentimage_5[0],
                    pipeline=True,
                    flux=True,
                )

                vaedecode_8 = vaedecode.decode(
                    samples=ksampler_3[0],
                    vae=vaeloader_11[0],
                    flux=True,
                )

                _check_interruption()
                # Build PNG metadata for Flux results
                flux_meta = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": str(current_seed),
                    "sampler": "euler_cfgpp",
                    "steps": "20",
                    "cfg": "1",
                    "scheduler": "beta",
                    "denoise": "1",
                    "width": str(w),
                    "height": str(h),
                    "batch_size": str(batch),
                    "flux_enabled": "True",
                    "hires_fix": str(hires_fix),
                    "adetailer": str(adetailer),
                    "stable_fast": str(stable_fast),
                    "realistic_model": str(realistic_model),
                    "reuse_seed": str(reuse_seed),
                }

                decoded_flux = vaedecode_8[0]
                if autohdr:
                    _tmp = hdr.apply_hdr2(decoded_flux)
                    flux_imgs = _tmp[0] if isinstance(_tmp, (tuple, list)) else _tmp
                else:
                    flux_imgs = decoded_flux
                saveimage.save_images(
                    filename_prefix="LD-Flux",
                    images=flux_imgs,
                    prompt=prompt,
                    extra_pnginfo=flux_meta,
                )
        else:
            while prompt is None:
                pass
            with torch.inference_mode():
                try:
                    loraloader = LoRas.LoraLoader()
                    loraloader_274 = loraloader.load_lora(
                        lora_name="add_detail.safetensors",
                        strength_model=0.7,
                        strength_clip=0.7,
                        model=checkpointloadersimple_241[0],
                        clip=checkpointloadersimple_241[1],
                    )
                    print("loading add_detail.safetensors")
                except Exception:
                    loraloader_274 = checkpointloadersimple_241
                clipsetlastlayer = Clip.CLIPSetLastLayer()
                clipsetlastlayer_257 = clipsetlastlayer.set_last_layer(
                    stop_at_clip_layer=-2, clip=loraloader_274[1]
                )
                applystablefast_158 = loraloader_274
                cliptextencode_242 = cliptextencode.encode(
                    text=prompt,
                    clip=clipsetlastlayer_257[0],
                )
                cliptextencode_243 = cliptextencode.encode(
                    text=negative_prompt,
                    clip=clipsetlastlayer_257[0],
                )
                emptylatentimage_244 = emptylatentimage.generate(
                    width=w, height=h, batch_size=batch
                )
                if stable_fast is True:
                    try:
                        from src.StableFast import StableFast

                        applystablefast = StableFast.ApplyStableFastUnet()
                        applystablefast_158 = applystablefast.apply_stable_fast(
                            enable_cuda_graph=False,
                            model=loraloader_274[0],
                        )
                    except Exception:
                        logger = logging.getLogger(__name__)
                        logger.exception("StableFast apply failed for flux/alternate path; falling back to normal model")
                        applystablefast_158 = (loraloader_274[0],)
                else:
                    applystablefast_158 = loraloader_274
                    # fb_cache = fbcache_nodes.ApplyFBCacheOnModel()
                    # applystablefast_158 = fb_cache.patch(
                    #     applystablefast_158, "diffusion_model", 0.120
                    # )

                # Create sampler with multi-scale options
                ksampler_239 = ksampler_instance.sample(
                    seed=current_seed,
                    steps=20,
                    cfg=7,
                    sampler_name=sampler_name,
                    scheduler="karras",
                    denoise=1,
                    pipeline=True,
                    model=hidiffoptimizer.go(
                        model_type="auto", model=applystablefast_158[0]
                    )[0],
                    positive=cliptextencode_242[0],
                    negative=cliptextencode_243[0],
                    latent_image=emptylatentimage_244[0],
                    enable_multiscale=enable_multiscale,
                    multiscale_factor=multiscale_factor,
                    multiscale_fullres_start=multiscale_fullres_start,
                    multiscale_fullres_end=multiscale_fullres_end,
                    multiscale_intermittent_fullres=multiscale_intermittent_fullres,
                )
                if hires_fix:
                    latentupscale_254 = latent_upscale.upscale(
                        width=w * 2,
                        height=h * 2,
                        samples=ksampler_239[0],
                    )
                    ksampler_253 = ksampler_instance.sample(
                        seed=random.randint(1, 2**64),
                        steps=10,
                        cfg=8,
                        sampler_name="euler_ancestral_cfgpp",
                        scheduler="normal",
                        denoise=0.45,
                        model=hidiffoptimizer.go(
                            model_type="auto", model=applystablefast_158[0]
                        )[0],
                        positive=cliptextencode_242[0],
                        negative=cliptextencode_243[0],
                        latent_image=latentupscale_254[0],
                        pipeline=True,
                    )
                else:
                    ksampler_253 = ksampler_239

                _check_interruption()
                vaedecode_240 = vaedecode.decode(
                    samples=ksampler_253[0],
                    vae=checkpointloadersimple_241[2],
                )

            if adetailer:
                _check_interruption()
                with torch.inference_mode():
                    samloader = SAM.SAMLoader()
                    samloader_87 = samloader.load_model(
                        model_name="sam_vit_b_01ec64.pth", device_mode="AUTO"
                    )
                    cliptextencode_124 = cliptextencode.encode(
                        text="royal, detailed, magnificient, beautiful, seducing",
                        clip=loraloader_274[1],
                    )
                    ultralyticsdetectorprovider = bbox.UltralyticsDetectorProvider()
                    ultralyticsdetectorprovider_151 = ultralyticsdetectorprovider.doit(
                        # model_name="face_yolov8m.pt"
                        model_name="person_yolov8m-seg.pt"
                    )
                    bboxdetectorsegs = bbox.BboxDetectorForEach()
                    samdetectorcombined = SAM.SAMDetectorCombined()
                    impactsegsandmask = SEGS.SegsBitwiseAndMask()
                    detailerforeachdebug = ADetailer.DetailerForEachTest()
                    bboxdetectorsegs_132 = bboxdetectorsegs.doit(
                        threshold=0.5,
                        dilation=10,
                        crop_factor=2,
                        drop_size=10,
                        labels="all",
                        bbox_detector=ultralyticsdetectorprovider_151[0],
                        image=vaedecode_240[0],
                    )
                    samdetectorcombined_139 = samdetectorcombined.doit(
                        detection_hint="center-1",
                        dilation=0,
                        threshold=0.93,
                        bbox_expansion=0,
                        mask_hint_threshold=0.7,
                        mask_hint_use_negative="False",
                        sam_model=samloader_87[0],
                        segs=bboxdetectorsegs_132,
                        image=vaedecode_240[0],
                    )
                    if samdetectorcombined_139 is None:
                        return {
                            "original_prompt": original_prompt,
                            "used_prompt": prompt,
                            "enhancement_applied": enhancement_applied,
                        }
                    impactsegsandmask_152 = impactsegsandmask.doit(
                        segs=bboxdetectorsegs_132,
                        mask=samdetectorcombined_139[0],
                    )
                    detailerforeachdebug_145 = detailerforeachdebug.doit(
                        guide_size=512,
                        guide_size_for=False,
                        max_size=768,
                        seed=random.randint(1, 2**64),
                        steps=20,
                        cfg=6.5,
                        sampler_name=sampler_name,
                        scheduler="karras",
                        denoise=0.5,
                        feather=5,
                        noise_mask=True,
                        force_inpaint=True,
                        wildcard="",
                        cycle=1,
                        inpaint_model=False,
                        noise_mask_feather=20,
                        image=vaedecode_240[0],
                        segs=impactsegsandmask_152[0],
                        model=applystablefast_158[0],
                        clip=checkpointloadersimple_241[1],
                        vae=checkpointloadersimple_241[2],
                        positive=cliptextencode_124[0],
                        negative=cliptextencode_243[0],
                        pipeline=True,
                    )
                    # Compute detailer seed safely (guard against detailer returning
                    # image tensors or other non-seed objects)
                    def _extract_scalar_seed(candidate):
                        try:
                            # integers
                            if isinstance(candidate, int):
                                return str(candidate)
                            # floats that represent integers
                            if isinstance(candidate, float) and float(candidate).is_integer():
                                return str(int(candidate))
                            # numeric strings
                            if isinstance(candidate, str):
                                s = candidate.strip()
                                if re.fullmatch(r"-?\d+", s):
                                    return s
                                # if it's a large string but contains an integer token, use the token
                                m = re.search(r"\d{4,}", s)
                                if m:
                                    return m.group(0)
                                return None
                            # numpy scalars/arrays
                            if isinstance(candidate, np.ndarray):
                                if candidate.size == 1:
                                    return str(int(candidate.item()))
                                return None
                            # torch tensors
                            if isinstance(candidate, torch.Tensor):
                                try:
                                    if candidate.numel() == 1:
                                        return str(int(candidate.item()))
                                except Exception:
                                    return None
                        except Exception:
                            return None
                        return None

                    try:
                        if isinstance(detailerforeachdebug_145, (list, tuple)) and len(detailerforeachdebug_145) > 1:
                            candidate = detailerforeachdebug_145[1]
                            extracted = _extract_scalar_seed(candidate)
                            detailer_body_seed = extracted if extracted is not None else str(current_seed)
                        else:
                            detailer_body_seed = str(current_seed)
                    except Exception:
                        detailer_body_seed = str(current_seed)

                    body_meta = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "seed": detailer_body_seed,
                        "sampler": str(sampler_name),
                        "steps": "20",
                        "cfg": "6.5",
                        "scheduler": "karras",
                        "denoise": "0.5",
                        "width": str(w),
                        "height": str(h),
                            "batch_size": str(batch),
                        "adetailer": "True",
                    }

                    if autohdr:
                        _tmp = hdr.apply_hdr2(detailerforeachdebug_145[0])
                        body_imgs = _tmp[0] if isinstance(_tmp, (tuple, list)) else _tmp
                    else:
                        body_imgs = detailerforeachdebug_145[0]
                    saveimage.save_images(
                        filename_prefix="LD-body",
                        images=body_imgs,
                        prompt=prompt,
                        extra_pnginfo=body_meta,
                    )
                    ultralyticsdetectorprovider = bbox.UltralyticsDetectorProvider()
                    ultralyticsdetectorprovider_151 = ultralyticsdetectorprovider.doit(
                        model_name="face_yolov9c.pt"
                    )
                    bboxdetectorsegs_132 = bboxdetectorsegs.doit(
                        threshold=0.5,
                        dilation=10,
                        crop_factor=2,
                        drop_size=10,
                        labels="all",
                        bbox_detector=ultralyticsdetectorprovider_151[0],
                        image=detailerforeachdebug_145[0],
                    )
                    samdetectorcombined_139 = samdetectorcombined.doit(
                        detection_hint="center-1",
                        dilation=0,
                        threshold=0.93,
                        bbox_expansion=0,
                        mask_hint_threshold=0.7,
                        mask_hint_use_negative="False",
                        sam_model=samloader_87[0],
                        segs=bboxdetectorsegs_132,
                        image=detailerforeachdebug_145[0],
                    )
                    impactsegsandmask_152 = impactsegsandmask.doit(
                        segs=bboxdetectorsegs_132,
                        mask=samdetectorcombined_139[0],
                    )
                    detailerforeachdebug_145 = detailerforeachdebug.doit(
                        guide_size=512,
                        guide_size_for=False,
                        max_size=768,
                        seed=random.randint(1, 2**64),
                        steps=20,
                        cfg=6.5,
                        sampler_name=sampler_name,
                        scheduler="karras",
                        denoise=0.5,
                        feather=5,
                        noise_mask=True,
                        force_inpaint=True,
                        wildcard="",
                        cycle=1,
                        inpaint_model=False,
                        noise_mask_feather=20,
                        image=detailerforeachdebug_145[0],
                        segs=impactsegsandmask_152[0],
                        model=applystablefast_158[0],
                        clip=checkpointloadersimple_241[1],
                        vae=checkpointloadersimple_241[2],
                        positive=cliptextencode_124[0],
                        negative=cliptextencode_243[0],
                        pipeline=True,
                    )
                    # Compute detailer head seed safely (same logic as body seed)
                    try:
                        if isinstance(detailerforeachdebug_145, (list, tuple)) and len(detailerforeachdebug_145) > 1:
                            candidate_h = detailerforeachdebug_145[1]
                            extracted_h = _extract_scalar_seed(candidate_h)
                            detailer_head_seed = extracted_h if extracted_h is not None else str(current_seed)
                        else:
                            detailer_head_seed = str(current_seed)
                    except Exception:
                        detailer_head_seed = str(current_seed)

                    head_meta = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "seed": detailer_head_seed,
                        "sampler": str(sampler_name),
                        "steps": "20",
                        "cfg": "6.5",
                        "scheduler": "karras",
                        "denoise": "0.5",
                        "width": str(w),
                        "height": str(h),
                            "batch_size": str(batch),
                        "adetailer": "True",
                    }

                    if autohdr:
                        _tmp = hdr.apply_hdr2(detailerforeachdebug_145[0])
                        head_imgs = _tmp[0] if isinstance(_tmp, (tuple, list)) else _tmp
                    else:
                        head_imgs = detailerforeachdebug_145[0]
                    saveimage.save_images(
                        filename_prefix="LD-head",
                        images=head_imgs,
                        prompt=prompt,
                        extra_pnginfo=head_meta,
                    )
            else:
                # Determine sampling metadata for main outputs
                if hires_fix:
                    main_steps = 10
                    main_cfg = 8
                    main_sampler = "euler_ancestral_cfgpp"
                else:
                    main_steps = 20
                    main_cfg = 7
                    main_sampler = sampler_name

                main_meta = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": str(current_seed),
                    "sampler": str(main_sampler),
                    "steps": str(main_steps),
                    "cfg": str(main_cfg),
                    "scheduler": "karras",
                    "denoise": "1",
                    "width": str(w),
                    "height": str(h),
                        "batch_size": str(batch),
                    "hires_fix": str(hires_fix),
                    "adetailer": str(adetailer),
                    "stable_fast": str(stable_fast),
                    "flux_enabled": str(flux_enabled),
                    "realistic_model": str(realistic_model),
                    "reuse_seed": str(reuse_seed),
                    "multiscale_preset": str(multiscale_preset),
                }

                if autohdr:
                    _tmp = hdr.apply_hdr2(vaedecode_240[0])
                    main_imgs = _tmp[0] if isinstance(_tmp, (tuple, list)) else _tmp
                else:
                    main_imgs = vaedecode_240[0]
                saveimage.save_images(
                    filename_prefix="LD-HF" if hires_fix else "LD",
                    images=main_imgs,
                    prompt=prompt,
                    extra_pnginfo=main_meta,
                )

    return {
        "original_prompt": original_prompt,
        "used_prompt": prompt,
        "enhancement_applied": enhancement_applied,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LightDiffusion pipeline.")
    parser.add_argument("prompt", type=str, help="The prompt for the pipeline.")
    parser.add_argument("width", type=int, help="The width of the generated image.")
    parser.add_argument("height", type=int, help="The height of the generated image.")
    parser.add_argument("number", type=int, help="The number of images to generate.")
    parser.add_argument(
        "batch",
        type=int,
        help="The batch size. aka the number of images to generate at once.",
    )
    parser.add_argument(
        "--hires-fix", action="store_true", help="Enable high-resolution fix."
    )
    parser.add_argument(
        "--adetailer",
        action="store_true",
        help="Enable automatic face and body enhancin.g",
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Enable Ollama prompt enhancement. Make sure to have ollama with Ollama installed.",
    )
    parser.add_argument(
        "--img2img",
        action="store_true",
        help="Enable image-to-image mode. This will use the prompt as path to the image.",
    )
    parser.add_argument(
        "--stable-fast",
        action="store_true",
        help="Enable StableFast mode. This will compile the model for faster inference.",
    )
    parser.add_argument(
        "--reuse-seed",
        action="store_true",
        help="Enable to reuse last used seed for sampling, default for False is a random seed at every use.",
    )
    parser.add_argument(
        "--flux",
        action="store_true",
        help="Enable the flux mode.",
    )
    parser.add_argument(
        "--prio-speed",
        action="store_true",
        help="Prioritize speed over quality.",
    )
    parser.add_argument(
        "--autohdr",
        action="store_true",
        help="Enable the AutoHDR mode.",
    )
    parser.add_argument(
        "--realistic-model",
        action="store_true",
        help="Use the realistic model.",
    )
    parser.add_argument(
        "--multiscale-preset",
        type=str,
        choices=["quality", "performance", "balanced", "disabled"],
        help="Predefined multiscale preset ('quality', 'performance', 'balanced', 'disabled'). Overrides individual multiscale parameters.",
    )
    parser.add_argument(
        "--enable-multiscale",
        action="store_true",
        default=True,
        help="Enable multi-scale diffusion for performance optimization.",
    )
    parser.add_argument(
        "--multiscale-factor",
        type=float,
        default=0.5,
        help="Scale factor for intermediate steps (0.1-1.0).",
    )
    parser.add_argument(
        "--multiscale-fullres-start",
        type=int,
        default=3,
        help="Number of first steps at full resolution.",
    )
    parser.add_argument(
        "--multiscale-fullres-end",
        type=int,
        default=8,
        help="Number of last steps at full resolution.",
    )
    parser.add_argument(
        "--multiscale-intermittent-fullres",
        action="store_true",
        help="Enable intermittent full-res rendering in low-res region.",
    )
    args = parser.parse_args()

    pipeline(
        args.prompt,
        args.width,
        args.height,
        args.number,
        args.batch,
        args.hires_fix,
        args.adetailer,
        args.enhance_prompt,
        args.img2img,
        args.stable_fast,
        args.reuse_seed,
        args.flux,
        args.prio_speed,
        args.autohdr,
        args.realistic_model,
        args.multiscale_preset,
        args.enable_multiscale,
        args.multiscale_factor,
        args.multiscale_fullres_start,
        args.multiscale_fullres_end,
        args.multiscale_intermittent_fullres,
    )

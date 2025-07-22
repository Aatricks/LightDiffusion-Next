import argparse
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

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

with open(os.path.join("./include/", "last_seed.txt"), "r") as f:
    last_seed = int(f.read())

Downloader.CheckAndDownload()


def pipeline(
    prompt: str,
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
    autohdr: bool = False,
    realistic_model: bool = False,
    # Multi-scale diffusion parameters
    multiscale_preset: str = None,
    enable_multiscale: bool = True,
    multiscale_factor: float = 0.5,
    multiscale_fullres_start: int = 3,
    multiscale_fullres_end: int = 8,
    multiscale_intermittent_fullres: bool = False,
) -> None:
    """#### Run the LightDiffusion pipeline.

    #### Args:
        - `prompt` (str): The prompt for the pipeline.
        - `w` (int): The width of the generated image.
        - `h` (int): The height of the generated image.
        - `hires_fix` (bool, optional): Enable high-resolution fix. Defaults to False.
        - `adetailer` (bool, optional): Enable automatic face and body enhancing. Defaults to False.
        - `enhance_prompt` (bool, optional): Enable Ollama prompt enhancement. Defaults to False.
        - `img2img` (bool, optional): Use LightDiffusion in Image to Image mode, the prompt input becomes the path to the input image. Defaults to False.
        - `stable_fast` (bool, optional): Enable Stable-Fast speedup offering a 70% speed improvement in return of a compilation time. Defaults to False.
        - `reuse_seed` (bool, optional): Reuse the last used seed, if False the seed will be kept random. Default to False.
        - `flux_enabled` (bool, optional): Enable the flux mode. Defaults to False.
        - `prio_speed` (bool, optional): Prioritize speed over quality. Defaults to False.
        - `autohdr` (bool, optional): Enable the AutoHDR mode. Defaults to False.
        - `realistic_model` (bool, optional): Use the realistic model. Defaults to False.
        - `multiscale_preset` (str, optional): Predefined multiscale preset ('quality', 'performance', 'balanced', 'disabled'). Overrides individual multiscale parameters. Defaults to None.
        - `enable_multiscale` (bool, optional): Enable multi-scale diffusion for performance optimization. Defaults to True.
        - `multiscale_factor` (float, optional): Scale factor for intermediate steps (0.1-1.0). Defaults to 0.5.
        - `multiscale_fullres_start` (int, optional): Number of first steps at full resolution. Defaults to 3.
        - `multiscale_fullres_end` (int, optional): Number of last steps at full resolution. Defaults to 8.
        - `multiscale_intermittent_fullres` (bool, optional): Enable intermittent full-res rendering in low-res region. Defaults to False.
    """
    global last_seed

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

    if reuse_seed:
        seed = last_seed

    else:
        seed = random.randint(1, 2**64)
        last_seed = seed
    with open(os.path.join("./include/", "last_seed.txt"), "w") as f:
        f.write(str(seed))
    if enhance_prompt:
        try:
            prompt = Enhancer.enhance_prompt(prompt)
        except:
            pass

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
    for _ in range(number):
        if img2img:
            img = Image.open(prompt)
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
                except:
                    loraloader_274 = checkpointloadersimple_241

                if stable_fast is True:
                    from src.StableFast import StableFast

                    applystablefast = StableFast.ApplyStableFastUnet()
                    applystablefast_158 = applystablefast.apply_stable_fast(
                        enable_cuda_graph=False,
                        model=loraloader_274[0],
                    )
                else:
                    applystablefast_158 = loraloader_274

                clipsetlastlayer = Clip.CLIPSetLastLayer()
                clipsetlastlayer_257 = clipsetlastlayer.set_last_layer(
                    stop_at_clip_layer=-2, clip=loraloader_274[1]
                )

                cliptextencode_242 = cliptextencode.encode(
                    text=prompt,
                    clip=clipsetlastlayer_257[0],
                )
                cliptextencode_243 = cliptextencode.encode(
                    text="(worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic), (embedding:EasyNegative), (embedding:badhandv4), (embedding:lr), (embedding:ng_deepnegative_v1_75t)",
                    clip=clipsetlastlayer_257[0],
                )
                upscalemodelloader = USDU_upscaler.UpscaleModelLoader()
                upscalemodelloader_244 = upscalemodelloader.load_model(
                    "RealESRGAN_x4plus.pth"
                )
                ultimatesdupscale_250 = ultimatesdupscale.upscale(
                    upscale_by=2,
                    seed=random.randint(1, 2**64),
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
                saveimage.save_images(
                    filename_prefix="LD-I2I",
                    images=hdr.apply_hdr2(ultimatesdupscale_250[0])
                    if autohdr
                    else ultimatesdupscale_250[0],
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
                    seed=random.randint(1, 2**64),
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

                saveimage.save_images(
                    filename_prefix="LD-Flux",
                    images=hdr.apply_hdr2(vaedecode_8[0])
                    if autohdr
                    else vaedecode_8[0],
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
                except:
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
                    text="(worst quality, low quality:1.4), (zombie, sketch, interlocked fingers, comic), (embedding:EasyNegative), (embedding:badhandv4), (embedding:lr), (embedding:ng_deepnegative_v1_75t)",
                    clip=clipsetlastlayer_257[0],
                )
                emptylatentimage_244 = emptylatentimage.generate(
                    width=w, height=h, batch_size=batch
                )
                if stable_fast is True:
                    from src.StableFast import StableFast

                    applystablefast = StableFast.ApplyStableFastUnet()
                    applystablefast_158 = applystablefast.apply_stable_fast(
                        enable_cuda_graph=False,
                        model=loraloader_274[0],
                    )
                else:
                    applystablefast_158 = loraloader_274
                    # fb_cache = fbcache_nodes.ApplyFBCacheOnModel()
                    # applystablefast_158 = fb_cache.patch(
                    #     applystablefast_158, "diffusion_model", 0.120
                    # )

                # Create sampler with multi-scale options
                ksampler_239 = ksampler_instance.sample(
                    seed=seed,
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

                vaedecode_240 = vaedecode.decode(
                    samples=ksampler_253[0],
                    vae=checkpointloadersimple_241[2],
                )

            if adetailer:
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
                        return
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
                    saveimage.save_images(
                        filename_prefix="LD-body",
                        images=hdr.apply_hdr2(detailerforeachdebug_145[0])
                        if autohdr
                        else detailerforeachdebug_145[0],
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
                    saveimage.save_images(
                        filename_prefix="LD-head",
                        images=hdr.apply_hdr2(detailerforeachdebug_145[0])
                        if autohdr
                        else detailerforeachdebug_145[0],
                    )
            else:
                saveimage.save_images(
                    filename_prefix="LD-HF" if hires_fix else "LD",
                    images=hdr.apply_hdr2(vaedecode_240[0])
                    if autohdr
                    else vaedecode_240[0],
                )


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

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch._dynamo
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.AutoDetailer import SAM, SEGS, ADetailer, bbox
from modules.AutoEncoders import VariationalAE
from modules.clip import Clip
from modules.Device import Device
from modules.FileManaging import Downloader, ImageSaver, Loader
from modules.hidiffusion import msw_msa_attention
from modules.Model import LoRas
from modules.Quantize import Quantizer
from modules.sample import sampling
from modules.UltimateSDUpscale import UltimateSDUpscale, USDU_upscaler
from modules.Utilities import Enhancer, Latent, upscale
from modules.WaveSpeed import fbcache_nodes

torch._dynamo.config.suppress_errors = True
torch.compiler.allow_in_graph

last_seed = 0

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
    """
    global last_seed
    if reuse_seed:
        seed = last_seed
    else:
        seed = random.randint(1, 2**64)
        last_seed = seed
    if enhance_prompt:
        try:
            prompt = Enhancer.enhance_prompt(prompt)
        except:
            pass
    sampler_name = "dpmpp_sde" if not prio_speed else "dpmpp_2m"
    ckpt = "./_internal/checkpoints/Meina V10 - baked VAE.safetensors"
    with torch.inference_mode():
        if not flux_enabled:
            checkpointloadersimple = Loader.CheckpointLoaderSimple()
            checkpointloadersimple_241 = checkpointloadersimple.load_checkpoint(
                ckpt_name=ckpt
            )
            hidiffoptimizer = msw_msa_attention.ApplyMSWMSAAttentionSimple()
        cliptextencode = Clip.CLIPTextEncode()
        emptylatentimage = Latent.EmptyLatentImage()
        ksampler_instance = sampling.KSampler2()
        vaedecode = VariationalAE.VAEDecode()
        saveimage = ImageSaver.SaveImage()
        latent_upscale = upscale.LatentUpscale()
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
                    from modules.StableFast import StableFast

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
                    filename_prefix="LD-i2i",
                    images=ultimatesdupscale_250[0],
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
                ksampler = sampling.KSampler2()
                unetloadergguf_10 = unetloadergguf.load_unet(
                    unet_name="flux1-dev-Q8_0.gguf"
                )
                fb_cache = fbcache_nodes.ApplyFBCacheOnModel()
                unetloadergguf_10 = fb_cache.patch(unetloadergguf_10, "diffusion_model", 0.120)
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
                    guidance=2.5,
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
                    sampler_name="euler",
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
                    filename_prefix="Flux", images=vaedecode_8[0]
                )
        else:
            while prompt is None:
                pass
            if Device.should_use_bf16():
                dtype = torch.bfloat16
            elif Device.should_use_fp16():
                dtype = torch.float16
            else:
                dtype = torch.float32
            with (
                torch.inference_mode(),
                torch.autocast(device_type="cuda", dtype=dtype),
            ):
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
                fb_cache = fbcache_nodes.ApplyFBCacheOnModel()
                applystablefast_158 = fb_cache.patch(applystablefast_158, "diffusion_model", 0.120)
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
                        sampler_name="euler_ancestral",
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
                        model=checkpointloadersimple_241[0],
                        clip=checkpointloadersimple_241[1],
                        vae=checkpointloadersimple_241[2],
                        positive=cliptextencode_124[0],
                        negative=cliptextencode_243[0],
                        pipeline=True,
                    )
                    saveimage.save_images(
                        filename_prefix="LD-refined",
                        images=detailerforeachdebug_145[0],
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
                        model=checkpointloadersimple_241[0],
                        clip=checkpointloadersimple_241[1],
                        vae=checkpointloadersimple_241[2],
                        positive=cliptextencode_124[0],
                        negative=cliptextencode_243[0],
                        pipeline=True,
                    )
                    saveimage.save_images(
                        filename_prefix="lD-2ndrefined",
                        images=detailerforeachdebug_145[0],
                    )
            else:
                saveimage.save_images(filename_prefix="LD", images=vaedecode_240[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LightDiffusion pipeline.")
    parser.add_argument("prompt", type=str, help="The prompt for the pipeline.")
    parser.add_argument("width", type=int, help="The width of the generated image.")
    parser.add_argument("height", type=int, help="The height of the generated image.")
    parser.add_argument("number", type=int, help="The number of images to generate.")
    parser.add_argument("batch", type=int, help="The batch size. aka the number of images to generate at once.")
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
        help="Enable image-to-image mode. This will use the image as the prompt.",
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
    )

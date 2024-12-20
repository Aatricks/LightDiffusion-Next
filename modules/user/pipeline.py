import os
import random
import sys
import argparse

import numpy as np
from PIL import Image
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.AutoEncoders import VariationalAE
from modules.clip import Clip
from modules.sample import sampling

from modules.Utilities import upscale
from modules.AutoDetailer import SAM, ADetailer, bbox, SEGS

from modules.FileManaging import ImageSaver, Loader
from modules.Model import LoRas
from modules.Utilities import Enhancer, Latent

def pipeline(prompt, w, h, hires_fix = False, adetailer = False):
    ckpt = "./_internal/checkpoints/Meina V10 - baked VAE.safetensors"
    with torch.inference_mode():
        checkpointloadersimple = Loader.CheckpointLoaderSimple()
        checkpointloadersimple_241 = checkpointloadersimple.load_checkpoint(
            ckpt_name=ckpt
        )
        cliptextencode = Clip.CLIPTextEncode()
        emptylatentimage = Latent.EmptyLatentImage()
        ksampler_instance = sampling.KSampler2()
        vaedecode = VariationalAE.VAEDecode()
        saveimage = ImageSaver.SaveImage()
        latent_upscale = upscale.LatentUpscale()
    try:
        prompt = Enhancer.enhance_prompt(prompt)
    except:
        pass
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
            width=w, height=h, batch_size=1
        )
        ksampler_239 = ksampler_instance.sample(
            seed=random.randint(1, 2**64),
            steps=40,
            cfg=7,
            sampler_name="dpm_adaptive",
            scheduler="karras",
            denoise=1,
            model=applystablefast_158[0],
            positive=cliptextencode_242[0],
            negative=cliptextencode_243[0],
            latent_image=emptylatentimage_244[0],
        )
        if hires_fix:
            latentupscale_254 = latent_upscale.upscale(
                upscale_method="bislerp",
                width=w * 2,
                height=h * 2,
                crop="disabled",
                samples=ksampler_239[0],
            )
            ksampler_253 = ksampler_instance.sample(
                seed=random.randint(1, 2**64),
                steps=10,
                cfg=8,
                sampler_name="euler_ancestral",
                scheduler="normal",
                denoise=0.45,
                model=applystablefast_158[0],
                positive=cliptextencode_242[0],
                negative=cliptextencode_243[0],
                latent_image=latentupscale_254[0],
            )
        else:
            ksampler_253 = ksampler_239
        if adetailer:
            try:
                samloader = SAM.SAMLoader()
                samloader_87 = samloader.load_model(
                    model_name="sam_vit_b_01ec64.pth", device_mode="AUTO"
                )

                cliptextencode_124 = cliptextencode.encode(
                    text="royal, detailed, magnificient, beautiful, seducing",
                    clip=loraloader_274[1],
                )

                ultralyticsdetectorprovider = ADetailer.UltralyticsDetectorProvider()
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
                    segs=bboxdetectorsegs_132[0],
                    image=vaedecode_240[0],
                )
                impactsegsandmask_152 = impactsegsandmask.doit(
                    segs=bboxdetectorsegs_132[0],
                    mask=samdetectorcombined_139[0],
                )
                detailerforeachdebug_145 = detailerforeachdebug.doit(
                    guide_size=512,
                    guide_size_for=False,
                    max_size=768,
                    seed=random.randint(1, 2**64),
                    steps=40,
                    cfg=6.5,
                    sampler_name="dpmpp_2m_sde",
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
                )
                saveimage.save_images(
                    filename_prefix="LD-refined",
                    images=detailerforeachdebug_145[0],
                )
                ultralyticsdetectorprovider = ADetailer.UltralyticsDetectorProvider()
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
                    segs=bboxdetectorsegs_132[0],
                    image=detailerforeachdebug_145[0],
                )
                impactsegsandmask_152 = impactsegsandmask.doit(
                    segs=bboxdetectorsegs_132[0],
                    mask=samdetectorcombined_139[0],
                )
                detailerforeachdebug_145 = detailerforeachdebug.doit(
                    guide_size=512,
                    guide_size_for=False,
                    max_size=768,
                    seed=random.randint(1, 2**64),
                    steps=40,
                    cfg=6.5,
                    sampler_name="dpmpp_2m_sde",
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
                )
                saveimage.save_images(
                    filename_prefix="lD-2ndrefined",
                    images=detailerforeachdebug_145[0],
                )
            except:
                pass
        else:
            vaedecode_240 = vaedecode.decode(
                samples=ksampler_253[0],
                vae=checkpointloadersimple_241[2],
            )
            saveimage.save_images(filename_prefix="LD", images=vaedecode_240[0])
            for image in vaedecode_240[0]:
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LightDiffusion pipeline.")
    parser.add_argument("prompt", type=str, help="The prompt for the pipeline.")
    parser.add_argument("width", type=int, help="The width of the generated image.")
    parser.add_argument("height", type=int, help="The height of the generated image.")
    parser.add_argument("--hires-fix", action="store_true", help="Enable high-resolution fix.")
    parser.add_argument("--adetailer", action="store_true", help="Enable automatic face and body enhancing")
    args = parser.parse_args()

    pipeline(args.prompt, args.width, args.height, args.hires_fix, args.adetailer)
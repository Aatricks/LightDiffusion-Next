import os
import random
import sys
import argparse

import torch
import torch._dynamo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.Device import Device
from modules.AutoEncoders import VariationalAE
from modules.clip import Clip
from modules.sample import sampling

from modules.Utilities import upscale
from modules.AutoDetailer import SAM, ADetailer, bbox, SEGS

from modules.FileManaging import ImageSaver, Loader
from modules.Model import LoRas
from modules.Utilities import Enhancer, Latent

torch._dynamo.config.suppress_errors = True
torch.compiler.allow_in_graph

# TODO: test compilation speedup
# @torch.compile
def pipeline(prompt, w, h, hires_fix=False, adetailer=False, enhance_prompt=False):
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
    if enhance_prompt:
        try:
            prompt = Enhancer.enhance_prompt(prompt)
        except:
            pass
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
        torch.no_grad(),
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
            width=w, height=h, batch_size=1
        )
        ksampler_239 = ksampler_instance.sample(
            seed=random.randint(1, 2**64),
            steps=40,
            cfg=7,
            sampler_name="dpm_adaptive",
            scheduler="karras",
            denoise=1,
            pipeline=True,
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

        vaedecode_240 = vaedecode.decode(
            samples=ksampler_253[0],
            vae=checkpointloadersimple_241[2],
        )

        if adetailer:
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
    parser.add_argument(
        "--hires-fix", action="store_true", help="Enable high-resolution fix."
    )
    parser.add_argument(
        "--adetailer",
        action="store_true",
        help="Enable automatic face and body enhancing",
    )
    parser.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="Enable llama3.2 prompt enhancement. Make sure to have ollama with llama3.2 installed.",
    )
    args = parser.parse_args()

    pipeline(
        args.prompt,
        args.width,
        args.height,
        args.hires_fix,
        args.adetailer,
        args.enhance_prompt,
    )

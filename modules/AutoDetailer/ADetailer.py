import math
import numpy as np
import torch


from modules.AutoDetailer import AD_util, bbox, tensor_util
from modules.AutoDetailer import SEGS
from modules.Utilities import util
from modules.AutoEncoders import VariationalAE
from modules.Device import Device
from modules.sample import ksampler_util, samplers, sampling, sampling_util

# FIXME: Improve slow inference times


import numpy as np
import torch


class DifferentialDiffusion:
    def apply(self, model):
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)
        return (model,)

    def forward(
        self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict
    ):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to)

        return (denoise_mask >= threshold).to(denoise_mask.dtype)


def to_latent_image(pixels, vae):
    x = pixels.shape[1]
    y = pixels.shape[2]
    return VariationalAE.VAEEncode().encode(vae, pixels)[0]


def calculate_sigmas2(model, sampler, scheduler, steps):
    return ksampler_util.calculate_sigmas(
        model.get_model_object("model_sampling"), scheduler, steps
    )


def get_noise_sampler(x, cpu, total_sigmas, **kwargs):
    if "extra_args" in kwargs and "seed" in kwargs["extra_args"]:
        sigma_min, sigma_max = total_sigmas[total_sigmas > 0].min(), total_sigmas.max()
        seed = kwargs["extra_args"].get("seed", None)
        return sampling_util.BrownianTreeNoiseSampler(
            x, sigma_min, sigma_max, seed=seed, cpu=cpu
        )
    return None


def ksampler2(sampler_name, total_sigmas, extra_options={}, inpaint_options={}):
    if sampler_name == "dpmpp_2m_sde":

        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs["noise_sampler"] = noise_sampler

            return samplers.sample_dpmpp_2m_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    else:
        return sampling.sampler_object(sampler_name)

    return sampling.KSAMPLER(sampler_function, extra_options, inpaint_options)


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = (
            input_latent["batch_index"] if "batch_index" in input_latent else None
        )
        return ksampler_util.prepare_noise(latent_image, self.seed, batch_inds)


def sample_with_custom_noise(
    model,
    add_noise,
    noise_seed,
    cfg,
    positive,
    negative,
    sampler,
    sigmas,
    latent_image,
    noise=None,
    callback=None,
):
    latent = latent_image
    latent_image = latent["samples"]

    out = latent.copy()
    out["samples"] = latent_image

    if noise is None:
        noise = Noise_RandomNoise(noise_seed).generate_noise(out)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    x0_output = {}

    disable_pbar = not util.PROGRESS_BAR_ENABLED

    device = Device.get_torch_device()

    noise = noise.to(device)
    latent_image = latent_image.to(device)
    if noise_mask is not None:
        noise_mask = noise_mask.to(device)

    samples = sampling.sample_custom(
        model,
        noise,
        cfg,
        sampler,
        sigmas,
        positive,
        negative,
        latent_image,
        noise_mask=noise_mask,
        disable_pbar=disable_pbar,
        seed=noise_seed,
    )

    samples = samples.to(Device.intermediate_device())

    out["samples"] = samples
    out_denoised = out
    return out, out_denoised


def separated_sample(
    model,
    add_noise,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    start_at_step,
    end_at_step,
    return_with_leftover_noise,
    sigma_ratio=1.0,
    sampler_opt=None,
    noise=None,
    callback=None,
    scheduler_func=None,
):

    total_sigmas = calculate_sigmas2(model, sampler_name, scheduler, steps)

    sigmas = total_sigmas

    if start_at_step is not None:
        sigmas = sigmas[start_at_step:] * sigma_ratio

    impact_sampler = ksampler2(sampler_name, total_sigmas)

    res = sample_with_custom_noise(
        model,
        add_noise,
        seed,
        cfg,
        positive,
        negative,
        impact_sampler,
        sigmas,
        latent_image,
        noise=noise,
        callback=callback,
    )

    return res[1]


def ksampler_wrapper(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise,
    refiner_ratio=None,
    refiner_model=None,
    refiner_clip=None,
    refiner_positive=None,
    refiner_negative=None,
    sigma_factor=1.0,
    noise=None,
    scheduler_func=None,
):

    # Use separated_sample instead of KSampler for `AYS scheduler`
    # refined_latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise * sigma_factor)[0]
    advanced_steps = math.floor(steps / denoise)
    start_at_step = advanced_steps - steps
    end_at_step = start_at_step + steps
    refined_latent = separated_sample(
        model,
        True,
        seed,
        advanced_steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        start_at_step,
        end_at_step,
        False,
        sigma_ratio=sigma_factor,
        noise=noise,
        scheduler_func=scheduler_func,
    )

    return refined_latent


def enhance_detail(
    image,
    model,
    clip,
    vae,
    guide_size,
    guide_size_for_bbox,
    max_size,
    bbox,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    denoise,
    noise_mask,
    force_inpaint,
    wildcard_opt=None,
    wildcard_opt_concat_mode=None,
    detailer_hook=None,
    refiner_ratio=None,
    refiner_model=None,
    refiner_clip=None,
    refiner_positive=None,
    refiner_negative=None,
    control_net_wrapper=None,
    cycle=1,
    inpaint_model=False,
    noise_mask_feather=0,
    scheduler_func=None,
):

    if noise_mask is not None:
        noise_mask = tensor_util.tensor_gaussian_blur_mask(
            noise_mask, noise_mask_feather
        )
        noise_mask = noise_mask.squeeze(3)

    h = image.shape[1]
    w = image.shape[2]

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    # for cropped_size
    upscale = guide_size / min(w, h)

    new_w = int(w * upscale)
    new_h = int(h * upscale)

    if new_w > max_size or new_h > max_size:
        upscale *= max_size / max(new_w, new_h)
        new_w = int(w * upscale)
        new_h = int(h * upscale)

    if upscale <= 1.0 or new_w == 0 or new_h == 0:
        print(f"Detailer: force inpaint")
        upscale = 1.0
        new_w = w
        new_h = h

    print(
        f"Detailer: segment upscale for ({bbox_w, bbox_h}) | crop region {w, h} x {upscale} -> {new_w, new_h}"
    )

    # upscale
    upscaled_image = tensor_util.tensor_resize(image, new_w, new_h)

    cnet_pils = None

    # prepare mask
    latent_image = to_latent_image(upscaled_image, vae)
    if noise_mask is not None:
        latent_image["noise_mask"] = noise_mask

    refined_latent = latent_image

    # ksampler
    for i in range(0, cycle):
        (
            model2,
            seed2,
            steps2,
            cfg2,
            sampler_name2,
            scheduler2,
            positive2,
            negative2,
            upscaled_latent2,
            denoise2,
        ) = (
            model,
            seed + i,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise,
        )
        noise = None

        refined_latent = ksampler_wrapper(
            model2,
            seed2,
            steps2,
            cfg2,
            sampler_name2,
            scheduler2,
            positive2,
            negative2,
            refined_latent,
            denoise2,
            refiner_ratio,
            refiner_model,
            refiner_clip,
            refiner_positive,
            refiner_negative,
            noise=noise,
            scheduler_func=scheduler_func,
        )

    # non-latent downscale - latent downscale cause bad quality
    try:
        # try to decode image normally
        refined_image = vae.decode(refined_latent["samples"])
    except Exception as e:
        # usually an out-of-memory exception from the decode, so try a tiled approach
        refined_image = vae.decode_tiled(
            refined_latent["samples"],
            tile_x=64,
            tile_y=64,
        )

    # downscale
    refined_image = tensor_util.tensor_resize(refined_image, w, h)

    # prevent mixing of device
    refined_image = refined_image.cpu()

    # don't convert to latent - latent break image
    # preserving pil is much better
    return refined_image, cnet_pils


class DetailerForEach:
    @staticmethod
    def do_detail(
        image,
        segs,
        model,
        clip,
        vae,
        guide_size,
        guide_size_for_bbox,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        denoise,
        feather,
        noise_mask,
        force_inpaint,
        wildcard_opt=None,
        detailer_hook=None,
        refiner_ratio=None,
        refiner_model=None,
        refiner_clip=None,
        refiner_positive=None,
        refiner_negative=None,
        cycle=1,
        inpaint_model=False,
        noise_mask_feather=0,
        scheduler_func_opt=None,
    ):

        image = image.clone()
        enhanced_alpha_list = []
        enhanced_list = []
        cropped_list = []
        cnet_pil_list = []

        segs = AD_util.segs_scale_match(segs, image.shape)
        new_segs = []

        wildcard_concat_mode = None
        wmode, wildcard_chooser = bbox.process_wildcard_for_segs(wildcard_opt)

        ordered_segs = segs[1]

        if (
            noise_mask_feather > 0
            and "denoise_mask_function" not in model.model_options
        ):
            model = DifferentialDiffusion().apply(model)[0]

        for i, seg in enumerate(ordered_segs):
            cropped_image = AD_util.crop_ndarray4(
                image.cpu().numpy(), seg.crop_region
            )  # Never use seg.cropped_image to handle overlapping area
            cropped_image = tensor_util.to_tensor(cropped_image)
            mask = tensor_util.to_tensor(seg.cropped_mask)
            mask = tensor_util.tensor_gaussian_blur_mask(mask, feather)

            is_mask_all_zeros = (seg.cropped_mask == 0).all().item()
            if is_mask_all_zeros:
                print(f"Detailer: segment skip [empty mask]")
                continue

            cropped_mask = seg.cropped_mask

            seg_seed, wildcard_item = wildcard_chooser.get(seg)

            seg_seed = seed + i if seg_seed is None else seg_seed

            cropped_positive = [
                [
                    condition,
                    {
                        k: (
                            crop_condition_mask(v, image, seg.crop_region)
                            if k == "mask"
                            else v
                        )
                        for k, v in details.items()
                    },
                ]
                for condition, details in positive
            ]

            cropped_negative = [
                [
                    condition,
                    {
                        k: (
                            crop_condition_mask(v, image, seg.crop_region)
                            if k == "mask"
                            else v
                        )
                        for k, v in details.items()
                    },
                ]
                for condition, details in negative
            ]

            orig_cropped_image = cropped_image.clone()
            enhanced_image, cnet_pils = enhance_detail(
                cropped_image,
                model,
                clip,
                vae,
                guide_size,
                guide_size_for_bbox,
                max_size,
                seg.bbox,
                seg_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                cropped_positive,
                cropped_negative,
                denoise,
                cropped_mask,
                force_inpaint,
                wildcard_opt=wildcard_item,
                wildcard_opt_concat_mode=wildcard_concat_mode,
                detailer_hook=detailer_hook,
                refiner_ratio=refiner_ratio,
                refiner_model=refiner_model,
                refiner_clip=refiner_clip,
                refiner_positive=refiner_positive,
                refiner_negative=refiner_negative,
                control_net_wrapper=seg.control_net_wrapper,
                cycle=cycle,
                inpaint_model=inpaint_model,
                noise_mask_feather=noise_mask_feather,
                scheduler_func=scheduler_func_opt,
            )

            if not (enhanced_image is None):
                # don't latent composite-> converting to latent caused poor quality
                # use image paste
                image = image.cpu()
                enhanced_image = enhanced_image.cpu()
                tensor_util.tensor_paste(
                    image,
                    enhanced_image,
                    (seg.crop_region[0], seg.crop_region[1]),
                    mask,
                )  # this code affecting to `cropped_image`.
                enhanced_list.append(enhanced_image)

            # Convert enhanced_pil_alpha to RGBA mode
            enhanced_image_alpha = tensor_util.tensor_convert_rgba(enhanced_image)
            new_seg_image = (
                enhanced_image.numpy()
            )  # alpha should not be applied to seg_image
            # Apply the mask
            mask = tensor_util.tensor_resize(
                mask, *tensor_util.tensor_get_size(enhanced_image)
            )
            tensor_util.tensor_putalpha(enhanced_image_alpha, mask)
            enhanced_alpha_list.append(enhanced_image_alpha)

            cropped_list.append(orig_cropped_image)  # NOTE: Don't use `cropped_image`

            new_seg = SEGS.SEG(
                new_seg_image,
                seg.cropped_mask,
                seg.confidence,
                seg.crop_region,
                seg.bbox,
                seg.label,
                seg.control_net_wrapper,
            )
            new_segs.append(new_seg)

        image_tensor = tensor_util.tensor_convert_rgb(image)

        cropped_list.sort(key=lambda x: x.shape, reverse=True)
        enhanced_list.sort(key=lambda x: x.shape, reverse=True)
        enhanced_alpha_list.sort(key=lambda x: x.shape, reverse=True)

        return (
            image_tensor,
            cropped_list,
            enhanced_list,
            enhanced_alpha_list,
            cnet_pil_list,
            (segs[0], new_segs),
        )


def empty_pil_tensor(w=64, h=64):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


class DetailerForEachTest(DetailerForEach):
    def doit(
        self,
        image,
        segs,
        model,
        clip,
        vae,
        guide_size,
        guide_size_for,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        denoise,
        feather,
        noise_mask,
        force_inpaint,
        wildcard,
        detailer_hook=None,
        cycle=1,
        inpaint_model=False,
        noise_mask_feather=0,
        scheduler_func_opt=None,
    ):

        (
            enhanced_img,
            cropped,
            cropped_enhanced,
            cropped_enhanced_alpha,
            cnet_pil_list,
            new_segs,
        ) = DetailerForEach.do_detail(
            image,
            segs,
            model,
            clip,
            vae,
            guide_size,
            guide_size_for,
            max_size,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            denoise,
            feather,
            noise_mask,
            force_inpaint,
            wildcard,
            detailer_hook,
            cycle=cycle,
            inpaint_model=inpaint_model,
            noise_mask_feather=noise_mask_feather,
            scheduler_func_opt=scheduler_func_opt,
        )

        cnet_pil_list = [empty_pil_tensor()]

        return (
            enhanced_img,
            cropped,
            cropped_enhanced,
            cropped_enhanced_alpha,
            cnet_pil_list,
        )

import math
import torch
from typing import Any, Dict, Optional, Tuple
from src.AutoDetailer import AD_util, bbox, tensor_util, SEGS
from src.Utilities import util
from src.AutoEncoders import VariationalAE
from src.Device import Device
from src.sample import ksampler_util, samplers, sampling, sampling_util


class DifferentialDiffusion:
    def apply(self, model):
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)
        return (model,)

    def forward(self, sigma, denoise_mask, extra_options):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        ts_from = model.inner_model.model_sampling.timestep(step_sigmas[0])
        ts_to = model.inner_model.model_sampling.timestep(model.inner_model.model_sampling.sigma_min)
        threshold = (model.inner_model.model_sampling.timestep(sigma[0]) - ts_to) / (ts_from - ts_to)
        return (denoise_mask >= threshold).to(denoise_mask.dtype)


def crop_condition_mask(mask, image, crop_region):
    x1, y1, x2, y2 = crop_region
    if len(mask.shape) == 4:
        return mask[:, y1:y2, x1:x2, :]
    elif len(mask.shape) == 3:
        return mask[y1:y2, x1:x2, :]
    elif len(mask.shape) == 2:
        return mask[y1:y2, x1:x2]
    raise ValueError(f"Unsupported mask shape: {mask.shape}")


def to_latent_image(pixels, vae):
    return VariationalAE.VAEEncode().encode(vae, pixels)[0]


def calculate_sigmas2(model, sampler, scheduler, steps):
    return ksampler_util.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)


def get_noise_sampler(x, cpu, total_sigmas, **kwargs):
    if "extra_args" in kwargs and "seed" in kwargs["extra_args"]:
        sigma_min, sigma_max = total_sigmas[total_sigmas > 0].min(), total_sigmas.max()
        return sampling_util.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=kwargs["extra_args"].get("seed"), cpu=cpu)
    return None


def ksampler2(sampler_name, total_sigmas, extra_options={}, inpaint_options={}, pipeline=False, disable_multiscale=True):
    if disable_multiscale:
        extra_options = {**extra_options, "enable_multiscale": False, "multiscale_factor": 1.0,
                        "multiscale_fullres_start": 0, "multiscale_fullres_end": 0, "multiscale_intermittent_fullres": False}
    if sampler_name == "dpmpp_2m_sde":
        def sample_dpmpp_sde(model, x, sigmas, pipeline, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler:
                kwargs["noise_sampler"] = noise_sampler
            return samplers.sample_dpmpp_2m_sde(model, x, sigmas, pipeline=pipeline, **kwargs)
        return sampling.KSAMPLER(sample_dpmpp_sde, extra_options, inpaint_options)
    return sampling.ksampler(sampler_name, pipeline=pipeline, extra_options=extra_options)


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        return ksampler_util.prepare_noise(input_latent["samples"], self.seed,
            input_latent.get("batch_index"), seeds_per_sample=None)


def sample_with_custom_noise(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image, noise=None, callback=None, pipeline=False):
    out = {**latent_image, "samples": latent_image["samples"]}
    if noise is None:
        noise = Noise_RandomNoise(noise_seed).generate_noise(out)
    device = Device.get_torch_device()
    noise, latent = noise.to(device), latent_image["samples"].to(device)
    noise_mask = latent_image.get("noise_mask")
    if noise_mask is not None:
        noise_mask = noise_mask.to(device)
    samples = sampling.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent,
        noise_mask=noise_mask, callback=callback, disable_pbar=not util.PROGRESS_BAR_ENABLED, seed=noise_seed, pipeline=pipeline)
    out["samples"] = samples.to(Device.intermediate_device())
    return out, out


def separated_sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                     start_at_step, end_at_step, return_with_leftover_noise, sigma_ratio=1.0, sampler_opt=None,
                     noise=None, callback=None, scheduler_func=None, pipeline=False):
    total_sigmas = calculate_sigmas2(model, sampler_name, scheduler, steps)
    sigmas = total_sigmas[start_at_step:] * sigma_ratio if start_at_step else total_sigmas
    return sample_with_custom_noise(model, add_noise, seed, cfg, positive, negative,
        ksampler2(sampler_name, total_sigmas, pipeline=pipeline), sigmas, latent_image, noise=noise, callback=callback, pipeline=pipeline)[1]


def ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                     refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None,
                     sigma_factor=1.0, noise=None, callback=None, scheduler_func=None, pipeline=False):
    advanced_steps = math.floor(steps / denoise)
    return separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler, positive, negative,
        latent_image, advanced_steps - steps, advanced_steps - steps + steps, False,
        sigma_ratio=sigma_factor, noise=noise, callback=callback, scheduler_func=scheduler_func, pipeline=pipeline)


def _compute_detailer_resize(width, height, guide_size, max_size):
    upscale = guide_size / min(width, height)
    new_w, new_h = int(width * upscale), int(height * upscale)
    if new_w > max_size or new_h > max_size:
        upscale *= max_size / max(new_w, new_h)
        new_w, new_h = int(width * upscale), int(height * upscale)
    # Round dimensions to nearest multiple of 8 for VAE compatibility.
    # Non-divisible-by-8 dimensions cause NaN in VAE encode when tiled
    # encoding is used, because round(dim * 0.125) != dim // 8.
    new_w = max(8, (new_w + 4) // 8 * 8)
    new_h = max(8, (new_h + 4) // 8 * 8)
    force_inpaint = False
    if upscale <= 1.0 or new_w == 0 or new_h == 0:
        force_inpaint = True
        upscale, new_w, new_h = 1.0, width, height
        # Also round when force inpaint to keep VAE compatibility
        new_w = max(8, (new_w + 4) // 8 * 8)
        new_h = max(8, (new_h + 4) // 8 * 8)
    return upscale, new_w, new_h, force_inpaint


def enhance_detail(image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, bbox, seed, steps, cfg,
                   sampler_name, scheduler, positive, negative, denoise, noise_mask, force_inpaint,
                   wildcard_opt=None, wildcard_opt_concat_mode=None, detailer_hook=None, refiner_ratio=None,
                   refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None,
                   control_net_wrapper=None, cycle=1, inpaint_model=False, noise_mask_feather=0,
                   callback=None, scheduler_func=None, pipeline=False):
    if noise_mask is not None:
        noise_mask = tensor_util.tensor_gaussian_blur_mask(noise_mask, noise_mask_feather).squeeze(3)
    h, w = image.shape[1], image.shape[2]
    upscale, new_w, new_h, force_inpaint = _compute_detailer_resize(w, h, guide_size, max_size)
    if force_inpaint:
        print("Detailer: force inpaint")
    print(f"Detailer: segment upscale for ({bbox[2]-bbox[0]}, {bbox[3]-bbox[1]}) | crop region {w, h} x {upscale} -> {new_w, new_h}")

    upscaled_image = tensor_util.tensor_resize(image, new_w, new_h)
    
    latent_image = to_latent_image(upscaled_image, vae)
    
    if noise_mask is not None:
        latent_image["noise_mask"] = noise_mask

    refined_latent = latent_image
    for i in range(cycle):
        refined_latent = ksampler_wrapper(model, seed + i, steps, cfg, sampler_name, scheduler, positive, negative,
            refined_latent, denoise, refiner_ratio, refiner_model, refiner_clip, refiner_positive, refiner_negative,
            noise=None, callback=callback, scheduler_func=scheduler_func, pipeline=pipeline)
    
    try:
        refined_image = vae.decode(refined_latent["samples"])
    except Exception:
        # Standard tile size for SDXL VAE to avoid artifacts
        refined_image = vae.decode_tiled(refined_latent["samples"], tile_x=256, tile_y=256)

    return tensor_util.tensor_resize(refined_image, w, h).cpu(), None


class DetailerForEach:
    @staticmethod
    def do_detail(image, segs, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg,
                  sampler_name, scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint,
                  wildcard_opt=None, detailer_hook=None, refiner_ratio=None, refiner_model=None, refiner_clip=None,
                  refiner_positive=None, refiner_negative=None, cycle=1, inpaint_model=False, noise_mask_feather=0,
                  callback=None, scheduler_func_opt=None, pipeline=False):
        image = image.clone()
        enhanced_alpha_list, enhanced_list, cropped_list, cnet_pil_list, new_segs = [], [], [], [], []
        segs = AD_util.segs_scale_match(segs, image.shape)
        wmode, wildcard_chooser = bbox.process_wildcard_for_segs(wildcard_opt)

        if noise_mask_feather > 0 and "denoise_mask_function" not in model.model_options:
            model = DifferentialDiffusion().apply(model)[0]

        for i, seg in enumerate(segs[1]):
            # Check for interrupt before each segment
            from src.user import app_instance
            app = getattr(app_instance, "app", None)
            if app and getattr(app, "interrupt_flag", False):
                print(f"Detailer: Interrupt requested, stopping at segment {i}")
                break

            cropped_image = tensor_util.to_tensor(AD_util.crop_ndarray4(image.cpu().numpy(), seg.crop_region))
            mask = tensor_util.tensor_gaussian_blur_mask(tensor_util.to_tensor(seg.cropped_mask), feather)
            if (seg.cropped_mask == 0).all().item():
                print("Detailer: segment skip [empty mask]")
                continue

            seg_seed, wildcard_item = wildcard_chooser.get(seg)
            seg_seed = seed + i if seg_seed is None else seg_seed

            crop_h, crop_w = int(cropped_image.shape[1]), int(cropped_image.shape[2])
            _, crop_new_w, crop_new_h, _ = _compute_detailer_resize(crop_w, crop_h, guide_size, max_size)

            def crop_cond(cond_list):
                if cond_list is None:
                    return None
                
                # Extract crop region coordinates
                x1, y1, x2, y2 = [int(round(c)) for c in seg.crop_region]
                
                res = []
                for entry in cond_list:
                    if isinstance(entry, (list, tuple)) and len(entry) > 1 and isinstance(entry[1], dict):
                        new_dict = entry[1].copy()
                        # Apply mask cropping if present
                        if "mask" in new_dict:
                            new_dict["mask"] = crop_condition_mask(new_dict["mask"], image, seg.crop_region)
                        
                        # CRITICAL: Preserve pooled_output for SDXL
                        if "pooled_output" in entry[1]:
                            new_dict["pooled_output"] = entry[1]["pooled_output"]
                        
                        # Inject SDXL size conditioning for the crop
                        # Use crop-local dimensions to match the actual sampling resolution.
                        new_dict["width"] = crop_new_w
                        new_dict["height"] = crop_new_h
                        new_dict["crop_w"] = 0
                        new_dict["crop_h"] = 0
                        new_dict["target_width"] = crop_new_w
                        new_dict["target_height"] = crop_new_h
                        
                        res.append([entry[0], new_dict])
                    else:
                        res.append(entry)
                return res

            orig_cropped_image = cropped_image.clone()
            enhanced_image, cnet_pils = enhance_detail(cropped_image, model, clip, vae, guide_size, guide_size_for_bbox,
                max_size, seg.bbox, seg_seed, steps, cfg, sampler_name, scheduler, crop_cond(positive), crop_cond(negative),
                denoise, seg.cropped_mask, force_inpaint, wildcard_opt=wildcard_item, wildcard_opt_concat_mode=None,
                detailer_hook=detailer_hook, refiner_ratio=refiner_ratio, refiner_model=refiner_model, refiner_clip=refiner_clip,
                refiner_positive=refiner_positive, refiner_negative=refiner_negative, control_net_wrapper=seg.control_net_wrapper,
                cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather,
                callback=callback, scheduler_func=scheduler_func_opt, pipeline=pipeline)

            if enhanced_image is not None:
                image = image.cpu()
                tensor_util.tensor_paste(image, enhanced_image.cpu(), (seg.crop_region[0], seg.crop_region[1]), mask)
                enhanced_list.append(enhanced_image)

            enhanced_image_alpha = tensor_util.tensor_convert_rgba(enhanced_image)
            mask = tensor_util.tensor_resize(mask, *tensor_util.tensor_get_size(enhanced_image))
            tensor_util.tensor_putalpha(enhanced_image_alpha, mask)
            enhanced_alpha_list.append(enhanced_image_alpha)
            cropped_list.append(orig_cropped_image)
            new_segs.append(SEGS.SEG(enhanced_image.numpy(), seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper))

        for lst in [cropped_list, enhanced_list, enhanced_alpha_list]:
            lst.sort(key=lambda x: x.shape, reverse=True)
        return tensor_util.tensor_convert_rgb(image), cropped_list, enhanced_list, enhanced_alpha_list, cnet_pil_list, (segs[0], new_segs)


def empty_pil_tensor(w=64, h=64):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


class DetailerForEachTest(DetailerForEach):
    def doit(self, image, segs, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg,
             sampler_name, scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint,
             wildcard, detailer_hook=None, cycle=1, inpaint_model=False, noise_mask_feather=0,
             callback=None, scheduler_func_opt=None, pipeline=False):
        if len(image.shape) == 4 and image.shape[0] > 1:
            batch_size = image.shape[0]
            results = [[], [], [], [], []]
            for i in range(batch_size):
                # Check for interrupt before each batch item
                from src.user import app_instance
                app = getattr(app_instance, "app", None)
                if app and getattr(app, "interrupt_flag", False):
                    print(f"ADetailer: Interrupt requested, stopping at batch item {i}")
                    break

                enhanced, cropped, enh, enh_alpha, cnet, _ = DetailerForEach.do_detail(
                    image[i:i+1], segs, model, clip, vae, guide_size, guide_size_for, max_size, seed + i, steps,
                    cfg, sampler_name, scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint,
                    wildcard, detailer_hook, cycle=cycle, inpaint_model=inpaint_model,
                    noise_mask_feather=noise_mask_feather, callback=callback, scheduler_func_opt=scheduler_func_opt, pipeline=pipeline)
                results[0].append(enhanced)
                results[1].extend(cropped)
                results[2].extend(enh)
                results[3].extend(enh_alpha)
                results[4].extend(cnet)
            return torch.cat(results[0], dim=0), results[1], results[2], results[3], results[4] or [empty_pil_tensor()]

        enhanced, cropped, enh, enh_alpha, cnet, _ = DetailerForEach.do_detail(
            image, segs, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
            scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint, wildcard, detailer_hook,
            cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather,
            callback=callback, scheduler_func_opt=scheduler_func_opt, pipeline=pipeline)
        return enhanced, cropped, enh, enh_alpha, [empty_pil_tensor()]

import math
import torch
from typing import Any, Dict, Optional, Tuple

from modules.AutoDetailer import AD_util, bbox, tensor_util
from modules.AutoDetailer import SEGS
from modules.Utilities import util
from modules.AutoEncoders import VariationalAE
from modules.Device import Device
from modules.sample import ksampler_util, samplers, sampling, sampling_util

# FIXME: Improve slow inference times


class DifferentialDiffusion:
    """#### Class for applying differential diffusion to a model."""

    def apply(self, model: torch.nn.Module) -> Tuple[torch.nn.Module]:
        """#### Apply differential diffusion to a model.

        #### Args:
            - `model` (torch.nn.Module): The input model.

        #### Returns:
            - `Tuple[torch.nn.Module]`: The modified model.
        """
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)
        return (model,)

    def forward(
        self,
        sigma: torch.Tensor,
        denoise_mask: torch.Tensor,
        extra_options: Dict[str, Any],
    ) -> torch.Tensor:
        """#### Forward function for differential diffusion.

        #### Args:
            - `sigma` (torch.Tensor): The sigma tensor.
            - `denoise_mask` (torch.Tensor): The denoise mask tensor.
            - `extra_options` (Dict[str, Any]): Additional options.

        #### Returns:
            - `torch.Tensor`: The processed denoise mask tensor.
        """
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to)

        return (denoise_mask >= threshold).to(denoise_mask.dtype)


def to_latent_image(pixels: torch.Tensor, vae: VariationalAE.VAE) -> torch.Tensor:
    """#### Convert pixels to a latent image using a VAE.

    #### Args:
        - `pixels` (torch.Tensor): The input pixel tensor.
        - `vae` (VariationalAE.VAE): The VAE model.

    #### Returns:
        - `torch.Tensor`: The latent image tensor.
    """
    pixels.shape[1]
    pixels.shape[2]
    return VariationalAE.VAEEncode().encode(vae, pixels)[0]


def calculate_sigmas2(
    model: torch.nn.Module, sampler: str, scheduler: str, steps: int
) -> torch.Tensor:
    """#### Calculate sigmas for a model.

    #### Args:
        - `model` (torch.nn.Module): The input model.
        - `sampler` (str): The sampler name.
        - `scheduler` (str): The scheduler name.
        - `steps` (int): The number of steps.

    #### Returns:
        - `torch.Tensor`: The calculated sigmas.
    """
    return ksampler_util.calculate_sigmas(
        model.get_model_object("model_sampling"), scheduler, steps
    )


def get_noise_sampler(
    x: torch.Tensor, cpu: bool, total_sigmas: torch.Tensor, **kwargs
) -> Optional[sampling_util.BrownianTreeNoiseSampler]:
    """#### Get a noise sampler.

    #### Args:
        - `x` (torch.Tensor): The input tensor.
        - `cpu` (bool): Whether to use CPU.
        - `total_sigmas` (torch.Tensor): The total sigmas tensor.
        - `kwargs` (dict): Additional arguments.

    #### Returns:
        - `Optional[sampling_util.BrownianTreeNoiseSampler]`: The noise sampler.
    """
    if "extra_args" in kwargs and "seed" in kwargs["extra_args"]:
        sigma_min, sigma_max = total_sigmas[total_sigmas > 0].min(), total_sigmas.max()
        seed = kwargs["extra_args"].get("seed", None)
        return sampling_util.BrownianTreeNoiseSampler(
            x, sigma_min, sigma_max, seed=seed, cpu=cpu
        )
    return None


def ksampler2(
    sampler_name: str,
    total_sigmas: torch.Tensor,
    extra_options: Dict[str, Any] = {},
    inpaint_options: Dict[str, Any] = {},
    pipeline: bool = False,
) -> sampling.KSAMPLER:
    """#### Get a ksampler.

    #### Args:
        - `sampler_name` (str): The sampler name.
        - `total_sigmas` (torch.Tensor): The total sigmas tensor.
        - `extra_options` (Dict[str, Any], optional): Additional options. Defaults to {}.
        - `inpaint_options` (Dict[str, Any], optional): Inpaint options. Defaults to {}.
        - `pipeline` (bool, optional): Whether to use pipeline. Defaults to False.

    #### Returns:
        - `sampling.KSAMPLER`: The ksampler.
    """
    if sampler_name == "dpmpp_2m_sde":

        def sample_dpmpp_sde(model, x, sigmas, pipeline, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs["noise_sampler"] = noise_sampler

            return samplers.sample_dpmpp_2m_sde(
                model, x, sigmas, pipeline=pipeline, **kwargs
            )

        sampler_function = sample_dpmpp_sde

    else:
        return sampling.sampler_object(sampler_name, pipeline=pipeline)

    return sampling.KSAMPLER(sampler_function, extra_options, inpaint_options)


class Noise_RandomNoise:
    """#### Class for generating random noise."""

    def __init__(self, seed: int):
        """#### Initialize the Noise_RandomNoise class.

        #### Args:
            - `seed` (int): The seed for random noise.
        """
        self.seed = seed

    def generate_noise(self, input_latent: Dict[str, torch.Tensor]) -> torch.Tensor:
        """#### Generate random noise.

        #### Args:
            - `input_latent` (Dict[str, torch.Tensor]): The input latent tensor.

        #### Returns:
            - `torch.Tensor`: The generated noise tensor.
        """
        latent_image = input_latent["samples"]
        batch_inds = (
            input_latent["batch_index"] if "batch_index" in input_latent else None
        )
        return ksampler_util.prepare_noise(latent_image, self.seed, batch_inds)


def sample_with_custom_noise(
    model: torch.nn.Module,
    add_noise: bool,
    noise_seed: int,
    cfg: int,
    positive: Any,
    negative: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    latent_image: Dict[str, torch.Tensor],
    noise: Optional[torch.Tensor] = None,
    callback: Optional[callable] = None,
    pipeline: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """#### Sample with custom noise.

    #### Args:
        - `model` (torch.nn.Module): The input model.
        - `add_noise` (bool): Whether to add noise.
        - `noise_seed` (int): The noise seed.
        - `cfg` (int): Classifier-Free Guidance Scale
        - `positive` (Any): The positive prompt.
        - `negative` (Any): The negative prompt.
        - `sampler` (Any): The sampler.
        - `sigmas` (torch.Tensor): The sigmas tensor.
        - `latent_image` (Dict[str, torch.Tensor]): The latent image tensor.
        - `noise` (Optional[torch.Tensor], optional): The noise tensor. Defaults to None.
        - `callback` (Optional[callable], optional): The callback function. Defaults to None.
        - `pipeline` (bool, optional): Whether to use pipeline. Defaults to False.

    #### Returns:
        - `Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]`: The sampled and denoised tensors.
    """
    latent = latent_image
    latent_image = latent["samples"]

    out = latent.copy()
    out["samples"] = latent_image

    if noise is None:
        noise = Noise_RandomNoise(noise_seed).generate_noise(out)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

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
        pipeline=pipeline,
    )

    samples = samples.to(Device.intermediate_device())

    out["samples"] = samples
    out_denoised = out
    return out, out_denoised


def separated_sample(
    model: torch.nn.Module,
    add_noise: bool,
    seed: int,
    steps: int,
    cfg: int,
    sampler_name: str,
    scheduler: str,
    positive: Any,
    negative: Any,
    latent_image: Dict[str, torch.Tensor],
    start_at_step: Optional[int],
    end_at_step: Optional[int],
    return_with_leftover_noise: bool,
    sigma_ratio: float = 1.0,
    sampler_opt: Optional[Dict[str, Any]] = None,
    noise: Optional[torch.Tensor] = None,
    callback: Optional[callable] = None,
    scheduler_func: Optional[callable] = None,
    pipeline: bool = False,
) -> Dict[str, torch.Tensor]:
    """#### Perform separated sampling.

    #### Args:
        - `model` (torch.nn.Module): The input model.
        - `add_noise` (bool): Whether to add noise.
        - `seed` (int): The seed for random noise.
        - `steps` (int): The number of steps.
        - `cfg` (int): Classifier-Free Guidance Scale
        - `sampler_name` (str): The sampler name.
        - `scheduler` (str): The scheduler name.
        - `positive` (Any): The positive prompt.
        - `negative` (Any): The negative prompt.
        - `latent_image` (Dict[str, torch.Tensor]): The latent image tensor.
        - `start_at_step` (Optional[int]): The step to start at.
        - `end_at_step` (Optional[int]): The step to end at.
        - `return_with_leftover_noise` (bool): Whether to return with leftover noise.
        - `sigma_ratio` (float, optional): The sigma ratio. Defaults to 1.0.
        - `sampler_opt` (Optional[Dict[str, Any]], optional): The sampler options. Defaults to None.
        - `noise` (Optional[torch.Tensor], optional): The noise tensor. Defaults to None.
        - `callback` (Optional[callable], optional): The callback function. Defaults to None.
        - `scheduler_func` (Optional[callable], optional): The scheduler function. Defaults to None.
        - `pipeline` (bool, optional): Whether to use pipeline. Defaults to False.

    #### Returns:
        - `Dict[str, torch.Tensor]`: The sampled tensor.
    """
    total_sigmas = calculate_sigmas2(model, sampler_name, scheduler, steps)

    sigmas = total_sigmas

    if start_at_step is not None:
        sigmas = sigmas[start_at_step:] * sigma_ratio

    impact_sampler = ksampler2(sampler_name, total_sigmas, pipeline=pipeline)

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
        pipeline=pipeline,
    )

    return res[1]


def ksampler_wrapper(
    model: torch.nn.Module,
    seed: int,
    steps: int,
    cfg: int,
    sampler_name: str,
    scheduler: str,
    positive: Any,
    negative: Any,
    latent_image: Dict[str, torch.Tensor],
    denoise: float,
    refiner_ratio: Optional[float] = None,
    refiner_model: Optional[torch.nn.Module] = None,
    refiner_clip: Optional[Any] = None,
    refiner_positive: Optional[Any] = None,
    refiner_negative: Optional[Any] = None,
    sigma_factor: float = 1.0,
    noise: Optional[torch.Tensor] = None,
    scheduler_func: Optional[callable] = None,
    pipeline: bool = False,
) -> Dict[str, torch.Tensor]:
    """#### Wrapper for ksampler.

    #### Args:
        - `model` (torch.nn.Module): The input model.
        - `seed` (int): The seed for random noise.
        - `steps` (int): The number of steps.
        - `cfg` (int): Classifier-Free Guidance Scale
        - `sampler_name` (str): The sampler name.
        - `scheduler` (str): The scheduler name.
        - `positive` (Any): The positive prompt.
        - `negative` (Any): The negative prompt.
        - `latent_image` (Dict[str, torch.Tensor]): The latent image tensor.
        - `denoise` (float): The denoise factor.
        - `refiner_ratio` (Optional[float], optional): The refiner ratio. Defaults to None.
        - `refiner_model` (Optional[torch.nn.Module], optional): The refiner model. Defaults to None.
        - `refiner_clip` (Optional[Any], optional): The refiner clip. Defaults to None.
        - `refiner_positive` (Optional[Any], optional): The refiner positive prompt. Defaults to None.
        - `refiner_negative` (Optional[Any], optional): The refiner negative prompt. Defaults to None.
        - `sigma_factor` (float, optional): The sigma factor. Defaults to 1.0.
        - `noise` (Optional[torch.Tensor], optional): The noise tensor. Defaults to None.
        - `scheduler_func` (Optional[callable], optional): The scheduler function. Defaults to None.
        - `pipeline` (bool, optional): Whether to use pipeline. Defaults to False.

    #### Returns:
        - `Dict[str, torch.Tensor]`: The refined latent tensor.
    """
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
        pipeline=pipeline,
    )

    return refined_latent


def enhance_detail(
    image: torch.Tensor,
    model: torch.nn.Module,
    clip: Any,
    vae: VariationalAE.VAE,
    guide_size: int,
    guide_size_for_bbox: bool,
    max_size: int,
    bbox: Tuple[int, int, int, int],
    seed: int,
    steps: int,
    cfg: int,
    sampler_name: str,
    scheduler: str,
    positive: Any,
    negative: Any,
    denoise: float,
    noise_mask: Optional[torch.Tensor],
    force_inpaint: bool,
    wildcard_opt: Optional[Any] = None,
    wildcard_opt_concat_mode: Optional[Any] = None,
    detailer_hook: Optional[callable] = None,
    refiner_ratio: Optional[float] = None,
    refiner_model: Optional[torch.nn.Module] = None,
    refiner_clip: Optional[Any] = None,
    refiner_positive: Optional[Any] = None,
    refiner_negative: Optional[Any] = None,
    control_net_wrapper: Optional[Any] = None,
    cycle: int = 1,
    inpaint_model: bool = False,
    noise_mask_feather: int = 0,
    scheduler_func: Optional[callable] = None,
    pipeline: bool = False,
) -> Tuple[torch.Tensor, Optional[Any]]:
    """#### Enhance detail of an image.

    #### Args:
        - `image` (torch.Tensor): The input image tensor.
        - `model` (torch.nn.Module): The model.
        - `clip` (Any): The clip model.
        - `vae` (VariationalAE.VAE): The VAE model.
        - `guide_size` (int): The guide size.
        - `guide_size_for_bbox` (bool): Whether to use guide size for bbox.
        - `max_size` (int): The maximum size.
        - `bbox` (Tuple[int, int, int, int]): The bounding box.
        - `seed` (int): The seed for random noise.
        - `steps` (int): The number of steps.
        - `cfg` (int): Classifier-Free Guidance Scale
        - `sampler_name` (str): The sampler name.
        - `scheduler` (str): The scheduler name.
        - `positive` (Any): The positive prompt.
        - `negative` (Any): The negative prompt.
        - `denoise` (float): The denoise factor.
        - `noise_mask` (Optional[torch.Tensor]): The noise mask tensor.
        - `force_inpaint` (bool): Whether to force inpaint.
        - `wildcard_opt` (Optional[Any], optional): The wildcard options. Defaults to None.
        - `wildcard_opt_concat_mode` (Optional[Any], optional): The wildcard concat mode. Defaults to None.
        - `detailer_hook` (Optional[callable], optional): The detailer hook. Defaults to None.
        - `refiner_ratio` (Optional[float], optional): The refiner ratio. Defaults to None.
        - `refiner_model` (Optional[torch.nn.Module], optional): The refiner model. Defaults to None.
        - `refiner_clip` (Optional[Any], optional): The refiner clip. Defaults to None.
        - `refiner_positive` (Optional[Any], optional): The refiner positive prompt. Defaults to None.
        - `refiner_negative` (Optional[Any], optional): The refiner negative prompt. Defaults to None.
        - `control_net_wrapper` (Optional[Any], optional): The control net wrapper. Defaults to None.
        - `cycle` (int, optional): The number of cycles. Defaults to 1.
        - `inpaint_model` (bool, optional): Whether to use inpaint model. Defaults to False.
        - `noise_mask_feather` (int, optional): The noise mask feather. Defaults to 0.
        - `scheduler_func` (Optional[callable], optional): The scheduler function. Defaults to None.
        - `pipeline` (bool, optional): Whether to use pipeline. Defaults to False.

    #### Returns:
        - `Tuple[torch.Tensor, Optional[Any]]`: The refined image tensor and optional cnet_pils.
    """
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
        print("Detailer: force inpaint")
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
            _upscaled_latent2,
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
            pipeline=pipeline,
        )

    # non-latent downscale - latent downscale cause bad quality
    try:
        # try to decode image normally
        refined_image = vae.decode(refined_latent["samples"])
    except Exception:
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
    """#### Class for detailing each segment of an image."""

    @staticmethod
    def do_detail(
        image: torch.Tensor,
        segs: Tuple[torch.Tensor, Any],
        model: torch.nn.Module,
        clip: Any,
        vae: VariationalAE.VAE,
        guide_size: int,
        guide_size_for_bbox: bool,
        max_size: int,
        seed: int,
        steps: int,
        cfg: int,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        denoise: float,
        feather: int,
        noise_mask: Optional[torch.Tensor],
        force_inpaint: bool,
        wildcard_opt: Optional[Any] = None,
        detailer_hook: Optional[callable] = None,
        refiner_ratio: Optional[float] = None,
        refiner_model: Optional[torch.nn.Module] = None,
        refiner_clip: Optional[Any] = None,
        refiner_positive: Optional[Any] = None,
        refiner_negative: Optional[Any] = None,
        cycle: int = 1,
        inpaint_model: bool = False,
        noise_mask_feather: int = 0,
        scheduler_func_opt: Optional[callable] = None,
        pipeline: bool = False,
    ) -> Tuple[torch.Tensor, list, list, list, list, Tuple[torch.Tensor, list]]:
        """#### Perform detailing on each segment of an image.

        #### Args:
            - `image` (torch.Tensor): The input image tensor.
            - `segs` (Tuple[torch.Tensor, Any]): The segments.
            - `model` (torch.nn.Module): The model.
            - `clip` (Any): The clip model.
            - `vae` (VariationalAE.VAE): The VAE model.
            - `guide_size` (int): The guide size.
            - `guide_size_for_bbox` (bool): Whether to use guide size for bbox.
            - `max_size` (int): The maximum size.
            - `seed` (int): The seed for random noise.
            - `steps` (int): The number of steps.
            - `cfg` (int): Classifier-Free Guidance Scale.
            - `sampler_name` (str): The sampler name.
            - `scheduler` (str): The scheduler name.
            - `positive` (Any): The positive prompt.
            - `negative` (Any): The negative prompt.
            - `denoise` (float): The denoise factor.
            - `feather` (int): The feather value.
            - `noise_mask` (Optional[torch.Tensor]): The noise mask tensor.
            - `force_inpaint` (bool): Whether to force inpaint.
            - `wildcard_opt` (Optional[Any], optional): The wildcard options. Defaults to None.
            - `detailer_hook` (Optional[callable], optional): The detailer hook. Defaults to None.
            - `refiner_ratio` (Optional[float], optional): The refiner ratio. Defaults to None.
            - `refiner_model` (Optional[torch.nn.Module], optional): The refiner model. Defaults to None.
            - `refiner_clip` (Optional[Any], optional): The refiner clip. Defaults to None.
            - `refiner_positive` (Optional[Any], optional): The refiner positive prompt. Defaults to None.
            - `refiner_negative` (Optional[Any], optional): The refiner negative prompt. Defaults to None.
            - `cycle` (int, optional): The number of cycles. Defaults to 1.
            - `inpaint_model` (bool, optional): Whether to use inpaint model. Defaults to False.
            - `noise_mask_feather` (int, optional): The noise mask feather. Defaults to 0.
            - `scheduler_func_opt` (Optional[callable], optional): The scheduler function. Defaults to None.
            - `pipeline` (bool, optional): Whether to use pipeline. Defaults to False.

        #### Returns:
            - `Tuple[torch.Tensor, list, list, list, list, Tuple[torch.Tensor, list]]`: The detailed image tensor, cropped list, enhanced list, enhanced alpha list, cnet PIL list, and new segments.
        """
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
                print("Detailer: segment skip [empty mask]")
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
                pipeline=pipeline,
            )

            if enhanced_image is not None:
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


def empty_pil_tensor(w: int = 64, h: int = 64) -> torch.Tensor:
    """#### Create an empty PIL tensor.

    #### Args:
        - `w` (int, optional): The width of the tensor. Defaults to 64.
        - `h` (int, optional): The height of the tensor. Defaults to 64.

    #### Returns:
        - `torch.Tensor`: The empty tensor.
    """
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


class DetailerForEachTest(DetailerForEach):
    """#### Test class for DetailerForEach."""

    def doit(
        self,
        image: torch.Tensor,
        segs: Any,
        model: torch.nn.Module,
        clip: Any,
        vae: VariationalAE.VAE,
        guide_size: int,
        guide_size_for: bool,
        max_size: int,
        seed: int,
        steps: int,
        cfg: Any,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        denoise: float,
        feather: int,
        noise_mask: Optional[torch.Tensor],
        force_inpaint: bool,
        wildcard: Optional[Any],
        detailer_hook: Optional[callable] = None,
        cycle: int = 1,
        inpaint_model: bool = False,
        noise_mask_feather: int = 0,
        scheduler_func_opt: Optional[callable] = None,
        pipeline: bool = False,
    ) -> Tuple[torch.Tensor, list, list, list, list]:
        """#### Perform detail enhancement for testing.

        #### Args:
            - `image` (torch.Tensor): The input image tensor.
            - `segs` (Any): The segments.
            - `model` (torch.nn.Module): The model.
            - `clip` (Any): The clip model.
            - `vae` (VariationalAE.VAE): The VAE model.
            - `guide_size` (int): The guide size.
            - `guide_size_for` (bool): Whether to use guide size for.
            - `max_size` (int): The maximum size.
            - `seed` (int): The seed for random noise.
            - `steps` (int): The number of steps.
            - `cfg` (Any): The configuration.
            - `sampler_name` (str): The sampler name.
            - `scheduler` (str): The scheduler name.
            - `positive` (Any): The positive prompt.
            - `negative` (Any): The negative prompt.
            - `denoise` (float): The denoise factor.
            - `feather` (int): The feather value.
            - `noise_mask` (Optional[torch.Tensor]): The noise mask tensor.
            - `force_inpaint` (bool): Whether to force inpaint.
            - `wildcard` (Optional[Any]): The wildcard options.
            - `detailer_hook` (Optional[callable], optional): The detailer hook. Defaults to None.
            - `cycle` (int, optional): The number of cycles. Defaults to 1.
            - `inpaint_model` (bool, optional): Whether to use inpaint model. Defaults to False.
            - `noise_mask_feather` (int, optional): The noise mask feather. Defaults to 0.
            - `scheduler_func_opt` (Optional[callable], optional): The scheduler function. Defaults to None.
            - `pipeline` (bool, optional): Whether to use pipeline. Defaults to False.

        #### Returns:
            - `Tuple[torch.Tensor, list, list, list, list]`: The enhanced image tensor, cropped list, cropped enhanced list, cropped enhanced alpha list, and cnet PIL list.
        """
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
            pipeline=pipeline,
        )

        cnet_pil_list = [empty_pil_tensor()]

        return (
            enhanced_img,
            cropped,
            cropped_enhanced,
            cropped_enhanced_alpha,
            cnet_pil_list,
        )

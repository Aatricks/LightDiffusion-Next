"""Ultimate SD Upscale - tiled upscaling with seam fix."""
from src.AutoEncoders import VariationalAE
from src.sample import sampling
from src.UltimateSDUpscale import USDU_upscaler, image_util
import torch
from PIL import ImageFilter, ImageDraw, Image
from enum import Enum
import math

state = USDU_upscaler.state


class UnsupportedModel(Exception):
    pass


class StableDiffusionProcessing:
    """Container for SD processing parameters."""
    def __init__(self, init_img: Image.Image, model, positive, negative, vae, seed, steps, cfg,
                 sampler_name, scheduler, denoise, upscale_by, uniform_tile_mode):
        self.init_images = [init_img]
        self.image_mask = None
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width, self.height = init_img.width, init_img.height
        self.model, self.positive, self.negative, self.vae = model, positive, negative, vae
        self.seed, self.steps, self.cfg = seed, steps, cfg
        self.sampler_name, self.scheduler, self.denoise = sampler_name, scheduler, denoise
        self.init_size = (init_img.width, init_img.height)
        self.upscale_by, self.uniform_tile_mode = upscale_by, uniform_tile_mode
        self.extra_generation_params = {}


class Processed:
    """Container for processed images."""
    def __init__(self, p, images, seed, info):
        self.images, self.seed, self.info = images, seed, info

    def infotext(self, p, index):
        return None


def fix_seed(p):
    pass


def process_images(p, pipeline=False):
    """Process tiles using inpainting."""
    image_mask = p.image_mask.convert("L")
    init_image = p.init_images[0]

    crop_region = image_util.get_crop_region(image_mask, p.inpaint_full_res_padding)
    x1, y1, x2, y2 = crop_region
    crop_ratio = (x2 - x1) / (y2 - y1)
    p_ratio = p.width / p.height
    
    if crop_ratio > p_ratio:
        target_width, target_height = x2 - x1, round((x2 - x1) / p_ratio)
    else:
        target_width, target_height = round((y2 - y1) * p_ratio), y2 - y1
    
    crop_region, _ = image_util.expand_crop(crop_region, image_mask.width, image_mask.height, target_width, target_height)
    tile_size = (p.width, p.height)

    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    tiles = [img.crop(crop_region) for img in USDU_upscaler.batch]
    initial_tile_size = tiles[0].size
    tiles = [t.resize(tile_size, Image.Resampling.LANCZOS) if t.size != tile_size else t for t in tiles]

    positive_cropped = image_util.crop_cond(p.positive, crop_region, p.init_size, init_image.size, tile_size)
    negative_cropped = image_util.crop_cond(p.negative, crop_region, p.init_size, init_image.size, tile_size)

    batched_tiles = torch.cat([image_util.pil_to_tensor(t) for t in tiles], dim=0)
    (latent,) = VariationalAE.VAEEncode().encode(p.vae, batched_tiles)
    
    # Auto-detect Flux for disabling multi-scale and setting correct flags
    model_sampling_obj = p.model.get_model_object("model_sampling")
    from src.sample.sampling import ModelSamplingFlux, ModelSamplingFlux2
    is_flux = isinstance(model_sampling_obj, (ModelSamplingFlux, ModelSamplingFlux2))
    is_flux2 = isinstance(model_sampling_obj, ModelSamplingFlux2)

    # Pass crop offsets for positional embedding coherence (Critical for Flux/DiT)
    model_options = p.model.model_options.copy()
    transformer_options = model_options.get("transformer_options", {}).copy()
    transformer_options["top"] = y1
    transformer_options["left"] = x1
    model_options["transformer_options"] = transformer_options

    (samples,) = sampling.common_ksampler(p.model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler,
                                          positive_cropped, negative_cropped, latent, denoise=p.denoise, 
                                          pipeline=pipeline, flux=is_flux, flux2=is_flux2,
                                          model_options=model_options)
    (decoded,) = VariationalAE.VAEDecode().decode(p.vae, samples)

    for i, tile_sampled in enumerate([image_util.tensor_to_pil(decoded, j) for j in range(len(decoded))]):
        init_image = USDU_upscaler.batch[i]
        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

        image_tile_only = Image.new("RGBA", init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])
        temp = image_tile_only.copy()
        temp.putalpha(image_mask.resize(temp.size))
        image_tile_only.paste(temp, image_tile_only)
        result = init_image.convert("RGBA")
        result.alpha_composite(image_tile_only)
        USDU_upscaler.batch[i] = result.convert("RGB")

    return Processed(p, [USDU_upscaler.batch[0]], p.seed, None)


class USDUMode(Enum):
    LINEAR, CHESS, NONE = 0, 1, 2


class USDUSFMode(Enum):
    NONE, BAND_PASS, HALF_TILE, HALF_TILE_PLUS_INTERSECTIONS = 0, 1, 2, 3


class USDUpscaler:
    """Main upscaler class."""
    def __init__(self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height):
        self.p, self.image = p, image
        self.scale_factor = math.ceil(max(p.width, p.height) / max(image.width, image.height))
        self.upscaler = USDU_upscaler.sd_upscalers[upscaler_index]
        self.redraw = USDURedraw()
        self.redraw.tile_width = tile_width or tile_height
        self.redraw.tile_height = tile_height or tile_width
        self.seams_fix = USDUSeamsFix()
        self.seams_fix.tile_width = self.redraw.tile_width
        self.seams_fix.tile_height = self.redraw.tile_height
        self.initial_info = None
        self.rows = math.ceil(self.p.height / self.redraw.tile_height)
        self.cols = math.ceil(self.p.width / self.redraw.tile_width)

    def get_factor(self, num):
        if num == 1: return 2
        for f in [4, 3, 2]:
            if num % f == 0: return f
        return 0

    def get_factors(self):
        scales, current = [], 1
        while current < self.scale_factor:
            f = self.get_factor(self.scale_factor // current)
            scales.append(f)
            current *= f
        self.scales = enumerate(scales)

    def upscale(self):
        print(f"Canva: {self.p.width}x{self.p.height}, Image: {self.image.width}x{self.image.height}, Scale: {self.scale_factor}")
        self.get_factors()
        for idx, val in self.scales:
            print(f"Upscaling iteration {idx + 1} with scale factor {val}")
            self.image = self.upscaler.scaler.upscale(self.image, val, self.upscaler.data_path)
        self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)

    def setup_redraw(self, mode, padding, mask_blur):
        self.redraw.mode = USDUMode(mode)
        self.redraw.enabled = self.redraw.mode != USDUMode.NONE
        self.redraw.padding = padding
        self.p.mask_blur = mask_blur

    def setup_seams_fix(self, padding, denoise, mask_blur, width, mode):
        self.seams_fix.padding, self.seams_fix.denoise = padding, denoise
        self.seams_fix.mask_blur, self.seams_fix.width = mask_blur, width
        self.seams_fix.mode = USDUSFMode(mode)
        self.seams_fix.enabled = self.seams_fix.mode != USDUSFMode.NONE

    def calc_jobs_count(self):
        global state
        redraw = (self.rows * self.cols) if self.redraw.enabled else 0
        seams = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols
        state.job_count = redraw + seams

    def print_info(self):
        print(f"Tile: {self.redraw.tile_width}x{self.redraw.tile_height}, Grid: {self.rows}x{self.cols}")

    def add_extra_info(self):
        self.p.extra_generation_params.update({
            "Ultimate SD upscale upscaler": self.upscaler.name,
            "Ultimate SD upscale tile_width": self.redraw.tile_width,
            "Ultimate SD upscale tile_height": self.redraw.tile_height,
        })

    def process(self, pipeline):
        USDU_upscaler.state.begin()
        self.calc_jobs_count()
        self.result_images = []
        if self.redraw.enabled:
            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols, pipeline)
            self.initial_info = self.redraw.initial_info
        self.result_images.append(self.image)
        if self.seams_fix.enabled:
            self.image = self.seams_fix.start(self.p, self.image, self.rows, self.cols, pipeline)
            self.initial_info = self.seams_fix.initial_info
            self.result_images.append(self.image)
        USDU_upscaler.state.end()


class USDURedraw:
    """Tile redraw functionality."""
    def init_draw(self, p, width, height):
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64
        mask = Image.new("L", (width, height), "black")
        return mask, ImageDraw.Draw(mask)

    def calc_rectangle(self, xi, yi):
        return xi * self.tile_width, yi * self.tile_height, (xi + 1) * self.tile_width, (yi + 1) * self.tile_height

    def linear_process(self, p, image, rows, cols, pipeline=False):
        global state
        mask, draw = self.init_draw(p, image.width, image.height)
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted: break
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images, p.image_mask = [image], mask
                processed = process_images(p, pipeline)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if processed.images: image = processed.images[0]
        p.width, p.height = image.width, image.height
        self.initial_info = processed.infotext(p, 0)
        return image

    def start(self, p, image, rows, cols, pipeline=False):
        self.initial_info = None
        return self.linear_process(p, image, rows, cols, pipeline)


class USDUSeamsFix:
    """Seam fixing functionality."""
    def init_draw(self, p):
        self.initial_info = None
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64

    def half_tile_process(self, p, image, rows, cols, pipeline=False):
        global state
        self.init_draw(p)
        processed = None
        gradient = Image.linear_gradient("L")
        
        row_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        row_gradient.paste(gradient.resize((self.tile_width, self.tile_height // 2), Image.BICUBIC), (0, 0))
        row_gradient.paste(gradient.rotate(180).resize((self.tile_width, self.tile_height // 2), Image.BICUBIC), (0, self.tile_height // 2))
        
        col_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        col_gradient.paste(gradient.rotate(90).resize((self.tile_width // 2, self.tile_height), Image.BICUBIC), (0, 0))
        col_gradient.paste(gradient.rotate(270).resize((self.tile_width // 2, self.tile_height), Image.BICUBIC), (self.tile_width // 2, 0))

        p.denoising_strength, p.mask_blur = self.denoise, self.mask_blur

        for yi in range(rows - 1):
            for xi in range(cols):
                p.width, p.height = self.tile_width, self.tile_height
                p.inpaint_full_res, p.inpaint_full_res_padding = True, self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(row_gradient, (xi * self.tile_width, yi * self.tile_height + self.tile_height // 2))
                p.init_images, p.image_mask = [image], mask
                processed = process_images(p, pipeline)
                if processed.images: image = processed.images[0]

        for yi in range(rows):
            for xi in range(cols - 1):
                p.width, p.height = self.tile_width, self.tile_height
                p.inpaint_full_res, p.inpaint_full_res_padding = True, self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(col_gradient, (xi * self.tile_width + self.tile_width // 2, yi * self.tile_height))
                p.init_images, p.image_mask = [image], mask
                processed = process_images(p, pipeline)
                if processed.images: image = processed.images[0]

        p.width, p.height = image.width, image.height
        if processed: self.initial_info = processed.infotext(p, 0)
        return image

    def start(self, p, image, rows, cols, pipeline=False):
        return self.half_tile_process(p, image, rows, cols, pipeline)


class Script(USDU_upscaler.Script):
    """Main script runner."""
    def run(self, p, _, tile_width, tile_height, mask_blur, padding, seams_fix_width, seams_fix_denoise,
            seams_fix_padding, upscaler_index, save_upscaled_image, redraw_mode, save_seams_fix_image,
            seams_fix_mask_blur, seams_fix_type, target_size_type, custom_width, custom_height, custom_scale, pipeline=False):
        fix_seed(p)
        USDU_upscaler.torch_gc()
        p.do_not_save_grid = p.do_not_save_samples = True
        p.inpaint_full_res = False
        p.inpainting_fill, p.n_iter, p.batch_size = 1, 1, 1

        init_img = image_util.flatten(p.init_images[0], USDU_upscaler.opts.img2img_background_color)
        p.width = math.ceil((init_img.width * custom_scale) / 64) * 64
        p.height = math.ceil((init_img.height * custom_scale) / 64) * 64

        upscaler = USDUpscaler(p, init_img, upscaler_index, save_upscaled_image, save_seams_fix_image, tile_width, tile_height)
        upscaler.upscale()
        upscaler.setup_redraw(redraw_mode, padding, mask_blur)
        upscaler.setup_seams_fix(seams_fix_padding, seams_fix_denoise, seams_fix_mask_blur, seams_fix_width, seams_fix_type)
        upscaler.print_info()
        upscaler.add_extra_info()
        upscaler.process(pipeline)
        return Processed(p, upscaler.result_images, p.seed, upscaler.initial_info or "")


# Monkey-patch overrides
_old_init = USDUpscaler.__init__
def _new_init(self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height):
    # Determine downscale factor from model (8 for SD, 16 for Flux)
    downscale_factor = 8
    try:
        latent_format = p.model.get_model_object("latent_format")
        if hasattr(latent_format, "downscale_factor"):
            downscale_factor = latent_format.downscale_factor
    except Exception:
        pass
        
    p.width = math.ceil((image.width * p.upscale_by) / downscale_factor) * downscale_factor
    p.height = math.ceil((image.height * p.upscale_by) / downscale_factor) * downscale_factor
    _old_init(self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height)
USDUpscaler.__init__ = _new_init

_old_redraw = USDURedraw.init_draw
def _new_redraw(self, p, width, height):
    mask, draw = _old_redraw(self, p, width, height)
    
    downscale_factor = 8
    try:
        latent_format = p.model.get_model_object("latent_format")
        if hasattr(latent_format, "downscale_factor"):
            downscale_factor = latent_format.downscale_factor
    except Exception:
        pass
        
    p.width = math.ceil((self.tile_width + self.padding) / downscale_factor) * downscale_factor
    p.height = math.ceil((self.tile_height + self.padding) / downscale_factor) * downscale_factor
    return mask, draw
USDURedraw.init_draw = _new_redraw

_old_seams = USDUSeamsFix.init_draw
def _new_seams(self, p):
    _old_seams(self, p)
    
    downscale_factor = 8
    try:
        latent_format = p.model.get_model_object("latent_format")
        if hasattr(latent_format, "downscale_factor"):
            downscale_factor = latent_format.downscale_factor
    except Exception:
        pass
        
    p.width = math.ceil((self.tile_width + self.padding) / downscale_factor) * downscale_factor
    p.height = math.ceil((self.tile_height + self.padding) / downscale_factor) * downscale_factor
USDUSeamsFix.init_draw = _new_seams

_old_upscale = USDUpscaler.upscale
def _new_upscale(self):
    _old_upscale(self)
    USDU_upscaler.batch = [self.image] + [img.resize((self.p.width, self.p.height), Image.LANCZOS) for img in USDU_upscaler.batch[1:]]
USDUpscaler.upscale = _new_upscale

MAX_RESOLUTION = 8192
MODES = {"Linear": USDUMode.LINEAR, "Chess": USDUMode.CHESS, "None": USDUMode.NONE}
SEAM_FIX_MODES = {"None": USDUSFMode.NONE, "Band Pass": USDUSFMode.BAND_PASS, "Half Tile": USDUSFMode.HALF_TILE, "Half Tile + Intersections": USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS}


class UltimateSDUpscale:
    """Main entry point for Ultimate SD Upscale."""
    def upscale(self, image, model, positive, negative, vae, upscale_by, seed, steps, cfg, sampler_name,
                scheduler, denoise, upscale_model, mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur, seam_fix_width, seam_fix_padding,
                force_uniform_tiles, pipeline=False):
        USDU_upscaler.sd_upscalers[0] = USDU_upscaler.UpscalerData()
        USDU_upscaler.actual_upscaler = upscale_model
        USDU_upscaler.batch = [image_util.tensor_to_pil(image, i) for i in range(len(image))]

        sdprocessing = StableDiffusionProcessing(
            image_util.tensor_to_pil(image), model, positive, negative, vae, seed, steps, cfg,
            sampler_name, scheduler, denoise, upscale_by, force_uniform_tiles)

        Script().run(
            p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height, mask_blur=mask_blur,
            padding=tile_padding, seams_fix_width=seam_fix_width, seams_fix_denoise=seam_fix_denoise,
            seams_fix_padding=seam_fix_padding, upscaler_index=0, save_upscaled_image=False,
            redraw_mode=MODES[mode_type], save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
            seams_fix_type=SEAM_FIX_MODES[seam_fix_mode], target_size_type=2, custom_width=None,
            custom_height=None, custom_scale=upscale_by, pipeline=pipeline)

        return (torch.cat([image_util.pil_to_tensor(img) for img in USDU_upscaler.batch], dim=0),)

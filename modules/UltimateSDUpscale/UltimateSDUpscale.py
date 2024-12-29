from modules.AutoEncoders import VariationalAE
from modules.sample import sampling
from modules.UltimateSDUpscale import USDU_upscaler, image_util
import torch
from PIL import ImageFilter, ImageDraw, Image
from enum import Enum
import math


class UnsupportedModel(Exception):
    pass


class StableDiffusionProcessing:
    def __init__(
        self,
        init_img,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_by,
        uniform_tile_mode,
    ):
        # Variables used by the USDU script
        self.init_images = [init_img]
        self.image_mask = None
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width = init_img.width
        self.height = init_img.height

        self.model = model
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height
        self.upscale_by = upscale_by
        self.uniform_tile_mode = uniform_tile_mode

        # Other required A1111 variables for the USDU script that is currently unused in this script
        self.extra_generation_params = {}


class Processed:
    def __init__(
        self, p: StableDiffusionProcessing, images: list, seed: int, info: str
    ):
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index):
        return None


def fix_seed(p: StableDiffusionProcessing):
    pass


def process_images(p: StableDiffusionProcessing) -> Processed:
    # Where the main image generation happens in A1111

    # Setup
    image_mask = p.image_mask.convert("L")
    init_image = p.init_images[0]

    # Locate the white region of the mask outlining the tile and add padding
    crop_region = image_util.get_crop_region(image_mask, p.inpaint_full_res_padding)

    x1, y1, x2, y2 = crop_region
    crop_width = x2 - x1
    crop_height = y2 - y1
    crop_ratio = crop_width / crop_height
    p_ratio = p.width / p.height
    if crop_ratio > p_ratio:
        target_width = crop_width
        target_height = round(crop_width / p_ratio)
    else:
        target_width = round(crop_height * p_ratio)
        target_height = crop_height
    crop_region, _ = image_util.expand_crop(
        crop_region,
        image_mask.width,
        image_mask.height,
        target_width,
        target_height,
    )
    tile_size = p.width, p.height

    # Blur the mask
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Crop the images to get the tiles that will be used for generation
    global batch
    tiles = [img.crop(crop_region) for img in batch]

    # Assume the same size for all images in the batch
    initial_tile_size = tiles[0].size

    # Resize if necessary
    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    # Crop conditioning
    positive_cropped = image_util.crop_cond(
        p.positive, crop_region, p.init_size, init_image.size, tile_size
    )
    negative_cropped = image_util.crop_cond(
        p.negative, crop_region, p.init_size, init_image.size, tile_size
    )

    # Encode the image
    vae_encoder = VariationalAE.VAEEncode()
    batched_tiles = torch.cat([image_util.pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = vae_encoder.encode(p.vae, batched_tiles)

    # Generate samples
    (samples,) = sampling.common_ksampler(
        p.model,
        p.seed,
        p.steps,
        p.cfg,
        p.sampler_name,
        p.scheduler,
        positive_cropped,
        negative_cropped,
        latent,
        denoise=p.denoise,
    )

    # Decode the sample
    vae_decoder = VariationalAE.VAEDecode()
    (decoded,) = vae_decoder.decode(p.vae, samples)

    # Convert the sample to a PIL image
    tiles_sampled = [image_util.tensor_to_pil(decoded, i) for i in range(len(decoded))]

    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = batch[i]

        # Resize back to the original size
        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(
                initial_tile_size, Image.Resampling.LANCZOS
            )

        # Put the tile into position
        image_tile_only = Image.new("RGBA", init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        # Add the mask as an alpha channel
        # Must make a copy due to the possibility of an edge becoming black
        temp = image_tile_only.copy()
        image_mask = image_mask.resize(temp.size)
        temp.putalpha(image_mask)
        temp.putalpha(image_mask)
        image_tile_only.paste(temp, image_tile_only)

        # Add back the tile to the initial image according to the mask in the alpha channel
        result = init_image.convert("RGBA")
        result.alpha_composite(image_tile_only)

        # Convert back to RGB
        result = result.convert("RGB")
        batch[i] = result

    processed = Processed(p, [batch[0]], p.seed, None)
    return processed


class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2


class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3


class USDUpscaler:
    def __init__(
        self,
        p,
        image,
        upscaler_index: int,
        save_redraw,
        save_seams_fix,
        tile_width,
        tile_height,
    ) -> None:
        self.p: StableDiffusionProcessing = p
        self.image: Image = image
        self.scale_factor = math.ceil(
            max(p.width, p.height) / max(image.width, image.height)
        )
        global sd_upscalers
        self.upscaler = sd_upscalers[upscaler_index]
        self.redraw = USDURedraw()
        self.redraw.save = save_redraw
        self.redraw.tile_width = tile_width if tile_width > 0 else tile_height
        self.redraw.tile_height = tile_height if tile_height > 0 else tile_width
        self.seams_fix = USDUSeamsFix()
        self.seams_fix.save = save_seams_fix
        self.seams_fix.tile_width = tile_width if tile_width > 0 else tile_height
        self.seams_fix.tile_height = tile_height if tile_height > 0 else tile_width
        self.initial_info = None
        self.rows = math.ceil(self.p.height / self.redraw.tile_height)
        self.cols = math.ceil(self.p.width / self.redraw.tile_width)

    def get_factor(self, num):
        # Its just return, don't need elif
        if num == 1:
            return 2
        if num % 4 == 0:
            return 4
        if num % 3 == 0:
            return 3
        if num % 2 == 0:
            return 2
        return 0

    def get_factors(self):
        scales = []
        current_scale = 1
        current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale < self.scale_factor:
            current_scale_factor = self.get_factor(self.scale_factor // current_scale)
            scales.append(current_scale_factor)
            current_scale = current_scale * current_scale_factor
        self.scales = enumerate(scales)

    def upscale(self):
        # Log info
        print(f"Canva size: {self.p.width}x{self.p.height}")
        print(f"Image size: {self.image.width}x{self.image.height}")
        print(f"Scale factor: {self.scale_factor}")
        # Get list with scale factors
        self.get_factors()
        # Upscaling image over all factors
        for index, value in self.scales:
            print(f"Upscaling iteration {index + 1} with scale factor {value}")
            self.image = self.upscaler.scaler.upscale(
                self.image, value, self.upscaler.data_path
            )
        # Resize image to set values
        self.image = self.image.resize(
            (self.p.width, self.p.height), resample=Image.LANCZOS
        )

    def setup_redraw(self, redraw_mode, padding, mask_blur):
        self.redraw.mode = USDUMode(redraw_mode)
        self.redraw.enabled = self.redraw.mode != USDUMode.NONE
        self.redraw.padding = padding
        self.p.mask_blur = mask_blur

    def setup_seams_fix(self, padding, denoise, mask_blur, width, mode):
        self.seams_fix.padding = padding
        self.seams_fix.denoise = denoise
        self.seams_fix.mask_blur = mask_blur
        self.seams_fix.width = width
        self.seams_fix.mode = USDUSFMode(mode)
        self.seams_fix.enabled = self.seams_fix.mode != USDUSFMode.NONE

    def calc_jobs_count(self):
        redraw_job_count = (self.rows * self.cols) if self.redraw.enabled else 0
        seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols
        global state
        state.job_count = redraw_job_count + seams_job_count

    def print_info(self):
        print(f"Tile size: {self.redraw.tile_width}x{self.redraw.tile_height}")
        print(f"Tiles amount: {self.rows * self.cols}")
        print(f"Grid: {self.rows}x{self.cols}")
        print(f"Redraw enabled: {self.redraw.enabled}")
        print(f"Seams fix mode: {self.seams_fix.mode.name}")

    def add_extra_info(self):
        self.p.extra_generation_params["Ultimate SD upscale upscaler"] = (
            self.upscaler.name
        )
        self.p.extra_generation_params["Ultimate SD upscale tile_width"] = (
            self.redraw.tile_width
        )
        self.p.extra_generation_params["Ultimate SD upscale tile_height"] = (
            self.redraw.tile_height
        )
        self.p.extra_generation_params["Ultimate SD upscale mask_blur"] = (
            self.p.mask_blur
        )
        self.p.extra_generation_params["Ultimate SD upscale padding"] = (
            self.redraw.padding
        )

    def process(self):
        global state
        state.begin()
        self.calc_jobs_count()
        self.result_images = []
        if self.redraw.enabled:
            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.redraw.initial_info
        self.result_images.append(self.image)

        if self.seams_fix.enabled:
            self.image = self.seams_fix.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.seams_fix.initial_info
            self.result_images.append(self.image)
        state.end()


class USDURedraw:
    def init_draw(self, p, width, height):
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        return mask, draw

    def calc_rectangle(self, xi, yi):
        x1 = xi * self.tile_width
        y1 = yi * self.tile_height
        x2 = xi * self.tile_width + self.tile_width
        y2 = yi * self.tile_height + self.tile_height

        return x1, y1, x2, y2

    def linear_process(self, p, image, rows, cols):
        global state
        mask, draw = self.init_draw(p, image.width, image.height)
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if len(processed.images) > 0:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        self.initial_info = None
        return self.linear_process(p, image, rows, cols)


class USDUSeamsFix:
    def init_draw(self, p):
        self.initial_info = None
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64

    def half_tile_process(self, p, image, rows, cols):
        global state
        self.init_draw(p)
        processed = None

        gradient = Image.linear_gradient("L")
        row_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        row_gradient.paste(
            gradient.resize(
                (self.tile_width, self.tile_height // 2), resample=Image.BICUBIC
            ),
            (0, 0),
        )
        row_gradient.paste(
            gradient.rotate(180).resize(
                (self.tile_width, self.tile_height // 2), resample=Image.BICUBIC
            ),
            (0, self.tile_height // 2),
        )
        col_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        col_gradient.paste(
            gradient.rotate(90).resize(
                (self.tile_width // 2, self.tile_height), resample=Image.BICUBIC
            ),
            (0, 0),
        )
        col_gradient.paste(
            gradient.rotate(270).resize(
                (self.tile_width // 2, self.tile_height), resample=Image.BICUBIC
            ),
            (self.tile_width // 2, 0),
        )

        p.denoising_strength = self.denoise
        p.mask_blur = self.mask_blur

        for yi in range(rows - 1):
            for xi in range(cols):
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(
                    row_gradient,
                    (
                        xi * self.tile_width,
                        yi * self.tile_height + self.tile_height // 2,
                    ),
                )

                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                if len(processed.images) > 0:
                    image = processed.images[0]

        for yi in range(rows):
            for xi in range(cols - 1):
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(
                    col_gradient,
                    (
                        xi * self.tile_width + self.tile_width // 2,
                        yi * self.tile_height,
                    ),
                )

                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                if len(processed.images) > 0:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        return self.half_tile_process(p, image, rows, cols)


class Script(USDU_upscaler.Script):
    def run(
        self,
        p,
        _,
        tile_width,
        tile_height,
        mask_blur,
        padding,
        seams_fix_width,
        seams_fix_denoise,
        seams_fix_padding,
        upscaler_index,
        save_upscaled_image,
        redraw_mode,
        save_seams_fix_image,
        seams_fix_mask_blur,
        seams_fix_type,
        target_size_type,
        custom_width,
        custom_height,
        custom_scale,
    ):
        # Init
        fix_seed(p)
        USDU_upscaler.torch_gc()

        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.inpaint_full_res = False

        p.inpainting_fill = 1
        p.n_iter = 1
        p.batch_size = 1

        seed = p.seed

        # Init image
        init_img = p.init_images[0]
        init_img = image_util.flatten(
            init_img, USDU_upscaler.opts.img2img_background_color
        )

        p.width = math.ceil((init_img.width * custom_scale) / 64) * 64
        p.height = math.ceil((init_img.height * custom_scale) / 64) * 64

        # Upscaling
        upscaler = USDUpscaler(
            p,
            init_img,
            upscaler_index,
            save_upscaled_image,
            save_seams_fix_image,
            tile_width,
            tile_height,
        )
        upscaler.upscale()

        # Drawing
        upscaler.setup_redraw(redraw_mode, padding, mask_blur)
        upscaler.setup_seams_fix(
            seams_fix_padding,
            seams_fix_denoise,
            seams_fix_mask_blur,
            seams_fix_width,
            seams_fix_type,
        )
        upscaler.print_info()
        upscaler.add_extra_info()
        upscaler.process()
        result_images = upscaler.result_images

        return Processed(
            p,
            result_images,
            seed,
            upscaler.initial_info if upscaler.initial_info is not None else "",
        )


#
# Instead of using multiples of 64, use multiples of 8
#

# Upscaler
old_init = USDUpscaler.__init__


def new_init(
    self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height
):
    p.width = math.ceil((image.width * p.upscale_by) / 8) * 8
    p.height = math.ceil((image.height * p.upscale_by) / 8) * 8
    old_init(
        self,
        p,
        image,
        upscaler_index,
        save_redraw,
        save_seams_fix,
        tile_width,
        tile_height,
    )


USDUpscaler.__init__ = new_init

# Redraw
old_setup_redraw = USDURedraw.init_draw


def new_setup_redraw(self, p, width, height):
    mask, draw = old_setup_redraw(self, p, width, height)
    p.width = math.ceil((self.tile_width + self.padding) / 8) * 8
    p.height = math.ceil((self.tile_height + self.padding) / 8) * 8
    return mask, draw


USDURedraw.init_draw = new_setup_redraw

# Seams fix
old_setup_seams_fix = USDUSeamsFix.init_draw


def new_setup_seams_fix(self, p):
    old_setup_seams_fix(self, p)
    p.width = math.ceil((self.tile_width + self.padding) / 8) * 8
    p.height = math.ceil((self.tile_height + self.padding) / 8) * 8


USDUSeamsFix.init_draw = new_setup_seams_fix

#
# Make the script upscale on a batch of images instead of one image
#

old_upscale = USDUpscaler.upscale


def new_upscale(self):
    old_upscale(self)
    global batch
    batch = [self.image] + [
        img.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
        for img in batch[1:]
    ]


USDUpscaler.upscale = new_upscale
MAX_RESOLUTION = 8192
# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": USDUMode.LINEAR,
    "Chess": USDUMode.CHESS,
    "None": USDUMode.NONE,
}
# The seam fix modes
SEAM_FIX_MODES = {
    "None": USDUSFMode.NONE,
    "Band Pass": USDUSFMode.BAND_PASS,
    "Half Tile": USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


class UltimateSDUpscale:
    def upscale(
        self,
        image,
        model,
        positive,
        negative,
        vae,
        upscale_by,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_model,
        mode_type,
        tile_width,
        tile_height,
        mask_blur,
        tile_padding,
        seam_fix_mode,
        seam_fix_denoise,
        seam_fix_mask_blur,
        seam_fix_width,
        seam_fix_padding,
        force_uniform_tiles,
    ):
        #
        # Set up A1111 patches
        #

        # Upscaler
        # An object that the script works with
        global sd_upscalers, actual_upscaler, batch
        sd_upscalers[0] = USDU_upscaler.UpscalerData()
        # Where the actual upscaler is stored, will be used when the script upscales using the Upscaler in UpscalerData
        actual_upscaler = upscale_model

        # Set the batch of images
        batch = [image_util.tensor_to_pil(image, i) for i in range(len(image))]

        # Processing
        sdprocessing = StableDiffusionProcessing(
            image_util.tensor_to_pil(image),
            model,
            positive,
            negative,
            vae,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
            upscale_by,
            force_uniform_tiles,
        )

        #
        # Running the script
        #
        script = Script()
        script.run(
            p=sdprocessing,
            _=None,
            tile_width=tile_width,
            tile_height=tile_height,
            mask_blur=mask_blur,
            padding=tile_padding,
            seams_fix_width=seam_fix_width,
            seams_fix_denoise=seam_fix_denoise,
            seams_fix_padding=seam_fix_padding,
            upscaler_index=0,
            save_upscaled_image=False,
            redraw_mode=MODES[mode_type],
            save_seams_fix_image=False,
            seams_fix_mask_blur=seam_fix_mask_blur,
            seams_fix_type=SEAM_FIX_MODES[seam_fix_mode],
            target_size_type=2,
            custom_width=None,
            custom_height=None,
            custom_scale=upscale_by,
        )

        # Return the resulting images
        images = [image_util.pil_to_tensor(img) for img in batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)

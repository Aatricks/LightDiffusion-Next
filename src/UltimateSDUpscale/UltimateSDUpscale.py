from src.AutoEncoders import VariationalAE
from src.sample import sampling
from src.UltimateSDUpscale import USDU_upscaler, image_util
import torch
from PIL import ImageFilter, ImageDraw, Image
from enum import Enum
import math

# taken from https://github.com/ssitu/ComfyUI_UltimateSDUpscale

state = USDU_upscaler.state

class UnsupportedModel(Exception):
    """#### Exception raised for unsupported models."""
    pass


class StableDiffusionProcessing:
    """#### Class representing the processing of Stable Diffusion images."""

    def __init__(
        self,
        init_img: Image.Image,
        model: torch.nn.Module,
        positive: str,
        negative: str,
        vae: VariationalAE.VAE,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        upscale_by: float,
        uniform_tile_mode: bool,
    ):
        """
        #### Initialize the StableDiffusionProcessing class.

        #### Args:
            - `init_img` (Image.Image): The initial image.
            - `model` (torch.nn.Module): The model.
            - `positive` (str): The positive prompt.
            - `negative` (str): The negative prompt.
            - `vae` (VariationalAE.VAE): The variational autoencoder.
            - `seed` (int): The seed.
            - `steps` (int): The number of steps.
            - `cfg` (float): The CFG scale.
            - `sampler_name` (str): The sampler name.
            - `scheduler` (str): The scheduler.
            - `denoise` (float): The denoise strength.
            - `upscale_by` (float): The upscale factor.
            - `uniform_tile_mode` (bool): Whether to use uniform tile mode.
        """
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
    """#### Class representing the processed images."""

    def __init__(
        self, p: StableDiffusionProcessing, images: list, seed: int, info: str
    ):
        """
        #### Initialize the Processed class.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `images` (list): The list of images.
            - `seed` (int): The seed.
            - `info` (str): The information string.
        """
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index: int) -> str:
        """
        #### Get the information text.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `index` (int): The index.

        #### Returns:
            - `str`: The information text.
        """
        return None


def fix_seed(p: StableDiffusionProcessing) -> None:
    """
    #### Fix the seed for reproducibility.

    #### Args:
        - `p` (StableDiffusionProcessing): The processing object.
    """
    pass


def process_images(p: StableDiffusionProcessing, pipeline: bool = False) -> Processed:
    """
    #### Process the images.

    #### Args:
        - `p` (StableDiffusionProcessing): The processing object.

    #### Returns:
        - `Processed`: The processed images.
    """
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
    tiles = [img.crop(crop_region) for img in USDU_upscaler.batch]

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
        pipeline=pipeline
    )

    # Decode the sample
    vae_decoder = VariationalAE.VAEDecode()
    (decoded,) = vae_decoder.decode(p.vae, samples)

    # Convert the sample to a PIL image
    tiles_sampled = [image_util.tensor_to_pil(decoded, i) for i in range(len(decoded))]

    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = USDU_upscaler.batch[i]

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
        USDU_upscaler.batch[i] = result

    processed = Processed(p, [USDU_upscaler.batch[0]], p.seed, None)
    return processed


class USDUMode(Enum):
    """#### Enum representing the modes for Ultimate SD Upscale."""
    LINEAR = 0
    CHESS = 1
    NONE = 2


class USDUSFMode(Enum):
    """#### Enum representing the seam fix modes for Ultimate SD Upscale."""
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3


class USDUpscaler:
    """#### Class representing the Ultimate SD Upscaler."""

    def __init__(
        self,
        p: StableDiffusionProcessing,
        image: Image.Image,
        upscaler_index: int,
        save_redraw: bool,
        save_seams_fix: bool,
        tile_width: int,
        tile_height: int,
    ) -> None:
        """
        #### Initialize the USDUpscaler class.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `image` (Image.Image): The image.
            - `upscaler_index` (int): The upscaler index.
            - `save_redraw` (bool): Whether to save the redraw.
            - `save_seams_fix` (bool): Whether to save the seams fix.
            - `tile_width` (int): The tile width.
            - `tile_height` (int): The tile height.
        """
        self.p: StableDiffusionProcessing = p
        self.image: Image = image
        self.scale_factor = math.ceil(
            max(p.width, p.height) / max(image.width, image.height)
        )
        self.upscaler = USDU_upscaler.sd_upscalers[upscaler_index]
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

    def get_factor(self, num: int) -> int:
        """
        #### Get the factor for a given number.

        #### Args:
            - `num` (int): The number.

        #### Returns:
            - `int`: The factor.
        """
        if num == 1:
            return 2
        if num % 4 == 0:
            return 4
        if num % 3 == 0:
            return 3
        if num % 2 == 0:
            return 2
        return 0

    def get_factors(self) -> None:
        """
        #### Get the list of scale factors.
        """
        scales = []
        current_scale = 1
        current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale < self.scale_factor:
            current_scale_factor = self.get_factor(self.scale_factor // current_scale)
            scales.append(current_scale_factor)
            current_scale = current_scale * current_scale_factor
        self.scales = enumerate(scales)

    def upscale(self) -> None:
        """
        #### Upscale the image.
        """
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

    def setup_redraw(self, redraw_mode: int, padding: int, mask_blur: int) -> None:
        """
        #### Set up the redraw.

        #### Args:
            - `redraw_mode` (int): The redraw mode.
            - `padding` (int): The padding.
            - `mask_blur` (int): The mask blur.
        """
        self.redraw.mode = USDUMode(redraw_mode)
        self.redraw.enabled = self.redraw.mode != USDUMode.NONE
        self.redraw.padding = padding
        self.p.mask_blur = mask_blur

    def setup_seams_fix(
        self, padding: int, denoise: float, mask_blur: int, width: int, mode: int
    ) -> None:
        """
        #### Set up the seams fix.

        #### Args:
            - `padding` (int): The padding.
            - `denoise` (float): The denoise strength.
            - `mask_blur` (int): The mask blur.
            - `width` (int): The width.
            - `mode` (int): The mode.
        """
        self.seams_fix.padding = padding
        self.seams_fix.denoise = denoise
        self.seams_fix.mask_blur = mask_blur
        self.seams_fix.width = width
        self.seams_fix.mode = USDUSFMode(mode)
        self.seams_fix.enabled = self.seams_fix.mode != USDUSFMode.NONE

    def calc_jobs_count(self) -> None:
        """
        #### Calculate the number of jobs.
        """
        redraw_job_count = (self.rows * self.cols) if self.redraw.enabled else 0
        seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols
        global state
        state.job_count = redraw_job_count + seams_job_count

    def print_info(self) -> None:
        """
        #### Print the information.
        """
        print(f"Tile size: {self.redraw.tile_width}x{self.redraw.tile_height}")
        print(f"Tiles amount: {self.rows * self.cols}")
        print(f"Grid: {self.rows}x{self.cols}")
        print(f"Redraw enabled: {self.redraw.enabled}")
        print(f"Seams fix mode: {self.seams_fix.mode.name}")

    def add_extra_info(self) -> None:
        """
        #### Add extra information.
        """
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

    def process(self, pipeline) -> None:
        """
        #### Process the image.
        """
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
    """#### Class representing the redraw functionality for Ultimate SD Upscale."""

    def init_draw(self, p: StableDiffusionProcessing, width: int, height: int) -> tuple:
        """
        #### Initialize the draw.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `width` (int): The width.
            - `height` (int): The height.

        #### Returns:
            - `tuple`: The mask and draw objects.
        """
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        return mask, draw

    def calc_rectangle(self, xi: int, yi: int) -> tuple:
        """
        #### Calculate the rectangle coordinates.

        #### Args:
            - `xi` (int): The x index.
            - `yi` (int): The y index.

        #### Returns:
            - `tuple`: The rectangle coordinates.
        """
        x1 = xi * self.tile_width
        y1 = yi * self.tile_height
        x2 = xi * self.tile_width + self.tile_width
        y2 = yi * self.tile_height + self.tile_height

        return x1, y1, x2, y2

    def linear_process(
        self, p: StableDiffusionProcessing, image: Image.Image, rows: int, cols: int, pipeline: bool = False
    ) -> Image.Image:
        """
        #### Perform linear processing.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `image` (Image.Image): The image.
            - `rows` (int): The number of rows.
            - `cols` (int): The number of columns.

        #### Returns:
            - `Image.Image`: The processed image.
        """
        global state
        mask, draw = self.init_draw(p, image.width, image.height)
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p, pipeline)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if len(processed.images) > 0:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p: StableDiffusionProcessing, image: Image.Image, rows: int, cols: int, pipeline: bool = False) -> Image.Image:
        """#### Start the redraw.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `image` (Image.Image): The image.
            - `rows` (int): The number of rows.
            - `cols` (int): The number of columns.
            
        #### Returns:
            - `Image.Image`: The processed image.
        """
        self.initial_info = None
        return self.linear_process(p, image, rows, cols, pipeline=pipeline)


class USDUSeamsFix:
    """#### Class representing the seams fix functionality for Ultimate SD Upscale."""

    def init_draw(self, p: StableDiffusionProcessing) -> None:
        """#### Initialize the draw.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
        """
        self.initial_info = None
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64

    def half_tile_process(
        self, p: StableDiffusionProcessing, image: Image.Image, rows: int, cols: int, pipeline: bool = False
    ) -> Image.Image:
        """#### Perform half-tile processing.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `image` (Image.Image): The image.
            - `rows` (int): The number of rows.
            - `cols` (int): The number of columns.

        #### Returns:
            - `Image.Image`: The processed image.
        """
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
                processed = process_images(p, pipeline)
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
                processed = process_images(p, pipeline)
                if len(processed.images) > 0:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def start(
        self, p: StableDiffusionProcessing, image: Image.Image, rows: int, cols: int, pipeline: bool = False
    ) -> Image.Image:
        """#### Start the seams fix process.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `image` (Image.Image): The image.
            - `rows` (int): The number of rows.
            - `cols` (int): The number of columns.

        #### Returns:
            - `Image.Image`: The processed image.
        """
        return self.half_tile_process(p, image, rows, cols, pipeline=pipeline)


class Script(USDU_upscaler.Script):
    """#### Class representing the script for Ultimate SD Upscale."""

    def run(
        self,
        p: StableDiffusionProcessing,
        _: None,
        tile_width: int,
        tile_height: int,
        mask_blur: int,
        padding: int,
        seams_fix_width: int,
        seams_fix_denoise: float,
        seams_fix_padding: int,
        upscaler_index: int,
        save_upscaled_image: bool,
        redraw_mode: int,
        save_seams_fix_image: bool,
        seams_fix_mask_blur: int,
        seams_fix_type: int,
        target_size_type: int,
        custom_width: int,
        custom_height: int,
        custom_scale: float,
        pipeline: bool = False,
    ) -> Processed:
        """#### Run the script.

        #### Args:
            - `p` (StableDiffusionProcessing): The processing object.
            - `_` (None): Unused parameter.
            - `tile_width` (int): The tile width.
            - `tile_height` (int): The tile height.
            - `mask_blur` (int): The mask blur.
            - `padding` (int): The padding.
            - `seams_fix_width` (int): The seams fix width.
            - `seams_fix_denoise` (float): The seams fix denoise strength.
            - `seams_fix_padding` (int): The seams fix padding.
            - `upscaler_index` (int): The upscaler index.
            - `save_upscaled_image` (bool): Whether to save the upscaled image.
            - `redraw_mode` (int): The redraw mode.
            - `save_seams_fix_image` (bool): Whether to save the seams fix image.
            - `seams_fix_mask_blur` (int): The seams fix mask blur.
            - `seams_fix_type` (int): The seams fix type.
            - `target_size_type` (int): The target size type.
            - `custom_width` (int): The custom width.
            - `custom_height` (int): The custom height.
            - `custom_scale` (float): The custom scale.

        #### Returns:
            - `Processed`: The processed images.
        """
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
        upscaler.process(pipeline=pipeline)
        result_images = upscaler.result_images

        return Processed(
            p,
            result_images,
            seed,
            upscaler.initial_info if upscaler.initial_info is not None else "",
        )


# Upscaler
old_init = USDUpscaler.__init__


def new_init(
    self: USDUpscaler,
    p: StableDiffusionProcessing,
    image: Image.Image,
    upscaler_index: int,
    save_redraw: bool,
    save_seams_fix: bool,
    tile_width: int,
    tile_height: int,
) -> None:
    """#### Initialize the USDUpscaler class with new settings.

    #### Args:
        - `self` (USDUpscaler): The USDUpscaler instance.
        - `p` (StableDiffusionProcessing): The processing object.
        - `image` (Image.Image): The image.
        - `upscaler_index` (int): The upscaler index.
        - `save_redraw` (bool): Whether to save the redraw.
        - `save_seams_fix` (bool): Whether to save the seams fix.
        - `tile_width` (int): The tile width.
        - `tile_height` (int): The tile height.
    """
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


def new_setup_redraw(
    self: USDURedraw, p: StableDiffusionProcessing, width: int, height: int
) -> tuple:
    """#### Set up the redraw with new settings.

    #### Args:
        - `self` (USDURedraw): The USDURedraw instance.
        - `p` (StableDiffusionProcessing): The processing object.
        - `width` (int): The width.
        - `height` (int): The height.

    #### Returns:
        - `tuple`: The mask and draw objects.
    """
    mask, draw = old_setup_redraw(self, p, width, height)
    p.width = math.ceil((self.tile_width + self.padding) / 8) * 8
    p.height = math.ceil((self.tile_height + self.padding) / 8) * 8
    return mask, draw


USDURedraw.init_draw = new_setup_redraw

# Seams fix
old_setup_seams_fix = USDUSeamsFix.init_draw


def new_setup_seams_fix(self: USDUSeamsFix, p: StableDiffusionProcessing) -> None:
    """#### Set up the seams fix with new settings.

    #### Args:
        - `self` (USDUSeamsFix): The USDUSeamsFix instance.
        - `p` (StableDiffusionProcessing): The processing object.
    """
    old_setup_seams_fix(self, p)
    p.width = math.ceil((self.tile_width + self.padding) / 8) * 8
    p.height = math.ceil((self.tile_height + self.padding) / 8) * 8


USDUSeamsFix.init_draw = new_setup_seams_fix

# Make the script upscale on a batch of images instead of one image
old_upscale = USDUpscaler.upscale


def new_upscale(self: USDUpscaler) -> None:
    """#### Upscale a batch of images.

    #### Args:
        - `self` (USDUpscaler): The USDUpscaler instance.
    """
    old_upscale(self)
    USDU_upscaler.batch = [self.image] + [
        img.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
        for img in USDU_upscaler.batch[1:]
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
    """#### Class representing the Ultimate SD Upscale functionality."""

    def upscale(
        self,
        image: torch.Tensor,
        model: torch.nn.Module,
        positive: str,
        negative: str,
        vae: VariationalAE.VAE,
        upscale_by: float,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        upscale_model: any,
        mode_type: str,
        tile_width: int,
        tile_height: int,
        mask_blur: int,
        tile_padding: int,
        seam_fix_mode: str,
        seam_fix_denoise: float,
        seam_fix_mask_blur: int,
        seam_fix_width: int,
        seam_fix_padding: int,
        force_uniform_tiles: bool,
        pipeline: bool = False,
    ) -> tuple:
        """#### Upscale the image.

        #### Args:
            - `image` (torch.Tensor): The image tensor.
            - `model` (torch.nn.Module): The model.
            - `positive` (str): The positive prompt.
            - `negative` (str): The negative prompt.
            - `vae` (VariationalAE.VAE): The variational autoencoder.
            - `upscale_by` (float): The upscale factor.
            - `seed` (int): The seed.
            - `steps` (int): The number of steps.
            - `cfg` (float): The CFG scale.
            - `sampler_name` (str): The sampler name.
            - `scheduler` (str): The scheduler.
            - `denoise` (float): The denoise strength.
            - `upscale_model` (any): The upscale model.
            - `mode_type` (str): The mode type.
            - `tile_width` (int): The tile width.
            - `tile_height` (int): The tile height.
            - `mask_blur` (int): The mask blur.
            - `tile_padding` (int): The tile padding.
            - `seam_fix_mode` (str): The seam fix mode.
            - `seam_fix_denoise` (float): The seam fix denoise strength.
            - `seam_fix_mask_blur` (int): The seam fix mask blur.
            - `seam_fix_width` (int): The seam fix width.
            - `seam_fix_padding` (int): The seam fix padding.
            - `force_uniform_tiles` (bool): Whether to force uniform tiles.

        #### Returns:
            - `tuple`: The resulting tensor.
        """
        # Set up A1111 patches

        # Upscaler
        # An object that the script works with
        USDU_upscaler.sd_upscalers[0] = USDU_upscaler.UpscalerData()
        # Where the actual upscaler is stored, will be used when the script upscales using the Upscaler in UpscalerData
        USDU_upscaler.actual_upscaler = upscale_model

        # Set the batch of images
        USDU_upscaler.batch = [image_util.tensor_to_pil(image, i) for i in range(len(image))]

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

        # Running the script
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
            pipeline=pipeline,
        )

        # Return the resulting images
        images = [image_util.pil_to_tensor(img) for img in USDU_upscaler.batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)

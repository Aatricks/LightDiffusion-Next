import os
import numpy as np
from PIL import Image

output_directory = "./_internal/output"


def get_output_directory() -> str:
    """#### Get the output directory.

    #### Returns:
        - `str`: The output directory.
    """
    global output_directory
    return output_directory


def get_save_image_path(
    filename_prefix: str, output_dir: str, image_width: int = 0, image_height: int = 0
) -> tuple:
    """#### Get the save image path.

    #### Args:
        - `filename_prefix` (str): The filename prefix.
        - `output_dir` (str): The output directory.
        - `image_width` (int, optional): The image width. Defaults to 0.
        - `image_height` (int, optional): The image height. Defaults to 0.

    #### Returns:
        - `tuple`: The full output folder, filename, counter, subfolder, and filename prefix.
    """

    def map_filename(filename: str) -> tuple:
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[: prefix_len + 1]
        try:
            digits = int(filename[prefix_len + 1 :].split("_")[0])
        except:
            digits = 0
        return (digits, prefix)

    def compute_vars(input: str, image_width: int, image_height: int) -> str:
        input = input.replace("%width%", str(image_width))
        input = input.replace("%height%", str(image_height))
        return input

    filename_prefix = compute_vars(filename_prefix, image_width, image_height)

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))

    full_output_folder = os.path.join(output_dir, subfolder)
    try:
        counter = (
            max(
                filter(
                    lambda a: a[1][:-1] == filename and a[1][-1] == "_",
                    map(map_filename, os.listdir(full_output_folder)),
                )
            )[0]
            + 1
        )
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return full_output_folder, filename, counter, subfolder, filename_prefix


MAX_RESOLUTION = 16384


class SaveImage:
    """#### Class for saving images."""

    def __init__(self):
        """#### Initialize the SaveImage class."""
        self.output_dir = get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    def save_images(
        self,
        images: list,
        filename_prefix: str = "LD",
        prompt: str = None,
        extra_pnginfo: dict = None,
    ) -> dict:
        """#### Save images to the output directory.

        #### Args:
            - `images` (list): The list of images.
            - `filename_prefix` (str, optional): The filename prefix. Defaults to "LD".
            - `prompt` (str, optional): The prompt. Defaults to None.
            - `extra_pnginfo` (dict, optional): Additional PNG info. Defaults to None.

        #### Returns:
            - `dict`: The saved images information.
        """
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[-2], images[0].shape[-1]
            )
        )
        results = list()
        for batch_number, image in enumerate(images):
            # Ensure correct shape by squeezing extra dimensions
            i = 255.0 * image.cpu().numpy()
            i = np.squeeze(i)  # Remove extra dimensions
            
            # Ensure we have a valid 3D array (height, width, channels)
            if i.ndim == 4:
                i = i.reshape(-1, i.shape[-2], i.shape[-1])
            
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            if filename_prefix == "LD-HF":
                full_output_folder = os.path.join(full_output_folder, "HiresFix")
            elif filename_prefix == "LD-I2I":
                full_output_folder = os.path.join(full_output_folder, "Img2Img")
            elif filename_prefix == "LD-Flux":
                full_output_folder = os.path.join(full_output_folder, "Flux")
            elif filename_prefix == "LD-Adetailer":
                full_output_folder = os.path.join(full_output_folder, "Adetailer")
            else:
                full_output_folder = os.path.join(full_output_folder, "Classic")
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"images": results}}
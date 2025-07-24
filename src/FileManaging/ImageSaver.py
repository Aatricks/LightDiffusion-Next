import os
import numpy as np
from PIL import Image

output_directory = "./output"


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
        except (ValueError, IndexError):
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
    subfolder_paths = [
        os.path.join(full_output_folder, x)
        for x in ["Classic", "HiresFix", "Img2Img", "Flux", "Adetailer"]
    ]
    for path in subfolder_paths:
        os.makedirs(path, exist_ok=True)
    # Find highest counter across all subfolders
    counter = 1
    for path in subfolder_paths:
        if os.path.exists(path):
            files = os.listdir(path)
            if files:
                numbers = [
                    map_filename(f)[0]
                    for f in files
                    if f.startswith(filename) and f.endswith(".png")
                ]
                if numbers:
                    counter = max(max(numbers) + 1, counter)

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
                filename_prefix,
                self.output_dir,
                images[0].shape[-2],
                images[0].shape[-1],
            )
        )
        results = list()
        for batch_number, image in enumerate(images):
            # Convert tensor to numpy and handle different dimensions
            i = image.cpu().numpy()

            # Handle batched tensors (4D: [batch, channels, height, width] or [batch, height, width, channels])
            if i.ndim == 4:
                # Process each image in the batch separately
                for sub_batch_idx in range(i.shape[0]):
                    sub_image = i[sub_batch_idx]  # Extract single image from batch

                    # Convert to HWC format if in CHW format
                    if sub_image.shape[0] in [1, 3, 4] and sub_image.shape[0] < min(
                        sub_image.shape[1], sub_image.shape[2]
                    ):
                        sub_image = np.transpose(sub_image, (1, 2, 0))  # CHW -> HWC

                    # Squeeze single channel dimension if present
                    if sub_image.shape[-1] == 1:
                        sub_image = sub_image.squeeze(-1)

                    # Scale to 0-255 range
                    sub_image_scaled = np.clip(sub_image * 255.0, 0, 255).astype(
                        np.uint8
                    )

                    img = Image.fromarray(sub_image_scaled)
                    metadata = None

                    filename_with_batch_num = filename.replace(
                        "%batch_num%", str(batch_number)
                    )
                    file = f"{filename_with_batch_num}_{counter:05}_.png"

                    # Save the image to appropriate subfolder
                    save_path = full_output_folder
                    if filename_prefix == "LD-HF":
                        save_path = os.path.join(full_output_folder, "HiresFix")
                    elif filename_prefix == "LD-I2I":
                        save_path = os.path.join(full_output_folder, "Img2Img")
                    elif filename_prefix == "LD-Flux":
                        save_path = os.path.join(full_output_folder, "Flux")
                    elif filename_prefix == "LD-head" or filename_prefix == "LD-body":
                        save_path = os.path.join(full_output_folder, "Adetailer")
                    else:
                        save_path = os.path.join(full_output_folder, "Classic")

                    img.save(
                        os.path.join(save_path, file),
                        pnginfo=metadata,
                        compress_level=self.compress_level,
                    )
                    results.append(
                        {"filename": file, "subfolder": subfolder, "type": self.type}
                    )
                    counter += 1
                continue  # Skip the rest of the loop for this batch

            # Handle 3D tensors (single image: [channels, height, width] or [height, width, channels])
            elif i.ndim == 3:
                # Convert to HWC format if in CHW format
                if i.shape[0] in [1, 3, 4] and i.shape[0] < min(i.shape[1], i.shape[2]):
                    i = np.transpose(i, (1, 2, 0))  # CHW -> HWC

                # Squeeze single channel dimension if present
                if i.shape[-1] == 1:
                    i = i.squeeze(-1)

            # Handle 2D tensors (grayscale: [height, width])
            elif i.ndim == 2:
                pass  # Already in correct format
            else:
                raise ValueError(f"Unexpected tensor dimensions: {i.shape}")

            # Scale to 0-255 range and convert to PIL Image
            i_scaled = np.clip(i * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(i_scaled)
            metadata = None

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            # Save the image to appropriate subfolder
            save_path = full_output_folder
            if filename_prefix == "LD-HF":
                save_path = os.path.join(full_output_folder, "HiresFix")
            elif filename_prefix == "LD-I2I":
                save_path = os.path.join(full_output_folder, "Img2Img")
            elif filename_prefix == "LD-Flux":
                save_path = os.path.join(full_output_folder, "Flux")
            elif filename_prefix == "LD-head" or filename_prefix == "LD-body":
                save_path = os.path.join(full_output_folder, "Adetailer")
            else:
                save_path = os.path.join(full_output_folder, "Classic")

            img.save(
                os.path.join(save_path, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"images": results}}

import os

import numpy as np
from PIL import Image


output_directory = "./_internal/output"

def get_output_directory():
    global output_directory
    return output_directory


def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
    def map_filename(filename):
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[: prefix_len + 1]
        try:
            digits = int(filename[prefix_len + 1 :].split("_")[0])
        except:
            digits = 0
        return (digits, prefix)

    def compute_vars(input, image_width, image_height):
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
    def __init__(self):
        self.output_dir = get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    def save_images(
        self, images, filename_prefix="LD", prompt=None, extra_pnginfo=None
    ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
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
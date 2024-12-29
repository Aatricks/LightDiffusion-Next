from typing import List
import torch


def bislerp(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """#### Perform bilinear interpolation on samples.

    #### Args:
        - `samples` (torch.Tensor): The input samples.
        - `width` (int): The target width.
        - `height` (int): The target height.

    #### Returns:
        - `torch.Tensor`: The interpolated samples.
    """

    def slerp(b1: torch.Tensor, b2: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """#### Perform spherical linear interpolation between two vectors.

        #### Args:
            - `b1` (torch.Tensor): The first vector.
            - `b2` (torch.Tensor): The second vector.
            - `r` (torch.Tensor): The interpolation ratio.

        #### Returns:
            - `torch.Tensor`: The interpolated vector.
        """

        c = b1.shape[-1]

        # norms
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        # normalize
        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms

        # zero when norms are zero
        b1_normalized[b1_norms.expand(-1, c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1, c) == 0.0] = 0.0

        # slerp
        dot = (b1_normalized * b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        # technically not mathematically correct, but more pleasing?
        res = (torch.sin((1.0 - r.squeeze(1)) * omega) / so).unsqueeze(
            1
        ) * b1_normalized + (torch.sin(r.squeeze(1) * omega) / so).unsqueeze(
            1
        ) * b2_normalized
        res *= (b1_norms * (1.0 - r) + b2_norms * r).expand(-1, c)

        # edge cases for same or polar opposites
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0 - r) + b2 * r)[dot < 1e-5 - 1]
        return res

    def generate_bilinear_data(
        length_old: int, length_new: int, device: torch.device
    ) -> List[torch.Tensor]:
        """#### Generate bilinear data for interpolation.

        #### Args:
            - `length_old` (int): The old length.
            - `length_new` (int): The new length.
            - `device` (torch.device): The device to use.

        #### Returns:
            - `torch.Tensor`: The ratios.
            - `torch.Tensor`: The first coordinates.
            - `torch.Tensor`: The second coordinates.
        """
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape(
            (1, 1, 1, -1)
        )
        coords_1 = torch.nn.functional.interpolate(
            coords_1, size=(1, length_new), mode="bilinear"
        )
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2 = (
            torch.arange(length_old, dtype=torch.float32, device=device).reshape(
                (1, 1, 1, -1)
            )
            + 1
        )
        coords_2[:, :, :, -1] -= 1
        coords_2 = torch.nn.functional.interpolate(
            coords_2, size=(1, length_new), mode="bilinear"
        )
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n, c, h, w = samples.shape
    h_new, w_new = (height, width)

    # linear w
    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = samples.gather(-1, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    # linear h
    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1, 1, -1, 1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1, 1, -1, 1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2, coords_1).movedim(1, -1).reshape((-1, c))
    pass_2 = result.gather(-2, coords_2).movedim(1, -1).reshape((-1, c))
    ratios = ratios.movedim(1, -1).reshape((-1, 1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)


def common_upscale(samples: List, width: int, height: int) -> torch.Tensor:
    """#### Upscales the given samples to the specified width and height using the specified method and crop settings.
    #### Args:
        - `samples` (list): The list of samples to be upscaled.
        - `width` (int): The target width for the upscaled samples.
        - `height` (int): The target height for the upscaled samples.
    #### Returns:
        - `torch.Tensor`: The upscaled samples.
    """
    s = samples
    return bislerp(s, width, height)


class LatentUpscale:
    """#### A class to upscale latent codes."""

    def upscale(self, samples: dict, width: int, height: int) -> tuple:
        """#### Upscales the given latent codes.

        #### Args:
            - `samples` (dict): The latent codes to be upscaled.
            - `width` (int): The target width for the upscaled samples.
            - `height` (int): The target height for the upscaled samples.

        #### Returns:
            - `tuple`: The upscaled samples.
        """
        if width == 0 and height == 0:
            s = samples
        else:
            s = samples.copy()
            width = max(64, width)
            height = max(64, height)

            s["samples"] = common_upscale(samples["samples"], width // 8, height // 8)
        return (s,)

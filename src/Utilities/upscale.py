import torch


def bislerp(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Bilinear spherical interpolation for latent upscaling."""

    def slerp(b1, b2, r):
        """Spherical linear interpolation."""
        c = b1.shape[-1]
        b1_n = torch.norm(b1, dim=-1, keepdim=True)
        b2_n = torch.norm(b2, dim=-1, keepdim=True)
        b1_norm = b1 / b1_n
        b2_norm = b2 / b2_n
        b1_norm[b1_n.expand(-1, c) == 0.0] = 0.0
        b2_norm[b2_n.expand(-1, c) == 0.0] = 0.0
        dot = (b1_norm * b2_norm).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)
        res = (torch.sin((1.0 - r.squeeze(1)) * omega) / so).unsqueeze(1) * b1_norm + \
              (torch.sin(r.squeeze(1) * omega) / so).unsqueeze(1) * b2_norm
        res *= (b1_n * (1.0 - r) + b2_n * r).expand(-1, c)
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0 - r) + b2 * r)[dot < 1e-5 - 1]
        return res

    def gen_bilinear(length_old, length_new, device):
        """Generate bilinear interpolation data."""
        c1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape(1, 1, 1, -1)
        c1 = torch.nn.functional.interpolate(c1, size=(1, length_new), mode="bilinear")
        ratios = c1 - c1.floor()
        c1 = c1.to(torch.int64)
        c2 = torch.arange(length_old, dtype=torch.float32, device=device).reshape(1, 1, 1, -1) + 1
        c2[:, :, :, -1] -= 1
        c2 = torch.nn.functional.interpolate(c2, size=(1, length_new), mode="bilinear").to(torch.int64)
        return ratios, c1, c2

    orig_dtype = samples.dtype
    samples = samples.float()
    n, c, h, w = samples.shape

    # Width interpolation
    ratios, c1, c2 = gen_bilinear(w, width, samples.device)
    p1 = samples.gather(-1, c1.expand(n, c, h, -1)).movedim(1, -1).reshape(-1, c)
    p2 = samples.gather(-1, c2.expand(n, c, h, -1)).movedim(1, -1).reshape(-1, c)
    result = slerp(p1, p2, ratios.expand(n, 1, h, -1).movedim(1, -1).reshape(-1, 1))
    result = result.reshape(n, h, width, c).movedim(-1, 1)

    # Height interpolation
    ratios, c1, c2 = gen_bilinear(h, height, samples.device)
    p1 = result.gather(-2, c1.reshape(1, 1, -1, 1).expand(n, c, -1, width)).movedim(1, -1).reshape(-1, c)
    p2 = result.gather(-2, c2.reshape(1, 1, -1, 1).expand(n, c, -1, width)).movedim(1, -1).reshape(-1, c)
    result = slerp(p1, p2, ratios.reshape(1, 1, -1, 1).expand(n, 1, -1, width).movedim(1, -1).reshape(-1, 1))
    return result.reshape(n, height, width, c).movedim(-1, 1).to(orig_dtype)


def common_upscale(samples: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Upscale samples using bislerp."""
    return bislerp(samples, width, height)


class LatentUpscale:
    """Upscale latent codes."""
    def upscale(self, samples: dict, width: int, height: int, upscale_method: str = "bislerp", 
                downscale_factor: int = 8) -> tuple:
        if width == 0 and height == 0:
            return (samples,)
        s = samples.copy()
        s["samples"] = common_upscale(samples["samples"], max(64, width) // downscale_factor, 
                                     max(64, height) // downscale_factor)
        return (s,)

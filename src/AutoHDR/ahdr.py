# Taken and adapted from https://github.com/SuperBeastsAI/ComfyUI-SuperBeasts

import numpy as np
from PIL import Image, ImageEnhance, ImageCms
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor as tv_to_tensor


sRGB_profile = ImageCms.createProfile("sRGB")
Lab_profile = ImageCms.createProfile("LAB")


def tensor2pil(image: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL image."""
    t = image.squeeze()
    if t.dim() == 3 and t.shape[-1] in [1, 3, 4]:
        t = t.permute(2, 0, 1)
    return to_pil_image(torch.clamp(t, 0, 1))


def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor [1,H,W,C]."""
    return tv_to_tensor(image).unsqueeze(0).permute(0, 2, 3, 1)

def adjust_shadows_non_linear(luminance, shadow_intensity, max_shadow_adjustment=1.5):
    lum = np.asarray(luminance, dtype=np.float32) / 255.0
    shadows = lum ** (1 / (1 + shadow_intensity * max_shadow_adjustment))
    return np.clip(shadows * 255, 0, 255).astype(np.uint8)


def adjust_highlights_non_linear(luminance, highlight_intensity, max_highlight_adjustment=1.5):
    lum = np.asarray(luminance, dtype=np.float32) / 255.0
    highlights = 1 - (1 - lum) ** (1 + highlight_intensity * max_highlight_adjustment)
    return np.clip(highlights * 255, 0, 255).astype(np.uint8)


def merge_adjustments_with_blend_modes(luminance, shadows, highlights, hdr_intensity, shadow_intensity, highlight_intensity):
    base = np.asarray(luminance, dtype=np.float32)
    scaled_shadow = shadow_intensity ** 2 * hdr_intensity
    scaled_highlight = highlight_intensity ** 2 * hdr_intensity
    shadow_mask = np.clip((1 - (base / 255)) ** 2, 0, 1)
    highlight_mask = np.clip((base / 255) ** 2, 0, 1)
    adj_shadows = np.clip(base * (1 - shadow_mask * scaled_shadow), 0, 255)
    adj_highlights = np.clip(base + (255 - base) * highlight_mask * scaled_highlight, 0, 255)
    adjusted = np.clip(adj_shadows + adj_highlights - base, 0, 255)
    final = np.clip(base * (1 - hdr_intensity) + adjusted * hdr_intensity, 0, 255).astype(np.uint8)
    return Image.fromarray(final)


def apply_gamma_correction(lum_array, gamma):
    if gamma == 0:
        return np.clip(lum_array, 0, 255).astype(np.uint8)
    gamma_corrected = 1 / (1.1 - gamma)
    return np.clip(255 * ((lum_array / 255) ** gamma_corrected), 0, 255).astype(np.uint8)
def apply_to_batch(func):
    """Decorator to apply function to each image in batch."""
    def wrapper(self, image, *args, **kwargs):
        results = [func(self, img, *args, **kwargs) for img in image]
        return (torch.cat(results, dim=0),)
    return wrapper


class HDREffects:
    @apply_to_batch
    def apply_hdr2(self, image, hdr_intensity=0.75, shadow_intensity=0.25, highlight_intensity=0.5, 
                   gamma_intensity=0.25, contrast=0.1, enhance_color=0.25):
        img = tensor2pil(image)
        try:
            # Preferred path using ICC profiles (Lab transform)
            img_lab = ImageCms.profileToProfile(img, sRGB_profile, Lab_profile, outputMode='LAB')
            luminance, a, b = img_lab.split()
            lum_array = np.asarray(luminance, dtype=np.float32)
            
            shadows_adj = adjust_shadows_non_linear(luminance, shadow_intensity)
            highlights_adj = adjust_highlights_non_linear(luminance, highlight_intensity)
            merged = merge_adjustments_with_blend_modes(lum_array, shadows_adj, highlights_adj, 
                                                         hdr_intensity, shadow_intensity, highlight_intensity)
            gamma_corr = Image.fromarray(apply_gamma_correction(np.asarray(merged), gamma_intensity)).resize(a.size)
            
            adjusted_lab = Image.merge('LAB', (gamma_corr, a, b))
            img_adjusted = ImageCms.profileToProfile(adjusted_lab, Lab_profile, sRGB_profile, outputMode='RGB')
            img_adjusted = ImageEnhance.Contrast(img_adjusted).enhance(1 + contrast)
            img_adjusted = ImageEnhance.Color(img_adjusted).enhance(1 + enhance_color * 0.2)
            return pil2tensor(img_adjusted)
        except Exception as e:
            # Fallback: some Pillow builds / platforms can't build the ICC transform
            # (OSError: cannot build transform). Use a safe RGB-based approximation
            # that adjusts contrast/color/brightness without relying on ImageCms.
            import logging
            logging.getLogger(__name__).warning(f"AutoHDR: profile transform failed ({e}); using RGB fallback")
            img_adjusted = ImageEnhance.Contrast(img).enhance(1 + contrast)
            img_adjusted = ImageEnhance.Color(img_adjusted).enhance(1 + enhance_color * 0.2)
            img_adjusted = ImageEnhance.Brightness(img_adjusted).enhance(1 + hdr_intensity * 0.1)
            return pil2tensor(img_adjusted)

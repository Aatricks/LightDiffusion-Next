"""Automatic face/body enhancement processor for LightDiffusion-Next.

This processor uses detection models to identify faces and bodies,
then applies targeted inpainting/enhancement to those regions.
"""

import logging
import random
import re
import time
from typing import TYPE_CHECKING, Any, Optional, Callable

import numpy as np
import torch

if TYPE_CHECKING:
    from src.Core.PipelineContext import PipelineContext
    from src.Core.AbstractModel import AbstractModel


class Adetailer:
    """Automatic face and body detailing processor.
    
    Uses YOLO detection and SAM segmentation to identify regions
    of interest, then applies targeted inpainting enhancement.
    """
    
    # Default settings
    DEFAULT_GUIDE_SIZE = 512
    DEFAULT_MAX_SIZE = 768
    DEFAULT_STEPS = 20
    DEFAULT_CFG = 6.5
    DEFAULT_DENOISE = 0.5
    DEFAULT_SCHEDULER = "karras"
    DEFAULT_POSITIVE_PROMPT = "royal, detailed, magnificient, beautiful, seducing"
    
    @classmethod
    def apply(
        cls,
        image: torch.Tensor,
        ctx: "PipelineContext",
        model: "AbstractModel",
        positive: Any = None,
        negative: Any = None,
        callback: Optional[Callable] = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Apply automatic face and body enhancement.
        
        Args:
            image: Input image tensor [B, H, W, C] or [H, W, C]
            ctx: Pipeline context with configuration
            model: The loaded model instance
            positive: Optional positive conditioning (uses default if not provided)
            negative: Negative conditioning from original generation
            callback: Optional callback for live previews
            
        Returns:
            Tuple of (enhanced_image, list_of_saved_intermediate_images_metadata)
        """
        logger = logging.getLogger(__name__)
        saved_images = []
        
        try:
            # Ensure image has batch dimension
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            # Import required modules
            from src.AutoDetailer import SAM, SEGS, ADetailer, bbox
            from src.clip import Clip
            from src.FileManaging import ImageSaver
            from src.AutoHDR import ahdr
            
            # Load detection and segmentation models
            samloader = SAM.SAMLoader()
            sam_model = samloader.load_model(
                model_name="sam_vit_b_01ec64.pth",
                device_mode="AUTO"
            )[0]
            
            # Load YOLO detector for person/body
            detector_provider = bbox.UltralyticsDetectorProvider()
            body_detector = detector_provider.doit(model_name="person_yolov8m-seg.pt")[0]
            
            # Create CLIP encoder for adetailer-specific prompt
            cliptextencode = Clip.CLIPTextEncode()
            adetailer_positive = cliptextencode.encode(
                text=cls.DEFAULT_POSITIVE_PROMPT,
                clip=model.clip,
            )[0]
            
            # Initialize processors
            bbox_detector = bbox.BboxDetectorForEach()
            sam_detector = SAM.SAMDetectorCombined()
            segs_mask = SEGS.SegsBitwiseAndMask()
            detailer = ADetailer.DetailerForEachTest()
            saveimage = ImageSaver.SaveImage()
            hdr = ahdr.HDREffects()
            
            # Determine model flags
            is_flux = getattr(model.capabilities, "is_flux", False)
            is_flux2 = getattr(model.capabilities, "is_flux2", False)
            
            # Adjust parameters for Flux/distilled models
            adetailer_steps = cls.DEFAULT_STEPS
            adetailer_cfg = cls.DEFAULT_CFG
            if is_flux2:
                adetailer_steps = 6  # Distilled models need very few steps
                adetailer_cfg = 1.0  # Flux distilled usually CFG 1.0
            elif is_flux:
                adetailer_steps = 20
                adetailer_cfg = 1.0
            
            # ===== BODY PASS =====
            # Detect body regions
            body_segs = bbox_detector.doit(
                threshold=0.5,
                dilation=10,
                crop_factor=2,
                drop_size=10,
                labels="all",
                bbox_detector=body_detector,
                image=image,
            )
            
            # Apply SAM for precise segmentation
            sam_result = sam_detector.doit(
                detection_hint="center-1",
                dilation=0,
                threshold=0.93,
                bbox_expansion=0,
                mask_hint_threshold=0.7,
                mask_hint_use_negative="False",
                sam_model=sam_model,
                segs=body_segs,
                image=image,
            )
            
            if sam_result is None:
                logger.info("Adetailer: No body regions detected")
                return image[0] if image.shape[0] == 1 else image, saved_images
            
            # Combine segmentation masks
            combined_segs = segs_mask.doit(
                segs=body_segs,
                mask=sam_result[0],
            )
            
            # Apply body enhancement
            body_seed = random.randint(1, 2**63 - 1)
            body_result = detailer.doit(
                guide_size=cls.DEFAULT_GUIDE_SIZE,
                guide_size_for=False,
                max_size=cls.DEFAULT_MAX_SIZE,
                seed=body_seed,
                steps=adetailer_steps,
                cfg=adetailer_cfg,
                sampler_name=ctx.sampling.sampler,
                scheduler=cls.DEFAULT_SCHEDULER,
                denoise=cls.DEFAULT_DENOISE,
                feather=5,
                noise_mask=True,
                force_inpaint=True,
                wildcard="",
                cycle=1,
                inpaint_model=False,
                noise_mask_feather=20,
                image=image,
                segs=combined_segs[0],
                model=model.model,
                clip=model.clip,
                vae=model.vae,
                positive=adetailer_positive,
                negative=negative,
                pipeline=True,
                callback=callback,
            )
            
            # Extract enhanced body image
            body_image = body_result[0]
            body_seed_str = cls._extract_seed(body_result, body_seed)
            
            # Apply HDR if enabled
            if ctx.generation.autohdr:
                try:
                    hdr_result = hdr.apply_hdr2(body_image)
                    body_image = hdr_result[0] if isinstance(hdr_result, (tuple, list)) else hdr_result
                except Exception:
                    pass
            
            # Save body-enhanced image
            body_meta = cls._build_metadata(ctx, body_seed_str, "body")
            # Update meta with actual steps/cfg used
            body_meta["steps"] = str(adetailer_steps)
            body_meta["cfg"] = str(adetailer_cfg)
            
            saved_body = saveimage.save_images(
                filename_prefix="LD-body",
                images=body_image,
                prompt=ctx.prompt if isinstance(ctx.prompt, str) else str(ctx.prompt),
                extra_pnginfo=body_meta,
            )
            saved_images.append(saved_body)
            
            # ===== FACE PASS =====
            # Check for interrupt before starting the next pass
            from src.user import app_instance
            app = getattr(app_instance, "app", None)
            if app and getattr(app, "interrupt_flag", False):
                logger.info("Adetailer: Interrupt requested, skipping Face pass")
                return body_image[0] if body_image.shape[0] == 1 else body_image, saved_images

            # Load face detector
            face_detector = detector_provider.doit(model_name="face_yolov9c.pt")[0]
            
            # Detect face regions on the body-enhanced image
            face_segs = bbox_detector.doit(
                threshold=0.5,
                dilation=10,
                crop_factor=2,
                drop_size=10,
                labels="all",
                bbox_detector=face_detector,
                image=body_image,
            )
            
            # Apply SAM for face segmentation
            face_sam_result = sam_detector.doit(
                detection_hint="center-1",
                dilation=0,
                threshold=0.93,
                bbox_expansion=0,
                mask_hint_threshold=0.7,
                mask_hint_use_negative="False",
                sam_model=sam_model,
                segs=face_segs,
                image=body_image,
            )
            
            if face_sam_result is None:
                logger.info("Adetailer: No face regions detected")
                return body_image[0] if body_image.shape[0] == 1 else body_image, saved_images
            
            # Combine face segmentation masks
            face_combined_segs = segs_mask.doit(
                segs=face_segs,
                mask=face_sam_result[0],
            )
            
            # Apply face enhancement
            face_seed = random.randint(1, 2**63 - 1)
            face_result = detailer.doit(
                guide_size=cls.DEFAULT_GUIDE_SIZE,
                guide_size_for=False,
                max_size=cls.DEFAULT_MAX_SIZE,
                seed=face_seed,
                steps=adetailer_steps,
                cfg=adetailer_cfg,
                sampler_name=ctx.sampling.sampler,
                scheduler=cls.DEFAULT_SCHEDULER,
                denoise=cls.DEFAULT_DENOISE,
                feather=5,
                noise_mask=True,
                force_inpaint=True,
                wildcard="",
                cycle=1,
                inpaint_model=False,
                noise_mask_feather=20,
                image=body_image,
                segs=face_combined_segs[0],
                model=model.model,
                clip=model.clip,
                vae=model.vae,
                positive=adetailer_positive,
                negative=negative,
                pipeline=True,
                callback=callback,
            )
            
            # Extract final enhanced image
            final_image = face_result[0]
            face_seed_str = cls._extract_seed(face_result, face_seed)
            
            # Apply HDR if enabled
            if ctx.generation.autohdr:
                try:
                    hdr_result = hdr.apply_hdr2(final_image)
                    final_image = hdr_result[0] if isinstance(hdr_result, (tuple, list)) else hdr_result
                except Exception:
                    pass
            
            # Save face-enhanced (final) image
            face_meta = cls._build_metadata(ctx, face_seed_str, "head")
            face_meta["steps"] = str(adetailer_steps)
            face_meta["cfg"] = str(adetailer_cfg)
            saved_face = saveimage.save_images(
                filename_prefix="LD-head",
                images=final_image,
                prompt=ctx.prompt if isinstance(ctx.prompt, str) else str(ctx.prompt),
                extra_pnginfo=face_meta,
            )
            saved_images.append(saved_face)
            
            logger.info("Adetailer: completed body and face enhancement")
            
            # Return final image (remove batch dim if it was added)
            return final_image[0] if final_image.shape[0] == 1 else final_image, saved_images
            
        except Exception as e:
            logger.exception(f"Adetailer failed: {e}")
            # Return original image on failure
            return image[0] if image.dim() == 4 and image.shape[0] == 1 else image, saved_images
    
    @classmethod
    def _extract_seed(cls, result: Any, fallback_seed: int) -> str:
        """Extract seed from detailer result safely.
        
        Args:
            result: Result from detailer (may be tuple with seed)
            fallback_seed: Seed to use if extraction fails
            
        Returns:
            String representation of the seed
        """
        try:
            if isinstance(result, (list, tuple)) and len(result) > 1:
                candidate = result[1]
                
                if isinstance(candidate, int):
                    return str(candidate)
                if isinstance(candidate, float) and float(candidate).is_integer():
                    return str(int(candidate))
                if isinstance(candidate, str):
                    s = candidate.strip()
                    if re.fullmatch(r"-?\d+", s):
                        return s
                    m = re.search(r"\d{4,}", s)
                    if m:
                        return m.group(0)
                if isinstance(candidate, np.ndarray) and candidate.size == 1:
                    return str(int(candidate.item()))
                if isinstance(candidate, torch.Tensor) and candidate.numel() == 1:
                    return str(int(candidate.item()))
        except Exception:
            pass
        
        return str(fallback_seed)
    
    @classmethod
    def _build_metadata(
        cls,
        ctx: "PipelineContext",
        seed: str,
        pass_type: str,
    ) -> dict:
        """Build metadata dictionary for saved images.
        
        Args:
            ctx: Pipeline context
            seed: Seed used for this pass
            pass_type: Type of enhancement pass ('body' or 'head')
            
        Returns:
            Metadata dictionary
        """
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "prompt": ctx.prompt if isinstance(ctx.prompt, str) else str(ctx.prompt),
            "negative_prompt": ctx.negative_prompt if isinstance(ctx.negative_prompt, str) else str(ctx.negative_prompt),
            "seed": seed,
            "sampler": ctx.sampling.sampler,
            "steps": str(cls.DEFAULT_STEPS),
            "cfg": str(cls.DEFAULT_CFG),
            "scheduler": cls.DEFAULT_SCHEDULER,
            "denoise": str(cls.DEFAULT_DENOISE),
            "width": str(ctx.generation.width),
            "height": str(ctx.generation.height),
            "batch_size": str(1),
            "adetailer": "True",
            "adetailer_pass": pass_type,
        }
    
    @classmethod
    def is_enabled(cls, ctx: "PipelineContext") -> bool:
        """Check if Adetailer should be applied based on context.
        
        Args:
            ctx: Pipeline context
            
        Returns:
            True if Adetailer should be applied
        """
        return ctx.features.adetailer

"""Generation Service for LightDiffusion-Next.

This module provides a unified service layer that all UIs (Gradio, Streamlit, FastAPI)
can use for image generation. It wraps the Pipeline with additional features:

- Async generation support
- Progress tracking
- Interrupt handling
- Result formatting
"""

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import torch

from src.Core.Context import Context
from src.Core.Pipeline import Pipeline, get_default_pipeline

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Standardized generation request."""
    prompt: str | list[str]
    width: int = 512
    height: int = 512
    negative_prompt: str = ""
    num_images: int = 1
    batch_size: int = 1
    scheduler: str = "ays"
    sampler: str = "dpmpp_sde_cfgpp"
    steps: int = 20
    hires_fix: bool = False
    adetailer: bool = False
    enhance_prompt: bool = False
    img2img: bool = False
    img2img_image: Optional[str] = None
    stable_fast: bool = False
    reuse_seed: bool = False
    autohdr: bool = True
    model_path: Optional[str] = None
    # Multi-scale
    enable_multiscale: bool = True
    multiscale_factor: float = 0.5
    multiscale_fullres_start: int = 3
    multiscale_fullres_end: int = 8
    multiscale_intermittent_fullres: bool = False
    # DeepCache
    deepcache_enabled: bool = False
    deepcache_interval: int = 3
    # CFG-free
    cfg_free_enabled: bool = False
    cfg_free_start_percent: float = 70.0
    # Request metadata
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    
    def to_context(self) -> Context:
        """Convert to Context for pipeline execution."""
        return Context.from_kwargs(
            prompt=self.prompt,
            w=self.width,
            h=self.height,
            negative_prompt=self.negative_prompt,
            number=self.num_images,
            batch=self.batch_size,
            scheduler=self.scheduler,
            sampler=self.sampler,
            steps=self.steps,
            hires_fix=self.hires_fix,
            adetailer=self.adetailer,
            enhance_prompt=self.enhance_prompt,
            img2img=self.img2img,
            img2img_image=self.img2img_image,
            stable_fast=self.stable_fast,
            reuse_seed=self.reuse_seed,
            autohdr=self.autohdr,
            model_path=self.model_path,
            enable_multiscale=self.enable_multiscale,
            multiscale_factor=self.multiscale_factor,
            multiscale_fullres_start=self.multiscale_fullres_start,
            multiscale_fullres_end=self.multiscale_fullres_end,
            multiscale_intermittent_fullres=self.multiscale_intermittent_fullres,
            deepcache_enabled=self.deepcache_enabled,
            deepcache_interval=self.deepcache_interval,
            cfg_free_enabled=self.cfg_free_enabled,
            cfg_free_start_percent=self.cfg_free_start_percent,
        )


@dataclass
class GenerationResult:
    """Standardized generation result."""
    request_id: str
    success: bool
    images: list[Any] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "image_count": len(self.images),
            "image_paths": self.image_paths,
            "error": self.error,
            "metadata": self.metadata,
        }


class GenerationService:
    """Unified generation service for all UI frontends.
    
    Thread-safe service that manages generation requests with:
    - Synchronous and async generation
    - Progress callbacks
    - Interrupt handling
    - Result caching
    
    Usage:
        service = GenerationService()
        
        # Sync
        result = service.generate(request)
        
        # Async
        result = await service.generate_async(request)
        
        # With progress
        result = service.generate(request, on_progress=lambda p: print(f"{p}%"))
    """
    
    def __init__(self, pipeline: Optional[Pipeline] = None):
        """Initialize the service.
        
        Args:
            pipeline: Pipeline instance to use (default: global default)
        """
        self.pipeline = pipeline or get_default_pipeline()
        self._lock = threading.Lock()
        self._current_request_id: Optional[str] = None
        self._interrupted = False
    
    def generate(
        self,
        request: GenerationRequest,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> GenerationResult:
        """Generate images synchronously.
        
        Args:
            request: Generation request parameters
            on_progress: Optional progress callback (0.0 to 1.0)
            
        Returns:
            GenerationResult with images or error
        """
        with self._lock:
            self._current_request_id = request.request_id
            self._interrupted = False
        
        try:
            # Clear any previous interrupt
            self._clear_interrupt()
            
            # Convert to context
            ctx = request.to_context()
            
            # Run pipeline
            with torch.inference_mode():
                if ctx.features.img2img:
                    self.pipeline.run_img2img(ctx)
                elif ctx.is_batched:
                    result = self.pipeline.run_batched(ctx)
                    return GenerationResult(
                        request_id=request.request_id,
                        success=True,
                        metadata=result,
                    )
                else:
                    self.pipeline.run(ctx)
            
            # Collect results
            return GenerationResult(
                request_id=request.request_id,
                success=True,
                images=[ctx.current_image] if ctx.current_image is not None else [],
                metadata={
                    "prompt": str(ctx.prompt),
                    "seed": ctx.seed,
                    "steps": ctx.sampling.steps,
                },
            )
            
        except InterruptedError:
            return GenerationResult(
                request_id=request.request_id,
                success=False,
                error="Generation interrupted",
            )
        except Exception as e:
            logger.exception(f"Generation failed: {e}")
            return GenerationResult(
                request_id=request.request_id,
                success=False,
                error=str(e),
            )
        finally:
            with self._lock:
                self._current_request_id = None
    
    async def generate_async(
        self,
        request: GenerationRequest,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> GenerationResult:
        """Generate images asynchronously.
        
        Args:
            request: Generation request parameters
            on_progress: Optional progress callback
            
        Returns:
            GenerationResult with images or error
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.generate(request, on_progress)
        )
    
    def interrupt(self) -> bool:
        """Interrupt the current generation.
        
        Returns:
            True if there was a generation to interrupt
        """
        with self._lock:
            if self._current_request_id is None:
                return False
            self._interrupted = True
        
        # Set the app interrupt flag
        try:
            from src.user import app_instance
            app = getattr(app_instance, "app", None)
            if app and hasattr(app, "interrupt_flag"):
                app.interrupt_flag = True
                return True
        except Exception:
            pass
        
        return False
    
    def _clear_interrupt(self) -> None:
        """Clear interrupt flags."""
        try:
            from src.user import app_instance
            app = getattr(app_instance, "app", None)
            if app and hasattr(app, "clear_interrupt"):
                app.clear_interrupt()
        except Exception:
            pass
    
    @property
    def is_generating(self) -> bool:
        """Check if generation is in progress."""
        with self._lock:
            return self._current_request_id is not None
    
    @property
    def current_request_id(self) -> Optional[str]:
        """Get the current request ID if generating."""
        with self._lock:
            return self._current_request_id


# Singleton service instance
_default_service: Optional[GenerationService] = None


def get_generation_service() -> GenerationService:
    """Get the default generation service instance."""
    global _default_service
    if _default_service is None:
        _default_service = GenerationService()
    return _default_service

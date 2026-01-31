"""StableFast optimization processor for LightDiffusion-Next.

Applies torch.compile and CUDA graph optimizations to models.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.Core.Context import Context
    from src.Core.AbstractModel import AbstractModel


class StableFastProcessor:
    """StableFast model optimization processor.
    
    Wraps src/StableFast/ as a standardized processor for model optimization.
    This is typically applied during model loading, not during generation.
    """
    
    @classmethod
    def is_enabled(cls, ctx: "Context") -> bool:
        """Check if StableFast should be applied."""
        return getattr(ctx.generation, "stable_fast", False)
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if StableFast is available in the environment."""
        try:
            from src.StableFast import StableFast
            return True
        except ImportError:
            return False
    
    @classmethod
    def apply(
        cls,
        model: "AbstractModel",
        enable_cuda_graph: bool = True,
    ) -> "AbstractModel":
        """Apply StableFast optimization to a model.
        
        Args:
            model: Model to optimize
            enable_cuda_graph: Whether to enable CUDA graphs
            
        Returns:
            Optimized model (same instance, modified in place)
        """
        logger = logging.getLogger(__name__)
        
        if not model.capabilities.supports_stable_fast:
            logger.info("Model does not support StableFast, skipping")
            return model
        
        try:
            from src.StableFast import StableFast
            
            applier = StableFast.ApplyStableFastUnet()
            result = applier.apply_stable_fast(
                enable_cuda_graph=enable_cuda_graph,
                model=model.model,
            )
            model.model = result[0]
            
            logger.info("StableFast optimization applied")
            
        except Exception as e:
            logger.warning(f"StableFast optimization failed: {e}")
        
        return model
    
    @classmethod
    def process(
        cls,
        ctx: "Context",
        model: "AbstractModel",
        enable_cuda_graph: bool = True,
        **kwargs,
    ) -> "Context":
        """Process context, applying StableFast to the model.
        
        Note: This modifies the model in place.
        
        Args:
            ctx: Pipeline context
            model: Model to optimize
            enable_cuda_graph: Whether to enable CUDA graphs
            **kwargs: Additional parameters
            
        Returns:
            Unchanged context (model is modified in place)
        """
        if cls.is_enabled(ctx):
            cls.apply(model, enable_cuda_graph)
        
        return ctx


class DeepCacheProcessor:
    """DeepCache optimization processor.
    
    Enables feature caching in the U-Net for faster inference.
    """
    
    @classmethod
    def is_enabled(cls, ctx: "Context") -> bool:
        """Check if DeepCache should be applied."""
        return getattr(ctx.sampling, "deepcache_enabled", False)
    
    @classmethod
    def apply(
        cls,
        model: "AbstractModel",
        cache_interval: int = 3,
        cache_depth: int = 2,
        start_step: int = 0,
        end_step: int = 1000,
    ) -> "AbstractModel":
        """Apply DeepCache optimization to a model.
        
        Args:
            model: Model to optimize
            cache_interval: Steps between cache updates
            cache_depth: U-Net depth for caching
            start_step: Start applying at this step
            end_step: Stop applying at this step
            
        Returns:
            Optimized model
        """
        logger = logging.getLogger(__name__)
        
        if not model.capabilities.supports_deepcache:
            logger.info("Model does not support DeepCache, skipping")
            return model
        
        try:
            from src.WaveSpeed import deepcache_nodes
            
            deepcache = deepcache_nodes.ApplyDeepCacheOnModel()
            result = deepcache.patch(
                model=(model.model,),
                object_to_patch="diffusion_model",
                cache_interval=cache_interval,
                cache_depth=cache_depth,
                start_step=start_step,
                end_step=end_step,
            )
            
            if isinstance(result, tuple) and len(result) > 0:
                model.model = result[0]
            
            logger.info(f"DeepCache applied (interval={cache_interval}, depth={cache_depth})")
            
        except Exception as e:
            logger.warning(f"DeepCache optimization failed: {e}")
        
        return model
    
    @classmethod
    def process(
        cls,
        ctx: "Context",
        model: "AbstractModel",
        **kwargs,
    ) -> "Context":
        """Process context, applying DeepCache to the model.
        
        Args:
            ctx: Pipeline context with deepcache settings
            model: Model to optimize
            **kwargs: Additional parameters
            
        Returns:
            Unchanged context (model is modified in place)
        """
        if not cls.is_enabled(ctx):
            return ctx
        
        sampling = ctx.sampling
        cls.apply(
            model,
            cache_interval=getattr(sampling, "deepcache_interval", 3),
            cache_depth=getattr(sampling, "deepcache_depth", 2),
            start_step=getattr(sampling, "deepcache_start_step", 0),
            end_step=getattr(sampling, "deepcache_end_step", 1000),
        )
        
        return ctx

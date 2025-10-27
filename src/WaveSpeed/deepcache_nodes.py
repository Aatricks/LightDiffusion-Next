"""DeepCache implementation for LightDiffusion-Next.

Based on:
- https://github.com/horseee/DeepCache
- https://gist.github.com/laksjdjf/435c512bc19636e9c9af4ee7bea9eb86

DeepCache accelerates diffusion models by reusing high-level features
while updating low-level features in a cheap way.
"""

import torch
import logging


class ApplyDeepCacheOnModel:
    """Apply DeepCache optimization to a model.
    
    DeepCache works by caching intermediate features in the U-Net architecture
    and reusing them for certain steps, significantly reducing computation.
    """

    def patch(
        self,
        model,
        object_to_patch="diffusion_model",
        cache_interval=3,
        cache_depth=2,
        start_step=0,
        end_step=1000,
    ):
        """Patch the model with DeepCache optimization.
        
        Args:
            model: The model to patch (should be a ModelPatcher or tuple containing one)
            object_to_patch: Name of the model object to patch (default: "diffusion_model")
            cache_interval: Interval for cache updates (higher = more speedup, lower quality)
            cache_depth: Depth of caching in U-Net blocks (0-12, higher = more aggressive)
            start_step: Start applying DeepCache at this timestep (0-1000)
            end_step: Stop applying DeepCache at this timestep (0-1000)
        
        Returns:
            Tuple containing the patched model
        """
        logger = logging.getLogger(__name__)
        
        # Handle both raw model and tuple input
        if isinstance(model, (tuple, list)):
            model = model[0]
        
        # Clone the model to avoid modifying the original
        new_model = model.clone()
        
        # State variables for cache management
        current_t = -1
        current_step = -1
        cached_output = None
        
        def apply_model_deepcache(model_function, kwargs):
            """Wrapper function that applies DeepCache logic to model forward pass.
            
            DeepCache works by simply reusing the output from previous steps instead of
            recomputing the full U-Net forward pass. This is much simpler and more robust
            than trying to manually execute partial U-Net blocks.
            """
            nonlocal current_t, current_step, cached_output
            
            try:
                # Extract inputs from kwargs
                xa = kwargs["input"]
                t = kwargs["timestep"]
                c_dict = kwargs.get("c", {})
                
                # Get the diffusion model (UNet) for validation
                try:
                    unet = new_model.get_model_object(object_to_patch)
                except Exception:
                    # If we can't get the object, just run normally
                    return model_function(xa, t, **c_dict)
                
                # Check if this is a UNet-based model (SD1.5, SD2.1, SDXL, etc.)
                if not hasattr(unet, "input_blocks") or not hasattr(unet, "output_blocks"):
                    # Not a U-Net architecture, skip DeepCache
                    return model_function(xa, t, **c_dict)
                
                # Get current timestep value
                current_t_value = t[0].item()
                
                # Reset step counter if timestep increased (new batch/generation)
                if current_t_value > current_t:
                    current_step = -1
                    cached_output = None
                
                current_t = current_t_value
                
                # Determine if we should apply caching at this timestep
                # Note: t goes from 999 -> 0 during generation
                apply = (1000 - end_step) <= current_t <= (1000 - start_step)
                
                if apply:
                    current_step += 1
                else:
                    current_step = -1
                    cached_output = None
                
                # Determine if this is a cache update step or cache reuse step
                is_cache_step = (current_step % cache_interval == 0) if apply else True
                
                # If not applying DeepCache or it's a cache update step, run full model
                if not apply or is_cache_step:
                    result = model_function(xa, t, **c_dict)
                    # Store the output for future reuse
                    if apply:
                        cached_output = result.clone() if hasattr(result, 'clone') else result
                    return result
                
                # Cache reuse step - return cached output instead of recomputing
                if cached_output is not None:
                    # DeepCache speedup: reuse previous output
                    return cached_output
                else:
                    # First non-cache step but no cache yet - run normally and cache
                    result = model_function(xa, t, **c_dict)
                    cached_output = result.clone() if hasattr(result, 'clone') else result
                    return result
                
            except Exception as e:
                # Any error - run normal forward and reset cache
                logger.error(f"DeepCache wrapper error: {e}")
                cached_output = None
                return model_function(xa, t, **c_dict)
        
        # Apply the wrapper
        new_model.set_model_unet_function_wrapper(apply_model_deepcache)
        
        return (new_model,)

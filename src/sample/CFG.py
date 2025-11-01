import math
import logging

import torch

from src.cond import cond, cond_util


def cfg_function(
    model: torch.nn.Module,
    cond_pred: torch.Tensor,
    uncond_pred: torch.Tensor,
    cond_scale: float,
    x: torch.Tensor,
    timestep: int,
    model_options: dict = {},
    cond: torch.Tensor = None,
    uncond: torch.Tensor = None,
) -> torch.Tensor:
    """#### Apply classifier-free guidance (CFG) to the model predictions.

    #### Args:
        - `model` (torch.nn.Module): The model.
        - `cond_pred` (torch.Tensor): The conditioned prediction.
        - `uncond_pred` (torch.Tensor): The unconditioned prediction.
        - `cond_scale` (float): The CFG scale.
        - `x` (torch.Tensor): The input tensor.
        - `timestep` (int): The current timestep.
        - `model_options` (dict, optional): Additional model options. Defaults to {}.
        - `cond` (torch.Tensor, optional): The conditioned tensor. Defaults to None.
        - `uncond` (torch.Tensor, optional): The unconditioned tensor. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The CFG result.
    """
    # Apply dynamic CFG rescaling if enabled
    if "cfg_guider" in model_options:
        cfg_guider = model_options["cfg_guider"]
        if hasattr(cfg_guider, "dynamic_cfg_rescaling") and cfg_guider.dynamic_cfg_rescaling:
            cond_scale = cfg_guider._apply_dynamic_cfg_rescaling(cond_pred, uncond_pred, cond_scale)
    
    # Check for custom sampler CFG function first
    if "sampler_cfg_function" in model_options:
        # Precompute differences to avoid redundant operations
        cond_diff = x - cond_pred
        uncond_diff = x - uncond_pred

        args = {
            "cond": cond_diff,
            "uncond": uncond_diff,
            "cond_scale": cond_scale,
            "timestep": timestep,
            "input": x,
            "sigma": timestep,
            "cond_denoised": cond_pred,
            "uncond_denoised": uncond_pred,
            "model": model,
            "model_options": model_options,
        }
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        # Standard CFG calculation - optimized to avoid intermediate tensor allocation
        # When cond_scale = 1.0, we can just return cond_pred without computation
        if math.isclose(cond_scale, 1.0):
            cfg_result = cond_pred
        else:
            # Fused operation: uncond_pred + (cond_pred - uncond_pred) * cond_scale
            # Equivalent to: uncond_pred * (1 - cond_scale) + cond_pred * cond_scale
            cfg_result = torch.lerp(uncond_pred, cond_pred, cond_scale)

    # Apply post-CFG functions if any
    post_cfg_functions = model_options.get("sampler_post_cfg_function", [])
    if post_cfg_functions:
        args = {
            "denoised": cfg_result,
            "cond": cond,
            "uncond": uncond,
            "model": model,
            "uncond_denoised": uncond_pred,
            "cond_denoised": cond_pred,
            "sigma": timestep,
            "model_options": model_options,
            "input": x,
        }

        # Apply each post-CFG function in sequence
        for fn in post_cfg_functions:
            cfg_result = fn(args)
            # Update the denoised result for the next function
            args["denoised"] = cfg_result

    return cfg_result


def sampling_function(
    model: torch.nn.Module,
    x: torch.Tensor,
    timestep: int,
    uncond: torch.Tensor,
    condo: torch.Tensor,
    cond_scale: float,
    model_options: dict = {},
    seed: int = None,
) -> torch.Tensor:
    """#### Perform sampling with CFG.

    #### Args:
        - `model` (torch.nn.Module): The model.
        - `x` (torch.Tensor): The input tensor.
        - `timestep` (int): The current timestep.
        - `uncond` (torch.Tensor): The unconditioned tensor.
        - `condo` (torch.Tensor): The conditioned tensor.
        - `cond_scale` (float): The CFG scale.
        - `model_options` (dict, optional): Additional model options. Defaults to {}.
        - `seed` (int, optional): The random seed. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The sampled tensor.
    """
    # Optimize conditional logic for uncond
    uncond_ = (
        None
        if (
            math.isclose(cond_scale, 1.0)
            and not model_options.get("disable_cfg1_optimization", False)
        )
        else uncond
    )

    # Batched CFG is enabled by default for performance
    # Both conditional and unconditional are computed in a single forward pass
    conds = [condo, uncond_]
    cond_outputs = cond.calc_cond_batch(model, conds, x, timestep, model_options)

    # Apply pre-CFG functions if any
    pre_cfg_functions = model_options.get("sampler_pre_cfg_function", [])
    if pre_cfg_functions:
        # Create args dictionary once
        args = {
            "conds": conds,
            "conds_out": cond_outputs,
            "cond_scale": cond_scale,
            "timestep": timestep,
            "input": x,
            "sigma": timestep,
            "model": model,
            "model_options": model_options,
        }

        # Apply each pre-CFG function
        for fn in pre_cfg_functions:
            cond_outputs = fn(args)
            args["conds_out"] = cond_outputs

    # Extract conditional and unconditional outputs explicitly for clarity
    cond_pred, uncond_pred = cond_outputs[0], cond_outputs[1]

    # Apply the CFG function
    return cfg_function(
        model,
        cond_pred,
        uncond_pred,
        cond_scale,
        x,
        timestep,
        model_options=model_options,
        cond=condo,
        uncond=uncond_,
    )


class CFGGuider:
    """#### Class for guiding the sampling process with CFG."""

    def __init__(
        self, 
        model_patcher, 
        flux=False,
        dynamic_cfg_rescaling=False,
        dynamic_cfg_method="variance",
        dynamic_cfg_percentile=95.0,
        dynamic_cfg_target_scale=7.0,
        adaptive_noise_enabled=False,
        adaptive_noise_method="complexity"
    ):
        """#### Initialize the CFGGuider.

        #### Args:
            - `model_patcher` (object): The model patcher.
            - `flux` (bool): Whether using Flux model.
            - `dynamic_cfg_rescaling` (bool): Enable dynamic CFG rescaling.
            - `dynamic_cfg_method` (str): Method for dynamic CFG ('variance' or 'range').
            - `dynamic_cfg_percentile` (float): Percentile for range method.
            - `dynamic_cfg_target_scale` (float): Target CFG scale.
            - `adaptive_noise_enabled` (bool): Enable adaptive noise scheduling.
            - `adaptive_noise_method` (str): Method for adaptive noise ('complexity' or 'attention').
        """
        self.model_patcher = model_patcher
        self.model_options = model_patcher.model_options
        self.original_conds = {}
        self.cfg = 1.0
        self.flux = flux
        # CFG-free sampling parameters
        self.cfg_free_enabled = False
        self.cfg_free_start_percent = 70.0
        self.original_cfg = 1.0
        self.sigmas = None
        
        # Dynamic CFG rescaling parameters
        self.dynamic_cfg_rescaling = dynamic_cfg_rescaling
        self.dynamic_cfg_method = dynamic_cfg_method
        self.dynamic_cfg_percentile = dynamic_cfg_percentile
        self.dynamic_cfg_target_scale = dynamic_cfg_target_scale
        
        # Adaptive noise scheduling parameters
        self.adaptive_noise_enabled = adaptive_noise_enabled
        self.adaptive_noise_method = adaptive_noise_method
        self.complexity_history = []
        self.base_sigmas = None

    def set_conds(self, positive, negative):
        """#### Set the conditions for CFG.

        #### Args:
            - `positive` (torch.Tensor): The positive condition.
            - `negative` (torch.Tensor): The negative condition.
        """
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        """#### Set the CFG scale.

        #### Args:
            - `cfg` (float): The CFG scale.
        """
        self.cfg = cfg
        self.original_cfg = cfg

    def set_cfg_free_params(self, enabled=False, start_percent=70.0):
        """#### Set CFG-free sampling parameters.

        #### Args:
            - `enabled` (bool): Whether to enable CFG-free sampling.
            - `start_percent` (float): Percentage (0-100) at which to start reducing CFG to 0.
        """
        self.cfg_free_enabled = enabled
        self.cfg_free_start_percent = max(0.0, min(100.0, start_percent))
        
        if enabled:
            logging.info(f"CFG-Free sampling ENABLED: will reduce CFG from {self.original_cfg} to 0 starting at {start_percent}% of steps")
            print(f"✓ CFG-Free sampling ACTIVE: CFG will gradually reduce to 0 starting at {start_percent:.0f}% of steps")

    def _update_cfg_for_sigma(self, sigma):
        """#### Update CFG value based on current sigma and CFG-free parameters.

        #### Args:
            - `sigma` (float): Current sigma/timestep value.
        """
        if not self.cfg_free_enabled or self.sigmas is None or len(self.sigmas) <= 1:
            return

        # Find the position of current sigma in the schedule
        # Sigmas go from high to low, so we need to find where we are
        total_steps = len(self.sigmas) - 1

        # Find closest sigma index
        current_step = 0
        min_diff = float('inf')
        for i, s in enumerate(self.sigmas):
            diff = abs(float(s) - float(sigma))
            if diff < min_diff:
                min_diff = diff
                current_step = i

        # Calculate current progress percentage
        if total_steps > 0:
            progress_percent = (current_step / total_steps) * 100.0

            if progress_percent >= self.cfg_free_start_percent:
                # Calculate how far we are into the CFG-free region
                remaining_percent = 100.0 - self.cfg_free_start_percent
                if remaining_percent > 0:
                    # Linear interpolation from original_cfg to 0
                    cfg_free_progress = (progress_percent - self.cfg_free_start_percent) / remaining_percent
                    new_cfg = self.original_cfg * (1.0 - cfg_free_progress)
                    # Ensure we don't go below 0
                    new_cfg = max(0.0, new_cfg)
                    
                    # Debug logging every 10% to confirm CFG-free is working
                    if current_step % max(1, total_steps // 10) == 0:
                        logging.info(f"CFG-Free: step {current_step}/{total_steps} ({progress_percent:.1f}%), CFG: {self.cfg:.2f} -> {new_cfg:.2f}")
                    
                    self.cfg = new_cfg
            else:
                # Before CFG-free region, use original CFG
                self.cfg = self.original_cfg

    def _apply_dynamic_cfg_rescaling(self, cond_pred, uncond_pred, cond_scale):
        """#### Apply dynamic CFG rescaling based on prediction statistics.
        
        #### Args:
            - `cond_pred` (torch.Tensor): Conditional prediction.
            - `uncond_pred` (torch.Tensor): Unconditional prediction.
            - `cond_scale` (float): Current CFG scale.
            
        #### Returns:
            - `float`: Adjusted CFG scale.
        """
        if not self.dynamic_cfg_rescaling:
            return cond_scale
            
        # Calculate the difference between conditional and unconditional predictions
        diff = cond_pred - uncond_pred
        
        if self.dynamic_cfg_method == "variance":
            # Variance-based rescaling: adjust CFG inversely to variance
            variance = torch.var(diff).item()
            
            # Normalize variance to a reasonable range (empirically tuned)
            # Higher variance = lower CFG to prevent oversaturation
            # Lower variance = higher CFG to boost the effect
            variance_normalized = min(variance / 0.1, 10.0)  # Clamp to reasonable range
            scale_adjustment = 1.0 / (1.0 + variance_normalized * 0.1)
            
            adjusted_scale = cond_scale * scale_adjustment
            logging.debug(f"Dynamic CFG (variance): {cond_scale:.2f} -> {adjusted_scale:.2f} (var={variance:.4f})")
            
        elif self.dynamic_cfg_method == "range":
            # Range-based rescaling: adjust CFG based on percentile range
            diff_flat = diff.flatten()
            
            # Calculate percentile range
            percentile_val = self.dynamic_cfg_percentile
            low = torch.quantile(diff_flat, (100 - percentile_val) / 100).item()
            high = torch.quantile(diff_flat, percentile_val / 100).item()
            range_val = high - low
            
            # Rescale to target CFG scale
            # If range is large, reduce CFG; if range is small, increase CFG
            target_range = 1.0  # Target normalized range
            scale_adjustment = target_range / max(range_val, 0.01)
            
            adjusted_scale = min(cond_scale * scale_adjustment, self.dynamic_cfg_target_scale)
            logging.debug(f"Dynamic CFG (range): {cond_scale:.2f} -> {adjusted_scale:.2f} (range={range_val:.4f})")
        else:
            adjusted_scale = cond_scale
            
        # Clamp to reasonable bounds
        adjusted_scale = max(1.0, min(adjusted_scale, 20.0))
        
        return adjusted_scale
    
    def _calculate_complexity_metric(self, prediction):
        """#### Calculate complexity metric for adaptive noise scheduling.
        
        #### Args:
            - `prediction` (torch.Tensor): Model prediction.
            
        #### Returns:
            - `float`: Complexity metric.
        """
        if not self.adaptive_noise_enabled:
            return 0.0
            
        if self.adaptive_noise_method == "complexity":
            # Measure complexity via gradient magnitude (edge detection proxy)
            # Higher gradients = more detail = higher complexity
            dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
            dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
            
            # Take the smaller spatial dimensions to match both gradients
            min_h = min(dx.shape[2], dy.shape[2])
            min_w = min(dx.shape[3], dy.shape[3])
            
            gradient_magnitude = (dx[:, :, :min_h, :min_w].abs() + dy[:, :, :min_h, :min_w].abs()).mean().item()
            return gradient_magnitude
            
        elif self.adaptive_noise_method == "attention":
            # Measure variance across spatial dimensions (attention proxy)
            # Higher variance = more focused attention = higher complexity
            spatial_variance = prediction.var(dim=[2, 3]).mean().item()
            return spatial_variance
        else:
            return 0.0

    def inner_set_conds(self, conds):
        """#### Set the internal conditions.

        #### Args:
            - `conds` (dict): The conditions.
        """
        for k in conds:
            self.original_conds[k] = cond.convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        """#### Call the CFGGuider to predict noise.

        #### Returns:
            - `torch.Tensor`: The predicted noise.
        """
        return self.predict_noise(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        """#### Predict noise using CFG.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `timestep` (int): The current timestep.
            - `model_options` (dict, optional): Additional model options. Defaults to {}.
            - `seed` (int, optional): The random seed. Defaults to None.

        #### Returns:
            - `torch.Tensor`: The predicted noise.
        """
        # Update CFG based on current sigma position if CFG-free is enabled
        if self.cfg_free_enabled:
            self._update_cfg_for_sigma(timestep)
        
        # Pass cfg_guider reference for dynamic rescaling
        model_options_with_guider = model_options.copy()
        model_options_with_guider["cfg_guider"] = self

        result = sampling_function(
            self.inner_model,
            x,
            timestep,
            self.conds.get("negative", None),
            self.conds.get("positive", None),
            self.cfg,
            model_options=model_options_with_guider,
            seed=seed,
        )
        
        # Calculate complexity metric for adaptive noise scheduling
        if self.adaptive_noise_enabled:
            complexity = self._calculate_complexity_metric(result)
            self.complexity_history.append(complexity)
            
        return result

    def inner_sample(
        self,
        noise,
        latent_image,
        device,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
        pipeline=False,
    ):
        """#### Perform the inner sampling process.

        #### Args:
            - `noise` (torch.Tensor): The noise tensor.
            - `latent_image` (torch.Tensor): The latent image tensor.
            - `device` (torch.device): The device to use.
            - `sampler` (object): The sampler object.
            - `sigmas` (torch.Tensor): The sigmas tensor.
            - `denoise_mask` (torch.Tensor): The denoise mask tensor.
            - `callback` (callable): The callback function.
            - `disable_pbar` (bool): Whether to disable the progress bar.
            - `seed` (int): The random seed.
            - `pipeline` (bool, optional): Whether to use the pipeline. Defaults to False.

        #### Returns:
            - `torch.Tensor`: The sampled tensor.
        """
        if (
            latent_image is not None and torch.count_nonzero(latent_image) > 0
        ):  # Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = cond.process_conds(
            self.inner_model,
            noise,
            self.conds,
            device,
            latent_image,
            denoise_mask,
            seed,
        )

        # Store sigmas for CFG-free sampling and adaptive noise scheduling
        self.sigmas = sigmas
        
        # Apply adaptive noise scheduling if enabled
        if self.adaptive_noise_enabled:
            # Store base sigmas on first run
            if self.base_sigmas is None:
                self.base_sigmas = sigmas.clone()
            
            # Modify sigmas based on complexity history
            if len(self.complexity_history) > 0:
                # Calculate average complexity
                avg_complexity = sum(self.complexity_history) / len(self.complexity_history)
                
                # Adjust sigma schedule based on complexity
                # Higher complexity = steeper noise schedule (more denoising power)
                # Lower complexity = gentler noise schedule (preserve details)
                complexity_factor = avg_complexity / max(0.01, avg_complexity + 0.1)
                
                # Apply exponential scaling to sigmas
                sigmas = self.base_sigmas * (1.0 + complexity_factor * 0.5)
                
                logging.debug(f"Adaptive noise: complexity={avg_complexity:.4f}, factor={complexity_factor:.4f}")

        extra_args = {"model_options": self.model_options, "seed": seed}

        samples = sampler.sample(
            self,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
            pipeline=pipeline,
        )
        return self.inner_model.process_latent_out(samples.to(torch.float32))

    def sample(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
        pipeline=False,
    ):
        """#### Perform the sampling process with CFG.

        #### Args:
            - `noise` (torch.Tensor): The noise tensor.
            - `latent_image` (torch.Tensor): The latent image tensor.
            - `sampler` (object): The sampler object.
            - `sigmas` (torch.Tensor): The sigmas tensor.
            - `denoise_mask` (torch.Tensor, optional): The denoise mask tensor. Defaults to None.
            - `callback` (callable, optional): The callback function. Defaults to None.
            - `disable_pbar` (bool, optional): Whether to disable the progress bar. Defaults to False.
            - `seed` (int, optional): The random seed. Defaults to None.
            - `pipeline` (bool, optional): Whether to use the pipeline. Defaults to False.

        #### Returns:
            - `torch.Tensor`: The sampled tensor.
        """
        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))

        self.inner_model, self.conds, self.loaded_models = cond_util.prepare_sampling(
            self.model_patcher, noise.shape, self.conds, flux_enabled=self.flux
        )
        device = self.model_patcher.load_device

        noise = noise.to(device)
        latent_image = latent_image.to(device)
        sigmas = sigmas.to(device)

        output = self.inner_sample(
            noise,
            latent_image,
            device,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            pipeline=pipeline,
        )

        # Use model cache to prevent cleanup if models should stay loaded
        from src.Device.ModelCache import get_model_cache
        get_model_cache().prevent_model_cleanup(self.conds, self.loaded_models)

        del self.inner_model
        del self.conds
        del self.loaded_models
        return output

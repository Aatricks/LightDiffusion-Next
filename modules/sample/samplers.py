import threading
import torch
from tqdm.auto import trange
from modules.Utilities import util


from modules.sample import sampling_util

disable_gui = False


@torch.no_grad()
def sample_euler_ancestral(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    pipeline=False,
    # Multi-scale parameters
    enable_multiscale=True,
    multiscale_factor=0.5,
    multiscale_fullres_start=3,
    multiscale_fullres_end=8,
    multiscale_intermittent_fullres=False,
):
    # Pre-calculate common values
    device = x.device
    global disable_gui
    disable_gui = pipeline

    if not disable_gui:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance

    # Multi-scale setup with validation
    original_shape = x.shape
    batch_size, channels, orig_h, orig_w = original_shape

    # Validate multi-scale parameters
    if enable_multiscale:
        if not (0.1 <= multiscale_factor <= 1.0):
            print(
                f"Warning: multiscale_factor {multiscale_factor} out of range [0.1, 1.0], disabling multi-scale"
            )
            enable_multiscale = False
        if multiscale_fullres_start < 0 or multiscale_fullres_end < 0:
            print("Warning: Invalid fullres step counts, disabling multi-scale")
            enable_multiscale = False

    # Calculate scaled dimensions (must be multiples of 8 for VAE compatibility)
    scale_h = (
        int(max(8, ((orig_h * multiscale_factor) // 8) * 8))
        if enable_multiscale
        else orig_h
    )
    scale_w = (
        int(max(8, ((orig_w * multiscale_factor) // 8) * 8))
        if enable_multiscale
        else orig_w
    )

    # Disable multi-scale for small images or short step counts
    n_steps = len(sigmas) - 1
    multiscale_active = (
        enable_multiscale
        and orig_h > 64
        and orig_w > 64
        and n_steps > (multiscale_fullres_start + multiscale_fullres_end)
        and (scale_h != orig_h or scale_w != orig_w)
    )

    if enable_multiscale and not multiscale_active:
        print(
            f"Multi-scale disabled: image too small ({orig_h}x{orig_w}) or insufficient steps ({n_steps})"
        )
    elif multiscale_active:
        print(
            f"Multi-scale active: {orig_h}x{orig_w} -> {scale_h}x{scale_w} (factor: {multiscale_factor})"
        )

    def downscale_tensor(tensor):
        """Downscale tensor using bilinear interpolation"""
        if not multiscale_active or tensor.shape[-2:] == (scale_h, scale_w):
            return tensor
        return torch.nn.functional.interpolate(
            tensor, size=(scale_h, scale_w), mode="bilinear", align_corners=False
        )

    def upscale_tensor(tensor):
        """Upscale tensor using bilinear interpolation"""
        if not multiscale_active or tensor.shape[-2:] == (orig_h, orig_w):
            return tensor
        return torch.nn.functional.interpolate(
            tensor, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )

    def should_use_fullres(step):
        """Determine if this step should use full resolution"""
        if not multiscale_active:
            return True

        # Always use full resolution for start and end steps
        if step < multiscale_fullres_start or step >= n_steps - multiscale_fullres_end:
            return True

        # Intermittent full-res: every 2nd step in low-res region if enabled
        if multiscale_intermittent_fullres:
            # Check if we're in the low-res region
            low_res_region_start = multiscale_fullres_start
            low_res_region_end = n_steps - multiscale_fullres_end
            if low_res_region_start <= step < low_res_region_end:
                # Calculate position within low-res region
                relative_step = step - low_res_region_start
                # Use full-res every 2nd step (0, 2, 4, ...)
                return relative_step % 2 == 0

        return False

    # Pre-allocate tensors and init noise sampler
    s_in = torch.ones((x.shape[0],), device=device)
    noise_sampler = (
        sampling_util.default_noise_sampler(x)
        if noise_sampler is None
        else noise_sampler
    )

    for i in trange(len(sigmas) - 1, disable=disable):
        if (
            not pipeline
            and hasattr(app_instance.app, "interrupt_flag")
            and app_instance.app.interrupt_flag
        ):
            return x

        if not pipeline:
            app_instance.app.progress.set(i / (len(sigmas) - 1))

        # Determine resolution for this step
        use_fullres = should_use_fullres(i)

        # Scale input for processing
        if use_fullres:
            x_process = x
            s_in_process = s_in
        else:
            x_process = downscale_tensor(x)
            s_in_process = torch.ones((x_process.shape[0],), device=device)

        # Model inference at appropriate resolution
        denoised = model(x_process, sigmas[i] * s_in_process, **(extra_args or {}))

        # Scale predictions back to original resolution if needed
        if not use_fullres:
            denoised = upscale_tensor(denoised)

        sigma_down, sigma_up = sampling_util.get_ancestral_step(
            sigmas[i], sigmas[i + 1], eta=eta
        )

        # Fused update step
        x = x + util.to_d(x, sigmas[i], denoised) * (sigma_down - sigmas[i])
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "denoised": denoised})

        if not pipeline and app_instance.app.previewer_var.get() and i % 5 == 0:
            threading.Thread(target=taesd.taesd_preview, args=(x,)).start()

    return x


@torch.no_grad()
def sample_euler(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    pipeline=False,
    # Multi-scale parameters
    enable_multiscale=True,
    multiscale_factor=0.5,
    multiscale_fullres_start=3,
    multiscale_fullres_end=8,
    multiscale_intermittent_fullres=False,
):
    # Pre-calculate common values
    device = x.device
    global disable_gui
    disable_gui = pipeline

    if not disable_gui:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance

    # Multi-scale setup with validation
    original_shape = x.shape
    batch_size, channels, orig_h, orig_w = original_shape

    # Validate multi-scale parameters
    if enable_multiscale:
        if not (0.1 <= multiscale_factor <= 1.0):
            print(
                f"Warning: multiscale_factor {multiscale_factor} out of range [0.1, 1.0], disabling multi-scale"
            )
            enable_multiscale = False
        if multiscale_fullres_start < 0 or multiscale_fullres_end < 0:
            print("Warning: Invalid fullres step counts, disabling multi-scale")
            enable_multiscale = False

    # Calculate scaled dimensions (must be multiples of 8 for VAE compatibility)
    scale_h = (
        int(max(8, ((orig_h * multiscale_factor) // 8) * 8))
        if enable_multiscale
        else orig_h
    )
    scale_w = (
        int(max(8, ((orig_w * multiscale_factor) // 8) * 8))
        if enable_multiscale
        else orig_w
    )

    # Disable multi-scale for small images or short step counts
    n_steps = len(sigmas) - 1
    multiscale_active = (
        enable_multiscale
        and orig_h > 64
        and orig_w > 64
        and n_steps > (multiscale_fullres_start + multiscale_fullres_end)
        and (scale_h != orig_h or scale_w != orig_w)
    )

    if enable_multiscale and not multiscale_active:
        print(
            f"Multi-scale disabled: image too small ({orig_h}x{orig_w}) or insufficient steps ({n_steps})"
        )
    elif multiscale_active:
        print(
            f"Multi-scale active: {orig_h}x{orig_w} -> {scale_h}x{scale_w} (factor: {multiscale_factor})"
        )

    def downscale_tensor(tensor):
        """Downscale tensor using bilinear interpolation"""
        if not multiscale_active or tensor.shape[-2:] == (scale_h, scale_w):
            return tensor
        return torch.nn.functional.interpolate(
            tensor, size=(scale_h, scale_w), mode="bilinear", align_corners=False
        )

    def upscale_tensor(tensor):
        """Upscale tensor using bilinear interpolation"""
        if not multiscale_active or tensor.shape[-2:] == (orig_h, orig_w):
            return tensor
        return torch.nn.functional.interpolate(
            tensor, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )

    def should_use_fullres(step):
        """Determine if this step should use full resolution"""
        if not multiscale_active:
            return True

        # Always use full resolution for start and end steps
        if step < multiscale_fullres_start or step >= n_steps - multiscale_fullres_end:
            return True

        # Intermittent full-res: every 2nd step in low-res region if enabled
        if multiscale_intermittent_fullres:
            # Check if we're in the low-res region
            low_res_region_start = multiscale_fullres_start
            low_res_region_end = n_steps - multiscale_fullres_end
            if low_res_region_start <= step < low_res_region_end:
                # Calculate position within low-res region
                relative_step = step - low_res_region_start
                # Use full-res every 2nd step (0, 2, 4, ...)
                return relative_step % 2 == 0

        return False

    # Pre-allocate tensors and cache parameters
    s_in = torch.ones((x.shape[0],), device=device)
    gamma_max = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_churn > 0 else 0

    for i in trange(len(sigmas) - 1, disable=disable):
        if (
            not pipeline
            and hasattr(app_instance.app, "interrupt_flag")
            and app_instance.app.interrupt_flag
        ):
            return x

        if not pipeline:
            app_instance.app.progress.set(i / (len(sigmas) - 1))

        # Determine resolution for this step
        use_fullres = should_use_fullres(i)

        # Combined sigma calculation and update
        sigma_hat = (
            sigmas[i] * (1 + (gamma_max if s_tmin <= sigmas[i] <= s_tmax else 0))
            if gamma_max > 0
            else sigmas[i]
        )

        if gamma_max > 0 and sigma_hat > sigmas[i]:
            x = (
                x
                + torch.randn_like(x) * s_noise * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
            )

        # Scale input for processing
        if use_fullres:
            x_process = x
            s_in_process = s_in
        else:
            x_process = downscale_tensor(x)
            s_in_process = torch.ones((x_process.shape[0],), device=device)

        # Model inference at appropriate resolution
        denoised = model(x_process, sigma_hat * s_in_process, **(extra_args or {}))

        # Scale predictions back to original resolution if needed
        if not use_fullres:
            denoised = upscale_tensor(denoised)

        x = x + util.to_d(x, sigma_hat, denoised) * (sigmas[i + 1] - sigma_hat)

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )

        if not pipeline and app_instance.app.previewer_var.get() and i % 5 == 0:
            threading.Thread(target=taesd.taesd_preview, args=(x, True)).start()

    return x


class Rescaler:
    def __init__(self, model, x, mode, **extra_args):
        self.model = model
        self.x = x
        self.mode = mode
        self.extra_args = extra_args

        self.latent_image, self.noise = model.latent_image, model.noise
        self.denoise_mask = self.extra_args.get("denoise_mask", None)

    def __enter__(self):
        if self.latent_image is not None:
            self.model.latent_image = torch.nn.functional.interpolate(
                input=self.latent_image, size=self.x.shape[2:4], mode=self.mode
            )
        if self.noise is not None:
            self.model.noise = torch.nn.functional.interpolate(
                input=self.latent_image, size=self.x.shape[2:4], mode=self.mode
            )
        if self.denoise_mask is not None:
            self.extra_args["denoise_mask"] = torch.nn.functional.interpolate(
                input=self.denoise_mask, size=self.x.shape[2:4], mode=self.mode
            )

        return self

    def __exit__(self, type, value, traceback):
        del self.model.latent_image, self.model.noise
        self.model.latent_image, self.model.noise = self.latent_image, self.noise


@torch.no_grad()
def dy_sampling_step_cfg_pp(
    x,
    model,
    sigma_next,
    i,
    sigma,
    sigma_hat,
    callback,
    current_cfg=7.5,
    cfg_x0_scale=1.0,
    **extra_args,
):
    """Dynamic sampling step with proper CFG++ handling"""
    # Track both conditional and unconditional denoised outputs
    uncond_denoised = None
    old_uncond_denoised = None

    def post_cfg_function(args):
        nonlocal uncond_denoised
        uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    # Process image in lower resolution
    original_shape = x.shape
    batch_size, channels, m, n = (
        original_shape[0],
        original_shape[1],
        original_shape[2] // 2,
        original_shape[3] // 2,
    )
    extra_row = x.shape[2] % 2 == 1
    extra_col = x.shape[3] % 2 == 1

    if extra_row:
        extra_row_content = x[:, :, -1:, :]
        x = x[:, :, :-1, :]
    if extra_col:
        extra_col_content = x[:, :, :, -1:]
        x = x[:, :, :, :-1]

    a_list = (
        x.unfold(2, 2, 2)
        .unfold(3, 2, 2)
        .contiguous()
        .view(batch_size, channels, m * n, 2, 2)
    )
    c = a_list[:, :, :, 1, 1].view(batch_size, channels, m, n)

    with Rescaler(model, c, "nearest-exact", **extra_args) as rescaler:
        denoised = model(c, sigma_hat * c.new_ones([c.shape[0]]), **rescaler.extra_args)

    if callback is not None:
        callback(
            {
                "x": c,
                "i": i,
                "sigma": sigma,
                "sigma_hat": sigma_hat,
                "denoised": denoised,
            }
        )

    # Apply proper CFG++ calculation
    if old_uncond_denoised is None:
        # First step - regular CFG
        cfg_denoised = uncond_denoised + (denoised - uncond_denoised) * current_cfg
    else:
        # CFG++ with momentum
        momentum = denoised
        uncond_momentum = uncond_denoised
        x0_coeff = cfg_x0_scale * current_cfg

        # Combined CFG++ update
        cfg_denoised = uncond_momentum + (momentum - uncond_momentum) * x0_coeff

    # Apply proper noise prediction and update
    d = util.to_d(c, sigma_hat, cfg_denoised)
    c = c + d * (sigma_next - sigma_hat)

    # Store updated pixels back in the original tensor
    d_list = c.view(batch_size, channels, m * n, 1, 1)
    a_list[:, :, :, 1, 1] = d_list[:, :, :, 0, 0]
    x = (
        a_list.view(batch_size, channels, m, n, 2, 2)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(batch_size, channels, 2 * m, 2 * n)
    )

    if extra_row or extra_col:
        x_expanded = torch.zeros(original_shape, dtype=x.dtype, device=x.device)
        x_expanded[:, :, : 2 * m, : 2 * n] = x
        if extra_row:
            x_expanded[:, :, -1:, : 2 * n + 1] = extra_row_content
        if extra_col:
            x_expanded[:, :, : 2 * m, -1:] = extra_col_content
        if extra_row and extra_col:
            x_expanded[:, :, -1:, -1:] = extra_col_content[:, :, -1:, :]
        x = x_expanded

    return x


@torch.no_grad()
def sample_euler_dy_cfg_pp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    s_gamma_start=0.0,
    s_gamma_end=0.0,
    s_extra_steps=True,
    pipeline=False,
    # CFG++ parameters
    cfg_scale=7.5,
    cfg_x0_scale=1.0,
    cfg_s_scale=1.0,
    cfg_min=1.0,
    **kwargs,
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    gamma_start = (
        round(s_gamma_start)
        if s_gamma_start > 1.0
        else (len(sigmas) - 1) * s_gamma_start
    )
    gamma_end = (
        round(s_gamma_end) if s_gamma_end > 1.0 else (len(sigmas) - 1) * s_gamma_end
    )
    n_steps = len(sigmas) - 1

    # CFG++ scheduling
    def get_cfg_scale(step):
        # Linear scheduling from cfg_scale to cfg_min
        progress = step / n_steps
        return cfg_scale + (cfg_min - cfg_scale) * progress

    old_uncond_denoised = None

    def post_cfg_function(args):
        nonlocal old_uncond_denoised
        old_uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    global disable_gui
    disable_gui = pipeline

    if not disable_gui:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance

    for i in trange(len(sigmas) - 1, disable=disable):
        if (
            not pipeline
            and hasattr(app_instance.app, "interrupt_flag")
            and app_instance.app.interrupt_flag
        ):
            return x

        if not pipeline:
            app_instance.app.progress.set(i / (len(sigmas) - 1))

        # Get current CFG scale
        current_cfg = get_cfg_scale(i)

        gamma = (
            max(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if gamma_start <= i < gamma_end and s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        sigma_hat = sigmas[i] * (gamma + 1)

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        uncond_denoised = extra_args.get("model_options", {}).get(
            "sampler_post_cfg_function", []
        )[-1]({"denoised": denoised, "uncond_denoised": None})

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                    "cfg_scale": current_cfg,
                }
            )

        # CFG++ calculation
        if old_uncond_denoised is None:
            # First step - regular CFG
            cfg_denoised = uncond_denoised + (denoised - uncond_denoised) * current_cfg
        else:
            # CFG++ with momentum
            x0_coeff = cfg_x0_scale * current_cfg

            # Simple momentum for Euler
            momentum = denoised
            uncond_momentum = uncond_denoised

            # Combined CFG++ update
            cfg_denoised = uncond_momentum + (momentum - uncond_momentum) * x0_coeff

        # Euler method with CFG++ denoised result
        d = util.to_d(x, sigma_hat, cfg_denoised)
        x = x + d * (sigmas[i + 1] - sigma_hat)

        # Store for momentum calculation
        old_uncond_denoised = uncond_denoised

        # Extra dynamic steps - pass the current CFG scale and predictions
        if sigmas[i + 1] > 0 and s_extra_steps:
            if i // 2 == 1:
                x = dy_sampling_step_cfg_pp(
                    x,
                    model,
                    sigmas[i + 1],
                    i,
                    sigmas[i],
                    sigma_hat,
                    callback,
                    current_cfg=current_cfg,  # Pass current CFG scale
                    cfg_x0_scale=cfg_x0_scale,  # Pass CFG++ x0 coefficient
                    **extra_args,
                )

        if not pipeline and app_instance.app.previewer_var.get() and i % 5 == 0:
            threading.Thread(target=taesd.taesd_preview, args=(x,)).start()

    return x


@torch.no_grad()
def sample_euler_ancestral_dy_cfg_pp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    s_gamma_start=0.0,
    s_gamma_end=0.0,
    pipeline=False,
    # CFG++ parameters
    cfg_scale=7.5,
    cfg_x0_scale=1.0,
    cfg_s_scale=1.0,
    cfg_min=1.0,
    **kwargs,
):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = (
        sampling_util.default_noise_sampler(x)
        if noise_sampler is None
        else noise_sampler
    )
    gamma_start = (
        round(s_gamma_start)
        if s_gamma_start > 1.0
        else (len(sigmas) - 1) * s_gamma_start
    )
    gamma_end = (
        round(s_gamma_end) if s_gamma_end > 1.0 else (len(sigmas) - 1) * s_gamma_end
    )
    n_steps = len(sigmas) - 1

    # CFG++ scheduling
    def get_cfg_scale(step):
        # Linear scheduling from cfg_scale to cfg_min
        progress = step / n_steps
        return cfg_scale + (cfg_min - cfg_scale) * progress

    old_uncond_denoised = None

    def post_cfg_function(args):
        nonlocal old_uncond_denoised
        old_uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    global disable_gui
    disable_gui = pipeline

    if not disable_gui:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if (
            not pipeline
            and hasattr(app_instance.app, "interrupt_flag")
            and app_instance.app.interrupt_flag
        ):
            return x

        if not pipeline:
            app_instance.app.progress.set(i / (len(sigmas) - 1))

        # Get current CFG scale
        current_cfg = get_cfg_scale(i)

        gamma = 2**0.5 - 1 if gamma_start <= i < gamma_end else 0.0
        sigma_hat = sigmas[i] * (gamma + 1)

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5

        denoised = model(x, sigma_hat * s_in, **extra_args)
        uncond_denoised = extra_args.get("model_options", {}).get(
            "sampler_post_cfg_function", []
        )[-1]({"denoised": denoised, "uncond_denoised": None})

        sigma_down, sigma_up = sampling_util.get_ancestral_step(
            sigmas[i], sigmas[i + 1], eta=eta
        )

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                    "cfg_scale": current_cfg,
                }
            )

        # CFG++ calculation
        if old_uncond_denoised is None or sigmas[i + 1] == 0:
            # First step or last step - regular CFG
            cfg_denoised = uncond_denoised + (denoised - uncond_denoised) * current_cfg
        else:
            # CFG++ with momentum
            x0_coeff = cfg_x0_scale * current_cfg

            # Simple momentum for Euler Ancestral
            momentum = denoised
            uncond_momentum = uncond_denoised

            # Combined CFG++ update
            cfg_denoised = uncond_momentum + (momentum - uncond_momentum) * x0_coeff

        # Euler ancestral method with CFG++ denoised result
        d = util.to_d(x, sigma_hat, cfg_denoised)
        x = x + d * (sigma_down - sigma_hat)

        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        # Store for momentum calculation
        old_uncond_denoised = uncond_denoised

        if not pipeline and app_instance.app.previewer_var.get() and i % 5 == 0:
            threading.Thread(target=taesd.taesd_preview, args=(x,)).start()

    return x


def set_model_options_post_cfg_function(
    model_options, post_cfg_function, disable_cfg1_optimization=False
):
    model_options["sampler_post_cfg_function"] = model_options.get(
        "sampler_post_cfg_function", []
    ) + [post_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options


@torch.no_grad()
def sample_dpmpp_2m_cfgpp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    pipeline=False,
    # CFG++ parameters
    cfg_scale=7.5,
    cfg_x0_scale=1.0,
    cfg_s_scale=1.0,
    cfg_min=1.0,
    # Multi-scale parameters
    enable_multiscale=True,
    multiscale_factor=0.5,
    multiscale_fullres_start=5,
    multiscale_fullres_end=8,
    multiscale_intermittent_fullres=True,
):
    """DPM-Solver++(2M) sampler with CFG++ optimizations"""
    # Pre-calculate common values and setup
    device = x.device
    global disable_gui
    disable_gui = pipeline

    if not disable_gui:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance

    # Multi-scale setup with validation
    original_shape = x.shape
    batch_size, channels, orig_h, orig_w = original_shape

    # Validate multi-scale parameters
    if enable_multiscale:
        if not (0.1 <= multiscale_factor <= 1.0):
            print(
                f"Warning: multiscale_factor {multiscale_factor} out of range [0.1, 1.0], disabling multi-scale"
            )
            enable_multiscale = False
        if multiscale_fullres_start < 0 or multiscale_fullres_end < 0:
            print("Warning: Invalid fullres step counts, disabling multi-scale")
            enable_multiscale = False

    # Calculate scaled dimensions (must be multiples of 8 for VAE compatibility)
    scale_h = (
        int(max(8, ((orig_h * multiscale_factor) // 8) * 8))
        if enable_multiscale
        else orig_h
    )
    scale_w = (
        int(max(8, ((orig_w * multiscale_factor) // 8) * 8))
        if enable_multiscale
        else orig_w
    )

    # Pre-allocate tensors and transform sigmas
    s_in = torch.ones((x.shape[0],), device=device)
    t_steps = -torch.log(sigmas)  # Fused calculation
    n_steps = len(sigmas) - 1

    # Disable multi-scale for small images or short step counts
    multiscale_active = (
        enable_multiscale
        and orig_h > 64
        and orig_w > 64
        and n_steps > (multiscale_fullres_start + multiscale_fullres_end)
        and (scale_h != orig_h or scale_w != orig_w)
    )

    if enable_multiscale and not multiscale_active:
        print(
            f"Multi-scale disabled: image too small ({orig_h}x{orig_w}) or insufficient steps ({n_steps})"
        )
    elif multiscale_active:
        print(
            f"Multi-scale active: {orig_h}x{orig_w} -> {scale_h}x{scale_w} (factor: {multiscale_factor})"
        )

    def downscale_tensor(tensor):
        """Downscale tensor using bilinear interpolation"""
        if not multiscale_active or tensor.shape[-2:] == (scale_h, scale_w):
            return tensor
        return torch.nn.functional.interpolate(
            tensor, size=(scale_h, scale_w), mode="bilinear", align_corners=False
        )

    def upscale_tensor(tensor):
        """Upscale tensor using bilinear interpolation"""
        if not multiscale_active or tensor.shape[-2:] == (orig_h, orig_w):
            return tensor
        return torch.nn.functional.interpolate(
            tensor, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )

    def should_use_fullres(step):
        """Determine if this step should use full resolution"""
        if not multiscale_active:
            return True

        # Always use full resolution for start and end steps
        if step < multiscale_fullres_start or step >= n_steps - multiscale_fullres_end:
            return True

        # Intermittent full-res: every 2nd step in low-res region if enabled
        if multiscale_intermittent_fullres:
            # Check if we're in the low-res region
            low_res_region_start = multiscale_fullres_start
            low_res_region_end = n_steps - multiscale_fullres_end
            if low_res_region_start <= step < low_res_region_end:
                # Calculate position within low-res region
                relative_step = step - low_res_region_start
                # Use full-res every 2nd step (0, 2, 4, ...)
                return relative_step % 2 == 0

        return False

    # Pre-calculate all needed values in one go
    sigma_steps = torch.exp(-t_steps)  # Fused calculation
    ratios = sigma_steps[1:] / sigma_steps[:-1]
    h_steps = t_steps[1:] - t_steps[:-1]

    # Pre-calculate CFG schedule for the entire sampling process
    steps = torch.arange(n_steps, device=device)
    cfg_values = cfg_scale + (cfg_min - cfg_scale) * (steps / n_steps)

    old_denoised = None
    old_uncond_denoised = None
    extra_args = {} if extra_args is None else extra_args

    # Define post-CFG function once outside the loop
    def post_cfg_function(args):
        nonlocal old_uncond_denoised
        old_uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    for i in trange(n_steps, disable=disable):
        if (
            not pipeline
            and hasattr(app_instance.app, "interrupt_flag")
            and app_instance.app.interrupt_flag
        ):
            return x

        if not pipeline:
            app_instance.app.progress.set(i / n_steps)

        # Determine resolution for this step
        use_fullres = should_use_fullres(i)

        # Scale input for processing
        if use_fullres:
            x_process = x
            s_in_process = s_in
        else:
            x_process = downscale_tensor(x)
            s_in_process = torch.ones((x_process.shape[0],), device=device)

        # Use pre-calculated CFG scale
        current_cfg = cfg_values[i]

        # Model inference at appropriate resolution
        denoised = model(x_process, sigmas[i] * s_in_process, **extra_args)
        uncond_denoised = extra_args.get("model_options", {}).get(
            "sampler_post_cfg_function", []
        )[-1]({"denoised": denoised, "uncond_denoised": None})

        # Scale predictions back to original resolution if needed
        if not use_fullres:
            denoised = upscale_tensor(denoised)
            uncond_denoised = upscale_tensor(uncond_denoised)

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                    "cfg_scale": current_cfg,
                }
            )

        # CFG++ update step using optimized operations
        if old_uncond_denoised is None or sigmas[i + 1] == 0:
            # First step or last step - use torch.lerp for efficient interpolation
            cfg_denoised = torch.lerp(uncond_denoised, denoised, current_cfg)
        else:
            # Fused momentum calculations
            h_ratio = h_steps[i - 1] / (2 * h_steps[i])
            h_ratio_plus_1 = 1 + h_ratio

            # Use fused multiply-add operations for momentum terms
            momentum = torch.addcmul(denoised * h_ratio_plus_1, old_denoised, -h_ratio)
            uncond_momentum = torch.addcmul(
                uncond_denoised * h_ratio_plus_1, old_uncond_denoised, -h_ratio
            )

            # Optimized interpolation for CFG++ update
            cfg_denoised = torch.lerp(
                uncond_momentum, momentum, current_cfg * cfg_x0_scale
            )

        # Apply update with pre-calculated expm1
        h_expm1 = torch.expm1(-h_steps[i])
        x = ratios[i] * x - h_expm1 * cfg_denoised

        old_denoised = denoised
        old_uncond_denoised = uncond_denoised

        # Preview updates
        if not pipeline and app_instance.app.previewer_var.get() and i % 5 == 0:
            threading.Thread(target=taesd.taesd_preview, args=(x,)).start()

    return x


@torch.no_grad()
def sample_dpmpp_sde_cfgpp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    r=1 / 2,
    pipeline=False,
    seed=None,
    # CFG++ parameters
    cfg_scale=7.5,
    cfg_x0_scale=1.0,
    cfg_s_scale=1.0,
    cfg_min=1.0,
    # Multi-scale parameters
    enable_multiscale=True,
    multiscale_factor=0.5,
    multiscale_fullres_start=5,
    multiscale_fullres_end=8,
    multiscale_intermittent_fullres=False,
):
    """DPM-Solver++ (SDE) with CFG++ optimizations and multi-scale diffusion"""
    # Pre-calculate common values
    device = x.device
    global disable_gui
    disable_gui = pipeline

    if not disable_gui:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance

    # Early return check
    if len(sigmas) <= 1:
        return x

    # Multi-scale setup with validation
    original_shape = x.shape
    batch_size, channels, orig_h, orig_w = original_shape

    # Validate multi-scale parameters
    if enable_multiscale:
        if not (0.1 <= multiscale_factor <= 1.0):
            print(
                f"Warning: multiscale_factor {multiscale_factor} out of range [0.1, 1.0], disabling multi-scale"
            )
            enable_multiscale = False
        if multiscale_fullres_start < 0 or multiscale_fullres_end < 0:
            print("Warning: Invalid fullres step counts, disabling multi-scale")
            enable_multiscale = False

    # Calculate scaled dimensions (must be multiples of 8 for VAE compatibility)
    scale_h = (
        int(max(8, ((orig_h * multiscale_factor) // 8) * 8))
        if enable_multiscale
        else orig_h
    )
    scale_w = (
        int(max(8, ((orig_w * multiscale_factor) // 8) * 8))
        if enable_multiscale
        else orig_w
    )

    # Disable multi-scale for small images or short step counts
    n_steps = len(sigmas) - 1
    multiscale_active = (
        enable_multiscale
        and orig_h > 64
        and orig_w > 64
        and n_steps > (multiscale_fullres_start + multiscale_fullres_end)
        and (scale_h != orig_h or scale_w != orig_w)
    )

    if enable_multiscale and not multiscale_active:
        print(
            f"Multi-scale disabled: image too small ({orig_h}x{orig_w}) or insufficient steps ({n_steps})"
        )
    elif multiscale_active:
        print(
            f"Multi-scale active: {orig_h}x{orig_w} -> {scale_h}x{scale_w} (factor: {multiscale_factor})"
        )

    def downscale_tensor(tensor):
        """Downscale tensor using bilinear interpolation"""
        if not multiscale_active or tensor.shape[-2:] == (scale_h, scale_w):
            return tensor
        return torch.nn.functional.interpolate(
            tensor, size=(scale_h, scale_w), mode="bilinear", align_corners=False
        )

    def upscale_tensor(tensor):
        """Upscale tensor using bilinear interpolation"""
        if not multiscale_active or tensor.shape[-2:] == (orig_h, orig_w):
            return tensor
        return torch.nn.functional.interpolate(
            tensor, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )

    def should_use_fullres(step):
        """Determine if this step should use full resolution"""
        if not multiscale_active:
            return True

        # Always use full resolution for start and end steps
        if step < multiscale_fullres_start or step >= n_steps - multiscale_fullres_end:
            return True

        # Intermittent full-res: every 2nd step in low-res region if enabled
        if multiscale_intermittent_fullres:
            # Check if we're in the low-res region
            low_res_region_start = multiscale_fullres_start
            low_res_region_end = n_steps - multiscale_fullres_end
            if low_res_region_start <= step < low_res_region_end:
                # Calculate position within low-res region
                relative_step = step - low_res_region_start
                # Use full-res every 2nd step (0, 2, 4, ...)
                return relative_step % 2 == 0

        return False  # Pre-allocate tensors and values

    s_in = torch.ones((x.shape[0],), device=device)
    extra_args = {} if extra_args is None else extra_args

    # CFG++ scheduling
    def get_cfg_scale(step):
        progress = step / n_steps
        return cfg_scale + (cfg_min - cfg_scale) * progress

    # Helper functions
    def sigma_fn(t):
        return (-t).exp()

    def t_fn(sigma):
        return -sigma.log()

    # Initialize noise sampler
    if noise_sampler is None:
        noise_sampler = sampling_util.BrownianTreeNoiseSampler(
            x, sigmas[sigmas > 0].min(), sigmas.max(), seed=seed, cpu=True
        )

    # Track previous predictions for momentum (stored at original resolution)
    old_denoised = None
    old_uncond_denoised = None

    def post_cfg_function(args):
        nonlocal old_uncond_denoised
        old_uncond_denoised = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = set_model_options_post_cfg_function(
        model_options, post_cfg_function, disable_cfg1_optimization=True
    )

    for i in trange(n_steps, disable=disable):
        if (
            not pipeline
            and hasattr(app_instance.app, "interrupt_flag")
            and app_instance.app.interrupt_flag
        ):
            return x

        if not pipeline:
            app_instance.app.progress.set(i / n_steps)

        # Determine resolution for this step
        use_fullres = should_use_fullres(i)

        # Scale input for processing
        if use_fullres:
            x_process = x
            s_in_process = s_in
        else:
            x_process = downscale_tensor(x)
            s_in_process = torch.ones((x_process.shape[0],), device=device)

        # Get current CFG scale
        current_cfg = get_cfg_scale(i)

        # Model inference at appropriate resolution
        denoised = model(x_process, sigmas[i] * s_in_process, **extra_args)
        uncond_denoised = extra_args.get("model_options", {}).get(
            "sampler_post_cfg_function", []
        )[-1]({"denoised": denoised, "uncond_denoised": None})

        # Scale predictions back to original resolution if needed
        if not use_fullres:
            denoised = upscale_tensor(denoised)
            uncond_denoised = upscale_tensor(uncond_denoised)

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                    "cfg_scale": current_cfg,
                }
            )

        if sigmas[i + 1] == 0:
            # Final step - regular CFG
            cfg_denoised = uncond_denoised + (denoised - uncond_denoised) * current_cfg
            x = x + util.to_d(x, sigmas[i], cfg_denoised) * (sigmas[i + 1] - sigmas[i])
        else:
            # Two-step update with CFG++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            s = t + (t_next - t) * r

            # Step 1 with CFG++
            sd, su = sampling_util.get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)

            if old_uncond_denoised is None:
                # First step - regular CFG
                cfg_denoised = (
                    uncond_denoised + (denoised - uncond_denoised) * current_cfg
                )
            else:
                # CFG++ with momentum (using properly scaled momentum terms)
                x0_coeff = cfg_x0_scale * current_cfg

                # Calculate momentum terms at original resolution
                h_ratio = (t - s_) / (2 * (t - t_next))
                momentum = (1 + h_ratio) * denoised - h_ratio * old_denoised
                uncond_momentum = (
                    1 + h_ratio
                ) * uncond_denoised - h_ratio * old_uncond_denoised

                # Combine with CFG++ scaling
                cfg_denoised = uncond_momentum + (momentum - uncond_momentum) * x0_coeff

            # Calculate x_2 for step 2
            noise_step1 = noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            x_2 = (
                (sigma_fn(s_) / sigma_fn(t)) * x
                - (t - s_).expm1() * cfg_denoised
                + noise_step1
            )

            # Step 2 inference - determine resolution
            use_fullres_step2 = should_use_fullres(i) if multiscale_active else True

            if use_fullres_step2:
                x_2_process = x_2
                s_in_process_2 = s_in
            else:
                x_2_process = downscale_tensor(x_2)
                s_in_process_2 = torch.ones((x_2_process.shape[0],), device=device)

            # Step 2 inference
            denoised_2 = model(x_2_process, sigma_fn(s) * s_in_process_2, **extra_args)
            uncond_denoised_2 = extra_args.get("model_options", {}).get(
                "sampler_post_cfg_function", []
            )[-1]({"denoised": denoised_2, "uncond_denoised": None})

            # Scale step 2 predictions back if needed
            if not use_fullres_step2:
                denoised_2 = upscale_tensor(denoised_2)
                uncond_denoised_2 = upscale_tensor(uncond_denoised_2)

            # Step 2 CFG++ combination
            if old_uncond_denoised is None:
                cfg_denoised_2 = (
                    uncond_denoised_2 + (denoised_2 - uncond_denoised_2) * current_cfg
                )
            else:
                momentum_2 = (1 + h_ratio) * denoised_2 - h_ratio * denoised
                uncond_momentum_2 = (
                    1 + h_ratio
                ) * uncond_denoised_2 - h_ratio * uncond_denoised
                cfg_denoised_2 = (
                    uncond_momentum_2 + (momentum_2 - uncond_momentum_2) * x0_coeff
                )

            # Final ancestral step
            sd, su = sampling_util.get_ancestral_step(
                sigma_fn(t), sigma_fn(t_next), eta
            )
            t_next_ = t_fn(sd)

            # Combined update with both predictions
            noise_final = noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
            x = (
                (sigma_fn(t_next_) / sigma_fn(t)) * x
                - (t - t_next_).expm1()
                * ((1 - 1 / (2 * r)) * cfg_denoised + (1 / (2 * r)) * cfg_denoised_2)
                + noise_final
            )

        old_denoised = denoised
        old_uncond_denoised = uncond_denoised

        # Preview updates
        if not pipeline and app_instance.app.previewer_var.get() and i % 5 == 0:
            threading.Thread(target=taesd.taesd_preview, args=(x,)).start()

    return x

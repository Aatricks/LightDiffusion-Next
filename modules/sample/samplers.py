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
):
    # Pre-calculate common values
    device = x.device
    global disable_gui
    disable_gui = pipeline

    if not disable_gui:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance

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

        # Combined model inference and step calculation
        denoised = model(x, sigmas[i] * s_in, **(extra_args or {}))
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
):
    # Pre-calculate common values
    device = x.device
    global disable_gui
    disable_gui = pipeline

    if not disable_gui:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance

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

        # Fused model inference and update step
        denoised = model(x, sigma_hat * s_in, **(extra_args or {}))
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


@torch.no_grad()
def sample_dpmpp_sde(
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
):
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

    # Pre-allocate tensors and values
    s_in = torch.ones((x.shape[0],), device=device)
    n_steps = len(sigmas) - 1
    extra_args = {} if extra_args is None else extra_args

    # Define helper functions
    def sigma_fn(t):
        return (-t).exp()

    def t_fn(sigma):
        return -sigma.log()

    # Initialize noise sampler
    if noise_sampler is None:
        noise_sampler = sampling_util.BrownianTreeNoiseSampler(
            x, sigmas[sigmas > 0].min(), sigmas.max(), seed=seed, cpu=True
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

        # Model inference
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "denoised": denoised})

        if sigmas[i + 1] == 0:
            # Single fused Euler step
            x = x + util.to_d(x, sigmas[i], denoised) * (sigmas[i + 1] - sigmas[i])
        else:
            # Fused DPM-Solver++ steps
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            s = t + (t_next - t) * r

            # Step 1 - Combined calculations
            sd, su = sampling_util.get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            x_2 = (
                (sigma_fn(s_) / sigma_fn(t)) * x
                - (t - s_).expm1() * denoised
                + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            )

            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2 - Combined calculations
            sd, su = sampling_util.get_ancestral_step(
                sigma_fn(t), sigma_fn(t_next), eta
            )
            t_next_ = t_fn(sd)

            # Final update in single calculation
            x = (
                (sigma_fn(t_next_) / sigma_fn(t)) * x
                - (t - t_next_).expm1()
                * ((1 - 1 / (2 * r)) * denoised + (1 / (2 * r)) * denoised_2)
                + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
            )

        # Preview updates
        if not pipeline and app_instance.app.previewer_var.get() and i % 5 == 0:
            threading.Thread(target=taesd.taesd_preview, args=(x,)).start()

    return x


@torch.no_grad()
def sample_dpmpp_2m(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    pipeline=False,
):
    """DPM-Solver++(2M) sampler with optimizations"""
    # Pre-calculate common values and setup
    device = x.device
    global disable_gui
    disable_gui = pipeline

    if not disable_gui:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance

    # Pre-allocate tensors and transform sigmas
    s_in = torch.ones((x.shape[0],), device=device)
    t_steps = -torch.log(sigmas)  # Fused calculation

    # Pre-calculate all needed values in one go
    sigma_steps = torch.exp(-t_steps)  # Fused calculation
    ratios = sigma_steps[1:] / sigma_steps[:-1]
    h_steps = t_steps[1:] - t_steps[:-1]

    old_denoised = None
    extra_args = {} if extra_args is None else extra_args

    for i in trange(len(sigmas) - 1, disable=disable):
        if (
            not pipeline
            and hasattr(app_instance.app, "interrupt_flag")
            and app_instance.app.interrupt_flag
        ):
            return x

        if not pipeline:
            app_instance.app.progress.set(i / (len(sigmas) - 1))

        # Fused model inference and update calculations
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )

        # Combined update step
        x = ratios[i] * x - (-h_steps[i]).expm1() * (
            denoised
            if old_denoised is None or sigmas[i + 1] == 0
            else (1 + h_steps[i - 1] / (2 * h_steps[i])) * denoised
            - (h_steps[i - 1] / (2 * h_steps[i])) * old_denoised
        )

        old_denoised = denoised

        # Preview updates
        if not pipeline and app_instance.app.previewer_var.get() and i % 5 == 0:
            threading.Thread(target=taesd.taesd_preview, args=(x,)).start()

    return x


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
):
    """DPM-Solver++(2M) sampler with CFG++ optimizations"""
    # Pre-calculate common values and setup
    device = x.device
    global disable_gui
    disable_gui = pipeline

    if not disable_gui:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance

    # Pre-allocate tensors and transform sigmas
    s_in = torch.ones((x.shape[0],), device=device)
    t_steps = -torch.log(sigmas)  # Fused calculation
    n_steps = len(sigmas) - 1

    # Pre-calculate all needed values in one go
    sigma_steps = torch.exp(-t_steps)  # Fused calculation
    ratios = sigma_steps[1:] / sigma_steps[:-1]
    h_steps = t_steps[1:] - t_steps[:-1]

    # CFG++ scheduling
    def get_cfg_scale(step):
        # Linear scheduling from cfg_scale to cfg_min
        progress = step / n_steps
        return cfg_scale + (cfg_min - cfg_scale) * progress

    old_denoised = None
    old_uncond_denoised = None
    extra_args = {} if extra_args is None else extra_args

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

        def post_cfg_function(args):
            nonlocal old_uncond_denoised
            old_uncond_denoised = args["uncond_denoised"]
            return args["denoised"]

        model_options = extra_args.get("model_options", {}).copy()
        extra_args["model_options"] = set_model_options_post_cfg_function(
            model_options, post_cfg_function, disable_cfg1_optimization=True
        )

        # Fused model inference and update calculations
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        uncond_denoised = extra_args.get("model_options", {}).get(
            "sampler_post_cfg_function", []
        )[-1]({"denoised": denoised, "uncond_denoised": None})

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

        # CFG++ update step
        if old_uncond_denoised is None or sigmas[i + 1] == 0:
            # First step or last step - regular update
            cfg_denoised = uncond_denoised + (denoised - uncond_denoised) * current_cfg
        else:
            # CFG++ combination with momentum
            x0_coeff = cfg_x0_scale * current_cfg
            s_coeff = cfg_s_scale * current_cfg

            # Momentum terms
            h_ratio = h_steps[i - 1] / (2 * h_steps[i])
            momentum = (1 + h_ratio) * denoised - h_ratio * old_denoised
            uncond_momentum = (
                1 + h_ratio
            ) * uncond_denoised - h_ratio * old_uncond_denoised

            # Combined update
            cfg_denoised = uncond_momentum + (momentum - uncond_momentum) * x0_coeff

        # Apply update
        x = ratios[i] * x - (-h_steps[i]).expm1() * cfg_denoised

        old_denoised = denoised
        old_uncond_denoised = uncond_denoised

        # Preview updates
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
):
    """DPM-Solver++ (SDE) with CFG++ optimizations"""
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

    # Pre-allocate tensors and values
    s_in = torch.ones((x.shape[0],), device=device)
    n_steps = len(sigmas) - 1
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

    # Track previous predictions
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

        # Get current CFG scale
        current_cfg = get_cfg_scale(i)

        # Model inference
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        uncond_denoised = extra_args.get("model_options", {}).get(
            "sampler_post_cfg_function", []
        )[-1]({"denoised": denoised, "uncond_denoised": None})

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
                # CFG++ with momentum
                x0_coeff = cfg_x0_scale * current_cfg
                s_coeff = cfg_s_scale * current_cfg

                # Calculate momentum terms
                h_ratio = (t - s_) / (2 * (t - t_next))
                momentum = (1 + h_ratio) * denoised - h_ratio * old_denoised
                uncond_momentum = (
                    1 + h_ratio
                ) * uncond_denoised - h_ratio * old_uncond_denoised

                # Combine with CFG++ scaling
                cfg_denoised = uncond_momentum + (momentum - uncond_momentum) * x0_coeff

            x_2 = (
                (sigma_fn(s_) / sigma_fn(t)) * x
                - (t - s_).expm1() * cfg_denoised
                + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            )

            # Step 2 inference
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            uncond_denoised_2 = extra_args.get("model_options", {}).get(
                "sampler_post_cfg_function", []
            )[-1]({"denoised": denoised_2, "uncond_denoised": None})

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
            x = (
                (sigma_fn(t_next_) / sigma_fn(t)) * x
                - (t - t_next_).expm1()
                * ((1 - 1 / (2 * r)) * cfg_denoised + (1 / (2 * r)) * cfg_denoised_2)
                + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su
            )

        old_denoised = denoised
        old_uncond_denoised = uncond_denoised

        # Preview updates
        if not pipeline and app_instance.app.previewer_var.get() and i % 5 == 0:
            threading.Thread(target=taesd.taesd_preview, args=(x,)).start()

    return x

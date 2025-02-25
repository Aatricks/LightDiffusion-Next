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
    noise_sampler = sampling_util.default_noise_sampler(x) if noise_sampler is None else noise_sampler

    for i in trange(len(sigmas) - 1, disable=disable):
        if not pipeline and hasattr(app_instance.app, 'interrupt_flag') and app_instance.app.interrupt_flag:
            return x

        if not pipeline:
            app_instance.app.progress.set(i/(len(sigmas)-1))

        # Combined model inference and step calculation
        denoised = model(x, sigmas[i] * s_in, **(extra_args or {}))
        sigma_down, sigma_up = sampling_util.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        
        # Fused update step
        x = x + util.to_d(x, sigmas[i], denoised) * (sigma_down - sigmas[i])
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'denoised': denoised})

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
    gamma_max = min(s_churn / (len(sigmas)-1), 2**0.5 - 1) if s_churn > 0 else 0

    for i in trange(len(sigmas) - 1, disable=disable):
        if not pipeline and hasattr(app_instance.app, 'interrupt_flag') and app_instance.app.interrupt_flag:
            return x

        if not pipeline:
            app_instance.app.progress.set(i/(len(sigmas)-1))

        # Combined sigma calculation and update
        sigma_hat = sigmas[i] * (1 + (gamma_max if s_tmin <= sigmas[i] <= s_tmax else 0)) if gamma_max > 0 else sigmas[i]
        
        if gamma_max > 0 and sigma_hat > sigmas[i]:
            x = x + torch.randn_like(x) * s_noise * (sigma_hat**2 - sigmas[i]**2)**0.5

        # Fused model inference and update step
        denoised = model(x, sigma_hat * s_in, **(extra_args or {}))
        x = x + util.to_d(x, sigma_hat, denoised) * (sigmas[i + 1] - sigma_hat)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})

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
    eta=1.,
    s_noise=1.,
    noise_sampler=None,
    r=1/2,
    pipeline=False,
    seed=None
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
    def sigma_fn(t): return (-t).exp()
    def t_fn(sigma): return -sigma.log()

    # Initialize noise sampler
    if noise_sampler is None:
        noise_sampler = sampling_util.BrownianTreeNoiseSampler(
            x, sigmas[sigmas > 0].min(), sigmas.max(), seed=seed, cpu=True
        )

    for i in trange(n_steps, disable=disable):
        if not pipeline and hasattr(app_instance.app, 'interrupt_flag') and app_instance.app.interrupt_flag:
            return x

        if not pipeline:
            app_instance.app.progress.set(i/n_steps)

        # Model inference 
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'denoised': denoised})

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
            x_2 = (sigma_fn(s_) / sigma_fn(t)) * x - (t - s_).expm1() * denoised + \
                  noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2 - Combined calculations  
            sd, su = sampling_util.get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            
            # Final update in single calculation
            x = (sigma_fn(t_next_) / sigma_fn(t)) * x - \
                (t - t_next_).expm1() * ((1 - 1/(2*r)) * denoised + (1/(2*r)) * denoised_2) + \
                noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su

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
        if not pipeline and hasattr(app_instance.app, 'interrupt_flag') and app_instance.app.interrupt_flag:
            return x

        if not pipeline:
            app_instance.app.progress.set(i/(len(sigmas)-1))
        
        # Fused model inference and update calculations
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        # Combined update step
        x = ratios[i] * x - (-h_steps[i]).expm1() * (
            denoised if old_denoised is None or sigmas[i + 1] == 0
            else (1 + h_steps[i-1]/(2*h_steps[i])) * denoised - 
                 (h_steps[i-1]/(2*h_steps[i])) * old_denoised
        )
            
        old_denoised = denoised
        
        # Preview updates
        if not pipeline and app_instance.app.previewer_var.get() and i % 5 == 0:
            threading.Thread(target=taesd.taesd_preview, args=(x,)).start()

    return x
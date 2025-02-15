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

    # Pre-allocate tensors
    s_in = torch.ones((x.shape[0],), device=device)
    n_steps = len(sigmas) - 1
    extra_args = {} if extra_args is None else extra_args
    
    # Cache noise sampler
    noise_sampler = sampling_util.default_noise_sampler(x) if noise_sampler is None else noise_sampler

    for i in trange(n_steps, disable=disable):
        if not pipeline and hasattr(app_instance.app, 'interrupt_flag') and app_instance.app.interrupt_flag:
            return x

        if not pipeline:
            app_instance.app.progress.set(i/n_steps)

        # Model inference
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        # Calculate steps
        sigma_down, sigma_up = sampling_util.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        
        # Update x
        d = util.to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'denoised': denoised})

        # Preview updates
        if not pipeline and app_instance.app.previewer_var.get() and i % 5 == 0:
            threading.Thread(target=taesd.taesd_preview, args=(x,)).start()

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

    # Pre-allocate tensors
    s_in = torch.ones((x.shape[0],), device=device)
    n_steps = len(sigmas) - 1
    extra_args = {} if extra_args is None else extra_args

    # Cache transformed values
    sigma_min = sigmas[sigmas > 0].min()
    sigma_max = sigmas.max()
    
    # Optimize math functions
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()

    # Initialize noise sampler
    if noise_sampler is None:
        noise_sampler = sampling_util.BrownianTreeNoiseSampler(
            x, sigma_min, sigma_max, seed=seed, cpu=True
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
            # Euler method
            d = util.to_d(x, sigmas[i], denoised)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
        else:
            # DPM-Solver++
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r

            # Step 1
            sd, su = sampling_util.get_ancestral_step(sigma_fn(t), sigma_fn(s), eta)
            s_ = t_fn(sd)
            
            sigma_ratio = sigma_fn(s_) / sigma_fn(t)
            x_2 = sigma_ratio * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(sigma_fn(t), sigma_fn(s)) * s_noise * su
            
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)

            # Step 2
            sd, su = sampling_util.get_ancestral_step(sigma_fn(t), sigma_fn(t_next), eta)
            t_next_ = t_fn(sd)
            
            fac = 1 / (2 * r)
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            
            sigma_ratio = sigma_fn(t_next_) / sigma_fn(t)
            x = sigma_ratio * x - (t - t_next_).expm1() * denoised_d
            x = x + noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * su

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
    # Pre-calculate common values
    device = x.device
    global disable_gui
    disable_gui = pipeline
    
    if not disable_gui:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance
        
    # Pre-allocate tensors
    s_in = torch.ones((x.shape[0],), device=device)
    n_steps = len(sigmas) - 1
    
    # Optimize math functions
    sigma_fn = torch.exp
    t_fn = lambda sigma: -torch.log(sigma)
    
    # Cache transformed values
    t_steps = t_fn(sigmas)
    sigma_steps = sigma_fn(-t_steps)
    
    # Pre-calculate ratios 
    ratios = sigma_steps[1:] / sigma_steps[:-1]
    h_steps = t_steps[1:] - t_steps[:-1]
    
    old_denoised = None
    
    for i in trange(n_steps, disable=disable):
        if not pipeline and hasattr(app_instance.app, 'interrupt_flag') and app_instance.app.interrupt_flag:
            return x

        if not pipeline:
            app_instance.app.progress.set(i/n_steps)
            
        # Get current values
        ratio = ratios[i]
        h = h_steps[i]
        
        # Model inference
        denoised = model(x, sigmas[i] * s_in, **(extra_args or {}))
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            
        # Main update step
        if old_denoised is None or sigmas[i + 1] == 0:
            x = ratio * x - (-h).expm1() * denoised
        else:
            h_last = t_steps[i] - t_steps[i-1] 
            r = h_last / h
            r_const = 1 / (2 * r)
            denoised_d = (1 + r_const) * denoised - r_const * old_denoised
            x = ratio * x - (-h).expm1() * denoised_d
            
        old_denoised = denoised
        
        # Preview updates
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

    # Pre-allocate tensors
    s_in = torch.ones((x.shape[0],), device=device)
    n_steps = len(sigmas) - 1
    extra_args = {} if extra_args is None else extra_args

    # Cache churn parameters
    use_churn = s_churn > 0
    if use_churn:
        gamma_max = min(s_churn / n_steps, 2**0.5 - 1)

    for i in trange(n_steps, disable=disable):
        if not pipeline and hasattr(app_instance.app, 'interrupt_flag') and app_instance.app.interrupt_flag:
            return x

        if not pipeline:
            app_instance.app.progress.set(i/n_steps)

        # Calculate sigma
        if use_churn:
            gamma = gamma_max if s_tmin <= sigmas[i] <= s_tmax else 0.0
            sigma_hat = sigmas[i] * (gamma + 1)
            
            if gamma > 0:
                eps = torch.randn_like(x) * s_noise
                x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        else:
            sigma_hat = sigmas[i]

        # Model inference
        denoised = model(x, sigma_hat * s_in, **extra_args)
        
        # Update x
        d = util.to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt

        if callback is not None:
            callback({
                "x": x,
                "i": i, 
                "sigma": sigmas[i],
                "sigma_hat": sigma_hat,
                "denoised": denoised,
            })

        # Preview updates  
        if not pipeline and app_instance.app.previewer_var.get() and i % 5 == 0:
            threading.Thread(target=taesd.taesd_preview, args=(x, True)).start()

    return x
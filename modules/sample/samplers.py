import threading
import torch
from tqdm.auto import trange, tqdm
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
    """#### Perform ancestral sampling using the Euler method.

    #### Args:
        - `model` (torch.nn.Module): The model to use for denoising.
        - `x` (torch.Tensor): The input tensor to be denoised.
        - `sigmas` (list or torch.Tensor): A list or tensor of sigma values for the noise schedule.
        - `extra_args` (dict, optional): Additional arguments to pass to the model. Defaults to None.
        - `callback` (callable, optional): A callback function to be called at each iteration. Defaults to None.
        - `disable` (bool, optional): If True, disables the progress bar. Defaults to None.
        - `eta` (float, optional): The eta parameter for the ancestral step. Defaults to 1.0.
        - `s_noise` (float, optional): The noise scaling factor. Defaults to 1.0.
        - `noise_sampler` (callable, optional): A function to sample noise. Defaults to None.

    #### Returns:
        - `torch.Tensor`: The denoised tensor after ancestral sampling.
    """
    global disable_gui
    disable_gui = True if pipeline is True else False
    if disable_gui is False:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = (
        sampling_util.default_noise_sampler(x)
        if noise_sampler is None
        else noise_sampler
    )
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if pipeline is False:
            if app_instance.app.interrupt_flag is True:
                break
            try:
                app_instance.app.title(f"LightDiffusion - {i}it")
            except:
                pass
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = sampling_util.get_ancestral_step(
            sigmas[i], sigmas[i + 1], eta=eta
        )
        d = util.to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
        if pipeline is False:
            if app_instance.app.previewer_var.get() is True:
                threading.Thread(target=taesd.taesd_preview, args=(x,)).start()
            else:
                pass
    return x


@torch.no_grad()
def sample_dpm_adaptive(
    model,
    x,
    sigma_min,
    sigma_max,
    extra_args=None,
    callback=None,
    disable=None,
    order=3,
    rtol=0.05,
    atol=0.0078,
    h_init=0.05,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
    accept_safety=0.81,
    eta=0.0,
    s_noise=1.0,
    noise_sampler=None,
    return_info=False,
    pipeline=False,
):
    """
    #### Samples from a diffusion probabilistic model using an adaptive step size solver.

    This function implements the DPM-Solver-12 and DPM-Solver-23 methods with adaptive step size as described in the paper
    https://arxiv.org/abs/2206.00927.

    #### Args:
        - `model` (torch.nn.Module): The diffusion model to sample from.
        - `x` (torch.Tensor): The initial tensor to start sampling from.
        - `sigma_min` (float): The minimum sigma value for the sampling process.
        - `sigma_max` (float): The maximum sigma value for the sampling process.
        - `extra_args` (dict, optional): Additional arguments to pass to the model. Default is None.
        - `callback` (callable, optional): A callback function to be called with progress information. Default is None.
        - `disable` (bool, optional): If True, disables the progress bar. Default is None.
        - `order` (int, optional): The order of the solver. Default is 3.
        - `rtol` (float, optional): Relative tolerance for adaptive step size. Default is 0.05.
        - `atol` (float, optional): Absolute tolerance for adaptive step size. Default is 0.0078.
        - `h_init` (float, optional): Initial step size. Default is 0.05.
        - `pcoeff` (float, optional): Coefficient for the predictor step. Default is 0.0.
        - `icoeff` (float, optional): Coefficient for the corrector step. Default is 1.0.
        - `dcoeff` (float, optional): Coefficient for the diffusion step. Default is 0.0.
        - `accept_safety` (float, optional): Safety factor for step acceptance. Default is 0.81.
        - `eta` (float, optional): Noise scale for the sampling process. Default is 0.0.
        - `s_noise` (float, optional): Scale of the noise to be added. Default is 1.0.
        - `noise_sampler` (callable, optional): A function to sample noise. Default is None.
        - `return_info` (bool, optional): If True, returns additional information about the sampling process. Default is False.

    #### Returns:
        - `torch.Tensor`: The sampled tensor.
        - `dict` (optional): Additional information about the sampling process if `return_info` is True.

    #### Raises:
        - `ValueError`: If sigma_min or sigma_max is less than or equal to 0.
    """
    global disable_gui
    disable_gui = True if pipeline is True else False
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must not be 0")
    with tqdm(disable=disable) as pbar:
        dpm_solver = sampling_util.DPMSolver(
            model, extra_args, eps_callback=pbar.update
        )
        if callback is not None:
            dpm_solver.info_callback = lambda info: callback(
                {
                    "sigma": dpm_solver.sigma(info["t"]),
                    "sigma_hat": dpm_solver.sigma(info["t_up"]),
                    **info,
                }
            )
        with torch.amp.autocast(device_type="cuda"):
            x, info = dpm_solver.dpm_solver_adaptive(
                x,
                dpm_solver.t(sigma_max.clone().detach()),
                dpm_solver.t(sigma_min.clone().detach()),
                order,
                rtol,
                atol,
                h_init,
                pcoeff,
                icoeff,
                dcoeff,
                accept_safety,
                eta,
                s_noise,
                noise_sampler,
                pipeline,
            )
    if return_info:
        return x, info
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    eta=1.0,
    s_noise=1.0,
    noise_sampler=None,
    solver_type="midpoint",
    pipeline=False,
):
    """
    #### Samples from a model using the DPM-Solver++(2M) SDE method.

    #### Args:
        - `model` (torch.nn.Module): The model to sample from.
        - `x` (torch.Tensor): The initial input tensor.
        - `sigmas` (torch.Tensor): A tensor of sigma values for the SDE.
        - `extra_args` (dict, optional): Additional arguments for the model. Default is None.
        - `callback` (callable, optional): A callback function to be called at each step. Default is None.
        - `disable` (bool, optional): If True, disables the progress bar. Default is None.
        - `eta` (float, optional): The eta parameter for the SDE. Default is 1.0.
        - `s_noise` (float, optional): The noise scale parameter. Default is 1.0.
        - `noise_sampler` (callable, optional): A noise sampler function. Default is None.
        - `solver_type` (str, optional): The type of solver to use ('midpoint' or 'heun'). Default is "midpoint".
        - `pipeline` (bool, optional): If True, disables the progress bar. Default is False.

    #### Returns:
        - `torch.Tensor`: The final sampled tensor.
    """
    global disable_gui
    disable_gui = True if pipeline is True else False
    if disable_gui is False:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = (
        sampling_util.BrownianTreeNoiseSampler(
            x, sigma_min, sigma_max, seed=seed, cpu=True
        )
        if noise_sampler is None
        else noise_sampler
    )
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_denoised = None
    h_last = None
    h = None
    for i in trange(len(sigmas) - 1, disable=disable):
        if pipeline is False:
            if app_instance.app.interrupt_flag is True:
                break
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h
            x = (
                sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x
                + (-h - eta_h).expm1().neg() * denoised
            )
            if old_denoised is not None:
                r = h_last / h
                if solver_type == "heun":
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (
                        1 / r
                    ) * (denoised - old_denoised)
                elif solver_type == "midpoint":
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (
                        denoised - old_denoised
                    )
            if eta:
                x = (
                    x
                    + noise_sampler(sigmas[i], sigmas[i + 1])
                    * sigmas[i + 1]
                    * (-2 * eta_h).expm1().neg().sqrt()
                    * s_noise
                )
        if pipeline is False:
            if app_instance.app.previewer_var.get() is True:
                threading.Thread(target=taesd.taesd_preview, args=(x,)).start()
            else:
                pass
        old_denoised = denoised
        h_last = h
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
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    global disable_gui
    disable_gui = True if pipeline is True else False
    if disable_gui is False:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        if pipeline is False:
            if app_instance.app.interrupt_flag is True:
                break
        if s_churn > 0:
            gamma = (
                min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
                if s_tmin <= sigmas[i] <= s_tmax
                else 0.0
            )
            sigma_hat = sigmas[i] * (gamma + 1)
        else:
            gamma = 0
            sigma_hat = sigmas[i]

        if gamma > 0:
            eps = torch.randn_like(x) * s_noise
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = util.to_d(x, sigma_hat, denoised)
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
        dt = sigmas[i + 1] - sigma_hat
        # Euler method
        x = x + d * dt
        if pipeline is False:
            if app_instance.app.previewer_var.get() is True:
                threading.Thread(target=taesd.taesd_preview, args=(x, True)).start()
            else:
                pass
    return x
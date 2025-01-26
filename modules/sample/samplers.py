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
    noise_sampler = sampling_util.default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        # Move interrupt check outside pipeline condition
        if not pipeline and hasattr(app_instance.app, 'interrupt_flag') and app_instance.app.interrupt_flag is True:
            return x

        if pipeline is False:
            try:
                app_instance.app.title(f"LightDiffusion - {i}it")
                app_instance.app.progress.set(((i)/(len(sigmas)-1)))
            except:
                pass

        # Rest of sampling code remains the same
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = sampling_util.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        d = util.to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up

        if pipeline is False:
            if app_instance.app.previewer_var.get() is True and i % 5 == 0:
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
    """
    #### Samples from a model using the DPM-Solver++(2M) SDE method.

    #### Args:
        - `model` (torch.nn.Module): The model to sample from.
        - `x` (torch.Tensor): The initial input tensor.
        - `sigmas` (torch.Tensor): A tensor of sigma values for the SDE.
        - `extra_args` (dict, optional): Additional arguments for the model. Default is None.
        - `callback` (callable, optional): A callback function to be called at each step. Default is None.
        - `disable` (bool, optional): If True, disables the progress bar. Default is None.
        - `pipeline` (bool, optional): If True, disables the progress bar. Default is False.

    #### Returns:
        - `torch.Tensor`: The final sampled tensor.
    """
    global disable_gui
    disable_gui = True if pipeline is True else False
    if disable_gui is False:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    def sigma_fn(t):
        return t.neg().exp()
    def t_fn(sigma):
        return sigma.log().neg()
    old_denoised = None

    for i in trange(len(sigmas) - 1, disable=disable):
        if not pipeline and hasattr(app_instance.app, 'interrupt_flag') and app_instance.app.interrupt_flag is True:
            return x

        if pipeline is False:
            app_instance.app.progress.set(((i)/(len(sigmas)-1)))
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
        if pipeline is False:
            if app_instance.app.previewer_var.get() is True and i % 5 == 0:
                threading.Thread(target=taesd.taesd_preview, args=(x,)).start()
            else:
                pass
    return x


@torch.no_grad()
def sample_euler(
    model: torch.nn.Module,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict = None,
    callback: callable = None,
    disable: bool = None,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    pipeline: bool = False,
):
    """#### Implements Algorithm 2 (Euler steps) from Karras et al. (2022).

    #### Args:
        - `model` (torch.nn.Module): The model to use for denoising.
        - `x` (torch.Tensor): The input tensor to be denoised.
        - `sigmas` (list or torch.Tensor): A list or tensor of sigma values for the noise schedule.
        - `extra_args` (dict, optional): Additional arguments to pass to the model. Defaults to None.
        - `callback` (callable, optional): A callback function to be called at each iteration. Defaults to None.
        - `disable` (bool, optional): If True, disables the progress bar. Defaults to None.
        - `s_churn` (float, optional): The churn rate. Defaults to 0.0.
        - `s_tmin` (float, optional): The minimum sigma value for churn. Defaults to 0.0.
        - `s_tmax` (float, optional): The maximum sigma value for churn. Defaults to float("inf").
        - `s_noise` (float, optional): The noise scaling factor. Defaults to 1.0.
        - `pipeline` (bool, optional): If True, disables the progress bar. Defaults to False.

    #### Returns:
        - `torch.Tensor`: The denoised tensor after Euler sampling.
    """
    global disable_gui
    disable_gui = True if pipeline is True else False
    if disable_gui is False:
        from modules.AutoEncoders import taesd
        from modules.user import app_instance

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        if not pipeline and hasattr(app_instance.app, 'interrupt_flag') and app_instance.app.interrupt_flag is True:
            return x

        if pipeline is False:
            app_instance.app.progress.set(((i)/(len(sigmas)-1)))
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
            if app_instance.app.previewer_var.get() is True and i % 5 == 0:
                threading.Thread(target=taesd.taesd_preview, args=(x, True)).start()
            else:
                pass
    return x

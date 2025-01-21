import logging
import math
import threading
import torch
import torchsde
from torch import nn

from modules.Utilities import util


disable_gui = False

logging_level = logging.INFO

logging.basicConfig(format="%(message)s", level=logging_level)


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    """#### Create a beta schedule.

    #### Args:
        - `schedule` (str): The schedule type.
        - `n_timestep` (int): The number of timesteps.
        - `linear_start` (float, optional): The linear start value. Defaults to 1e-4.
        - `linear_end` (float, optional): The linear end value. Defaults to 2e-2.
        - `cosine_s` (float, optional): The cosine s value. Defaults to 8e-3.

    #### Returns:
        - `list`: The beta schedule.
    """
    betas = (
        torch.linspace(
            linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
        )
        ** 2
    )
    return betas


def checkpoint(func, inputs, params, flag):
    """#### Create a checkpoint.

    #### Args:
        - `func` (callable): The function to checkpoint.
        - `inputs` (list): The inputs to the function.
        - `params` (list): The parameters of the function.
        - `flag` (bool): The checkpoint flag.

    #### Returns:
        - `any`: The checkpointed output.
    """
    return func(*inputs)

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """#### Create a timestep embedding.

    #### Args:
        - `timesteps` (torch.Tensor): The timesteps.
        - `dim` (int): The embedding dimension.
        - `max_period` (int, optional): The maximum period. Defaults to 10000.
        - `repeat_only` (bool, optional): Whether to repeat only. Defaults to False.

    #### Returns:
        - `torch.Tensor`: The timestep embedding.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding

def timestep_embedding_flux(t: torch.Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """#### Create a timestep embedding.

    #### Args:
        - `timesteps` (torch.Tensor): The timesteps.
        - `dim` (int): The embedding dimension.
        - `max_period` (int, optional): The maximum period. Defaults to 10000.
        - `repeat_only` (bool, optional): Whether to repeat only. Defaults to False.

    #### Returns:
        - `torch.Tensor`: The timestep embedding.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """#### Get the sigmas for Karras sampling.

    constructs the noise schedule of Karras et al. (2022).

    #### Args:
        - `n` (int): The number of sigmas.
        - `sigma_min` (float): The minimum sigma value.
        - `sigma_max` (float): The maximum sigma value.
        - `rho` (float, optional): The rho value. Defaults to 7.0.
        - `device` (str, optional): The device to use. Defaults to "cpu".

    #### Returns:
        - `torch.Tensor`: The sigmas.
    """
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return util.append_zero(sigmas).to(device)


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    """
    #### Calculate the ancestral step in a diffusion process.

    This function computes the values of `sigma_down` and `sigma_up` based on the
    input parameters `sigma_from`, `sigma_to`, and `eta`. These values are used
    in the context of diffusion models to determine the next step in the process.

    #### Parameters:
        - `sigma_from` (float): The starting value of sigma.
        - `sigma_to` (float): The target value of sigma.
        - `eta` (float, optional): A scaling factor for the step size. Default is 1.0.

    #### Returns:
    - `tuple`: A tuple containing `sigma_down` and `sigma_up`:
        - `sigma_down` (float): The computed value of sigma for the downward step.
        - `sigma_up` (float): The computed value of sigma for the upward step.
    """
    sigma_up = min(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x):
    """
    #### Returns a noise sampling function that generates random noise with the same shape as the input tensor `x`.

    #### Args:
        - `x` (torch.Tensor): The input tensor whose shape will be used to generate random noise.

    #### Returns:
        - `function`: A function that takes two arguments, `sigma` and `sigma_next`, and returns a tensor of random noise
                  with the same shape as `x`.
    """
    return lambda sigma, sigma_next: torch.randn_like(x)


class BatchedBrownianTree:
    """#### A class to represent a batched Brownian tree for stochastic differential equations.

    #### Attributes:
        - `cpu_tree` : bool
            Indicates if the tree is on CPU.
        - `sign` : int
            Sign indicating the order of t0 and t1.
        - `batched` : bool
            Indicates if the tree is batched.
        - `trees` : list
            List of BrownianTree instances.

    #### Methods:
        - `__init__(x, t0, t1, seed=None, **kwargs)`:
            Initializes the BatchedBrownianTree with given parameters.
        - `sort(a, b)`:
            Static method to sort two values and return them along with a sign.
        - `__call__(t0, t1)`:
            Calls the Brownian tree with given time points t0 and t1.
    """

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        self.cpu_tree = True
        if "cpu" in kwargs:
            self.cpu_tree = kwargs.pop("cpu")
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get("w0", torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, []).item()
        self.batched = True
        seed = [seed]
        self.batched = False
        self.trees = [
            torchsde.BrownianTree(t0.cpu(), w0.cpu(), t1.cpu(), entropy=s, **kwargs)
            for s in seed
        ]

    @staticmethod
    def sort(a, b):
        """#### Sort two values and return them along with a sign.

        #### Args:
            - `a` (float): The first value.
            - `b` (float): The second value.

        #### Returns:
            - `tuple`: A tuple containing the sorted values and a sign:
        """
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        """#### Call the Brownian tree with given time points t0 and t1.

        #### Args:
            - `t0` (torch.Tensor): The starting time point.
            - `t1` (torch.Tensor): The target time point.

        #### Returns:
            - `torch.Tensor`: The Brownian tree values.
        """
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack(
            [
                tree(t0.cpu().float(), t1.cpu().float()).to(t0.dtype).to(t0.device)
                for tree in self.trees
            ]
        ) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """#### A class to sample noise using a Brownian tree approach.

    #### Attributes:
        - `transform` (callable): A function to transform the sigma values.
        - `tree` (BatchedBrownianTree): An instance of the BatchedBrownianTree class.

    #### Methods:
        - `__init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False)`:
            Initializes the BrownianTreeNoiseSampler with the given parameters.
        - `__call__(self, sigma, sigma_next)`:
            Samples noise between the given sigma values.
    """

    def __init__(
        self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x, cpu=False
    ):
        """#### Initializes the BrownianTreeNoiseSampler with the given parameters.

        #### Args:
            - `x` (Tensor): The initial tensor.
            - `sigma_min` (float): The minimum sigma value.
            - `sigma_max` (float): The maximum sigma value.
            - `seed` (int, optional): The seed for random number generation. Defaults to None.
            - `transform` (callable, optional): A function to transform the sigma values. Defaults to identity function.
            - `cpu` (bool, optional): Whether to use CPU for computations. Defaults to False.
        """
        self.transform = transform
        t0, t1 = (
            self.transform(torch.as_tensor(sigma_min)),
            self.transform(torch.as_tensor(sigma_max)),
        )
        self.tree = BatchedBrownianTree(x, t0, t1, seed, cpu=cpu)

    def __call__(self, sigma, sigma_next):
        """#### Samples noise between the given sigma values.

        #### Args:
            - `sigma` (float): The current sigma value.
            - `sigma_next` (float): The next sigma value.

        #### Returns:
            - `Tensor`: The sampled noise.
        """
        t0, t1 = (
            self.transform(torch.as_tensor(sigma)),
            self.transform(torch.as_tensor(sigma_next)),
        )
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


class PIDStepSizeController:
    """#### A PID (Proportional-Integral-Derivative) Step Size Controller for adaptive step size selection.

    #### Attributes:
        - `h` (float): Initial step size.
        - `b1` (float): Coefficient for the proportional term.
        - `b2` (float): Coefficient for the integral term.
        - `b3` (float): Coefficient for the derivative term.
        - `accept_safety` (float): Safety factor for accepting a proposed step size.
        - `eps` (float): Small value to prevent division by zero.
        - `errs` (list): List to store inverse errors for PID control.

    #### Methods:
        - `__init__(self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8)`:
            Initializes the PIDStepSizeController with given parameters.
        - `limiter(self, x)`:
            Limits the factor to prevent excessive changes in step size.
        - `propose_step(self, error)`:
            Proposes a new step size based on the given error and updates internal state.
    """

    def __init__(
        self, h, pcoeff, icoeff, dcoeff, order=1, accept_safety=0.81, eps=1e-8
    ):
        self.h = h
        self.b1 = (pcoeff + icoeff + dcoeff) / order
        self.b2 = -(pcoeff + 2 * dcoeff) / order
        self.b3 = dcoeff / order
        self.accept_safety = accept_safety
        self.eps = eps
        self.errs = []

    def limiter(self, x):
        """#### Limit the factor to prevent excessive changes in step size.

        #### Args:
            - `x` (float): The factor to limit.

        #### Returns:
            - `float`: The limited factor.
        """
        return 1 + math.atan(x - 1)

    def propose_step(self, error):
        """#### Propose a new step size based on the given error and update the internal state.

        #### Args:
            - `error` (float): The error value.

        #### Returns:
            - `bool`: True if the proposed step size is accepted, False otherwise.
        """
        inv_error = 1 / (float(error) + self.eps)
        if not self.errs:
            self.errs = [inv_error, inv_error, inv_error]
        self.errs[0] = inv_error
        factor = (
            self.errs[0] ** self.b1 * self.errs[1] ** self.b2 * self.errs[2] ** self.b3
        )
        factor = self.limiter(factor)
        accept = factor >= self.accept_safety
        if accept:
            self.errs[2] = self.errs[1]
            self.errs[1] = self.errs[0]
        self.h *= factor
        return accept


class DPMSolver(nn.Module):
    """#### DPMSolver is a class for solving differential equations using the DPM-Solver algorithm.

    #### Args:
        - `model` (nn.Module): The model to be used for solving the differential equations.
        - `extra_args` (dict, optional): Additional arguments to be passed to the model. Defaults to None.
        - `eps_callback` (callable, optional): A callback function to be called after computing epsilon. Defaults to None.
        - `info_callback` (callable, optional): A callback function to be called with information about the solver's progress. Defaults to None.

    #### Methods:
        - `t(sigma)`:
            Converts sigma to time t.
        - `sigma(t)`:
            Converts time t to sigma.
        - `eps(eps_cache, key, x, t, *args, **kwargs)`:
            Computes the epsilon value for the given inputs and caches the result.
        - `dpm_solver_2_step(x, t, t_next, r1=1/2, eps_cache=None)`:
            Performs a 2-step DPM-Solver update.
        - `dpm_solver_3_step(x, t, t_next, r1=1/3, r2=2/3, eps_cache=None)`:
            Performs a 3-step DPM-Solver update.
        - `dpm_solver_adaptive(x, t_start, t_end, order=3, rtol=0.05, atol=0.0078, h_init=0.05, pcoeff=0.0, icoeff=1.0, dcoeff=0.0, accept_safety=0.81, eta=0.0, s_noise=1.0, noise_sampler=None)`:
            Performs an adaptive DPM-Solver update with error control and step size adaptation.
    """

    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        super().__init__()
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma):
        """#### Convert sigma to time t.

        #### Args:
            - `sigma` (torch.Tensor): The sigma value.

        #### Returns:
            - `torch.Tensor`: The time t.
        """
        return -sigma.log()

    def sigma(self, t):
        """#### Convert time t to sigma.

        #### Args:
            - `t` (torch.Tensor): The time t.

        #### Returns:
            - `torch.Tensor`: The sigma value.
        """
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        """#### Compute the epsilon value for the given inputs and cache the result.

        #### Args:
            - `eps_cache` (dict): The cache for epsilon values.
            - `key` (str): The key for the cache.
            - `x` (torch.Tensor): The input tensor.
            - `t` (torch.Tensor): The time t.

        #### Returns:
            - `tuple`: A tuple containing the epsilon value and the updated cache.
        """
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (
            x - self.model(x, sigma, *args, **self.extra_args, **kwargs)
        ) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        """#### Perform a 2-step DPM-Solver update.

        #### Args:
            -`x` (torch.Tensor): The input tensor.
            -`t` (torch.Tensor): The current time t.
            -`t_next` (torch.Tensor): The target time t.
            -`r1` (float, optional): The ratio for the first step. Defaults to 1/2.
            -`eps_cache` (dict, optional): The cache for epsilon values. Defaults to None.

        #### Returns:
            - `tuple`: A tuple containing the updated tensor and the updated cache.
        """
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, "eps_r1", u1, s1)
        x_2 = (
            x
            - self.sigma(t_next) * h.expm1() * eps
            - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps)
        )
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        """#### Perform a 3-step DPM-Solver update.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `t` (torch.Tensor): The current time t.
            - `t_next` (torch.Tensor): The target time t.
            - `r1` (float, optional): The ratio for the first step. Defaults to 1/3.
            - `r2` (float, optional): The ratio for the second step. Defaults to 2/3.
            - `eps_cache` (dict, optional): The cache for epsilon values. Defaults to None.

        #### Returns:
            - `tuple`: A tuple containing the updated tensor and the updated cache.
        """
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, "eps_r1", u1, s1)
        u2 = (
            x
            - self.sigma(s2) * (r2 * h).expm1() * eps
            - self.sigma(s2)
            * (r2 / r1)
            * ((r2 * h).expm1() / (r2 * h) - 1)
            * (eps_r1 - eps)
        )
        eps_r2, eps_cache = self.eps(eps_cache, "eps_r2", u2, s2)
        x_3 = (
            x
            - self.sigma(t_next) * h.expm1() * eps
            - self.sigma(t_next) / r2 * (h.expm1() / h - 1) * (eps_r2 - eps)
        )
        return x_3, eps_cache

    def dpm_solver_adaptive(
        self,
        x,
        t_start,
        t_end,
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
        pipeline=False,
    ):
        """#### Perform an adaptive DPM-Solver update with error control and step size adaptation.

        #### Args:
            - `x` (torch.Tensor): The input tensor.
            - `t_start` (torch.Tensor): The starting time t.
            - `t_end` (torch.Tensor): The target time t.
            - `order` (int, optional): The order of the DPM-Solver. Defaults to 3.
            - `rtol` (float, optional): The relative tolerance for error control. Defaults to 0.05.
            - `atol` (float, optional): The absolute tolerance for error control. Defaults to 0.0078.
            - `h_init` (float, optional): The initial step size. Defaults to 0.05.
            - `pcoeff` (float, optional): Coefficient for the proportional term in the PID controller. Defaults to 0.0.
            - `icoeff` (float, optional): Coefficient for the integral term in the PID controller. Defaults to 1.0.
            - `dcoeff` (float, optional): Coefficient for the derivative term in the PID controller. Defaults to 0.0.
            - `accept_safety` (float, optional): Safety factor for accepting a proposed step size. Defaults to 0.81.
            - `eta` (float, optional): The eta parameter for the ancestral step. Defaults to 0.0.
            - `s_noise` (float, optional): The noise scaling factor. Defaults to 1.0.
            - `noise_sampler` (callable, optional): A function to sample noise. Defaults to None.

        #### Returns:
            - `tuple`: A tuple containing the updated tensor and information about the solver's progress.
        """
        global disable_gui
        disable_gui = True if pipeline is True else False
        if disable_gui is False:
            from modules.AutoEncoders import taesd
            from modules.user import app_instance
        noise_sampler = (
            default_noise_sampler(x) if noise_sampler is None else noise_sampler
        )
        forward = t_end > t_start
        h_init = abs(h_init) * (1 if forward else -1)
        atol = torch.tensor(atol)
        rtol = torch.tensor(rtol)
        s = t_start
        x_prev = x
        accept = True
        pid = PIDStepSizeController(
            h_init, pcoeff, icoeff, dcoeff, 1.5 if eta else order, accept_safety
        )
        info = {"steps": 0, "nfe": 0, "n_accept": 0, "n_reject": 0}
        while s < t_end - 1e-5 if forward else s > t_end + 1e-5:
            if not pipeline and hasattr(app_instance.app, 'interrupt_flag') and app_instance.app.interrupt_flag is True:
                return x
            if disable_gui is False:
                try:
                    app_instance.app.title(f"LightDiffusion - {info['steps']*3}it")
                    app_instance.app.progress.set((info["steps"]*3)/100)
                except:
                    pass
            eps_cache = {}
            t = (
                torch.minimum(t_end, s + pid.h)
                if forward
                else torch.maximum(t_end, s + pid.h)
            )
            t_, su = t, 0.0
            eps, eps_cache = self.eps(eps_cache, "eps", x, s)
            x - self.sigma(s) * eps
            x_low, eps_cache = self.dpm_solver_2_step(
                x, s, t_, r1=1 / 3, eps_cache=eps_cache
            )
            x_high, eps_cache = self.dpm_solver_3_step(x, s, t_, eps_cache=eps_cache)
            delta = torch.maximum(atol, rtol * torch.maximum(x_low.abs(), x_prev.abs()))
            error = torch.linalg.norm((x_low - x_high) / delta) / x.numel() ** 0.5
            accept = pid.propose_step(error)
            if accept:
                x_prev = x_low
                x = x_high + su * s_noise * noise_sampler(self.sigma(s), self.sigma(t))
                s = t
                info["n_accept"] += 1
            else:
                info["n_reject"] += 1
            info["nfe"] += order
            info["steps"] += 1
            if disable_gui is False:
                if app_instance.app.previewer_var.get() is True:
                    threading.Thread(target=taesd.taesd_preview, args=(x,)).start()
                else:
                    pass
        if disable_gui is False:
            try:
                app_instance.app.title("LightDiffusion")
            except:
                pass
        return x, info

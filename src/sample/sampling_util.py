import logging
import math
import threading
import torch
import torchsde
from torch import nn

from src.Utilities import util


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

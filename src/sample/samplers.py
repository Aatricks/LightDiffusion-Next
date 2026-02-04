"""Functional interface for samplers."""
import torch
from src.sample.BaseSampler import DPMPPSDESampler, EulerSampler, EulerAncestralSampler, DPMPP2MSampler

def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, pipeline=False, **kwargs):
    sampler = DPMPPSDESampler(pipeline=pipeline, **kwargs)
    return sampler.sample(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)

def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, pipeline=False, **kwargs):
    sampler = EulerSampler(pipeline=pipeline, **kwargs)
    return sampler.sample(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)

def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, pipeline=False, **kwargs):
    sampler = EulerAncestralSampler(pipeline=pipeline, **kwargs)
    return sampler.sample(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)

def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None, pipeline=False, **kwargs):
    sampler = DPMPP2MSampler(pipeline=pipeline, **kwargs)
    return sampler.sample(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)

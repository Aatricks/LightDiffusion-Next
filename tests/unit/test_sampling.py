import torch
from unittest.mock import patch

from src.sample import sampling


class DummyModel:
    def get_model_object(self, name):
        class MS:
            def __init__(self):
                import torch
                # minimal sigmas to avoid heavy computation
                self.sigmas = torch.linspace(1, 0, 20)
                self.sigma_min = 0.0
                self.sigma_max = 1.0
            def timestep(self, sigma):
                return sigma
        return MS()


def test_ksampler_sets_scheduler_before_set_steps(monkeypatch):
    ks = sampling.KSampler()
    captured = {}

    def fake_set_steps(steps, denoise=None):
        # capture scheduler value visible inside set_steps
        captured['scheduler'] = ks.scheduler

    # Patch ks.set_steps to avoid running heavy logic and to capture scheduler
    monkeypatch.setattr(ks, 'set_steps', fake_set_steps)

    # Patch common_ksampler to no-op to prevent further sampling behavior
    monkeypatch.setattr(sampling, 'common_ksampler', lambda *args, **kwargs: {})

    # Call sample with scheduler passed explicitly
    ks.sample(model=DummyModel(), steps=20, scheduler='simple', sampler_name='euler')

    assert captured.get('scheduler') == 'simple', "KSampler.sample must set the provided scheduler before calling set_steps"

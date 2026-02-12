"""Pytest-style tests for ays_scheduler (converted from unittest).

Moved from `tests/test_ays_sigmas.py` and converted to pytest parametrize.
"""

import pytest
from src.sample.ays_scheduler import ays_scheduler


@pytest.mark.parametrize("steps,model,expected_len", [
    (10, "SD15", 11),
    (13, "SD15", 14),
    (2,  "SD15", 3),
    (30, "SD15", 31),
    (20, "SDXL", 21),
])
def test_ays_sigmas_step_counts(steps, model, expected_len):
    sigmas = ays_scheduler(None, steps, model)
    assert len(sigmas) == expected_len

import torch
import unittest
from src.sample.ays_scheduler import ays_scheduler

class TestAYSSigmas(unittest.TestCase):
    def test_ays_step_counts(self):
        # Test exact match (SD15 10 steps)
        sigmas = ays_scheduler(None, 10, "SD15")
        self.assertEqual(len(sigmas), 11, "Should return 11 sigmas for 10 steps (exact match)")
        
        # Test interpolation (SD15 12 steps) - 12 is in schedules too, let's try 13
        sigmas = ays_scheduler(None, 13, "SD15")
        self.assertEqual(len(sigmas), 14, "Should return 14 sigmas for 13 steps (interpolated)")
        
        # Test small step count (underflow)
        sigmas = ays_scheduler(None, 2, "SD15")
        self.assertEqual(len(sigmas), 3, "Should return 3 sigmas for 2 steps (resampled from 4)")
        
        # Test large step count (overflow)
        sigmas = ays_scheduler(None, 30, "SD15")
        self.assertEqual(len(sigmas), 31, "Should return 31 sigmas for 30 steps (resampled from 25)")
        
        # Test SDXL
        sigmas = ays_scheduler(None, 20, "SDXL")
        self.assertEqual(len(sigmas), 21, "Should return 21 sigmas for 20 steps (SDXL)")

if __name__ == "__main__":
    unittest.main()

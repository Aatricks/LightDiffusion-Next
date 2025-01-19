import unittest
import os
from modules.user.pipeline import pipeline

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.test_prompt = "a cute cat, high quality, detailed"
        self.test_img_path = "../_internal/Flux_00001.png"  # Make sure this test image exists
        
    def test_basic_generation_small(self):
        pipeline(self.test_prompt, 128, 128, number=1)
        # Check if output files exist
        
    def test_basic_generation_medium(self):
        pipeline(self.test_prompt, 512, 512, number=1)
        
    def test_basic_generation_large(self):
        pipeline(self.test_prompt, 1024, 1024, number=1)
        
    def test_hires_fix(self):
        pipeline(self.test_prompt, 512, 512, number=1, hires_fix=True)
        
    def test_adetailer(self):
        pipeline(
            "a portrait of a person, high quality", 
            512, 
            512, 
            number=1, 
            adetailer=True
        )
        
    def test_enhance_prompt(self):
        pipeline(
            self.test_prompt, 
            512, 
            512, 
            number=1, 
            enhance_prompt=True
        )
        
    def test_img2img(self):
        # Skip if test image doesn't exist
        if not os.path.exists(self.test_img_path):
            self.skipTest("Test image not found")
            
        pipeline(
            self.test_img_path,
            512,
            512,
            number=1,
            img2img=True
        )
        
    def test_stable_fast(self):
        resolutions = [(128, 128), (512, 512), (1024, 1024)]
        for w, h in resolutions:
            pipeline(
                self.test_prompt,
                w,
                h,
                number=1,
                stable_fast=True
            )
            
    def test_reuse_seed(self):
        pipeline(
            self.test_prompt,
            512,
            512,
            number=2,
            reuse_seed=True
        )
        
    def test_flux_mode(self):
        resolutions = [(128, 128), (512, 512), (1024, 1024)]
        for w, h in resolutions:
            pipeline(
                self.test_prompt,
                w,
                h,
                number=1,
                flux_enabled=True
            )

if __name__ == '__main__':
    unittest.main()
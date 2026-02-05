
import unittest
from PIL import Image
import io
import base64
import time
import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.user.app_instance import AppInstance

class TestPreviewPipeline(unittest.TestCase):
    def setUp(self):
        self.app = AppInstance()

    def test_update_image_lazy(self):
        """Test that update_image stores images without converting to base64 immediately."""
        img1 = Image.new('RGB', (64, 64), color='red')
        img2 = Image.new('RGB', (64, 64), color='blue')
        
        self.app.update_image([img1, img2], step=5, total_steps=20)
        
        # Verify internal state
        self.assertEqual(len(self.app.preview_images), 2)
        self.assertEqual(self.app.preview_base64_cache, [])
        self.assertEqual(self.app.current_step, 5)
        self.assertEqual(self.app.total_steps, 20)
        self.assertGreater(self.app.last_preview_time, 0)

    def test_metadata_check(self):
        """Test lightweight metadata check."""
        img1 = Image.new('RGB', (64, 64), color='red')
        self.app.update_image([img1], step=2, total_steps=10)
        
        meta = self.app.get_preview_metadata()
        self.assertEqual(meta["step"], 2)
        self.assertEqual(meta["total_steps"], 10)
        self.assertTrue(meta["has_images"])
        self.assertGreater(meta["timestamp"], 0)

    def test_lazy_base64_conversion(self):
        """Test that base64 conversion happens on demand and is cached."""
        img1 = Image.new('RGB', (32, 32), color='green')
        self.app.update_image([img1], step=1, total_steps=10)
        
        # First call should trigger conversion
        previews = self.app.get_latest_previews()
        self.assertEqual(len(previews["base64"]), 1)
        self.assertTrue(previews["base64"][0].startswith("data:image/webp;base64,"))
        
        # Cache should now be populated
        cache_before = self.app.preview_base64_cache
        self.assertGreater(len(cache_before), 0)
        
        # Second call should use cache
        previews2 = self.app.get_latest_previews()
        self.assertIs(previews2["base64"], self.app.preview_base64_cache)
        
        # Invalidate cache with new update
        self.app.update_image([img1], step=2, total_steps=10)
        self.assertEqual(self.app.preview_base64_cache, [])

if __name__ == "__main__":
    unittest.main()

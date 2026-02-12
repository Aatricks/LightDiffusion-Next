import importlib

import pytest


def test_image_saver_respects_env_var(monkeypatch):
    # Set custom env var and reload the module to pick it up
    monkeypatch.setenv("LD_MAX_IMAGES_PER_SAVE", "1024")
    import src.FileManaging.ImageSaver as ImageSaver

    importlib.reload(ImageSaver)
    assert ImageSaver.MAX_IMAGES_PER_SAVE == 1024

    # Clean up: remove env var and reload to restore default
    monkeypatch.delenv("LD_MAX_IMAGES_PER_SAVE", raising=False)
    importlib.reload(ImageSaver)
    assert ImageSaver.MAX_IMAGES_PER_SAVE == 16

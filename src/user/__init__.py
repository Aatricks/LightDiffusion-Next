"""User-facing runtime modules for LightDiffusion-Next.

The package intentionally avoids importing submodules at import time so callers
can load `src.user.app_instance` without also triggering pipeline setup.
"""

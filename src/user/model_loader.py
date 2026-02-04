"""Legacy model loader for backward compatibility with tests."""
from typing import Any, Tuple

def load_model_for_pipeline(model_path: str = None, **kwargs) -> Tuple[str, Tuple[Any, Any, Any]]:
    """
    Legacy function used by tests to mock model loading.
    In real usage, the Pipeline class handles model loading internally.
    """
    # This is just a stub for tests to patch
    return ("SD15", (None, None, None))

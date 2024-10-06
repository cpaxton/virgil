# # (c) 2024 by Chris Paxton

from .gemma import Gemma
from .base import Backend


def get_backend(name: str) -> Backend:
    """Get a backend by name.

    Args:
        name (str): The name of the backend.

    Returns:
        Backend: The backend instance.
    """
    if name == "gemma":
        return Gemma()
    else:
        raise ValueError(f"Unknown backend: {name}")

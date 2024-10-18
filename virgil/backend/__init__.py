# # (c) 2024 by Chris Paxton

from .gemma import Gemma
from .base import Backend
from .llama import Llama


backend_list = ["gemma", "gemma2b", "gemma-2b-it",
                "llama-3.2-1B"]


def get_backend(name: str) -> Backend:
    """Get a backend by name.

    Args:
        name (str): The name of the backend.

    Returns:
        Backend: The backend instance.
    """
    if name == "gemma" or name == "gemma2b" or name == "gemma-2b-it":
        return Gemma()
    elif name == "llama" or name == "llama-3.2-1B":
        return Llama(model_name="meta-llama/Llama-3.2-1B")
    else:
        raise ValueError(f"Unknown backend: {name}")

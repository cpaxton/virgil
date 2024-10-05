from .gemma import Gemma


def get_backend(name: str):
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

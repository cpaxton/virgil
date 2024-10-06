# # (c) 2024 by Chris Paxton

from .base import ImageGenerator
from .diffuser import DiffuserImageGenerator

__all__ = ["ImageGenerator", "DiffuserImageGenerator"]


def create_image_generator(generator: str, **kwargs) -> ImageGenerator:
    """
    Create an image generator based on the given generator name and parameters.

    Args:
        generator (str): The name of the image generator to create.
        **kwargs: Additional keyword arguments to pass to the image generator constructor.

    Returns:
        ImageGenerator: The created image generator.
    """
    if generator == "diffuser":
        return DiffuserImageGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown image generator: {generator}")

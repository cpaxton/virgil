# # (c) 2024 by Chris Paxton

from abc import ABC, abstractmethod
from PIL import Image


class ImageGenerator(ABC):
    """
    An abstract base class for image generators.
    """

    def __init__(self, height: int = 512, width: int = 512):
        """
        Initialize the AbstractImageGenerator with basic image parameters.

        Args:
            height (int): The height of the generated image. Defaults to 512.
            width (int): The width of the generated image. Defaults to 512.
        """
        self.height = height
        self.width = width

    @abstractmethod
    def generate(self, prompt: str) -> Image.Image:
        """
        Generate an image based on the given text prompt.

        Args:
            prompt (str): The text description of the image to generate.

        Returns:
            Image.Image: The generated image.
        """
        pass

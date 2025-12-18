# Copyright 2024 Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# (c) 2024 by Chris Paxton

from abc import ABC, abstractmethod
from PIL import Image
from virgil.image.base import ImageGenerator


class ImageService(ABC):
    """
    Abstract service for image generation operations.
    This abstraction allows the image generation functionality to be used
    by various agents (Discord bots, MCP servers, etc.) without tight coupling.
    """

    @abstractmethod
    def generate_image(self, prompt: str) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt (str): The text description of the image to generate.

        Returns:
            Image.Image: The generated PIL Image.
        """
        pass


class VirgilImageService(ImageService):
    """
    Implementation of ImageService using Virgil's ImageGenerator classes.
    """

    def __init__(self, image_generator: ImageGenerator):
        """
        Initialize the service with an image generator.

        Args:
            image_generator (ImageGenerator): The image generator to use.
        """
        self.image_generator = image_generator

    def generate_image(self, prompt: str) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt (str): The text description of the image to generate.

        Returns:
            Image.Image: The generated PIL Image.
        """
        return self.image_generator.generate(prompt)

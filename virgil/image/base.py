# # Copyright 2024 Chris Paxton
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

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

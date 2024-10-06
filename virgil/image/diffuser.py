# # (c) 2024 by Chris Paxton

from PIL import Image
import torch
from diffusers import AutoPipelineForText2Image
from .base import ImageGenerator


class DiffuserImageGenerator(ImageGenerator):
    """
    A class for generating images from text prompts using the Diffusers library.
    """

    def __init__(self, height: int = 512, width: int = 512, num_inference_steps: int = 20):
        """
        Initialize the DiffuserImageGenerator with a pre-trained model and image generation parameters.

        Args:
            height (int): The height of the generated image. Defaults to 512.
            width (int): The width of the generated image. Defaults to 512.
            num_inference_steps (int): The number of denoising steps. Defaults to 20.
        """
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps

        self.pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16").to("cuda")

    def generate(self, prompt: str) -> Image.Image:
        """
        Generate an image based on the given text prompt.

        Args:
            prompt (str): The text description of the image to generate.

        Returns:
            Image.Image: The generated image.
        """
        result = self.pipeline(prompt=prompt, height=self.height, width=self.width, num_inference_steps=self.num_inference_steps)
        return result.images[0]

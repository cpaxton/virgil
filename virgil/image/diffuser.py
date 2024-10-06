# (c) 2024 by Chris Paxton

from PIL import Image
import torch
from diffusers import AutoPipelineForText2Image
from virgil.image.base import ImageGenerator
from virgil.image.siglip import SigLIPAligner


class DiffuserImageGenerator(ImageGenerator):
    """
    A class for generating images from text prompts using the Diffusers library.
    """

    def __init__(self, height: int = 512, width: int = 512, num_inference_steps: int = 50, guidance_scale: float = 7.5) -> None:
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
        self.guidance_scale = guidance_scale

        self.pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16").to("cuda")

    def generate(self, prompt: str) -> Image.Image:
        """
        Generate an image based on the given text prompt.

        Args:
            prompt (str): The text description of the image to generate.

        Returns:
            Image.Image: The generated image.
        """
        result = self.pipeline(prompt=prompt, height=self.height, width=self.width, num_inference_steps=self.num_inference_steps, return_images=True)
        return result.images[0]


if __name__ == "__main__":
    prompt = "A blobfish, its gelatinous body slumped and discolored, resting on a bed of seaweed in a dark, deep-sea environment. The blobfish''s face is a pale, almost translucent, and its eyes are wide and vacant. The background is a dark, inky blue, with faint bioluminescent creatures swimming in the distance."
    generator = DiffuserImageGenerator()
    image = generator.generate(prompt)
    image.save("blobfish.png")

    # prompt = "A sea turtle wearing a scuba diving suit, working as a marine biologist."
    # image = generator.generate(prompt)
    # image.save("sea_turtle.png")

    # prompt = "Picture of A sea turtle wearing a scuba diving suit, working as a marine biologist."
    # image = generator.generate(prompt)
    # image.save("sea_turtle2.png")

    aligner = SigLIPAligner()

    # Blobfish tests
    prompt = "A blobfish, its gelatinous body slumped and discolored, resting on a bed of seaweed in a dark, deep-sea environment. The blobfish''s face is a pale, almost translucent, and its eyes are wide and vacant. The background is a dark, inky blue, with faint bioluminescent creatures swimming in the distance."
    aligner.check_alignment("blobfish.png", prompt)
    score, image = aligner.search(generator, prompt, num_tries=25)
    print(score)
    image.save("blobfish_search.png")

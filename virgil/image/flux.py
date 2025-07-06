# Copyright (c) 2025 Chris Paxton

import click
import torch
from diffusers import FluxPipeline
from PIL import Image
from transformers import BitsAndBytesConfig

from virgil.image.base import ImageGenerator


class FluxImageGenerator(ImageGenerator):
    def __init__(
        self, height: int = 1536, width: int = 1536, quantization: str = "int4"
    ) -> None:
        """Create a new FluxImageGenerator. This class generates images using the FLUX model.

        Args:
            height (int): The height of the generated image. Defaults to 1536.
            width (int): The width of the generated image. Defaults to 1536.
            quantization (str): The quantization method to use. Options: "int4", "int8", or None. Defaults to "int4".

        Raises:
            ValueError: If an unknown quantization method is provided.
        """
        super().__init__(height, width)

        # Configure quantization
        if quantization == "int4":
            # Configure 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quantization == "int8":
            # Configure 8-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16
            )
        elif quantization is None:
            quantization_config = None
            torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown quantization method: {quantization}")

        # Load the model with the quantization configuration
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch_dtype
            if "torch_dtype" in locals()
            else torch.bfloat16,  # Set torch_dtype based on quantization
            quantization_config=quantization_config,
            use_safetensors=True,
        )
        self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, prompt: str) -> Image.Image:
        with torch.inference_mode():
            image = self.pipe(
                prompt=prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=12,  # Reduced steps with new rectified flow
                guidance_scale=5.0,  # Optimal for v1.2
                output_type="pil",
            ).images[0]
        return image


@click.command()
@click.option("--height", default=1536, help="Height of the generated image.")
@click.option("--width", default=1536, help="Width of the generated image.")
@click.option("--quantization", default="int4", help="Quantization method (int4, int8, or None).")
@click.option("--prompt", default="", help="Prompt for image generation.")
@click.option("--output", default="generated_image.png", help="Output filename for the generated image.")
def main(height: int = 1536, width: int = 1536, quantization: str = "int4", prompt: str = "", output: str = "generated_image.png") -> None:
    """Main function to generate an image using the FluxImageGenerator."""
    generator = FluxImageGenerator(height=height, width=width, quantization=quantization)
    if len(prompt) == 0:
        prompt = "A beautiful sunset over a calm sea, with vibrant colors reflecting on the water."
    image = generator.generate(prompt)
    image.save(output)

if __name__ == "__main__":
    main()

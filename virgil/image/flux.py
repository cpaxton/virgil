# Copyright (c) 2025 Chris Paxton

import click
import torch
from diffusers import FluxPipeline
from PIL import Image
from transformers import BitsAndBytesConfig

from virgil.image.base import ImageGenerator


class FluxImageGenerator(ImageGenerator):
    def __init__(
        self,
        height: int = 1536,
        width: int = 1536,
        quantization: str = "int4"
    ) -> None:
        """Initialize the FLUX model image generator.

        Args:
            height: Height of generated images (default: 1536)
            width: Width of generated images (default: 1536)
            quantization: Quantization method ("int4", "int8", or None)

        Raises:
            ValueError: For unknown quantization methods
            RuntimeError: If no GPU is available when required
        """
        super().__init__(height, width)
        self.device = self._get_device()
        torch_dtype = torch.bfloat16
        quantization_config = None

        # Configure quantization
        if quantization == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        elif quantization == "int8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch_dtype,
            )
        elif quantization is not None:
            raise ValueError(f"Unsupported quantization: '{quantization}'. "
                             "Use 'int4', 'int8', or None.")

        # Load model pipeline
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            use_safetensors=True,
        ).to(self.device)
        
        self.pipe.set_progress_bar_config(disable=True)

    def _get_device(self) -> str:
        """Validate and return available compute device."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device not available. GPU required for image generation.")
        return "cuda"

    def generate(self, prompt: str) -> Image.Image:
        """Generate image from text prompt.
        
        Args:
            prompt: Text prompt to visualize
            
        Returns:
            Generated PIL image
        """
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=12,
                guidance_scale=5.0,
                output_type="pil",
            )
        return result.images[0]


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

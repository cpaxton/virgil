# Copyright (c) 2025 Chris Paxton

import click
import torch
from diffusers import FluxPipeline
from PIL import Image
from transformers import BitsAndBytesConfig


from transformers import (
    T5EncoderModel,
    BitsAndBytesConfig as TransformersBitsAndBytesConfig,
)
from diffusers.models import FluxTransformer2DModel
import gc


class FluxImageGenerator:
    def __init__(
        self,
        height: int = 1536,
        width: int = 1536,
        quantization: str = "int4",
        cpu_offload: bool = True,
        inference_steps: int = 50,
        guidance_scale: float = 3.5,
    ) -> None:
        self.height = height
        self.width = width
        self.device = self._get_device()

        # Used for generation
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale

        if quantization == "int4":
            # Configure 4-bit quantization (NF4 + nested quantization)
            diffusers_quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",  # Optimal for normal-distributed weights
                bnb_4bit_compute_dtype=torch.float16,  # Faster computation
                bnb_4bit_use_double_quant=True,  # Nested quantization for extra savings :cite[1]
            )

            transformers_quant_config = TransformersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif quantization == "int8":
            # Configure 8-bit quantization
            diffusers_quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,  # Nested quantization for extra savings
                bnb_8bit_compute_dtype=torch.float16,
            )
            transformers_quant_config = TransformersBitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.float16,
            )
        else:
            # No quantization
            diffusers_quant_config = None
            transformers_quant_config = None

        # Load quantized components
        text_encoder = T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="text_encoder_2",
            quantization_config=transformers_quant_config,
            torch_dtype=torch.float16,
        )

        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            quantization_config=diffusers_quant_config,
            torch_dtype=torch.float16,
        )

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            text_encoder_2=text_encoder,
            torch_dtype=torch.float16,
        )

        self.cpu_offload = cpu_offload
        if self.cpu_offload:
            # Enable CPU offloading for memory efficiency
            self.pipe.enable_model_cpu_offload()
        else:
            # Only move to device if not quantized (quantized models load on GPU by default)
            if diffusers_quant_config is None:
                self.pipe = self.pipe.to(self.device)

        self.pipe.set_progress_bar_config(disable=True)

    def _get_device(self) -> str:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device not available. GPU required for image generation."
            )
        return "cuda"

    def generate(self, prompt: str) -> Image.Image:
        # Garbage collection and empty cache to free up memory
        gc.collect()
        torch.cuda.empty_cache()
        # Generate the image using the Flux pipeline
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=self.inference_steps,
                guidance_scale=self.guidance_scale,
                output_type="pil",
            )
        return result.images[0]


@click.command()
@click.option("--height", default=512, help="Height of the generated image.")
@click.option("--width", default=512, help="Width of the generated image.")
@click.option(
    "--quantization", default="int4", help="Quantization method (int4, int8, or None)."
)
@click.option("--prompt", default="", help="Prompt for image generation.")
@click.option(
    "--output",
    default="generated_image.png",
    help="Output filename for the generated image.",
)
def main(
    height: int = 1536,
    width: int = 1536,
    quantization: str = "int4",
    prompt: str = "",
    output: str = "generated_image.png",
) -> None:
    """Main function to generate an image using the FluxImageGenerator."""
    generator = FluxImageGenerator(
        height=height, width=width, quantization=quantization
    )
    if len(prompt) == 0:
        prompt = "A beautiful sunset over a calm sea, with vibrant colors reflecting on the water."
    print("Generating image with prompt:", prompt)
    image = generator.generate(prompt)
    print("Saving generated image to:", output)
    image.save(output)


if __name__ == "__main__":
    main()

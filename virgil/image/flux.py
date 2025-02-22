
import torch
from diffusers import FluxProPipeline
from PIL import Image
from abc import ABC, abstractmethod

from virgil.image.base import ImageGenerator

class FluxImageGenerator(ImageGenerator):
    def __init__(self, height: int = 1536, width: int = 1536):
        super().__init__(height, width)
        self.pipe = FluxProPipeline.from_pretrained(
            "black-forest-labs/FLUX.1.2-pro-ultra",
            torch_dtype=torch.bfloat16,
            variant="fp16",
            use_safetensors=True
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
                guidance_scale=5.0,      # Optimal for v1.2
                output_type="pil"
            ).images[0]
        return image


if __name__ == "__main__":

    generator = FluxImageGenerator()
    blobfish = True

    if blobfish:
        prompt = "A blobfish, its gelatinous body slumped and discolored, resting on a bed of seaweed in a dark, deep-sea environment. The blobfish''s face is a pale, almost translucent, and its eyes are wide and vacant. The background is a dark, inky blue, with faint bioluminescent creatures swimming in the distance."
        image = generator.generate(prompt)
        image.save("blobfish.png")


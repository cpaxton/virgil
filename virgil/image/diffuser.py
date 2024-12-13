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

# (c) 2024 by Chris Paxton

from PIL import Image
import torch
from diffusers import AutoPipelineForText2Image
from virgil.image.base import ImageGenerator
from virgil.image.siglip import SigLIPAligner


from diffusers import AutoPipelineForText2Image
import torch
from PIL import Image


class DiffuserImageGenerator(ImageGenerator):
    """
    A class for generating images from text prompts using the Diffusers library,
    with support for various small and efficient models.
    """

    MODEL_OPTIONS = {
        "base": "stabilityai/stable-diffusion-xl-base-1.0",
        "turbo": "stabilityai/sdxl-turbo",
        "turbo_3.5": "stabilityai/stable-diffusion-3.5-large-turbo",
        "small": "segmind/SSD-1B",
        "tiny": "segmind/tiny-sd",
        "small_sd3": "stabilityai/stable-diffusion-3-medium",
    }

    def __init__(self, height: int = 512, width: int = 512, num_inference_steps: int = 50, guidance_scale: float = 7.5, model: str = "base", xformers: bool = False) -> None:
        """
        Initialize the DiffuserImageGenerator with a pre-trained model and image generation parameters.

        Args:
            height (int): The height of the generated image. Defaults to 512.
            width (int): The width of the generated image. Defaults to 512.
            num_inference_steps (int): The number of denoising steps. Defaults to 50.
            guidance_scale (float): The guidance scale for generation. Defaults to 7.5.
            model (str): The model to use. Options: "base", "turbo", "small", "tiny", "small_sd3". Defaults to "base".
            xformers (bool): Whether to use xformers for memory efficiency. Defaults to False.
        """
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        if model not in self.MODEL_OPTIONS:
            raise ValueError(f"Unknown model: {model}. Available options are: {', '.join(self.MODEL_OPTIONS.keys())}")

        model_name = self.MODEL_OPTIONS[model]

        # Adjust parameters for specific models
        if model == "turbo":
            self.num_inference_steps = min(4, num_inference_steps)
            self.guidance_scale = 0.0 if guidance_scale == 7.5 else guidance_scale

        if xformers:
            try:
                import xformers
            except ImportError:
                print("Tried to use Xformers but it is not installed. Please install it following instructions at: https://github.com/facebookresearch/xformers")
                xformers = False

        # Load the model
        print("[Diffuser] Loading model...")
        self.pipeline = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

        # Place the model on the GPU if available
        print("[Diffuser] Placing model on GPU if available...")
        if torch.cuda.is_available():
            self.pipeline = self.pipeline.to("cuda")
        elif torch.backends.mps.is_available():
            self.pipeline = self.pipeline.to("mps")
        else:
            self.pipeline = self.pipeline.to("cpu")
        print("...done.")

        # Change to channels-last format for speed
        # print("[Diffuser] Converting models to channels-last format for speed...")
        # self.pipeline.unet.to(memory_format=torch.channels_last)
        # self.pipeline.vae.to(memory_format=torch.channels_last)
        # print("...done.")

        # Fuse the QKV projections for memory efficiency
        # print("[Diffuser] Fusing QKV projections for memory efficiency...")
        # self.pipeline.fuse_qkv_projections()
        # print("...done.")

        # Compile the models
        # print("[Diffuser] Compiling models for speed...")
        # self.pipeline.unet = torch.compile(self.pipeline.unet, mode="max-autotune", fullgraph=True)
        # self.pipeline.vae.decode = torch.compile(self.pipeline.vae.decode, mode="max-autotune", fullgraph=True)
        # print("...done.")

        # Convert the model to float16 for memory efficiency
        # self.pipeline = self.pipeline.to(torch.float16)

        # Ensure all model components are in FP16
        # self.pipeline.unet = self.pipeline.unet.to(torch.float16)
        # self.pipeline.text_encoder = self.pipeline.text_encoder.to(torch.float16)
        # self.pipeline.vae = self.pipeline.vae.to(torch.float16)

        # Optional: Enable memory efficient attention
        if xformers:
            print("[Diffuser] Enabling memory efficient attention via xformers...")
            self.pipeline.enable_xformers_memory_efficient_attention()
            print("...done.")

    def generate(self, prompt: str, negative_prompt: str = "blurry, bad quality, duplicated") -> Image.Image:
        """
        Generate an image based on the given text prompt.

        Args:
            prompt (str): The text description of the image to generate.
            negative_prompt (str): The negative prompt to guide generation away from certain attributes.

        Returns:
            Image.Image: The generated image.
        """
        with torch.no_grad():
            # with torch.cuda.amp.autocast(dtype=torch.float16):
            result = self.pipeline(prompt=prompt, negative_prompt=negative_prompt, height=self.height, width=self.width, num_inference_steps=self.num_inference_steps, guidance_scale=self.guidance_scale)
        return result.images[0]


if __name__ == "__main__":
    import torch

    print("CuDNN version:", torch.backends.cudnn.version())

    # generator = DiffuserImageGenerator()
    # Sized for Discord banner
    # generator = DiffuserImageGenerator(height=500, width=500)
    generator = DiffuserImageGenerator(height=200, width=200, num_inference_steps=4, guidance_scale=0.0, model="turbo", xformers=True)
    aligner = SigLIPAligner()

    blobfish = False
    friend = True

    if blobfish:
        prompt = "A blobfish, its gelatinous body slumped and discolored, resting on a bed of seaweed in a dark, deep-sea environment. The blobfish''s face is a pale, almost translucent, and its eyes are wide and vacant. The background is a dark, inky blue, with faint bioluminescent creatures swimming in the distance."
        image = generator.generate(prompt)
        image.save("blobfish.png")

        # prompt = "A sea turtle wearing a scuba diving suit, working as a marine biologist."
        # image = generator.generate(prompt)
        # image.save("sea_turtle.png")

        # prompt = "Picture of A sea turtle wearing a scuba diving suit, working as a marine biologist."
        # image = generator.generate(prompt)
        # image.save("sea_turtle2.png")

        # Blobfish tests
        prompt = "A blobfish, its gelatinous body slumped and discolored, resting on a bed of seaweed in a dark, deep-sea environment. The blobfish''s face is a pale, almost translucent, and its eyes are wide and vacant. The background is a dark, inky blue, with faint bioluminescent creatures swimming in the distance."

        aligner.check_alignment("blobfish.png", prompt + " A beautiful, high-quality image.")
        score, image = aligner.search(generator, prompt + " A beautiful, high-quality image.", num_tries=25)
        print(score)
        image.save("blobfish_search.png")
    if friend:
        # prompt = "a vibrant, neon-colored synth that has been transformed into a character. Its body is sleek and metallic, adorned with glowing circuits and buttons reminiscent of a vintage computer console. The arms extend outwards, each ending in a curved handle that resembles a stylized guitar. Futuristic, high-quality, artisitic image."
        prompt = "A giant, glowing onion with a mischievous grin, surrounded by tiny, dancing emojis; high-quality, vibrant, and whimsical."
        prompt = "A giant, glowing onion, but it's not just any onion. It's a sentient onion, radiating a warm, fuzzy light. Its skin is a vibrant, shifting fuchsia that changes colors with every passing second, creating a mesmerizing visual experience.  Around the onion, a swirling galaxy of dancing emojis bursts forth:  funky, colorful emojis in a chaotic yet harmonious ballet, representing the fun and weirdness of the app itself."
        image = generator.generate(prompt)
        image.save("onion.png")
        aligner.check_alignment("onion.png", prompt)
        score, image = aligner.search(generator, prompt, num_tries=25)
        print(score)
        image.save("onion_search.png")

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
import virgil.utils.log as log
import click
from diffusers import BitsAndBytesConfig
from diffusers.models import FluxTransformer2DModel


class DiffuserImageGenerator(ImageGenerator):
    """
    A class for generating images from text prompts using the Diffusers library,
    with support for various models including future improvements.
    """

    # Preset model aliases for convenience
    MODEL_ALIASES = {
        "base": "stabilityai/stable-diffusion-xl-base-1.0",
        "turbo": "stabilityai/sdxl-turbo",
        "turbo_3.5": "stabilityai/stable-diffusion-3.5-large-turbo",
        "small": "segmind/SSD-1B",
        "tiny": "segmind/tiny-sd",
        "small_sd3": "stabilityai/stable-diffusion-3-medium",
        # New additions
        "lightning": "ByteDance/SDXL-Lightning",  # Ultra-fast generation (4-8 steps)
        "hyper": "stabilityai/sdxl-turbo",  # Alias for consistency (same as turbo)
        "playground": "playgroundai/playground-v2.5-1024px-aesthetic",  # High-quality aesthetic
        "pixart": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",  # Transformer-based efficiency
        "würstchen": "warp-ai/wuerstchen",  # Efficient latent diffusion
        "sd3_medium": "stabilityai/stable-diffusion-3-medium",  # Consistent naming
    }

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        model: str = "base",
        xformers: bool = False,
        quantization: str = "none",  # Options: "none", "int4", "int8"
        **pipeline_kwargs,  # Additional pipeline arguments
    ) -> None:
        """
        Initialize the image generator.

        Args:
            height: Height of generated images (default: 512)
            width: Width of generated images (default: 512)
            num_inference_steps: Denoising steps (default: 50)
            guidance_scale: Classifier-free guidance scale (default: 7.5)
            model: Model alias or direct HuggingFace repository ID
            xformers: Enable memory-efficient attention (requires xformers)
            pipeline_kwargs: Additional arguments for pipeline initialization
        """
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        # Resolve model identifier
        model_name = self.MODEL_ALIASES.get(model, model)

        # Auto-adjust parameters for turbo models
        if "turbo" in model_name.lower():
            self.num_inference_steps = min(4, num_inference_steps)
            if guidance_scale == 7.5:  # Only override default value
                self.guidance_scale = 0.0

        # Handle turbo-like models
        if model in ["turbo", "hyper", "lightning"]:
            self.num_inference_steps = min(4, num_inference_steps)

            # Special guidance scale handling
            if guidance_scale == 7.5:  # Only override default
                if model == "lightning":
                    self.guidance_scale = 1.0  # SDXL-Lightning prefers 1.0
                else:
                    self.guidance_scale = 0.0  # Other turbos use 0.0

        # Handle SD3 models
        elif "sd3" in model:
            self.num_inference_steps = min(28, num_inference_steps)
            if guidance_scale == 7.5:
                self.guidance_scale = 5.0
        elif "playground" in model:
            # Inference steps should be 20-25
            self.num_inference_steps = min(20, num_inference_steps)
            if guidance_scale == 7.5:
                self.guidance_scale = 4.0

        # Quantization configuration
        # Prepare quantization parameters
        load_dtype = torch.float16
        bnb_4bit_compute_dtype = torch.float16
        quant_config = None

        if quantization in ["int4", "int8"]:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=(quantization == "int4"),
                load_in_8bit=(quantization == "int8"),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=load_dtype,
                bnb_8bit_compute_dtype=load_dtype,
            )
            print(f"[Diffuser] Using {quantization} quantization")

        # First load the transformer model
        transformer = None
        if "flux" in model:
            # Load the quantized transformer model
            transformer = FluxTransformer2DModel.from_pretrained(
                "fused_transformer",
                quantization_config=quant_config,
                torch_dtype=bnb_4bit_compute_dtype,
            )

        print(f"[Diffuser] Loading model: {model_name}")
        try:
            # Load pipeline with quantization config
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                model_name,
                torch_dtype=load_dtype,
                variant="fp16",
                use_safetensors=True,
                quantization_config=quant_config,  # Correct parameter name
                **pipeline_kwargs,
            )
        except ImportError:
            print("Quantization requires bitsandbytes: pip install bitsandbytes")
        except Exception as e:
            log.error(f"Standard load failed ({e}), attempting fallback...")
            # Fallback without quantization/variant
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                model_name,
                transformer=transformer,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                **pipeline_kwargs,
            )

        # Device placement
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # New: Special handling for quantized models
        if quantization in ["int4", "int8"]:
            if device != "cuda":
                print(
                    f"Warning: {quantization} quantization requires CUDA. Disabling quantization."
                )
            elif not isinstance(self.pipeline, torch.nn.Module):
                # Workaround for diffusers device placement bug
                self.pipeline.to("cuda")
            else:
                self.pipeline = self.pipeline.to(device)
        else:
            self.pipeline = self.pipeline.to(device)

        print(f"[Diffuser] Using device: {device}")

        # Enable optimizations
        self._enable_optimizations(xformers)

    def _enable_optimizations(self, xformers: bool) -> None:
        """Apply performance optimizations where supported"""
        try:
            # Channels-last memory format
            if hasattr(self.pipeline, "unet"):
                self.pipeline.unet.to(memory_format=torch.channels_last)
            if hasattr(self.pipeline, "vae"):
                self.pipeline.vae.to(memory_format=torch.channels_last)

            # Memory-efficient attention
            if xformers:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("[Diffuser] Enabled xformers memory efficient attention")
        except Exception as e:
            print(f"[Diffuser] Optimization warning: {str(e)}")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, bad quality, duplicated",
        **generation_kwargs,  # Additional generation arguments
    ) -> Image.Image:
        """
        Generate image from text prompt.

        Args:
            prompt: Text description of desired image
            negative_prompt: Negative guidance prompt
            generation_kwargs: Additional arguments for generation

        Returns:
            Generated PIL Image
        """
        # Merge class parameters with runtime parameters
        base_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": self.height,
            "width": self.width,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
        }
        params = {**base_params, **generation_kwargs}

        with torch.no_grad():
            result = self.pipeline(**params)
        return result.images[0]


def test_diffuser():
    """
    Test the DiffuserImageGenerator with a sample prompt.
    """

    import torch

    print("CuDNN version:", torch.backends.cudnn.version())

    # generator = DiffuserImageGenerator()
    # Sized for Discord banner
    # generator = DiffuserImageGenerator(height=500, width=500)
    generator = DiffuserImageGenerator(
        height=200,
        width=200,
        num_inference_steps=4,
        guidance_scale=0.0,
        model="turbo",
        xformers=True,
    )
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

        aligner.check_alignment(
            "blobfish.png", prompt + " A beautiful, high-quality image."
        )
        score, image = aligner.search(
            generator, prompt + " A beautiful, high-quality image.", num_tries=25
        )
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


@click.command()
@click.option("--height", default=512, help="Height of the generated image.")
@click.option("--width", default=512, help="Width of the generated image.")
@click.option(
    "--num_inference_steps",
    "--steps",
    "--num-inference-steps",
    "--num-steps",
    "--num_steps",
    "--inference-steps",
    "--inference_steps",
    default=50,
    help="Number of inference steps for generation.",
)
@click.option(
    "--guidance_scale",
    "--guidance",
    "--guidance-scale",
    default=7.5,
    help="Guidance scale for image generation.",
)
@click.option(
    "--model",
    "-m",
    default="turbo",
    type=click.Choice(DiffuserImageGenerator.MODEL_ALIASES.keys()),
    help="Model to use for image generation.",
)
@click.option(
    "--xformers",
    is_flag=True,
    help="Use xformers for memory-efficient attention.",
)
@click.option(
    "--prompt",
    "-p",
    default="A beautiful sunset over a calm sea, with vibrant colors reflecting on the water.",
    help="Text prompt for image generation.",
)
@click.option(
    "--output",
    "--output-file",
    "--output-filename",
    "-o",
    default="generated_image.png",
    help="Output filename for the generated image.",
)
@click.option(
    "--quantization",
    "-q",
    default="none",
    type=click.Choice(["none", "int4", "int8"]),
    help="Quantization method to use (none, int4, int8).",
)
@click.option("--num-images", "-n", default=1, help="Number of images to generate.")
def main(
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    model: str = "base",
    xformers: bool = False,
    prompt: str = "A beautiful sunset over a calm sea, with vibrant colors reflecting on the water.",
    output: str = "generated_image.png",
    num_images: int = 1,
    quantization: str = "none",  # Options: "none", "int4", "int8"
) -> None:
    """Main function to generate an image using the DiffuserImageGenerator."""
    generator = DiffuserImageGenerator(
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        model=model,
        xformers=xformers,
        quantization=quantization,
    )

    if num_images < 1:
        raise ValueError("Number of images to generate must be at least 1.")

    if len(prompt) == 0:
        prompt = "A beautiful sunset over a calm sea, with vibrant colors reflecting on the water."

    # Extract file extension from output
    output_extension = output.split(".")[-1].lower()
    if output_extension not in ["png", "jpg", "jpeg"]:
        raise ValueError("Output file must be a PNG or JPG image.")

    if num_images > 1:
        images = []
        for i in range(num_images):
            image = generator.generate(prompt)
            images.append(image)
            image.save(f"{output.split('.')[0]}_{i + 1}.{output_extension}")
    else:
        image = generator.generate(prompt)
        image.save(output)


if __name__ == "__main__":
    main()
    # test_diffuser()  # Uncomment to run the test function

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

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from .base import VideoBackend
from typing import List, Optional
import click
from PIL import Image
from transformers import BitsAndBytesConfig

# Try to import the specific WanPipeline if available
try:
    from diffusers import WanPipeline

    PipelineClass = WanPipeline
    WANPIPELINE_AVAILABLE = True
except ImportError:
    PipelineClass = DiffusionPipeline
    WANPIPELINE_AVAILABLE = False


class Wan21(VideoBackend):
    """A wrapper for the Wan-2.1 model with proper quantization support."""

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-14B",
        torch_dtype: torch.dtype = torch.float16,
        variant: str = "fp16",
        quantization: Optional[str] = None,
        offload: bool = False,
        sequential_offload: bool = False,
    ):
        """
        Initialize the Wan21 backend with advanced memory management.

        Args:
            model_id (str): The ID of the model to use.
            torch_dtype (torch.dtype): The torch data type to use.
            variant (str): The model variant to use (e.g., "fp16").
            quantization (str): Quantization mode ('int4' or 'int8')
            offload (bool): Enable model CPU offloading
            sequential_offload (bool): Enable sequential CPU offloading (more memory efficient)
        """
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "variant": variant,
        }

        # Configure quantization using BitsAndBytesConfig
        if quantization:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=(quantization == "int4"),
                load_in_8bit=(quantization == "int8"),
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"  # Automatic device placement

        # Initialize pipeline
        self.pipe = PipelineClass.from_pretrained(model_id, **load_kwargs)

        # Apply offloading strategies
        if offload or sequential_offload:
            if not quantization:  # Offloading doesn't work with quantization
                if sequential_offload:
                    self.pipe.enable_sequential_cpu_offload()
                else:
                    self.pipe.enable_model_cpu_offload()
            else:
                print(
                    "Warning: Offloading is not compatible with quantization - using quantization device mapping"
                )
        elif not quantization:
            # Move to GPU if not using quantization or offloading
            self.pipe.to("cuda")

        self.supported_modes = ["text", "image"]

    def __call__(
        self,
        prompt: Optional[str] = None,
        image: Optional[Image.Image] = None,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_path: str = "output.mp4",
        **kwargs,
    ) -> List[str]:
        """
        Generate a video from text prompt OR an initial image.

        Args:
            prompt (str): Text prompt for video generation
            image (PIL.Image): Initial image for video generation
            num_frames (int): Number of frames to generate
            num_inference_steps (int): Inference steps
            guidance_scale (float): Guidance scale
            output_path (str): Output video path

        Returns:
            List[str]: Path to generated video
        """
        # Validate input
        if not (prompt or image):
            raise ValueError("Either prompt or image must be provided")
        if prompt and image:
            raise ValueError("Specify only one of prompt or image")

        # Prepare generation parameters
        gen_kwargs = {
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            **kwargs,
        }

        # Run appropriate generation mode
        if image:
            gen_kwargs["image"] = image
            result = self.pipe(**gen_kwargs)
        else:
            gen_kwargs["prompt"] = prompt
            result = self.pipe(**gen_kwargs)

        # Handle different pipeline output formats
        if hasattr(result, "frames"):
            video_frames = result.frames[0]
        elif hasattr(result, "videos"):
            video_frames = result.videos[0]
        elif isinstance(result, list) and len(result) > 0:
            video_frames = result[0]
        else:
            raise ValueError(f"Unexpected pipeline output format: {type(result)}")

        # Save and return result
        export_to_video(video_frames, output_path)
        return [output_path]


@click.command()
@click.option("--prompt", type=str, help="Text prompt for video generation")
@click.option("--image-path", type=click.Path(exists=True), help="Initial image path")
@click.option(
    "--output-path", default="output.mp4", type=click.Path(), help="Output video path"
)
@click.option("--num-frames", default=16, type=int, help="Number of video frames")
@click.option("--num-steps", default=50, type=int, help="Inference steps")
@click.option("--guidance-scale", default=7.5, type=float, help="Guidance scale")
@click.option(
    "--model-id", default="Wan-AI/Wan2.1-T2V-14B", help="Hugging Face model ID"
)
@click.option(
    "--quantization", type=click.Choice(["int4", "int8"]), help="Quantization mode"
)
@click.option(
    "--dtype",
    default="float16",
    type=click.Choice(["float32", "float16", "bfloat16"]),
    help="Torch data type",
)
@click.option("--offload", is_flag=True, help="Enable model CPU offloading")
@click.option(
    "--sequential-offload",
    is_flag=True,
    help="Enable sequential CPU offloading (more memory efficient)",
)
def generate_video(
    prompt: str,
    image_path: str,
    output_path: str,
    num_frames: int,
    num_steps: int,
    guidance_scale: float,
    model_id: str,
    quantization: str,
    dtype: str,
    offload: bool,
    sequential_offload: bool,
):
    """CLI for Wan2.1 video generation with advanced memory options"""
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    # Load image if provided
    image = None
    if image_path:
        image = Image.open(image_path)

    # Initialize pipeline
    generator = Wan21(
        model_id=model_id,
        torch_dtype=dtype_map[dtype],
        quantization=quantization,
        offload=offload,
        sequential_offload=sequential_offload,
    )

    # Generate video
    result = generator(
        prompt=prompt,
        image=image,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        output_path=output_path,
    )

    click.echo(f"Video generated at: {result[0]}")


if __name__ == "__main__":
    generate_video()

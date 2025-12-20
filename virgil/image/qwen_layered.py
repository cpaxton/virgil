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

"""
Qwen Image Layered generator for Virgil.

This module provides image generation using Qwen's Image Layered model,
which can generate layered images from text prompts or input images.
See: https://huggingface.co/Qwen/Qwen-Image-Layered
"""

from PIL import Image
import torch
from typing import Optional
from virgil.image.base import ImageGenerator
import virgil.utils.log as log


class QwenLayeredImageGenerator(ImageGenerator):
    """
    A class for generating layered images using Qwen's Image Layered model.

    This generator can create layered images from text prompts or input images.
    The model generates multiple layers that can be composited together.

    Model: https://huggingface.co/Qwen/Qwen-Image-Layered
    """

    def __init__(
        self,
        height: int = 640,
        width: int = 640,
        num_inference_steps: int = 50,
        layers: int = 4,
        resolution: int = 640,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = " ",
        num_images_per_prompt: int = 1,
        cfg_normalize: bool = True,
        use_en_prompt: bool = True,
        model: str = "Qwen/Qwen-Image-Layered",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> None:
        """
        Initialize the Qwen Image Layered generator.

        Args:
            height: Height of generated images (default: 640)
            width: Width of generated images (default: 640)
            num_inference_steps: Number of denoising steps (default: 50)
            layers: Number of layers to generate (default: 4)
            resolution: Resolution bucket (640 or 1024, default: 640 recommended)
            true_cfg_scale: CFG scale for generation (default: 4.0)
            negative_prompt: Negative prompt for generation (default: " ")
            num_images_per_prompt: Number of images per prompt (default: 1)
            cfg_normalize: Whether to enable CFG normalization (default: True)
            use_en_prompt: Automatic caption language if user doesn't provide caption (default: True)
            model: HuggingFace model identifier (default: "Qwen/Qwen-Image-Layered")
            device: Device to run on (default: "cuda" if available, else "cpu")
            dtype: Data type for model (default: torch.bfloat16)
            **kwargs: Additional pipeline arguments
        """
        super().__init__(height=height, width=width)

        self.num_inference_steps = num_inference_steps
        self.layers = layers
        self.resolution = resolution
        self.true_cfg_scale = true_cfg_scale
        self.negative_prompt = negative_prompt
        self.num_images_per_prompt = num_images_per_prompt
        self.cfg_normalize = cfg_normalize
        self.use_en_prompt = use_en_prompt
        self.model_name = model
        self.dtype = dtype

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Validate resolution
        if resolution not in [640, 1024]:
            log.warning(
                f"Resolution {resolution} not in recommended buckets [640, 1024]. Using 640."
            )
            self.resolution = 640

        # Initialize pipeline (lazy loading)
        self.pipeline = None
        self._pipeline_kwargs = kwargs

    def _load_pipeline(self):
        """Lazy load the pipeline."""
        if self.pipeline is not None:
            return

        try:
            from diffusers import QwenImageLayeredPipeline

            log.info(f"Loading Qwen Image Layered model: {self.model_name}")
            self.pipeline = QwenImageLayeredPipeline.from_pretrained(self.model_name)
            self.pipeline = self.pipeline.to(self.device, self.dtype)
            self.pipeline.set_progress_bar_config(disable=None)
            log.info("Qwen Image Layered pipeline loaded successfully")
        except ImportError:
            raise ImportError(
                "QwenImageLayeredPipeline not found. "
                "Make sure you have the latest version of diffusers installed: "
                "pip install diffusers --upgrade"
            )
        except Exception as e:
            log.error(f"Failed to load Qwen Image Layered pipeline: {e}")
            raise

    def generate(
        self,
        prompt: str,
        input_image: Optional[Image.Image] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate a layered image from a text prompt or input image.

        Args:
            prompt: Text description of the image to generate
            input_image: Optional input image for image-to-image generation.
                        If None, generates from text prompt only.
            seed: Random seed for generation (default: None for random)

        Returns:
            Image.Image: The first generated layer image (composited if multiple layers)
        """
        self._load_pipeline()

        # Prepare generator
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        # Prepare inputs
        inputs = {
            "prompt": prompt,
            "generator": generator,
            "true_cfg_scale": self.true_cfg_scale,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
            "num_images_per_prompt": self.num_images_per_prompt,
            "layers": self.layers,
            "resolution": self.resolution,
            "cfg_normalize": self.cfg_normalize,
            "use_en_prompt": self.use_en_prompt,
            **self._pipeline_kwargs,
        }

        # Add input image if provided
        if input_image is not None:
            # Ensure image is RGBA for layered generation
            if input_image.mode != "RGBA":
                input_image = input_image.convert("RGBA")
            inputs["image"] = input_image

        # Generate
        try:
            with torch.inference_mode():
                output = self.pipeline(**inputs)
                output_images = output.images[0]  # Get first batch

            # If multiple layers, composite them
            if isinstance(output_images, list) and len(output_images) > 1:
                # Composite layers: start with first layer
                result = output_images[0].convert("RGBA")

                # Blend subsequent layers
                for layer in output_images[1:]:
                    layer_rgba = layer.convert("RGBA")
                    result = Image.alpha_composite(result, layer_rgba)

                return result.convert("RGB")
            elif isinstance(output_images, list) and len(output_images) == 1:
                return output_images[0].convert("RGB")
            else:
                # Single image
                return output_images.convert("RGB")

        except Exception as e:
            log.error(f"Error generating image with Qwen Image Layered: {e}")
            raise

    def generate_layers(
        self,
        prompt: str,
        input_image: Optional[Image.Image] = None,
        seed: Optional[int] = None,
    ) -> list[Image.Image]:
        """
        Generate all layers separately (without compositing).

        Args:
            prompt: Text description of the image to generate
            input_image: Optional input image for image-to-image generation
            seed: Random seed for generation

        Returns:
            list[Image.Image]: List of all generated layer images
        """
        self._load_pipeline()

        # Prepare generator
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        # Prepare inputs
        inputs = {
            "prompt": prompt,
            "generator": generator,
            "true_cfg_scale": self.true_cfg_scale,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
            "num_images_per_prompt": self.num_images_per_prompt,
            "layers": self.layers,
            "resolution": self.resolution,
            "cfg_normalize": self.cfg_normalize,
            "use_en_prompt": self.use_en_prompt,
            **self._pipeline_kwargs,
        }

        # Add input image if provided
        if input_image is not None:
            if input_image.mode != "RGBA":
                input_image = input_image.convert("RGBA")
            inputs["image"] = input_image

        # Generate
        try:
            with torch.inference_mode():
                output = self.pipeline(**inputs)
                output_images = output.images[0]

            # Return all layers
            return [img.convert("RGB") for img in output_images]

        except Exception as e:
            log.error(f"Error generating layers with Qwen Image Layered: {e}")
            raise

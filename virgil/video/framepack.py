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
from diffusers.utils import export_to_video, load_image
from .base import VideoBackend
from typing import List
from diffusers import (
    HunyuanVideoFramepackPipeline,
    HunyuanVideoFramepackTransformer3DModel,
)
from transformers import SiglipImageProcessor, SiglipVisionModel


class Framepack(VideoBackend):
    """
    A wrapper for the FramePack model.
    """

    def __init__(
        self,
        # model_id: str = "cerspense/zeroscope_v2_576w",  # Updated default model
        # torch_dtype: torch.dtype = torch.float16,
        # variant: str = "fp16",
    ):
        """
        Initialize the Framepack backend.

        Args:
            model_id (str): The ID of the model to use.
            torch_dtype (torch.dtype): The torch data type to use.
            variant (str): The model variant to use (e.g., "fp16").
        """

        transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
            "lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16
        )
        feature_extractor = SiglipImageProcessor.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
        )
        image_encoder = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl",
            subfolder="image_encoder",
            torch_dtype=torch.float16,
        )
        self.pipe = HunyuanVideoFramepackPipeline.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            transformer=transformer,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            torch_dtype=torch.float16,
        )
        self.pipe.to("cuda")

        # Enable memory optimizations
        self.enable_memory_optimizations = True  # Set to False to disable optimizations
        if self.enable_memory_optimizations:
            self.pipe.enable_model_cpu_offload()
            self.pipe.vae.enable_tiling()

    def __call__(
        self,
        prompt: str,
        image_path: str,
        num_frames: int = 91,
        num_inference_steps: int = 30,
        guidance_scale: float = 9.0,
        output_path: str | None = None,
        fps: int = 30,
        **kwargs,
    ) -> List[str]:
        """
        Generate a video from a text prompt and an initial image.

        Args:
            prompt (str): The text prompt to generate the video from.
            image_path (str): The path to the initial image.
            num_frames (int): The number of frames to generate.
            num_inference_steps (int): The number of inference steps to use.
            guidance_scale (float): The guidance scale to use.
            output_path (str): The path to save the generated video.

        Returns:
            List[str]: A list containing the path to the generated video.
        """

        image = load_image(image_path)
        video_frames = self.pipe(
            prompt=prompt,
            image=image,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(0),
            sampling_type="inverted_anti_drifting",
            **kwargs,
        ).frames[0]
        if output_path is not None:
            export_to_video(video_frames, output_path, fps=fps)
        return [output_path]

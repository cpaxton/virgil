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
from diffusers.utils import export_to_video, load_image
from .base import VideoBackend
from typing import List


class Framepack(VideoBackend):
    """
    A wrapper for the FramePack model.
    """

    def __init__(
        self,
        model_id: str = "lllyasviel/FramePackI2V_HY",
        torch_dtype: torch.dtype = torch.float16,
        variant: str = "fp16",
    ):
        """
        Initialize the Framepack backend.

        Args:
            model_id (str): The ID of the model to use.
            torch_dtype (torch.dtype): The torch data type to use.
            variant (str): The model variant to use (e.g., "fp16").
        """
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            variant=variant,
        )
        self.pipe.to("cuda")

    def __call__(
        self,
        prompt: str,
        image_path: str,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_path: str = "output.mp4",
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
            **kwargs,
        ).frames[0]

        export_to_video(video_frames, output_path)
        return [output_path]

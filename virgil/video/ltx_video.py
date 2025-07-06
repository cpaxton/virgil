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
from diffusers import LTXPipeline
from diffusers.utils import export_to_video
from .base import VideoBackend
from typing import List

from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    LTXVideoTransformer3DModel,
)
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel


class LTXVideo(VideoBackend):
    """
    A wrapper for the Lightricks LTX-Video model.
    """

    def __init__(
        self,
        model_id: str = "lightricks/ltx-video",
        torch_dtype: torch.dtype = torch.float16,
        variant: str = "fp16",
    ):
        """
        Initialize the LTXVideo backend.

        Args:
            model_id (str): The ID of the model to use.
            torch_dtype (torch.dtype): The torch data type to use.
            variant (str): The model variant to use (e.g., "fp16").
        """

        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        text_encoder_8bit = T5EncoderModel.from_pretrained(
            "Lightricks/LTX-Video",
            subfolder="text_encoder",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
        )

        quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
        transformer_8bit = LTXVideoTransformer3DModel.from_pretrained(
            "Lightricks/LTX-Video",
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
        )

        self.pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            text_encoder=text_encoder_8bit,
            transformer=transformer_8bit,
            torch_dtype=torch.float16,
            device_map="balanced",
        )
        self.pipe.to("cuda")

    def __call__(
        self,
        prompt: str,
        num_frames: int = 16,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_path: str = "output.mp4",
        **kwargs,
    ) -> List[str]:
        """
        Generate a video from a text prompt.

        Args:
            prompt (str): The text prompt to generate the video from.
            num_frames (int): The number of frames to generate.
            num_inference_steps (int): The number of inference steps to use.
            guidance_scale (float): The guidance scale to use.
            output_path (str): The path to save the generated video.

        Returns:
            List[str]: A list containing the path to the generated video.
        """
        video_frames = self.pipe(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs,
        ).frames[0]

        export_to_video(video_frames, output_path)
        return [output_path]

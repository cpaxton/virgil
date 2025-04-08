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

from typing import Optional

import torch
from .gemma import Gemma
from .base import Backend
from .llama import Llama
from .qwen import Qwen, qwen_sizes, qwen_specializations


backend_list = [
    "gemma",
    "gemma2b",
    "gemma-2b-it",
    "gemma-3-27b-it",
    "llama-3.2-1B",
    "qwen",
    "qwen-0.5B-Instruct",
    "qwen-0.5B-Coder",
    "qwen-0.5B-Math",
    "qwen-1.5B-Instruct",
    "qwen-1.5B-Coder",
    "qwen-1.5B-Math",
    "qwen-3B-Instruct",
    "qwen-3B-Coder",
    "qwen-3B-Math",
    "qwen-7B-Instruct",
    "qwen-7B-Coder",
    "qwen-7B-Math",
    "qwen-14B-Instruct",
    "qwen-14B-Coder",
    "qwen-14B-Math",
    "qwen-1.5B-Deepseek",
    "qwen-7B-Deepseek",
    "qwen-14B-Deepseek",
    "qwen-32B-Deepseek",
]


def get_backend(name: str, use_flash_attention: bool = False, **kwargs) -> Backend:
    """Get a backend by name.

    Args:
        name (str): The name of the backend.
        use_flash_attention (bool): Whether to use Flash Attention. Defaults to False. Only used for Gemma.

    Returns:
        Backend: The backend instance, used for interfacing with an LLM.
    """
    name = name.lower()
    if name.startswith("gemma"):
        if name == "gemma2b" or name == "gemma-2b-it":
            gemma_kwargs = kwargs
            gemma_kwargs["quantization"] = "int8" if torch.cuda.is_available() else None
            gemma_kwargs["use_flash_attention"] = True if torch.cuda.is_available() else False
        else:
            gemma_kwargs = kwargs
            gemma_kwargs["quantization"] = "int4" if torch.cuda.is_available() else None
            gemma_kwargs["use_flash_attention"] = True if torch.cuda.is_available() else False
        return Gemma(**gemma_kwargs)
    elif name == "llama" or name == "llama-3.2-1B":
        return Llama(model_name="meta-llama/Llama-3.2-1B", **kwargs)
    elif name.startswith("qwen"):
        # get the size and specialization
        # This lets us use the same model for different sizes and specializations (e.g. 1B-Instruct, 1B-Coder, 1B-Math)

        # TODO: process this better
        if name == "qwen":
            size = "3B"
            specialization = "Instruct"
        # if one of the sizes is in the name...
        elif any(size.lower() in name for size in qwen_sizes):
            for size in qwen_sizes:
                if size.lower() in name:
                    break
            else:
                # if we didn't find a size, default to 1.5B
                size = "1.5B"
            if any(spec.lower() in name for spec in qwen_specializations):
                for spec in qwen_specializations:
                    if spec.lower() in name:
                        specialization = spec
                        break
                else:
                    specialization = "Instruct"
            else:
                specialization = "Instruct"
            print(f"Size: {size}, Specialization: {specialization}")
        else:
            size = "1.5B"
            specialization = "Instruct"

        if specialization == "Deepseek":
            model_name = f"deepseek-ai/DeepSeek-R1-Distill-Qwen-{size}"
        else:
            model_name = f"Qwen/Qwen2.5-{size}-{specialization}"

        qwen_kwargs = kwargs
        # qwen_kwargs["quantization"] = "int8" if torch.cuda.is_available() else None
        return Qwen(model_name=model_name, **qwen_kwargs)
    else:
        raise ValueError(f"Unknown backend: {name}")

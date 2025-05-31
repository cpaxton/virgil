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
import torch
from .gemma import Gemma
from .base import Backend
from .llama import Llama
from .qwen import (
    Qwen,
    qwen_sizes,
    qwen_specializations,
    qwen_releases,
    qwen25_sizes,
    qwen30_sizes,
)
import virgil.util.log as logger

qwens = []
qwens = [
    f"qwen{release}-{size}-{spec}"
    for size in qwen_sizes
    for spec in qwen_specializations
    for release in qwen_releases
]


backend_list = [
    "gemma",
    "gemma2b",
    "gemma-2b-it",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "llama-3.2-1B",
    "qwen",
] + qwens


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
            gemma_kwargs["use_flash_attention"] = (
                True if torch.cuda.is_available() else False
            )
            gemma_kwargs["variant"] = "google/gemma-2-2b-it"
        else:
            gemma_kwargs = kwargs
            gemma_kwargs["quantization"] = "int4" if torch.cuda.is_available() else None
            gemma_kwargs["use_flash_attention"] = (
                True if torch.cuda.is_available() else False
            )
            if not name.endswith("-it"):
                name += "-it"
            gemma_kwargs["variant"] = "google/" + name
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
            release = "3"
        else:
            release = "2.5"
            size = "1.5B"
            specialization = "Instruct"

            # Parse qwen{release}-{size}-{specialization} or qwen{release}-{size}
            # e.g. qwen3-1.5B-Instruct, qwen2.5-7B-Coder, qwen2.5-14B-Math
            parts = name.split("-")
            if len(parts) == 3:
                release = parts[0][4:]
                size = parts[1]
                specialization = parts[2].capitalize()
                logger.info(f"Using Qwen model: Qwen{release}-{size}-{specialization}")

        # Parse size and specialization
        if release == "2.5":
            if size.upper() not in qwen25_sizes:
                raise ValueError(
                    f"Unknown size: {size}. Available sizes for Qwen 2.5: {qwen25_sizes}"
                )
            if specialization == "Deepseek":
                model_name = f"deepseek-ai/DeepSeek-R1-Distill-Qwen-{size}"
            elif specialization not in qwen_specializations:
                raise ValueError(
                    f"Unknown specialization: {specialization}. Available specializations: {qwen_specializations}"
                )
            model_name = f"Qwen/Qwen{release}-{size}-{specialization}"
        elif release == "3":
            # No specializations
            if size.upper() not in qwen30_sizes:
                raise ValueError(
                    f"Unknown size: {size}. Available sizes for Qwen 3: {qwen30_sizes}"
                )
            if specialization != "Deepseek":
                model_name = f"Qwen/Qwen{release}-{size}"
            else:
                if size != "8B":
                    logger.warning(
                        f"Deepseek specialization only available for 8B size in Qwen 3, using 8B instead of {size}"
                    )
                model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8b"
        else:
            raise ValueError(
                f"Unknown release: {release}. Available releases: {qwen_releases}"
            )

        qwen_kwargs = kwargs
        # qwen_kwargs["quantization"] = "int8" if torch.cuda.is_available() else None
        return Qwen(model_name=model_name, **qwen_kwargs)
    else:
        raise ValueError(f"Unknown backend: {name}")

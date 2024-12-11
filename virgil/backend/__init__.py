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

from .gemma import Gemma
from .base import Backend
from .llama import Llama
from .qwen import Qwen, qwen_sizes, qwen_specializations


backend_list = [
    "gemma",
    "gemma2b",
    "gemma-2b-it",
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
]


def get_backend(name: str, use_flash_attention: bool = False) -> Backend:
    """Get a backend by name.

    Args:
        name (str): The name of the backend.

    Returns:
        Backend: The backend instance.
    """
    name = name.lower()
    if name == "gemma" or name == "gemma2b" or name == "gemma-2b-it":
        return Gemma(use_flash_attention=use_flash_attention)
    elif name == "llama" or name == "llama-3.2-1B":
        return Llama(model_name="meta-llama/Llama-3.2-1B")
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

        return Qwen(model_name=f"Qwen/Qwen2.5-{size}-{specialization}")
    else:
        raise ValueError(f"Unknown backend: {name}")

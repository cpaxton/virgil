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

import warnings

import torch

# Enable TF32 for matmul (inductor recommends; PyTorch 2.9 warns internally)
if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*new API settings to control TF32.*"
        )
        torch.set_float32_matmul_precision("high")

from .base import Backend
from .ernie import Ernie, get_ernie_model_names
from .gemma import Gemma, get_gemma_model_names
from .llama import Llama, get_llama_model_names
from .phi import Phi, get_phi_model_names
from .qwen import Qwen, get_qwen_model_names
from .smollm import SmolLM, get_smollm_model_names
from .tinyllama import TinyLlama, get_tinyllama_model_names

backend_list = (
    get_llama_model_names()
    + get_gemma_model_names()
    + get_qwen_model_names()
    + get_ernie_model_names()
    + get_phi_model_names()
    + get_tinyllama_model_names()
    + get_smollm_model_names()
)


def get_backend(name: str, **kwargs) -> Backend:
    """
    Get a backend by name.

    This function acts as a factory for creating backend instances. It determines the
    correct backend class based on the model name and passes along any additional
    keyword arguments to the class's constructor.

    Args:
        name (str): The name of the backend (e.g., "gemma-2-9b-it", "qwen3-8b").
        **kwargs: Additional arguments to pass to the backend constructor, such as
                  `quantization`, `temperature`, etc.

    Returns:
        Backend: An instance of the requested backend.

    Raises:
        ValueError: If the backend name is unknown.
    """
    name = name.lower()
    kwargs["model_name"] = name

    if name.startswith("gemma"):
        return Gemma(**kwargs)
    elif name.startswith("ernie"):
        return Ernie(**kwargs)
    elif name.startswith("llama"):
        return Llama(**kwargs)
    elif name.startswith("qwen"):
        return Qwen(**kwargs)
    elif name.startswith("phi"):
        return Phi(**kwargs)
    elif name.startswith("tinyllama"):
        return TinyLlama(**kwargs)
    elif name.startswith("smollm"):
        return SmolLM(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {name}")

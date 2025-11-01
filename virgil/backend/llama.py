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
from transformers import BitsAndBytesConfig, pipeline

from virgil.backend.base import Backend

llama_variants = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
]

name_to_variant = {
    "llama": "meta-llama/Llama-3.2-1B",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
}


def get_llama_model_names() -> list[str]:
    """Get a list of available Llama model names."""
    return list(name_to_variant.keys())


class Llama(Backend):
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        quantization: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        if model_path:
            model_id = model_path
        elif model_name in name_to_variant:
            model_id = name_to_variant[model_name]
        elif model_name.startswith("meta-llama/"):
            model_id = model_name
        else:
            model_id = model_name  # Assume it's a valid HuggingFace model ID

        model_kwargs = {"dtype": torch.bfloat16}

        if quantization:
            if quantization == "int8":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            elif quantization == "int4":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True
                )
            else:
                raise ValueError(f"Unknown quantization: {quantization}")

        pipeline_kwargs = {}
        if torch.cuda.is_available():
            pipeline_kwargs["device_map"] = "auto"
        elif torch.backends.mps.is_available():
            pipeline_kwargs["device"] = "mps"

        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs=model_kwargs,
            **pipeline_kwargs,
        )
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

    def __call__(self, messages, max_new_tokens: int = 256, *args, **kwargs) -> list:
        """Generate a response to a list of messages."""
        with torch.no_grad():
            return self.pipe(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
            )

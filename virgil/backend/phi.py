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
from transformers import (
    BitsAndBytesConfig,
    pipeline,
)

from virgil.backend.base import Backend

phi_variants = [
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/Phi-3-small-8k-instruct",
    "microsoft/Phi-3-small-128k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-3-medium-128k-instruct",
]

name_to_variant = {
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-mini-4k": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-mini-128k": "microsoft/Phi-3-mini-128k-instruct",
    "phi-3-small": "microsoft/Phi-3-small-8k-instruct",
    "phi-3-small-8k": "microsoft/Phi-3-small-8k-instruct",
    "phi-3-small-128k": "microsoft/Phi-3-small-128k-instruct",
    "phi-3-medium": "microsoft/Phi-3-medium-4k-instruct",
    "phi-3-medium-4k": "microsoft/Phi-3-medium-4k-instruct",
    "phi-3-medium-128k": "microsoft/Phi-3-medium-128k-instruct",
}


def get_phi_model_names() -> list[str]:
    """Get a list of available Phi model names."""
    return list(name_to_variant.keys())


class Phi(Backend):
    """Use Microsoft Phi-3 models for text generation."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        quantization: Optional[str] = "int4",
        model_path: Optional[str] = None,
    ) -> None:
        """Initialize the Phi backend."""
        if model_path:
            variant = model_path
        elif model_name in name_to_variant:
            variant = name_to_variant[model_name]
        elif model_name.startswith("microsoft/"):
            variant = model_name
        else:
            raise ValueError(
                f"Unknown Phi model: {model_name}. Supported: {list(name_to_variant.keys())}"
            )

        print(f"Loading Phi variant: {variant}")

        model_kwargs = {"dtype": torch.bfloat16, "trust_remote_code": True}

        if quantization:
            quantization = quantization.lower()
            if quantization in ["int8", "int4"]:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=(quantization == "int4"),
                    load_in_8bit=(quantization == "int8"),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                raise ValueError(f"Unknown quantization method: {quantization}")

        pipeline_kwargs = {}
        if torch.cuda.is_available():
            pipeline_kwargs["device_map"] = "auto"
        elif torch.backends.mps.is_available():
            pipeline_kwargs["device"] = "mps"

        print("[Phi] loading the model...")
        self.pipe = pipeline(
            "text-generation",
            model=variant,
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

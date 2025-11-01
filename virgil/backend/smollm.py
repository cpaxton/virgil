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

smollm_variants = [
    "huggingface/SmolLM-135M-Instruct",
    "huggingface/SmolLM-360M-Instruct",
    "huggingface/SmolLM-1.7B-Instruct",
    "huggingface/SmolLM-2.7B-Instruct",
]

name_to_variant = {
    "smollm-135m": "huggingface/SmolLM-135M-Instruct",
    "smollm-360m": "huggingface/SmolLM-360M-Instruct",
    "smollm-1.7b": "huggingface/SmolLM-1.7B-Instruct",
    "smollm-2.7b": "huggingface/SmolLM-2.7B-Instruct",
}


def get_smollm_model_names() -> list[str]:
    """Get a list of available SmolLM model names."""
    return list(name_to_variant.keys())


class SmolLM(Backend):
    """Use HuggingFace SmolLM models for text generation."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        quantization: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        """Initialize the SmolLM backend."""
        if model_path:
            variant = model_path
        elif model_name in name_to_variant:
            variant = name_to_variant[model_name]
        elif model_name.startswith("huggingface/"):
            variant = model_name
        else:
            raise ValueError(
                f"Unknown SmolLM model: {model_name}. Supported: {list(name_to_variant.keys())}"
            )

        print(f"Loading SmolLM variant: {variant}")

        # SmolLM models are very small, so quantization is optional
        # but can still be useful for larger variants
        model_kwargs = {"dtype": torch.bfloat16}

        if quantization:
            quantization = quantization.lower()
            if quantization == "int8":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            elif quantization == "int4":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
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

        print("[SmolLM] loading the model...")
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

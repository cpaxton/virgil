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

tinyllama_variants = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
]

name_to_variant = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tinyllama-chat": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tinyllama-intermediate": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
}


def get_tinyllama_model_names() -> list[str]:
    """Get a list of available TinyLlama model names."""
    return list(name_to_variant.keys())


class TinyLlama(Backend):
    """Use TinyLlama models for text generation."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        quantization: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        """Initialize the TinyLlama backend."""
        if model_path:
            variant = model_path
        elif model_name in name_to_variant:
            variant = name_to_variant[model_name]
        elif model_name.startswith("TinyLlama/"):
            variant = model_name
        else:
            raise ValueError(
                f"Unknown TinyLlama model: {model_name}. Supported: {list(name_to_variant.keys())}"
            )

        print(f"Loading TinyLlama variant: {variant}")

        # Default to int8 quantization for TinyLlama (very small model)
        if quantization is None:
            quantization = "int8"

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

        print("[TinyLlama] loading the model...")
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

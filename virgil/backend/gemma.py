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

variants = [
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-1-7b-it",
    "google/gemma-1-3b-it",
]

name_to_variant = {variant.split("/")[-1]: variant for variant in variants}


def get_gemma_model_names() -> list[str]:
    """Get a list of available Gemma model names."""
    return list(name_to_variant.keys())


def supports_flash_attention() -> bool:
    """Check if the current device supports Flash Attention."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return major >= 8
    return False


class Gemma(Backend):
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        quantization: Optional[str] = None,
        use_flash_attention: bool = True,
    ) -> None:
        """Initialize the Gemma backend."""
        if model_name in name_to_variant:
            variant = name_to_variant[model_name]
        elif model_name.startswith("google/"):
            variant = model_name
        else:
            raise ValueError(
                f"Unknown Gemma model: {model_name}. Supported: {list(name_to_variant.keys())}"
            )

        print(f"Loading Gemma variant: {variant}")

        # Set default quantization if not provided
        if quantization is None:
            quantization = "int8" if "2b" in variant else "int4"

        model_kwargs = {"dtype": torch.bfloat16}
        if quantization:
            print(f"[Gemma] quantizing the model to {quantization}")
            if quantization == "int8":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            elif quantization == "int4":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True
                )
            else:
                raise ValueError(f"Unknown quantization method: {quantization}")

        if supports_flash_attention() and use_flash_attention:
            print("[Gemma] using Flash Attention")
            model_kwargs["attn_implementation"] = "flash_attention_2"

        pipeline_kwargs = {}
        if torch.cuda.is_available():
            pipeline_kwargs["device_map"] = "auto"
        elif torch.backends.mps.is_available():
            pipeline_kwargs["device"] = "mps"

        print("[Gemma] loading the model...")
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

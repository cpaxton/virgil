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
from transformers import pipeline, BitsAndBytesConfig
from typing import Optional
from virgil.backend.base import Backend

variants = [
    "google/gemma-2-2b-it",
    "google/gemma-2-2b-en",
    "google/gemma-1-7b-it",
    "google/gemma-1-7b-en",
    "google/gemma-1-3b-it",
    "google/gemma-1-3b-en",
    "google/gemma-3-27b-it",
]

class Gemma(Backend):
    def __init__(self, temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True,
                 quantization: Optional[str] = "int8", use_flash_attention: bool = True,
                 variant: str = "google/gemma-3-27b-it") -> None:
        """Initialize the Gemma backend.

        Args:
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling parameter.
            do_sample (bool): Whether to sample or not.
            quantization (Optional[str]): Optional quantization method.
            use_flash_attention (bool): Whether to use Flash Attention.
        """

        if quantization is not None:
            print(f"[Gemma] quantizing the model to {quantization}")
            if quantization == "int8":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization == "int4":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            else:
                raise ValueError(f"Unknown quantization method: {quantization}")
        else:
            quantization_config = None

        model_kwargs = {"torch_dtype": torch.bfloat16}
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        if supports_flash_attention() and use_flash_attention:
            print("[Gemma] using Flash Attention from Flash-Attn")
            model_kwargs["attn_implementation"] = "flash_attention_2"

        pipeline_kwargs = {}
        if torch.backends.mps.is_available():
            pipeline_kwargs["device"] = torch.device("mps")

        print("[Gemma] loading the model...")
        self.pipe = pipeline("text-generation", model=variant, model_kwargs=model_kwargs, **pipeline_kwargs)
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        # print("[Gemma] compile the model for speed...")
        # self.pipe.model = torch.compile(self.pipe.model, mode="max-autotune", fullgraph=True)
        # print("...done")

    def __call__(self, messages, max_new_tokens: int = 256, *args, **kwargs) -> list:
        """Generate a response to a list of messages.

        Args:
            messages (List[str]): A list of messages.
            max_new_tokens (int): The maximum number of tokens to generate.

        Returns:
            List[str]: A list of generated responses.
        """
        with torch.no_grad():
            return self.pipe(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
            )


if __name__ == "__main__":
    gemma = Gemma()
    print(gemma("The meaning of life is:"))


def supports_flash_attention() -> bool:
    """Check if the current device supports Flash Attention.

    Returns:
        bool: Whether Flash Attention is supported.
    """
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major >= 8
    return False

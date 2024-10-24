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

qwen_sizes = ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]
qwen_specializations = ["Instruct", "Coder", "Math"]


class Qwen(Backend):
    """Use the Qwen model to generate responses to messages."""

    def __init__(self, model_name: Optional[str] = None, size: Optional[str] = None, specialization="Instruct", temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True, quantization: Optional[str] = "AWQ") -> None:
        if size is None:
            size = "1.5B"
        size = size.upper()
        # Check if the size is valid
        if size not in qwen_sizes:
            raise ValueError(f"Unknown size: {size}. Available sizes: {qwen_sizes}")
        # Check if the specialization is valid
        if specialization not in qwen_specializations:
            raise ValueError(f"Unknown specialization: {specialization}. Available specializations: {qwen_specializations}")
        # Check if the model name is valid
        if model_name is None:
            model_name = "Qwen/Qwen2.5-{size}-{specialization}"

        if quantization is not None:
            quantization = quantization.lower()
            if quantization in ["AWQ", "int4"]:
                if quantization == "AWQ":
                    model_name += "-AWQ"
                elif quantization == "int4":
                    model_name += "-Int4"
                else:
                    raise ValueError(f"Unknown quantization method: {quantization}")
            else:
                raise ValueError(f"Unknown quantization method: {quantization}")

        self.pipe = pipeline("text-generation", model=model_name, torch_dtype="auto", device_map="auto")
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

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
    llm = Qwen(model_name=None, size="1B", specialization="Instruct")
    print(llm("The key to life is"))

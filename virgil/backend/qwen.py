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
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from virgil.backend.base import Backend
import virgil.utils.log as logger

qwen_sizes = ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]
qwen_specializations = ["Instruct", "Coder", "Math"]


class Qwen(Backend):
    """Use the Qwen model to generate responses to messages."""

    def __init__(self, model_name: Optional[str] = None, size: Optional[str] = None, specialization="Instruct", temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True, quantization: Optional[str] = "int4", model_path: str = None) -> None:
        """Initialize the Qwen backend.

        Args:
            model_name (Optional[str]): The name of the model to use.
            size (Optional[str]): The size of the model to use.
            specialization (str): The specialization of the model to use.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling parameter.
            do_sample (bool): Whether to sample or not.
            quantization (Optional[str]): Optional quantization method.
            model_path (str): The path to the model weights. Optional; HuggingFace will download the model if not provided.
        """
        if size is None:
            size = "1.5B"
        size = size.upper()
        # Check if the size is valid
        if size not in qwen_sizes:
            raise ValueError(f"Unknown size: {size}. Available sizes: {qwen_sizes}")
        # Check if the specialization is valid
        if specialization not in qwen_specializations:
            raise ValueError(f"Unknown specialization: {specialization}. Available specializations: {qwen_specializations}")

        model_kwargs = {"torch_dtype": "auto"}
        if model_path is not None and len(model_path) > 0:
            # model_kwargs["model_path"] = model_path
            model_id = model_path
        else:
            # Check if the model name is valid
            if model_name is None:
                model_name = f"Qwen/Qwen2.5-{size}-{specialization}"
            # model_kwargs["model"] = model_name
            model_id = model_name

        quantization_config = None
        if quantization is not None:
            quantization = quantization.lower()
            # Note: there were supposed to be other options but this is the only one that worked this way
            if quantization == "awq":
                model_kwargs["torch_dtype"] = torch.float16
                try:
                    import awq
                except ImportError:
                    logger.error("To use quantization, please install the autoawq package.")
                    quantization = ""
                if quantization == "awq":
                    model_name += "-AWQ"
                elif len(quantization) > 0:
                    raise ValueError(f"Unknown quantization method: {quantization}")
            elif quantization in ["int8", "int4"]:
                try:
                    import bitsandbytes  # noqa: F401
                except ImportError:
                    raise ImportError("bitsandbytes required for int4/int8 quantization: pip install bitsandbytes")

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=(quantization == "int4"),
                    load_in_8bit=(quantization == "int8"),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                model_kwargs["quantization_config"] = quantization_config
            else:
                raise ValueError(f"Unknown quantization method: {quantization}")

        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        if torch.backends.mps.is_available():
            model_kwargs["device"] = "mps"  # Metal Performance Shaders for Apple GPUas

        # Set up optional quantization
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config


        # Load the model and tokenizer with the quantization configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None,
            torch_dtype=torch.bfloat16 if quantization_config is None else None  # added to avoid errors when no quantization
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs=model_kwargs # keep this
        )

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
    llm = Qwen(model_name=None, size="7B", specialization="Instruct", quantization="int4")
    print(llm("The key to life is"))

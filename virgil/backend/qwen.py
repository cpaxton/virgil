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

from typing import Optional, Tuple, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from virgil.backend.base import Backend

qwen25_sizes = ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"]
qwen_specializations = ["Instruct", "Coder", "Math", "Deepseek"]
qwen30_sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]


def get_qwen_model_names() -> list[str]:
    """Get a list of available Qwen model names."""
    names = []
    # Qwen 2.5
    for size in qwen25_sizes:
        for spec in qwen_specializations:
            names.append(f"qwen2.5-{size}-{spec}".lower())
    # Qwen 3
    for size in qwen30_sizes:
        names.append(f"qwen3-{size}".lower())
    return names


class Qwen(Backend):
    """Use the Qwen model to generate responses to messages."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        quantization: Optional[str] = "int4",
        model_path: Optional[str] = None,
        compile_model: bool = True,
        repetition_penalty: float = 1.1,
    ) -> None:
        """Initialize the Qwen backend."""
        model_id = self._get_model_id(model_name)
        if model_path:
            model_id = model_path

        print(f"Loading model: {model_id}")

        model_kwargs = {"dtype": "auto"}
        if quantization:
            quantization = quantization.lower()
            if quantization == "int4":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,  # Nested quantization
                    bnb_4bit_quant_type="nf4",  # Optimal quantization type
                    bnb_4bit_compute_dtype=torch.bfloat16,  # Faster computation
                )
            elif quantization == "int8":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_use_double_quant=True,  # Nested quantization
                    bnb_8bit_compute_dtype=torch.bfloat16,  # Faster computation
                )
            else:
                raise ValueError(f"Unknown quantization method: {quantization}")

        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        elif torch.backends.mps.is_available():
            model_kwargs["device_map"] = "mps"

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty

        # Compile model for faster inference (PyTorch 2.0+)
        if compile_model and hasattr(torch, "compile"):
            try:
                print("[Qwen] Compiling model for faster inference...")
                self.model = torch.compile(
                    self.model, mode="reduce-overhead", fullgraph=False
                )
                print("[Qwen] Model compilation successful")
            except Exception as e:
                print(f"[Qwen] Model compilation failed (continuing without): {e}")
        elif compile_model:
            print("[Qwen] torch.compile not available (requires PyTorch 2.0+)")

        # Enable KV cache support
        self._supports_kv_cache = True

        # Track processed conversation for incremental generation
        self._cached_input_ids = None

        # Warmup the model for faster first inference
        self._warmup_model()

    def reset_cache(self):
        """Reset the KV cache state."""
        self._cached_input_ids = None

    def _warmup_model(self):
        """Warmup the model with a dummy forward pass to optimize first inference."""
        try:
            print("[Qwen] Warming up model...")
            device = next(self.model.parameters()).device
            # Create a small dummy input
            dummy_input = self.tokenizer(
                "Hello", return_tensors="pt", add_special_tokens=False
            ).to(device)
            with torch.inference_mode():
                # Single forward pass to warm up CUDA kernels, memory allocators, etc.
                _ = self.model.generate(
                    dummy_input["input_ids"],
                    max_new_tokens=1,
                    do_sample=False,
                    use_cache=True,
                )
            print("[Qwen] Model warmup complete")
        except Exception as e:
            print(f"[Qwen] Model warmup failed (continuing): {e}")

    def _get_model_id(self, name: str) -> str:
        """Construct the HuggingFace model ID from the model name."""
        if name.startswith("qwen/"):
            return name

        parts = name.lower().split("-")
        if len(parts) == 1 and parts[0] == "qwen":  # Default model
            return "Qwen/Qwen3-8B"

        release_part = parts[0]
        size_part = parts[1].upper()

        if release_part == "qwen3":
            if size_part not in qwen30_sizes:
                raise ValueError(
                    f"Unknown size: {size_part}. Available for Qwen 3: {qwen30_sizes}"
                )
            return f"Qwen/Qwen3-{size_part}"
        elif release_part == "qwen2.5":
            if size_part not in qwen25_sizes:
                raise ValueError(
                    f"Unknown size: {size_part}. Available for Qwen 2.5: {qwen25_sizes}"
                )
            spec_part = parts[2].capitalize()
            if spec_part not in qwen_specializations:
                raise ValueError(
                    f"Unknown specialization: {spec_part}. Available: {qwen_specializations}"
                )
            if spec_part == "Deepseek":
                return f"deepseek-ai/DeepSeek-R1-Distill-Qwen-{size_part}"
            else:
                return f"Qwen/Qwen2.5-{size_part}-{spec_part}"
        else:
            raise ValueError(f"Unknown Qwen release from name: {name}")

    def __call__(self, messages, max_new_tokens: int = 512, *args, **kwargs) -> list:
        """Generate a response to a list of messages."""
        with torch.inference_mode():  # More efficient than no_grad() for inference
            return self.pipe(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                repetition_penalty=self.repetition_penalty,
            )

    def generate_with_cache(
        self,
        messages,
        max_new_tokens: int = 512,
        past_key_values: Optional[Any] = None,
        *args,
        **kwargs,
    ) -> Tuple[list, Optional[Any]]:
        """Generate a response with KV cache support.

        Args:
            messages: The messages to generate a response for.
            max_new_tokens: Maximum number of new tokens to generate.
            past_key_values: Previous KV cache (not used in current implementation).

        Returns:
            Tuple of (output, new_past_key_values).
        """
        # Use pipeline with use_cache=True (default) for efficient generation
        with torch.inference_mode():  # More efficient than no_grad() for inference
            output = self.pipe(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                repetition_penalty=self.repetition_penalty,
            )

        return output, None

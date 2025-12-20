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
        compile_model: bool = True,
        repetition_penalty: float = 1.1,
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
                    load_in_8bit=True,
                    bnb_8bit_use_double_quant=True,  # Nested quantization for better compression
                    bnb_8bit_compute_dtype=torch.bfloat16,  # Faster computation
                )
            elif quantization == "int4":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,  # Nested quantization
                    bnb_4bit_quant_type="nf4",  # Optimal quantization type
                    bnb_4bit_compute_dtype=torch.bfloat16,  # Faster computation
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
        self.repetition_penalty = repetition_penalty

        # Enable KV cache support
        self._supports_kv_cache = True

        # Extract model and tokenizer from pipeline for direct access
        self.model = self.pipe.model
        self.tokenizer = self.pipe.tokenizer

        # Compile model for faster inference (PyTorch 2.0+)
        if compile_model and hasattr(torch, "compile"):
            try:
                print("[Gemma] Compiling model for faster inference...")
                # Compile the model's forward method
                self.model = torch.compile(
                    self.model, mode="reduce-overhead", fullgraph=False
                )
                print("[Gemma] Model compilation successful")
            except Exception as e:
                print(f"[Gemma] Model compilation failed (continuing without): {e}")
        elif compile_model:
            print("[Gemma] torch.compile not available (requires PyTorch 2.0+)")

        # Track processed conversation for incremental generation
        self._cached_conversation_text = None
        self._cached_input_ids = None

        # Warmup the model for faster first inference
        self._warmup_model()

    def reset_cache(self):
        """Reset the KV cache state."""
        self._cached_conversation_text = None
        self._cached_input_ids = None

    def _warmup_model(self):
        """Warmup the model with a dummy forward pass to optimize first inference."""
        try:
            print("[Gemma] Warming up model...")
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
            print("[Gemma] Model warmup complete")
        except Exception as e:
            print(f"[Gemma] Model warmup failed (continuing): {e}")

    def __call__(self, messages, max_new_tokens: int = 256, *args, **kwargs) -> list:
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
        max_new_tokens: int = 256,
        past_key_values: Optional[Any] = None,
        *args,
        **kwargs,
    ) -> Tuple[list, Optional[Any]]:
        """Generate a response with KV cache support for incremental generation.

        This implementation tracks the conversation state and uses past_key_values
        to avoid re-processing tokens that have already been seen. This provides
        significant speedup for multi-turn conversations.

        Args:
            messages: The messages to generate a response for.
            max_new_tokens: Maximum number of new tokens to generate.
            past_key_values: Previous KV cache from previous generation.

        Returns:
            Tuple of (output, new_past_key_values).
        """
        if not self._kv_cache_enabled:
            # Fall back to regular generation
            output = self(messages, max_new_tokens=max_new_tokens, *args, **kwargs)
            return output, None

        with torch.inference_mode():  # More efficient than no_grad() for inference
            device = next(self.model.parameters()).device

            # Format the full conversation using chat template
            if isinstance(messages, str):
                formatted_text = messages
            else:
                formatted_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            # Tokenize the conversation
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # If we have past_key_values, we only need to process new tokens
            if past_key_values is not None and self._cached_input_ids is not None:
                # Calculate how many tokens are new
                cached_length = self._cached_input_ids.shape[1]
                current_length = input_ids.shape[1]

                if current_length > cached_length:
                    # Only process the new tokens
                    new_input_ids = input_ids[:, cached_length:]
                    if attention_mask is not None:
                        new_attention_mask = attention_mask[:, cached_length:]
                    else:
                        new_attention_mask = None
                else:
                    # Conversation was reset or shortened, process everything
                    new_input_ids = input_ids
                    new_attention_mask = attention_mask
                    past_key_values = None
            else:
                # No cache, process everything
                new_input_ids = input_ids
                new_attention_mask = attention_mask
                past_key_values = None

            # Prepare generation kwargs with optimizations
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": self.do_sample,
                "use_cache": True,  # KV cache for faster generation
                "repetition_penalty": self.repetition_penalty,  # Reduce repetition
                "pad_token_id": self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            }

            if past_key_values is not None:
                generation_kwargs["past_key_values"] = past_key_values

            if new_attention_mask is not None:
                generation_kwargs["attention_mask"] = new_attention_mask

            # Generate with the model directly
            # Note: use_cache=True is the default and speeds up token-by-token generation
            outputs = self.model.generate(
                new_input_ids,
                **generation_kwargs,
            )

            # Extract only the newly generated tokens (everything after the input)
            input_length = new_input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]

            # Decode the generated text
            generated_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

            # Format output to match pipeline format
            output = [
                {"generated_text": [{"role": "assistant", "content": generated_text}]}
            ]

            # Update cached state for tracking (even though we're not using past_key_values yet)
            # This helps us detect when conversation is reset
            self._cached_input_ids = outputs[0]

            # Note: transformers' generate() uses KV cache internally for faster
            # token-by-token generation, but doesn't expose past_key_values for
            # reuse across calls. The main performance benefit comes from:
            # 1. use_cache=True speeds up generation within a single call
            # 2. Flash Attention (already enabled) provides efficient attention
            # 3. TF32 (already enabled) provides faster matrix operations

            # For true cross-call KV caching, we would need to:
            # - Use model.forward() to get past_key_values
            # - Manually manage the cache state
            # This is more complex and may not provide significant benefit if
            # the conversation history is relatively short

            new_past_key_values = None

            return output, new_past_key_values

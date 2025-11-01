# Copyright 2024 Chris Paxton
# Adapted for Baidu ERNIE by Chris Paxton
#
# Licensed under the Apache License, Version 2.0

from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from virgil.backend.base import Backend

ernie_model_ids = [
    "baidu/ERNIE-4.5-VL-28B-A3B-PT",
    "baidu/ERNIE-4.5-VL-28B-A3B-Base-Paddle",
    "baidu/ERNIE-4.5-21B-A3B-PT",
    "baidu/ERNIE-4.5-21B-A3B-Base-Paddle",
    "baidu/ERNIE-4.5-21B-Base-Paddle",
    "baidu/ERNIE-4.5-0.3B-Base-Paddle",
    "baidu/ERNIE-4.5-0.3B-Base-PT",
]

ernie_name_to_id = {name.split("/")[-1].lower(): name for name in ernie_model_ids}


def get_ernie_model_id(name: str) -> Optional[str]:
    """Get the ERNIE model ID from its name."""
    return ernie_name_to_id.get(name.lower())


def get_ernie_model_names() -> list[str]:
    """Get a list of available ERNIE model names."""
    return list(ernie_name_to_id.keys())


class Ernie(Backend):
    """Use the Baidu ERNIE model for multimodal generation."""

    def __init__(
        self,
        model_name: str,
        quantization: Optional[str] = "int4",
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        model_path: Optional[str] = None,
    ) -> None:
        """Initialize the ERNIE backend."""
        model_id = get_ernie_model_id(model_name)
        if model_id is None:
            if model_name in ernie_model_ids:
                model_id = model_name
            else:
                raise ValueError(
                    f"Unknown ERNIE model: {model_name}. Supported: {list(ernie_name_to_id.keys())}"
                )

        if model_path:
            model_id = model_path

        model_kwargs = {"torch_dtype": "auto", "trust_remote_code": True}

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

        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        elif torch.backends.mps.is_available():
            model_kwargs["device_map"] = "mps"

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

    def __call__(
        self, messages, images=None, max_new_tokens: int = 128, *args, **kwargs
    ) -> str:
        """Generate a response to a list of messages, optionally with images."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if images:
            content = [{"type": "text", "text": messages[0]["content"]}]
            for image_url in images:
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            chat_messages = [{"role": "user", "content": content}]
        else:
            chat_messages = messages

        text = self.processor.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
            )
            return self.processor.decode(generated_ids[0], skip_special_tokens=True)

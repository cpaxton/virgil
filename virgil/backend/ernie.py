# Copyright 2024 Chris Paxton
# Adapted for Baidu ERNIE by [Your Name]
#
# Licensed under the Apache License, Version 2.0

import torch
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
)
from typing import Optional

from virgil.backend.base import Backend

ernie_model_ids = [
    "baidu/ERNIE-4.5-VL-28B-A3B-PT",  # Post-trained
    "baidu/ERNIE-4.5-VL-28B-A3B-Base-Paddle",  # Base pre-trained
]

ernie_name_to_id = {
    "ernie-4.5-vl-28b-a3b-pt": "baidu/ERNIE-4.5-VL-28B-A3B-PT",
    "ernie-4.5-vl-28b-a3b-base-paddle": "baidu/ERNIE-4.5-VL-28B-A3B-Base-Paddle",
}


def get_ernie_model_id(name: str) -> Optional[str]:
    """Get the ERNIE model ID from its name.

    Args:
        name (str): The name of the ERNIE model.

    Returns:
        Optional[str]: The HuggingFace model ID if found, otherwise None.
    """
    return ernie_name_to_id.get(name.lower(), None)


def get_ernie_model_names() -> list[str]:
    """Get a list of available ERNIE model names.

    Returns:
        list[str]: List of ERNIE model names.
    """
    return list(ernie_name_to_id.keys())


class Ernie(Backend):
    """Use the Baidu ERNIE 4.5-VL-28B-A3B model for multimodal generation."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        quantization: Optional[str] = "int4",
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        model_path: Optional[str] = None,
    ) -> None:
        """Initialize the ERNIE backend.

        Args:
            model_name (Optional[str]): The HuggingFace model ID to use.
            quantization (Optional[str]): Quantization method ("int4", "int8", or None).
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling parameter.
            do_sample (bool): Whether to sample or not.
            model_path (str): Local path to model weights (optional).
        """
        if model_name is None:
            model_name = ernie_model_ids[0]

        model_id = model_path or model_name

        model_kwargs = {"torch_dtype": "auto", "trust_remote_code": True}

        quantization_config = None
        if quantization is not None:
            quantization = quantization.lower()
            if quantization in ["int8", "int4"]:
                try:
                    import bitsandbytes  # noqa: F401
                except ImportError:
                    raise ImportError(
                        "bitsandbytes required for int4/int8 quantization: pip install bitsandbytes"
                    )
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=(quantization == "int4"),
                    load_in_8bit=(quantization == "int8"),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model_kwargs["quantization_config"] = quantization_config
            else:
                raise ValueError(f"Unknown quantization method: {quantization}")

        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        elif torch.backends.mps.is_available():
            model_kwargs["device"] = "mps"

        # Load model, tokenizer, and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

    def __call__(
        self, messages, images=None, max_new_tokens: int = 128, *args, **kwargs
    ) -> list:
        """
        Generate a response to a list of messages, optionally with images.

        Args:
            messages (List[str] or List[dict]): List of messages or multimodal message dicts.
            images (List[str] or None): List of image URLs or local paths.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            List[str]: Generated responses.
        """
        # Prepare input for ERNIE's multimodal API
        if isinstance(messages, str):
            messages = [messages]

        # For multimodal, use the chat template and processor
        chat_messages = []
        if images:
            chat_messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": messages[0]},
                        {"type": "image_url", "image_url": {"url": images[0]}},
                    ],
                }
            )
        else:
            chat_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": messages[0]}],
                }
            )

        text = self.processor.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        image_inputs, video_inputs = self.processor.process_vision_info(chat_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
            )
            output_text = self.processor.decode(generated_ids)
        return output_text


if __name__ == "__main__":
    llm = Ernie(model_name="baidu/ERNIE-4.5-VL-28B-A3B-PT", quantization="int4")
    print(
        llm(
            "Describe the image.",
            images=[
                "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example1.jpg"
            ],
        )
    )

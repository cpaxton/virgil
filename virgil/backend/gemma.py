# # (c) 2024 by Chris Paxton

import torch
from transformers import pipeline

from .base import Backend


class Gemma(Backend):
    def __init__(self):
        self.pipe = pipeline(
            "text-generation",
            model="google/gemma-2-2b-it",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",  # replace with "mps" to run on a Mac device
        )

    def __call__(self, messages, max_new_tokens: int = 256, *args, **kwargs):
        return self.pipe(messages, max_new_tokens=max_new_tokens)

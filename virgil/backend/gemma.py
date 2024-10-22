# # (c) 2024 by Chris Paxton

import torch
from transformers import pipeline

from virgil.backend.base import Backend


class Gemma(Backend):
    def __init__(self, temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True, quantization: Optional[str] = "int8") -> None:
        """Initialize the Gemma backend.

        Args:
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling parameter.
            do_sample (bool): Whether to sample or not.
            quantization (Optional[str]): Optional quantization method.
        """

        if quantization is not None:
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

        self.pipe = pipeline(
            "text-generation",
            model="google/gemma-2-2b-it",
            model_kwargs=model_kwargs,
            device="cuda",  # replace with "mps" to run on a Mac device
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
    gemma = Gemma()
    print(gemma("The meaning of life is:"))

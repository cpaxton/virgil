# # Copyright 2024 Chris Paxton
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# # (c) 2024 by Chris Paxton

import torch
from transformers import pipeline
from typing import Optional

from virgil.backend.base import Backend


class Llama(Backend):
    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True) -> None:
        if model_name is None:
            model_name = "meta-llama/Llama-3.2-1B"
        self.pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")
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
    llama = Llama()
    print(llama("The key to life is"))

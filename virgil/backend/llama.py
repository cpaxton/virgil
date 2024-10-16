import torch
from transformers import pipeline
from typimg import Optional

class Llama(Backend):
    def __init__(self, model_id: Optional[str] = None, temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True) -> None:
        if model_id is None:
            model_id = "meta-llama/Llama-3.2-1B"
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto")
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
            return self.pipe(messages, max_new_tokens=max_new_tokens,
                             temperature=self.temperature,
                             top_p=self.top_p,
                             do_sample=self.do_sample,)

if __name__ == '__main__':
    llama = Llama()
    print(llama("The key to life is"))

# Description: This file contains the code for generating a meme based on a prompt.
#
import pkg_resources

from virgil.backend import get_backend

def load_prompt() -> str:
    """Load the prompt from the prompt.txt file.

    Returns:
        str: The prompt text.
    """

    file_path = pkg_resources.resource_filename('virgil.meme', 'prompt.txt')

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    return content

class MemeGenerator:
    def __init__(self, backend: str = "gemma"):
        self.backend = get_backend(backend)
        self.base_prompt = load_prompt()

    def generate_meme(self, prompt: str) -> str:
        """Generate a meme based on the prompt.

        Args:
            prompt (str): The prompt for the meme.

        Returns:
            str: The generated meme.
        """
        
        prompt = self.base_prompt.format(prompt=prompt)

        return self.backend.generate_meme(prompt)

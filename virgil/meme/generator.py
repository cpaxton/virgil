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

# Description: This file contains the code for generating a meme based on a prompt.
#
import pkg_resources

from virgil.backend import get_backend


def load_prompt() -> str:
    """Load the prompt from the prompt.txt file.

    Returns:
        str: The prompt text.
    """

    file_path = pkg_resources.resource_filename("virgil.meme", "prompt.txt")

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    return content


class MemeGenerator:
    def __init__(self, backend: str = "gemma", memory_manager=None):
        self.backend = get_backend(backend)
        self.base_prompt = load_prompt()
        self.memory_manager = memory_manager

    def generate_meme(self, prompt: str) -> str:
        """Generate a meme based on the prompt.

        Args:
            prompt (str): The prompt for the meme.

        Returns:
            str: The generated meme.
        """
        # Get relevant memories if memory manager is available
        memories_text = ""
        if self.memory_manager:
            relevant_memories = self.memory_manager.get_memories_for_query(prompt)
            if relevant_memories:
                memories_text = f"\n\nRelevant memories for this meme:\n----\n{relevant_memories}\n----\nEnd memories.\n"
            else:
                memories_text = "\n\n(No relevant memories found)\n"

        prompt = self.base_prompt.format(prompt=prompt, memories=memories_text)

        return self.backend.generate_meme(prompt)

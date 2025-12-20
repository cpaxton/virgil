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
from pathlib import Path

from virgil.backend import get_backend


def load_prompt() -> str:
    """Load the prompt from the prompt.txt file.

    Returns:
        str: The prompt text.
    """
    # Get the directory where this file is located
    current_dir = Path(__file__).parent
    file_path = current_dir / "prompt.txt"

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
                memories_text = f"\n\nRelevant memories for this meme (retrieved dynamically based on context):\n----\n{relevant_memories}\n----\nEnd memories.\n"
            # If no relevant memories, leave empty (don't show "No relevant memories found")

        formatted_prompt = self.base_prompt.format(
            prompt=prompt, memories=memories_text
        )

        # Convert prompt to messages format for backend
        messages = [{"role": "user", "content": formatted_prompt}]

        # Generate using backend
        result = self.backend(messages, max_new_tokens=512)

        # Extract text from result - backends return list of dicts with "generated_text"
        # Format: [{"generated_text": [{"role": "assistant", "content": "..."}]}]
        # Same format as ChatWrapper uses
        if isinstance(result, str):
            return result.strip()
        elif isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                generated_text = result[0].get("generated_text", [])
                if isinstance(generated_text, list) and len(generated_text) > 0:
                    # Get the last message (assistant response)
                    last_msg = generated_text[-1]
                    if isinstance(last_msg, dict):
                        return last_msg.get("content", str(last_msg)).strip()
                    else:
                        return str(last_msg).strip()
                elif isinstance(generated_text, str):
                    return generated_text.strip()
                else:
                    return str(generated_text).strip()
            else:
                return str(result[0]).strip()
        else:
            return str(result).strip()

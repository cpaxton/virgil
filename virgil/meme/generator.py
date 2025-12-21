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
from typing import Union, Optional

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
    def __init__(
        self,
        backend: Union[str, object, None] = None,
        memory_manager=None,
        backend_name: Optional[str] = None,
    ):
        """
        Initialize the MemeGenerator.

        Args:
            backend: Backend instance to reuse (avoids loading duplicate models in GPU memory),
                    or backend name string (for backward compatibility). If None, uses backend_name.
            memory_manager: Optional memory manager for RAG support.
            backend_name: Backend name string to use if backend is None. Defaults to "gemma".
                        Only used if backend is None.

        Note: For backward compatibility, if backend is a string, it's treated as a backend name.
              To pass a backend instance, ensure it's an object (not a string).
        """
        # Handle backward compatibility: if backend is a string, treat it as backend_name
        if isinstance(backend, str):
            backend_name = backend
            backend = None

        # Reuse backend instance if provided (to avoid duplicate GPU memory usage)
        # Otherwise create a new backend instance
        if backend is not None:
            self.backend = backend
        else:
            # Use backend_name if provided, otherwise default to "gemma"
            final_backend_name = backend_name if backend_name is not None else "gemma"
            self.backend = get_backend(final_backend_name)
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

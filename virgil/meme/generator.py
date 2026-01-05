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
from typing import Union, Optional, Tuple
import re

from virgil.backend import get_backend
from PIL import Image


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
        image_service=None,
    ):
        """
        Initialize the MemeGenerator.

        Args:
            backend: Backend instance to reuse (avoids loading duplicate models in GPU memory),
                    or backend name string (for backward compatibility). If None, uses backend_name.
            memory_manager: Optional memory manager for RAG support.
            backend_name: Backend name string to use if backend is None. Defaults to "gemma".
                        Only used if backend is None.
            image_service: Optional image service to use for generating meme template images.
                          If provided, will be used to generate images. If None, only text will be generated.

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
        self.image_service = image_service

    def generate_meme(self, prompt: str) -> Tuple[Optional[Image.Image], str]:
        """Generate a meme based on the prompt.

        Args:
            prompt (str): The prompt for the meme.

        Returns:
            Tuple[Optional[Image.Image], str]: A tuple of (image, caption).
                - image: The generated meme image template, or None if image_service is not available
                - caption: The meme caption text
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
        response_text = ""
        if isinstance(result, str):
            response_text = result.strip()
        elif isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                generated_text = result[0].get("generated_text", [])
                if isinstance(generated_text, list) and len(generated_text) > 0:
                    # Get the last message (assistant response)
                    last_msg = generated_text[-1]
                    if isinstance(last_msg, dict):
                        response_text = last_msg.get("content", str(last_msg)).strip()
                    else:
                        response_text = str(last_msg).strip()
                elif isinstance(generated_text, str):
                    response_text = generated_text.strip()
                else:
                    response_text = str(generated_text).strip()
            else:
                response_text = str(result[0]).strip()
        else:
            response_text = str(result).strip()

        # Remove <think> tags if present (both closed and unclosed)
        response_text = re.sub(
            r"<think>.*?</think>", "", response_text, flags=re.DOTALL
        )
        # Also remove unclosed <think> tags
        response_text = re.sub(r"<think>.*$", "", response_text, flags=re.DOTALL)
        response_text = response_text.strip()

        # Parse IMAGE: and CAPTION: from response
        image_prompt = None
        caption = response_text  # Default to full response if parsing fails

        # Try to extract IMAGE: and CAPTION: sections
        image_match = re.search(
            r"IMAGE:\s*(.+?)(?=CAPTION:|$)", response_text, re.DOTALL | re.IGNORECASE
        )
        caption_match = re.search(
            r"CAPTION:\s*(.+?)$", response_text, re.DOTALL | re.IGNORECASE
        )

        if image_match:
            image_prompt = image_match.group(1).strip()
        if caption_match:
            caption = caption_match.group(1).strip()
        elif image_match:
            # If we have IMAGE but no CAPTION, use everything after IMAGE as caption
            caption = response_text[image_match.end() :].strip()

        # Generate image if image_service is available
        image = None
        if self.image_service and image_prompt:
            try:
                image = self.image_service.generate_image(image_prompt)
            except Exception as e:
                print(f"Warning: Failed to generate meme image: {e}")
                # Fallback to using prompt as image description if parsing worked
                if image_prompt:
                    try:
                        image = self.image_service.generate_image(prompt)
                    except Exception:
                        pass

        return (image, caption)

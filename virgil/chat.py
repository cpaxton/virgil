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

# (c) 2024 by Chris Paxton

import click
import timeit
from termcolor import colored

from virgil.backend import Backend, backend_list


class ChatWrapper:
    def __init__(
        self,
        backend: Backend,
        max_history_length: int = 50,
        preserve: int = 2,
        use_kv_cache: bool = True,
    ) -> None:
        self.backend = backend
        self.max_history_length = max_history_length
        self.preserve = preserve
        if self.preserve > self.max_history_length:
            raise ValueError(
                f"Preserve must be less than or equal to max_history_length. Got preserve={self.preserve} and max_history_length={self.max_history_length}"
            )
        self.conversation_history = []
        # KV cache management
        self.use_kv_cache = use_kv_cache and self.backend.supports_kv_cache()
        if self.use_kv_cache:
            self.backend.enable_kv_cache(True)
            self._past_key_values = None
            self._processed_message_count = 0
        else:
            self._past_key_values = None
            self._processed_message_count = 0

    def add_conversation_history(self, role: str, content: str, verbose: bool = False):
        """Add a message to the conversation history.

        Args:
            role (str): The role of the speaker.
            content (str): The content of the message.
        """

        # Roles must alternate
        added = False
        if len(self.conversation_history) > 0:
            # Get previous role
            prev_role = self.conversation_history[-1]["role"]
            if prev_role == role:
                # Just concatenate it
                self.conversation_history[-1]["content"] += "\n" + content
                added = True

        if not added:
            self.conversation_history.append({"role": role, "content": content})

            # Trim the conversation history, preserving the first `preserve` messages
            # TODO: handle self.preserve > 0
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history = (
                    self.conversation_history[: self.preserve]
                    + self.conversation_history[
                        -(self.max_history_length - self.preserve) :
                    ]
                )

        if verbose:
            print("Conversation history:")
            for i, message in enumerate(self.conversation_history):
                print(f"{i} {message['role']}: {message['content']}")

    def clear(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self._past_key_values = None
        self._processed_message_count = 0

    def __len__(self):
        return len(self.conversation_history)

    def prompt(
        self,
        msg: str,
        verbose: bool = False,
        max_new_tokens: int = 2000,
        assistant_history_prefix: str = "",
    ) -> str:
        """Prompt the LLM with a message.

        Args:
            msg (str): The message to prompt the LLM with.
        """

        if verbose:
            print()

        self.add_conversation_history("user", msg)

        messages = self.conversation_history.copy()
        t0 = timeit.default_timer()

        # Use KV cache if available and enabled
        if self.use_kv_cache:
            # Check if we need to reset cache (history was trimmed)
            current_message_count = len(self.conversation_history)
            if current_message_count < self._processed_message_count:
                # History was trimmed, reset cache
                if verbose:
                    print(colored("History trimmed, resetting KV cache", "yellow"))
                self._past_key_values = None
                self._processed_message_count = 0
                # Reset backend cache if it has a reset method
                if hasattr(self.backend, "reset_cache"):
                    self.backend.reset_cache()

            # Use incremental generation with KV cache
            outputs, new_past_key_values = self.backend.generate_with_cache(
                messages,
                max_new_tokens=max_new_tokens,
                past_key_values=self._past_key_values,
            )

            # Update cache for next time
            if new_past_key_values is not None:
                self._past_key_values = new_past_key_values
                self._processed_message_count = (
                    current_message_count + 1
                )  # +1 for assistant response
        else:
            # Fall back to regular generation
            outputs = self.backend(messages, max_new_tokens=max_new_tokens)

        t1 = timeit.default_timer()

        if isinstance(outputs, str):
            assistant_response = outputs.strip()
        else:
            assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

        # Add the assistant's response to the conversation history
        self.add_conversation_history(
            "assistant", assistant_history_prefix + assistant_response
        )

        if verbose:
            # Print the assistant's response
            print()
            print(colored("User prompt:\n", "green") + msg)
            print()
            print(colored("Response:\n", "blue") + assistant_response)
            print("----------------")
            print(f"Generator time taken: {t1 - t0:.2f} seconds")
            if self.use_kv_cache:
                print(colored("KV cache: enabled", "green"))
            print("----------------")

        return assistant_response


@click.command()
@click.option(
    "--max-history-length",
    default=50,
    help="The maximum length of the conversation history.",
)
@click.option(
    "--preserve",
    default=2,
    help="The number of messages to preserve in the conversation history.",
)
@click.option("--verbose", is_flag=True, help="Print verbose output.")
@click.option(
    "--backend",
    "--model",
    default="gemma-2-2b-it",
    help="The backend to use.",
    type=click.Choice(backend_list, case_sensitive=False),
)
@click.option(
    "--prompt",
    default="",
    help="The prompt (as a text file) to start the conversation with.",
)
@click.option("--path", default="", help="Optional path to model weights.")
def main(
    max_history_length: int,
    preserve: int,
    verbose: bool,
    backend: str,
    prompt: str,
    path: str = "",
) -> None:
    from virgil.backend import get_backend

    kwargs = {}
    if len(path) > 0:
        kwargs["model_path"] = path
    backend = get_backend(backend, **kwargs)
    chat = ChatWrapper(
        backend=backend, max_history_length=max_history_length, preserve=preserve
    )

    prompt_text = None
    if len(prompt) > 0:
        with open(prompt, "r") as f:
            prompt_text = f.read()

    print("Chatting with the assistant. Leave empty to exit.")
    while True:
        if prompt_text is not None:
            user_input = prompt_text
            prompt_text = None
            print(colored("Prompt: ", "green"), user_input)
        else:
            user_input = input(colored("You: ", "green"))
        if not user_input:
            break
        res = chat.prompt(user_input, verbose=verbose)
        print(colored("Assistant:", "blue"), res)


if __name__ == "__main__":
    main()

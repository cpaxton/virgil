# # (c) 2024 by Chris Paxton

import timeit
from termcolor import colored

from virgil.backend import Backend


class ChatWrapper:
    def __init__(self, backend: Backend, max_history_length: int = 50, preserve: int = 2) -> None:
        self.backend = backend
        self.max_history_length = max_history_length
        self.preserve = preserve
        if self.preserve > self.max_history_length:
            raise ValueError(f"Preserve must be less than or equal to max_history_length. Got preserve={self.preserve} and max_history_length={self.max_history_length}")
        self.conversation_history = []

    def add_conversation_history(self, role: str, content: str):
        """Add a message to the conversation history.

        Args:
            role (str): The role of the speaker.
            content (str): The content of the message.
        """
        self.conversation_history.append({"role": role, "content": content})

        # Trim the conversation history, preserving the first `preserve` messages
        # TODO: handle self.preserve > 0
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[:self.preserve] + self.conversation_history[-(self.max_history_length - self.preserve):]

    def clear(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def __len__(self):
        return len(self.conversation_history)

    def prompt(self, msg: str, verbose: bool = False) -> str:
        """Prompt the LLM with a message.

        Args:
            msg (str): The message to prompt the LLM with.
        """

        if verbose:
            print()

        self.add_conversation_history("user", msg)

        messages = self.conversation_history.copy()
        t0 = timeit.default_timer()
        outputs = self.backend(messages, max_new_tokens=256)
        t1 = timeit.default_timer()
        assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

        # Add the assistant's response to the conversation history
        self.add_conversation_history("assistant", assistant_response)

        if verbose:
            # Print the assistant's response
            print()
            print(colored("User prompt:\n", "green") + msg)
            print()
            print(colored("Response:\n", "blue") + assistant_response)
            print("----------------")
            print(f"Generator time taken: {t1-t0:.2f} seconds")
            print("----------------")

        return assistant_response

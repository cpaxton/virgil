# # (c) 2024 by Chris Paxton

import timeit
from typing import Optional
from termcolor import colored

from virgil.backend import Backend
from virgil.parser import Parser


class ChatWrapper:
    def __init__(self, backend: Backend, parser: Optional[Parser] = None) -> None:
        self.backend = backend
        self.conversation_history = []

    def add_conversation_history(self, role: str, content: str):
        """Add a message to the conversation history.

        Args:
            role (str): The role of the speaker.
            content (str): The content of the message.
        """
        self.conversation_history.append({"role": role, "content": content})

    def prompt(self, msg: str) -> str:
        """Prompt the LLM with a message.

        Args:
            msg (str): The message to prompt the LLM with.
        """

        print()

        self.add_conversation_history("user", msg)

        messages = self.conversation_history.copy()
        t0 = timeit.default_timer()
        outputs = self.backend(messages, max_new_tokens=256)
        t1 = timeit.default_timer()
        assistant_response = outputs[0]["generated_text"][-1]["content"].strip()

        # Add the assistant's response to the conversation history
        self.add_conversation_history("assistant", assistant_response)

        # Print the assistant's response
        print()
        print(colored("User prompt:\n", "green") + msg)
        print()
        print(colored("Response:\n", "blue") + assistant_response)
        print("----------------")
        print(f"Generator time taken: {t1-t0:.2f} seconds")
        print("----------------")

        return assistant_response

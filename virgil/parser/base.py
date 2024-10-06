# # (c) 2024 by Chris Paxton

# Description: Base class for all parsers

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Parser(ABC):
    """Base class for all parsers. Contains methods for parsing text as well as utility functions common to many parsers, like pruning text to a certain length or key."""

    def __init__(self, chat):
        self.chat = chat

    @abstractmethod
    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        pass

    def prompt(self, msg: Optional[str] = None, *args, **kwargs) -> str:
        """Prompt the LLM and return the parsed response."""
        raise NotImplementedError

    def prune_to(self, text: str, max_tokens: int):
        """Prune the text to a maximum number of tokens.

        Args:
            text (str): The text to prune.
            max_tokens (int): The maximum number of tokens to prune to.

        Returns:
            str: The pruned text.
        """
        return text[:max_tokens]

    def prune_to_key(self, text: str, key: str):
        """Prune the text to a key. For example, if the key is "END", the text will be pruned to the first instance of "END".

        Args:
            text (str): The text to prune.
            key (str): The key to prune to.

        Returns:
            str: The pruned text.
        """
        idx = text.index(key)
        if idx == -1:
            return text
        return text[:idx]

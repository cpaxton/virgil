# # (c) 2024 by Chris Paxton

# Description: Base class for all parsers

from abc import ABC, abstractmethod


class Parser(ABC):
    """Base class for all parsers. Contains methods for parsing text as well as utility functions common to many parsers, like pruning text to a certain length or key."""

    @abstractmethod
    def parse(self, text: str):
        return text

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

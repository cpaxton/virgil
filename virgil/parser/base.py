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

# (c) 2024 by Chris Paxton

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

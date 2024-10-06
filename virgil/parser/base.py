# # (c) 2024 by Chris Paxton

# Description: Base class for all parsers

from abc import ABC, abstractmethod


class Parser(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def parse(self, text: str):
        return text

from abc import ABC, abstractmethod


class BaseAudioGenerator(ABC):
    @abstractmethod
    def generate(self, text: str, output_path: str):
        pass

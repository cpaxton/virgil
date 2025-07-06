from virgil.audio.base import BaseAudioGenerator
from virgil.audio.main import AudioGenerator


def get_audio_generator(generator: str = "speecht5", **kwargs) -> BaseAudioGenerator:
    if generator == "speecht5":
        return AudioGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown audio generator: {generator}")

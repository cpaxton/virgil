from virgil.audio.base import BaseAudioGenerator
from virgil.audio.speech import AudioGenerator
from virgil.audio.music import MusicGenerator


def get_audio_generator(generator: str = "speecht5", **kwargs) -> BaseAudioGenerator:
    if generator in AudioGenerator.MODEL_ALIASES:
        return AudioGenerator(model=generator, **kwargs)
    elif generator in MusicGenerator.MODEL_ALIASES:
        return MusicGenerator(model=generator, **kwargs)
    else:
        raise ValueError(f"Unknown audio generator: {generator}")

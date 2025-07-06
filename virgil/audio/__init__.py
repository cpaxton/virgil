from virgil.audio.base import BaseAudioGenerator
from virgil.audio.main import AudioGenerator


def get_audio_generator(config_path: str = None) -> BaseAudioGenerator:
    return AudioGenerator(config_path)

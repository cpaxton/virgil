from virgil.audio.base import BaseAudioGenerator


class AudioGenerator(BaseAudioGenerator):
    def __init__(self, config_path: str = None):
        pass

    def generate(self, text: str, output_path: str):
        print(f"Generating audio for '{text}' at {output_path}")

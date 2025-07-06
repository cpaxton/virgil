from virgil.audio.base import BaseAudioGenerator
from transformers import pipeline
import torch
import soundfile as sf


class MusicGenerator(BaseAudioGenerator):
    MODEL_ALIASES = {
        "musicgen": "facebook/musicgen-large",
    }

    def __init__(self, model: str = "musicgen"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = self.MODEL_ALIASES.get(model, model)
        self.pipe = pipeline("text-to-audio", model=model_id, device=self.device)

    def generate(self, text: str, output_path: str):
        music = self.pipe(text)
        sf.write(output_path, music["audio"], samplerate=music["sampling_rate"])

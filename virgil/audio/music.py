import click
import torch
import torchaudio
import soundfile as sf
from transformers import pipeline

from audiocraft.models import MusicGen
from virgil.audio.base import BaseAudioGenerator


class MusicGenerator(BaseAudioGenerator):
    MODEL_ALIASES = {
        "musicgen-small": "facebook/musicgen-small",
        "musicgen-medium": "facebook/musicgen-medium",
        "musicgen-large": "facebook/musicgen-large",
        "musicgen-stereo-small": "facebook/musicgen-stereo-small",
        "musicgen-stereo-medium": "facebook/musicgen-stereo-medium",
        "musicgen-stereo-large": "facebook/musicgen-stereo-large",
        "suno-bark": "suno/bark",
        "musicgen-melody": "meta-audio/musicgen-melody",
    }

    def __init__(self, model: str = "musicgen-small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        model_id = self.MODEL_ALIASES.get(model, model)
        if model == "suno-bark":
            self.pipe = pipeline("text-to-speech", model=model_id, device=self.device)
        elif model == "musicgen-melody":
            self.pipe = MusicGen.get_pretrained(model_id, device=self.device)
        else:
            self.pipe = pipeline("text-to-audio", model=model_id, device=self.device)

    def generate(self, text: str, output_path: str, melody_path: str = None):
        if self.model == "musicgen-melody":
            if melody_path is None:
                raise ValueError("Melody path must be provided for musicgen-melody")
            melody, sr = torchaudio.load(melody_path)
            self.pipe.set_generation_params(duration=10)
            music = self.pipe.generate_with_chroma(
                [text], melody[None].expand(1, -1, -1), sr
            )
            sf.write(
                output_path, music.cpu().squeeze(), samplerate=self.pipe.sample_rate
            )
        else:
            music = self.pipe(text)
            sf.write(
                output_path, music["audio"].squeeze(), samplerate=music["sampling_rate"]
            )


@click.command()
@click.option(
    "--text",
    "-t",
    required=True,
    help="The text prompt to generate music from.",
)
@click.option(
    "--output-path",
    "-o",
    required=True,
    help="The path to save the generated music file (e.g., music.wav).",
)
@click.option(
    "--model",
    "-m",
    default="musicgen-small",
    type=click.Choice(list(MusicGenerator.MODEL_ALIASES.keys())),
    help="The model to use for music generation.",
)
@click.option(
    "--melody-path",
    "-mp",
    default=None,
    help="Path to the melody file for musicgen-melody.",
)
def cli(text: str, output_path: str, model: str, melody_path: str):
    """Generates music from a text prompt."""
    generator = MusicGenerator(model=model)
    generator.generate(text=text, output_path=output_path, melody_path=melody_path)
    click.echo(f"Music generated and saved to {output_path}")


if __name__ == "__main__":
    cli()

import click
import torch
import soundfile as sf
from transformers import pipeline

from virgil.audio.base import BaseAudioGenerator


class MusicGenerator(BaseAudioGenerator):
    MODEL_ALIASES = {
        "musicgen-small": "facebook/musicgen-small",
        "musicgen-medium": "facebook/musicgen-medium",
        "musicgen-large": "facebook/musicgen-large",
        "musicgen-stereo-small": "facebook/musicgen-stereo-small",
        "musicgen-stereo-medium": "facebook/musicgen-stereo-medium",
        "musicgen-stereo-large": "facebook/musicgen-stereo-large",
    }

    def __init__(self, model: str = "musicgen-small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = self.MODEL_ALIASES.get(model, model)
        self.pipe = pipeline("text-to-audio", model=model_id, device=self.device)

    def generate(self, text: str, output_path: str):
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
def cli(text: str, output_path: str, model: str):
    """Generates music from a text prompt."""
    generator = MusicGenerator(model=model)
    generator.generate(text=text, output_path=output_path)
    click.echo(f"Music generated and saved to {output_path}")


if __name__ == "__main__":
    cli()

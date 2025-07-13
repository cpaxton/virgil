import click
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from virgil.audio.base import BaseAudioGenerator


class AudioGenerator(BaseAudioGenerator):
    MODEL_ALIASES = {
        "speecht5": "microsoft/speecht5_tts",
    }

    def __init__(
        self,
        model: str = "speecht5",
        vocoder: str = "microsoft/speecht5_hifigan",
        speaker_id: int = 174,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = self.MODEL_ALIASES.get(model, model)
        self.processor = SpeechT5Processor.from_pretrained(model_id)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_id).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder).to(self.device)
        self.embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )
        self.speaker_id = speaker_id

    def generate(self, text: str, output_path: str):
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        speaker_embeddings = (
            torch.tensor(self.embeddings_dataset[self.speaker_id]["xvector"])
            .unsqueeze(0)
            .to(self.device)
        )

        speech = self.model.generate_speech(
            inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder
        )

        sf.write(output_path, speech.cpu().numpy(), samplerate=16000)


@click.command()
@click.option(
    "--text",
    "-t",
    required=True,
    help="The text to convert to speech.",
)
@click.option(
    "--output-path",
    "-o",
    required=True,
    help="The path to save the generated speech file (e.g., speech.wav).",
)
@click.option(
    "--model",
    "-m",
    default="speecht5",
    help="The model to use for speech generation (e.g., 'speecht5').",
)
@click.option(
    "--vocoder",
    "-v",
    default="microsoft/speecht5_hifigan",
    help="The vocoder to use for speech generation.",
)
@click.option(
    "--speaker-id",
    "-s",
    default=174,
    type=int,
    help="The speaker ID to use for speech generation.",
)
def cli(text: str, output_path: str, model: str, vocoder: str, speaker_id: int):
    """Generates speech from text."""
    generator = AudioGenerator(model=model, vocoder=vocoder, speaker_id=speaker_id)
    generator.generate(text=text, output_path=output_path)
    click.echo(f"Speech generated and saved to {output_path}")


if __name__ == "__main__":
    cli()

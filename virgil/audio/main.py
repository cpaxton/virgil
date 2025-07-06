from virgil.audio.base import BaseAudioGenerator
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf


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

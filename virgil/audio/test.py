import unittest
from virgil.audio import get_audio_generator
import os


class TestAudioGenerator(unittest.TestCase):
    def test_get_audio_generator(self):
        generator = get_audio_generator()
        self.assertIsNotNone(generator)

    def test_generate_audio(self):
        generator = get_audio_generator()
        output_path = "/tmp/test_audio.wav"
        generator.generate("Hello, this is a test.", output_path)
        self.assertTrue(os.path.exists(output_path))

    def test_generate_audio_with_speaker(self):
        generator = get_audio_generator(speaker_id=100)
        output_path = "/tmp/test_audio_speaker.wav"
        generator.generate(
            "Hello, this is a test with a different speaker.", output_path
        )
        self.assertTrue(os.path.exists(output_path))

    def test_generate_music(self):
        generator = get_audio_generator(generator="musicgen")
        output_path = "/tmp/test_music.wav"
        generator.generate("A beautiful piano melody.", output_path)
        self.assertTrue(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()

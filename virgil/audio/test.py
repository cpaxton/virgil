import unittest
from virgil.audio import get_audio_generator


class TestAudioGenerator(unittest.TestCase):
    def test_get_audio_generator(self):
        generator = get_audio_generator()
        self.assertIsNotNone(generator)

    def test_generate_audio(self):
        generator = get_audio_generator()
        generator.generate("Hello, world!", "/tmp/hello.wav")


if __name__ == "__main__":
    unittest.main()

# # (c) 2024 by Chris Paxton

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


class SigLIPAligner:
    def __init__(self, model_name="google/siglip-base-patch16-224"):
        """
        Initialize the SigLIPAligner with a specified model.

        Args:
            model_name (str): The name of the SigLIP model to use.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.max_length = self.model.config.text_config.max_position_embeddings

    def check_alignment(self, image: Image, text: str) -> float:
        """
        Check the alignment between an image and text.

        Args:
            image (str or PIL.Image.Image): The image file path or a PIL Image object.
            text (str): The text to compare with the image.

        Returns:
            float: The alignment score between the image and text.
        """
        # Load and preprocess the image
        if isinstance(image, str):
            image = Image.open(image)

        # Prepare inputs
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True, max_length=self.max_length, truncation=True).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Calculate similarity
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        similarity = (image_embeds @ text_embeds.T).item()

        return similarity

    def search(self, image_generator, prompt: str, num_tries: int = 10) -> Image:
        """Generate an image based on the prompt, then return the image with the highest alignment score."""
        best_score = -1
        best_image = None

        for i in range(num_tries):
            print("Generating image attempt", i + 1)
            print("Prompt:", prompt)
            image = image_generator.generate(prompt)
            score = self.check_alignment(image, prompt)
            print("Alignment score:", score)
            if score > best_score:
                best_score = score
                best_image = image

        return best_score, best_image


# Example usage
if __name__ == "__main__":
    aligner = SigLIPAligner()

    # Example with a PIL Image object
    from PIL import Image
    import requests

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    text = "Two cats sitting on a couch"

    score = aligner.check_alignment(image, text)
    print(f"Alignment score: {score}")

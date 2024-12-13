import torch
from diffusers import AutoPipelineForText2Image
from transformers import AutoTokenizer

# Set up the pipeline
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)

# Move the pipeline to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = pipeline.to(device)

# Set up the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")

# Define the prompt
prompt = "A serene landscape with a mountain lake at sunset, photorealistic style"

# Generate the image
image = pipeline(prompt=prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

# Save the generated image
image.save("generated_image.png")

print("Image generated and saved as 'generated_image.png'")


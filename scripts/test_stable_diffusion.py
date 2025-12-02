# Copyright 2024 Chris Paxton
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from diffusers import AutoPipelineForText2Image
from transformers import AutoTokenizer

# Set up the pipeline
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = AutoPipelineForText2Image.from_pretrained(
    model_id, dtype=torch.float16, variant="fp16", use_safetensors=True
)

# Move the pipeline to GPU if available - also supports Apple mps
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

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

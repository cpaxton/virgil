# Qwen Image Layered Generator

This module provides image generation using Qwen's Image Layered model, which can generate layered images from text prompts or input images.

## Model Information

- **Model**: [Qwen/Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered)
- **Type**: Layered image generation (text-to-image and image-to-image)
- **Resolution**: Supports 640x640 and 1024x1024 (640 recommended)

## Features

- **Layered Generation**: Generates multiple compositable layers
- **Text-to-Image**: Generate images from text prompts
- **Image-to-Image**: Enhance or modify existing images
- **Automatic Compositing**: Automatically composites layers in the `generate()` method
- **Layer Access**: Use `generate_layers()` to get individual layers

## Usage

### Basic Usage

```python
from virgil.image import QwenLayeredImageGenerator

# Create generator
generator = QwenLayeredImageGenerator(
    height=640,
    width=640,
    num_inference_steps=50,
    layers=4,
    resolution=640,
)

# Generate image from text prompt
image = generator.generate("a beautiful sunset over mountains")
image.save("sunset.png")

# Generate all layers separately
layers = generator.generate_layers("a cat wearing a hat")
for i, layer in enumerate(layers):
    layer.save(f"layer_{i}.png")
```

### Image-to-Image

```python
from PIL import Image

# Load input image
input_image = Image.open("input.png").convert("RGBA")

# Generate layered output
image = generator.generate(
    prompt="make it more vibrant",
    input_image=input_image,
)
image.save("output.png")
```

### With Friend Bot

```bash
python virgil/friend/friend.py --image-generator qwen-layered --backend gemma
```

### With MCP Server

```bash
python -m virgil.friend.friend --mcp-only --image-generator qwen-layered
```

### With CLI Tool

```bash
python -m virgil.mcp.cli generate-image "a beautiful landscape" --image-generator qwen-layered -o output.png
```

## Parameters

- `height` (int): Height of generated images (default: 640)
- `width` (int): Width of generated images (default: 640)
- `num_inference_steps` (int): Number of denoising steps (default: 50)
- `layers` (int): Number of layers to generate (default: 4)
- `resolution` (int): Resolution bucket - 640 or 1024 (default: 640, recommended)
- `true_cfg_scale` (float): CFG scale for generation (default: 4.0)
- `negative_prompt` (str): Negative prompt (default: " ")
- `num_images_per_prompt` (int): Number of images per prompt (default: 1)
- `cfg_normalize` (bool): Enable CFG normalization (default: True)
- `use_en_prompt` (bool): Automatic caption language (default: True)
- `model` (str): HuggingFace model identifier (default: "Qwen/Qwen-Image-Layered")
- `device` (str): Device to run on (default: "cuda" if available)
- `dtype` (torch.dtype): Data type for model (default: torch.bfloat16)

## Requirements

- `diffusers` library with Qwen Image Layered support
- `torch` with CUDA support (recommended)
- Latest version of diffusers: `pip install diffusers --upgrade`

## Notes

- The model generates multiple layers that can be composited together
- Resolution 640 is recommended for best performance
- The `generate()` method automatically composites layers into a single image
- Use `generate_layers()` to access individual layers for custom compositing
- Image-to-image mode requires RGBA input images

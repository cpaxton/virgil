# Testing Backend Models

This guide provides instructions for testing each backend model available in Virgil, along with what to expect from each model.

> **Quick Reference**: See [model-quick-reference.md](./model-quick-reference.md) for a complete list of all 66+ available models.

## Table of Contents

- [Microsoft Phi-3 Models](#microsoft-phi-3-models)
- [TinyLlama](#tinyllama)
- [SmolLM](#smollm)
- [Gemma Models](#gemma-models)
- [Qwen Models](#qwen-models)
- [Llama Models](#llama-models)
- [ERNIE Models](#ernie-models)

## Microsoft Phi-3 Models

### Available Models

- `phi-3-mini` (3.8B, 4k context)
- `phi-3-mini-4k` (3.8B, 4k context)
- `phi-3-mini-128k` (3.8B, 128k context)
- `phi-3-small` (7B, 8k context)
- `phi-3-small-8k` (7B, 8k context)
- `phi-3-small-128k` (7B, 128k context)
- `phi-3-medium` (14B, 4k context)
- `phi-3-medium-4k` (14B, 4k context)
- `phi-3-medium-128k` (14B, 128k context)

### Testing Command

```bash
python -m virgil.chat --backend phi-3-mini --verbose
```

### What to Expect

- **Quality**: Phi-3 models are state-of-the-art small language models from Microsoft, known for excellent reasoning capabilities despite their small size
- **Speed**: Fast inference, especially with quantization (defaults to int4)
- **Performance**: Phi-3-mini performs exceptionally well for a 3.8B model, often matching or exceeding larger models on reasoning tasks
- **Use Cases**: General conversation, coding assistance, math problems, reasoning tasks
- **Context Length**: Choose 4k/8k variants for standard use, 128k variants for long-context tasks

### Example Test Prompts

- "Explain quantum computing in simple terms"
- "Write a Python function to calculate fibonacci numbers"
- "Solve this math problem: If a train travels 120 km in 2 hours, what's its average speed?"

---

## TinyLlama

### Available Models

- `tinyllama` (1.1B)
- `tinyllama-1.1b` (1.1B)
- `tinyllama-chat` (1.1B, chat-optimized)
- `tinyllama-intermediate` (1.1B, intermediate checkpoint)

### Testing Command

```bash
python -m virgil.chat --backend tinyllama --verbose
```

### What to Expect

- **Quality**: Basic conversational abilities, suitable for lightweight applications
- **Speed**: Very fast inference due to small model size (1.1B parameters)
- **Performance**: Good for simple tasks, limited on complex reasoning
- **Use Cases**: Quick responses, simple Q&A, lightweight applications, testing
- **Memory**: Requires minimal GPU memory, can run on consumer hardware

### Example Test Prompts

- "What is the capital of France?"
- "Write a simple hello world program"
- "Tell me a short joke"

---

## SmolLM

### Available Models

- `smollm-135m` (135M parameters)
- `smollm-360m` (360M parameters)
- `smollm-1.7b` (1.7B parameters)
- `smollm-2.7b` (2.7B parameters)

### Testing Command

```bash
python -m virgil.chat --backend smollm-1.7b --verbose
```

### What to Expect

- **Quality**: HuggingFace's efficient small models optimized for speed and resource usage
- **Speed**: Extremely fast, especially the smaller variants (135M, 360M)
- **Performance**: The 1.7B and 2.7B variants offer better quality while remaining lightweight
- **Use Cases**: Edge devices, rapid prototyping, resource-constrained environments
- **Memory**: Minimal memory footprint, great for testing and development

### Example Test Prompts

- "Hello, how are you?"
- "What is 2+2?"
- "Name three colors"

---

## Gemma Models

### Available Models

- `gemma-2-2b-it` (2B, instruction-tuned)
- `gemma-2-9b-it` (9B, instruction-tuned)
- `gemma-1-7b-it` (7B, instruction-tuned)
- `gemma-1-3b-it` (3B, instruction-tuned)

### Testing Command

```bash
python -m virgil.chat --backend gemma-2-2b-it --verbose
```

### What to Expect

- **Quality**: Google's open-source models with strong performance on instruction following
- **Speed**: Fast inference, especially with Flash Attention support (automatically enabled on compatible GPUs)
- **Performance**: Good balance between quality and speed, well-suited for chat applications
- **Use Cases**: General conversation, instruction following, creative writing
- **Features**: Supports Flash Attention 2 for faster inference on Ampere+ GPUs

### Example Test Prompts

- "Write a haiku about programming"
- "Explain how neural networks work"
- "Help me write an email to thank someone"

---

## Qwen Models

### Available Models

Qwen 2.5 models (sizes: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B):
- `qwen2.5-{size}-instruct`
- `qwen2.5-{size}-coder`
- `qwen2.5-{size}-math`
- `qwen2.5-{size}-deepseek`

Qwen 3 models (sizes: 0.6B, 1.7B, 4B, 8B, 14B, 32B):
- `qwen3-{size}`

### Testing Command

```bash
# Test a small Qwen 3 model
python -m virgil.chat --backend qwen3-1.7b --verbose

# Test a specialized Qwen 2.5 model
python -m virgil.chat --backend qwen2.5-3b-coder --verbose
```

### What to Expect

- **Quality**: Alibaba's models with strong multilingual capabilities and specialized variants
- **Speed**: Efficient inference, good performance across sizes
- **Performance**: 
  - Base models: Good general performance
  - Coder variants: Excellent for code generation and programming tasks
  - Math variants: Strong mathematical reasoning
  - Deepseek variants: Advanced reasoning capabilities
- **Use Cases**: 
  - Multilingual conversations
  - Code generation (coder variants)
  - Math problems (math variants)
  - Complex reasoning (deepseek variants)
- **Specializations**: Choose specialized variants based on your task

### Example Test Prompts

- General: "Tell me about artificial intelligence"
- Coder: "Write a Python function to sort a list"
- Math: "Solve: x^2 + 5x + 6 = 0"
- Deepseek: "Explain the implications of quantum entanglement"

---

## Llama Models

### Available Models

- `llama` (defaults to 3.2-1B)
- `llama-3.2-1b` (1B)
- `llama-3.2-3b` (3B)
- `llama-3.1-8b` (8B)

### Testing Command

```bash
python -m virgil.chat --backend llama-3.2-1b --verbose
```

### What to Expect

- **Quality**: Meta's Llama models, well-known for strong performance
- **Speed**: Fast inference, especially the smaller variants
- **Performance**: Good general-purpose capabilities, strong instruction following
- **Use Cases**: General conversation, instruction following, various NLP tasks
- **Reputation**: Llama models are widely used and benchmarked

### Example Test Prompts

- "What are the benefits of renewable energy?"
- "Write a story about a robot learning to paint"
- "Explain the water cycle"

---

## ERNIE Models

### Available Models

- `ernie-4.5-0.3b-base-paddle` (0.3B)
- `ernie-4.5-0.3b-base-pt` (0.3B, PyTorch)
- `ernie-4.5-21b-base-paddle` (21B)
- `ernie-4.5-21b-a3b-base-paddle` (21B)
- `ernie-4.5-21b-a3b-pt` (21B, PyTorch)
- `ernie-4.5-vl-28b-a3b-base-paddle` (28B, vision-language)
- `ernie-4.5-vl-28b-a3b-pt` (28B, vision-language, PyTorch)

### Testing Command

```bash
python -m virgil.chat --backend ernie-4.5-0.3b-base-pt --verbose
```

### What to Expect

- **Quality**: Baidu's ERNIE models, with strong performance on Chinese and multilingual tasks
- **Speed**: The 0.3B variant is extremely fast, larger models offer better quality
- **Performance**: 
  - Base models: Good general performance
  - VL (Vision-Language) models: Can process images (requires images parameter)
- **Use Cases**: 
  - Multilingual conversations (especially Chinese)
  - Vision-language tasks (VL variants)
  - Resource-constrained environments (0.3B variant)
- **Special Features**: VL models support multimodal input (text + images)

### Example Test Prompts

- "你好，请介绍一下你自己" (Chinese: "Hello, please introduce yourself")
- "What is artificial intelligence?"
- "Describe this image" (for VL models, requires image input)

---

## General Testing Tips

### Using Verbose Mode

Add `--verbose` to see:
- Detailed conversation history
- Generation timing information
- Full prompts and responses

### Testing with Custom Model Paths

If you have local model weights:

```bash
python -m virgil.chat --backend phi-3-mini --path /path/to/local/model --verbose
```

### Testing with Prompt Files

Use a text file as initial prompt:

```bash
python -m virgil.chat --backend gemma-2-2b-it --prompt my_prompt.txt --verbose
```

### Adjusting Generation Parameters

You can modify generation parameters in the backend initialization (temperature, top_p, etc.) or adjust `max_new_tokens` in the chat interface.

### Performance Comparison

To compare models, test the same prompt across different backends:

```bash
# Test Phi-3
python -m virgil.chat --backend phi-3-mini --prompt test_prompt.txt --verbose

# Test Gemma
python -m virgil.chat --backend gemma-2-2b-it --prompt test_prompt.txt --verbose

# Test TinyLlama
python -m virgil.chat --backend tinyllama --prompt test_prompt.txt --verbose
```

### Memory Considerations

- **Small models** (< 2B): Should run on most GPUs with 8GB+ VRAM
- **Medium models** (2B-7B): May require quantization (int4/int8) on consumer GPUs
- **Large models** (> 7B): Will likely need quantization or high-end GPUs

Most models default to quantization when appropriate. You can disable it by setting `quantization=None` in the backend code if needed.

---

## Troubleshooting

### Model Not Found

If you get a "model not found" error:
1. Ensure you have internet connection (models download from HuggingFace)
2. Check that the model name matches exactly (case-insensitive)
3. Verify the model is in the `backend_list` in `virgil/backend/__init__.py`

### Out of Memory

If you encounter CUDA out of memory errors:
1. The model should auto-quantize, but you can verify quantization is enabled
2. Try a smaller model variant
3. Close other GPU-intensive applications
4. Reduce `max_new_tokens` in your prompts

### Slow Generation

- Check if quantization is enabled (should speed things up)
- Ensure you're using GPU (CUDA) if available
- Try a smaller model variant
- Reduce `max_new_tokens` parameter

### Import Errors

Ensure all dependencies are installed:
```bash
pip install transformers torch accelerate bitsandbytes
```

---

## Quick Reference: All Available Models

Run this to see all available models:

```python
from virgil.backend import backend_list
print("\n".join(sorted(backend_list)))
```

Or check the `backend_list` in `virgil/backend/__init__.py`.


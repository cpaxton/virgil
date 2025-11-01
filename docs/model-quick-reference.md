# Model Quick Reference

This document provides a quick reference of all available models in Virgil, organized by family.

## Microsoft Phi-3 Models (8 models)

- `phi-3-mini` - 3.8B, 4k context (recommended)
- `phi-3-mini-4k` - 3.8B, 4k context
- `phi-3-mini-128k` - 3.8B, 128k context
- `phi-3-small` - 7B, 8k context
- `phi-3-small-8k` - 7B, 8k context
- `phi-3-small-128k` - 7B, 128k context
- `phi-3-medium` - 14B, 4k context
- `phi-3-medium-4k` - 14B, 4k context
- `phi-3-medium-128k` - 14B, 128k context

**Quick Test**: `python -m virgil.chat --backend phi-3-mini`

## TinyLlama Models (4 models)

- `tinyllama` - 1.1B (recommended)
- `tinyllama-1.1b` - 1.1B
- `tinyllama-chat` - 1.1B, chat-optimized
- `tinyllama-intermediate` - 1.1B, intermediate checkpoint

**Quick Test**: `python -m virgil.chat --backend tinyllama`

## SmolLM Models (4 models)

- `smollm-135m` - 135M parameters
- `smollm-360m` - 360M parameters
- `smollm-1.7b` - 1.7B parameters (recommended)
- `smollm-2.7b` - 2.7B parameters

**Quick Test**: `python -m virgil.chat --backend smollm-1.7b`

## Gemma Models (4 models)

- `gemma-2-2b-it` - 2B, instruction-tuned (recommended)
- `gemma-2-9b-it` - 9B, instruction-tuned
- `gemma-1-7b-it` - 7B, instruction-tuned
- `gemma-1-3b-it` - 3B, instruction-tuned

**Quick Test**: `python -m virgil.chat --backend gemma-2-2b-it`

## Qwen Models (40+ models)

### Qwen 2.5 Models
**Format**: `qwen2.5-{size}-{specialization}`

**Sizes**: `0.5b`, `1.5b`, `3b`, `7b`, `14b`, `32b`, `72b`

**Specializations**:
- `instruct` - General instruction following
- `coder` - Code generation
- `math` - Mathematical reasoning
- `deepseek` - Advanced reasoning

**Examples**:
- `qwen2.5-3b-instruct`
- `qwen2.5-7b-coder`
- `qwen2.5-1.5b-math`
- `qwen2.5-7b-deepseek`

### Qwen 3 Models
**Format**: `qwen3-{size}`

**Sizes**: `0.6b`, `1.7b`, `4b`, `8b`, `14b`, `32b`

**Examples**:
- `qwen3-1.7b`
- `qwen3-8b`

**Quick Test**: `python -m virgil.chat --backend qwen3-1.7b`

## Llama Models (4 models)

- `llama` - Defaults to 3.2-1B
- `llama-3.2-1b` - 1B parameters
- `llama-3.2-3b` - 3B parameters
- `llama-3.1-8b` - 8B parameters

**Quick Test**: `python -m virgil.chat --backend llama-3.2-1b`

## ERNIE Models (7 models)

- `ernie-4.5-0.3b-base-paddle` - 0.3B, PaddlePaddle
- `ernie-4.5-0.3b-base-pt` - 0.3B, PyTorch (recommended for small)
- `ernie-4.5-21b-base-paddle` - 21B, PaddlePaddle
- `ernie-4.5-21b-a3b-base-paddle` - 21B, A3B, PaddlePaddle
- `ernie-4.5-21b-a3b-pt` - 21B, A3B, PyTorch
- `ernie-4.5-vl-28b-a3b-base-paddle` - 28B, Vision-Language, PaddlePaddle
- `ernie-4.5-vl-28b-a3b-pt` - 28B, Vision-Language, PyTorch

**Quick Test**: `python -m virgil.chat --backend ernie-4.5-0.3b-base-pt`

---

## Testing All Models

To test any model, use:

```bash
python -m virgil.chat --backend <model-name> --verbose
```

Replace `<model-name>` with any model from the lists above.

### Example: Test All Phi-3 Variants

```bash
python -m virgil.chat --backend phi-3-mini --verbose
python -m virgil.chat --backend phi-3-mini-128k --verbose
python -m virgil.chat --backend phi-3-small --verbose
python -m virgil.chat --backend phi-3-medium --verbose
```

### Example: Test Small Models

```bash
# Very small models
python -m virgil.chat --backend smollm-135m --verbose
python -m virgil.chat --backend tinyllama --verbose
python -m virgil.chat --backend ernie-4.5-0.3b-base-pt --verbose

# Small but capable
python -m virgil.chat --backend phi-3-mini --verbose
python -m virgil.chat --backend gemma-2-2b-it --verbose
python -m virgil.chat --backend qwen3-1.7b --verbose
```

## Getting All Available Models Programmatically

```python
from virgil.backend import backend_list
print("\n".join(sorted(backend_list)))
```

Or run:
```bash
python -c "from virgil.backend import backend_list; print('\n'.join(sorted(backend_list)))"
```

## Model Selection Guide

### For Speed (Fastest)
- `smollm-135m` or `smollm-360m` - Extremely fast
- `tinyllama` - Very fast
- `ernie-4.5-0.3b-base-pt` - Very fast

### For Quality (Best Performance)
- `phi-3-mini` - Excellent for 3.8B model
- `phi-3-small` - Great quality at 7B
- `gemma-2-9b-it` - Strong 9B model
- `qwen2.5-7b-instruct` - Good general purpose

### For Specific Tasks
- **Code**: `qwen2.5-7b-coder` or `qwen2.5-3b-coder`
- **Math**: `qwen2.5-7b-math` or `qwen2.5-3b-math`
- **Reasoning**: `qwen2.5-7b-deepseek`
- **Multilingual**: Any Qwen model
- **Vision**: `ernie-4.5-vl-28b-a3b-pt` (requires image input)

### For Limited Resources
- Start with `smollm-135m` or `tinyllama`
- If you have ~4GB VRAM: `phi-3-mini`, `gemma-2-2b-it`
- If you have ~8GB VRAM: `phi-3-small`, `gemma-2-9b-it`

All models support automatic quantization (int4/int8) to reduce memory usage.


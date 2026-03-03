#!/usr/bin/env python3
"""
Test script for LLM embedding with TF32 compatibility.

The error (when using fp32_precision): "RuntimeError: PyTorch is checking whether
allow_tf32_new is enabled for cuBlas matmul, Current status indicate that you have
used mix of the legacy and new APIs to set the TF32 status."

Root cause: torch.compile (inductor) reads the LEGACY API (allow_tf32). Using the
NEW API (fp32_precision) in backend/__init__.py causes a mix - PyTorch 2.9+ does
not support mixing these.

Fix: Use only the legacy API (allow_tf32) in backend/__init__.py.
"""

import os
import sys

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def test_llm_embedding_with_compile():
    """Verify LLM embedding works with torch.compile (uses legacy TF32 API)."""
    if not torch.cuda.is_available():
        print("Skipping: CUDA not available")
        return

    from virgil.backend import get_backend
    from virgil.friend.memory import MemoryManager

    print("Loading Qwen backend (with torch.compile)...")
    backend = get_backend("qwen3.5-0.8b", compile_model=True, quantization=None)

    print("Creating MemoryManager with use_llm_embeddings=True...")
    memory = MemoryManager(
        mode="rag",
        use_llm_embeddings=True,
        llm_backend=backend,
    )

    print("Adding memory (triggers LLM embedding extraction)...")
    try:
        memory.add_memory("User said hi")
        print("SUCCESS: LLM embedding works (no TF32 error)")
    except RuntimeError as e:
        if "allow_tf32" in str(e) or "tf32" in str(e).lower():
            print(f"FAIL: TF32 API mix error: {e}")
            raise
        else:
            raise


if __name__ == "__main__":
    print("=" * 60)
    print("Test: LLM embedding with torch.compile (TF32 compatibility)")
    print("=" * 60)
    try:
        test_llm_embedding_with_compile()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

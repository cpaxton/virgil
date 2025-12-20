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

# (c) 2024 by Chris Paxton

from typing import Optional, Tuple, Any


class Backend:
    """Base class for backends."""

    def __init__(self):
        self._supports_kv_cache = False
        self._kv_cache_enabled = False

    def supports_kv_cache(self) -> bool:
        """Check if this backend supports KV caching.

        Returns:
            bool: True if KV caching is supported, False otherwise.
        """
        return self._supports_kv_cache

    def enable_kv_cache(self, enable: bool = True):
        """Enable or disable KV caching for this backend.

        Args:
            enable (bool): Whether to enable KV caching. Defaults to True.
        """
        if enable and not self._supports_kv_cache:
            raise ValueError("This backend does not support KV caching")
        self._kv_cache_enabled = enable

    def generate_with_cache(
        self,
        messages,
        max_new_tokens: int = 256,
        past_key_values: Optional[Any] = None,
        *args,
        **kwargs,
    ) -> Tuple[Any, Optional[Any]]:
        """Generate a response with KV cache support.

        This method should be implemented by backends that support KV caching.
        It should return both the generated output and the new past_key_values.

        Args:
            messages: The messages to generate a response for.
            max_new_tokens: Maximum number of new tokens to generate.
            past_key_values: Previous KV cache from previous generation.
            *args, **kwargs: Additional arguments.

        Returns:
            Tuple of (output, new_past_key_values). If KV cache is not supported,
            new_past_key_values will be None.
        """
        # Default implementation: fall back to regular generation
        output = self(messages, max_new_tokens=max_new_tokens, *args, **kwargs)
        return output, None

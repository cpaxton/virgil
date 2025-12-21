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

"""
Memory management system with optional RAG (Retrieval-Augmented Generation) support.

Supports three modes:
- static: All memories included in prompt (backward compatible)
- rag: Dynamic semantic retrieval based on query
- hybrid: RAG for large sets, static for small sets
"""

import os
import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict
import numpy as np


@dataclass
class Memory:
    """Represents a single memory entry."""

    id: str
    content: str
    embedding: Optional[List[float]] = None  # Stored as list for JSON serialization
    metadata: Optional[Dict] = None
    created_at: Optional[str] = None
    accessed_at: Optional[str] = None
    access_count: int = 0

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        """Create Memory from dictionary.

        Note: Embeddings are NOT loaded from JSON - they are regenerated lazily
        when needed since they're deterministic from content. This keeps JSON files small.
        """
        return cls(
            id=data["id"],
            content=data["content"],
            embedding=None,  # Don't load embeddings - regenerate lazily
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at"),
            accessed_at=data.get("accessed_at"),
            access_count=data.get("access_count", 0),
        )


class MemoryManager:
    """
    Manages memories with optional RAG support.

    Modes:
    - "static": Current behavior (all memories in prompt)
    - "rag": Dynamic retrieval based on semantic similarity
    - "hybrid": RAG + fallback to static for small memory sets
    """

    def __init__(
        self,
        mode: str = "static",
        storage_file: str = "memories.json",
        legacy_file: str = "memory.txt",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_memories: int = 10,
        similarity_threshold: float = 0.3,
        use_gpu: bool = True,
        hybrid_threshold: int = 20,  # Use static mode if fewer than this many memories
    ):
        """
        Initialize the memory manager.

        Args:
            mode: "static", "rag", or "hybrid"
            storage_file: Path to JSON file for storing memories
            legacy_file: Path to legacy memory.txt file (for migration)
            embedding_model: Model name for sentence transformers
            max_memories: Maximum memories to retrieve in RAG mode
            similarity_threshold: Minimum similarity score (0.0-1.0)
            use_gpu: Whether to use GPU for embeddings
            hybrid_threshold: Memory count threshold for hybrid mode
        """
        self.mode = mode
        self.storage_file = storage_file
        self.legacy_file = legacy_file
        self.embedding_model_name = embedding_model
        self.max_memories = max_memories
        self.similarity_threshold = similarity_threshold
        self.use_gpu = use_gpu
        self.hybrid_threshold = hybrid_threshold

        self.memories: Dict[str, Memory] = {}
        self._embedding_model = None
        self._embedding_cache = {}  # Cache embeddings in memory

        # Load existing memories
        self._load_memories()

    def _load_memories(self):
        """Load memories from storage file or migrate from legacy format."""
        # Check if new format exists
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        # Old format: list of memories
                        for mem_data in data:
                            memory = Memory.from_dict(mem_data)
                            self.memories[memory.id] = memory
                    elif isinstance(data, dict) and "memories" in data:
                        # New format: dict with metadata
                        # Embeddings are NOT loaded - they will be regenerated lazily when needed
                        for mem_data in data["memories"]:
                            memory = Memory.from_dict(mem_data)
                            # Don't load embeddings from JSON - regenerate lazily
                            memory.embedding = None
                            self.memories[memory.id] = memory
            except Exception as e:
                print(f"Error loading memories from {self.storage_file}: {e}")
                # Try to migrate from legacy format
                self._migrate_from_legacy()
        elif os.path.exists(self.legacy_file):
            # Migrate from legacy memory.txt format
            self._migrate_from_legacy()

    def _migrate_from_legacy(self):
        """Migrate memories from legacy memory.txt format."""
        print(f"Migrating memories from {self.legacy_file} to {self.storage_file}...")
        if not os.path.exists(self.legacy_file):
            return

        try:
            with open(self.legacy_file, "r") as f:
                lines = f.read().split("\n")

            migrated_count = 0
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    memory = Memory(
                        id=str(uuid.uuid4()),
                        content=line,
                        metadata={"migrated": True, "source": "memory.txt"},
                    )
                    self.memories[memory.id] = memory
                    migrated_count += 1

            if migrated_count > 0:
                self._save_memories()
                print(f"Migrated {migrated_count} memories from legacy format.")
        except Exception as e:
            print(f"Error migrating from legacy format: {e}")

    def _save_memories(self):
        """Save memories to storage file.

        Embeddings are NOT persisted - they are regenerated on load since they're
        deterministic from the content. This keeps the JSON file small and efficient.
        """
        try:
            # Save memories WITHOUT embeddings to keep JSON file small
            # Embeddings will be regenerated on load since they're deterministic
            memories_data = {
                "version": "1.0",
                "mode": self.mode,
                "memories": [
                    {
                        "id": mem.id,
                        "content": mem.content,
                        "metadata": mem.metadata,
                        "created_at": mem.created_at,
                        "accessed_at": mem.accessed_at,
                        "access_count": mem.access_count,
                        # Embedding is NOT saved - will be regenerated on load
                    }
                    for mem in self.memories.values()
                ],
            }
            with open(self.storage_file, "w") as f:
                json.dump(memories_data, f, indent=2)
        except Exception as e:
            print(f"Error saving memories: {e}")

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None and self.mode in ("rag", "hybrid"):
            try:
                from sentence_transformers import SentenceTransformer

                print(f"Loading embedding model: {self.embedding_model_name}")
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                if self.use_gpu:
                    # Try to use GPU if available
                    import torch

                    if torch.cuda.is_available():
                        self._embedding_model = self._embedding_model.cuda()
                        print("Using GPU for embeddings")
                    else:
                        print("GPU not available, using CPU")
            except ImportError:
                print(
                    "Warning: sentence-transformers not installed. Install with: pip install sentence-transformers"
                )
                print("Falling back to static mode")
                self.mode = "static"
            except Exception as e:
                print(f"Error loading embedding model: {e}")
                print("Falling back to static mode")
                self.mode = "static"
        return self._embedding_model

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text, using cache if available."""
        # Check cache first
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        model = self._get_embedding_model()
        if model is None:
            return None

        try:
            embedding = model.encode(text, convert_to_numpy=True)
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            self._embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def add_memory(self, content: str, metadata: Optional[Dict] = None) -> Memory:
        """
        Add a new memory.

        Args:
            content: Memory content
            metadata: Optional metadata dictionary

        Returns:
            The created Memory object
        """
        memory = Memory(
            id=str(uuid.uuid4()),
            content=content,
            metadata=metadata or {},
        )

        # Generate embedding if in RAG mode
        if self.mode in ("rag", "hybrid"):
            embedding = self._get_embedding(content)
            if embedding is not None:
                memory.embedding = embedding.tolist()

        self.memories[memory.id] = memory
        self._save_memories()
        return memory

    def remove_memory(self, content: str) -> bool:
        """
        Remove a memory by exact content match.

        Args:
            content: Memory content to remove

        Returns:
            True if memory was removed, False if not found
        """
        for mem_id, memory in list(self.memories.items()):
            if memory.content == content:
                del self.memories[mem_id]
                # Also remove from cache
                if content in self._embedding_cache:
                    del self._embedding_cache[content]
                self._save_memories()
                return True
        return False

    def get_relevant_memories(
        self, query: str, max_results: Optional[int] = None
    ) -> List[Memory]:
        """
        Retrieve relevant memories for a query using semantic similarity.

        Args:
            query: Query text to find relevant memories for
            max_results: Maximum number of results (defaults to self.max_memories)

        Returns:
            List of Memory objects sorted by relevance
        """
        if not self.memories:
            return []

        max_results = max_results or self.max_memories

        # Generate query embedding first
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            # Fallback to all memories if embedding fails
            return list(self.memories.values())[:max_results]

        # Generate embeddings lazily - only as we need them for similarity computation
        # Embeddings are cached in _embedding_cache, so regenerating is fast if already computed
        similarities = []
        for memory in self.memories.values():
            # Generate embedding lazily if not already cached
            if memory.embedding is None and self.mode in ("rag", "hybrid"):
                # Check cache first (fast), then generate if needed
                embedding = self._get_embedding(memory.content)
                if embedding is not None:
                    memory.embedding = embedding.tolist()
                else:
                    continue  # Skip if embedding generation failed

            # Convert to numpy array (use cached embedding from memory object)
            mem_embedding = np.array(memory.embedding)
            # Compute cosine similarity
            similarity = np.dot(query_embedding, mem_embedding)

            if similarity >= self.similarity_threshold:
                similarities.append((similarity, memory))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Update access tracking
        for _, memory in similarities[:max_results]:
            memory.access_count += 1
            memory.accessed_at = datetime.now().isoformat()

        # Return top results
        return [memory for _, memory in similarities[:max_results]]

    def get_all_memories(self) -> List[str]:
        """
        Get all memory contents as strings (for static mode).

        Returns:
            List of memory content strings
        """
        return [memory.content for memory in self.memories.values()]

    def get_memories_for_query(self, query: str) -> str:
        """
        Get formatted memories for a query based on current mode.

        Args:
            query: Query text

        Returns:
            Formatted string of memories (one per line)
        """
        if self.mode == "static":
            memories = self.get_all_memories()
        elif self.mode == "hybrid":
            # Use static if small set, otherwise RAG
            if len(self.memories) < self.hybrid_threshold:
                memories = self.get_all_memories()
            else:
                memories = [m.content for m in self.get_relevant_memories(query)]
        else:  # rag mode
            memories = [m.content for m in self.get_relevant_memories(query)]

        # Filter out empty strings and join with newlines
        memories = [m for m in memories if m.strip()]
        return "\n".join(memories) if memories else ""

    def __len__(self) -> int:
        """Return number of memories."""
        return len(self.memories)

    def clear(self):
        """Clear all memories."""
        self.memories.clear()
        self._embedding_cache.clear()
        self._save_memories()

    def consolidate_memories(
        self,
        similarity_threshold: float = 0.85,
        use_llm: bool = False,
        backend=None,
    ) -> Dict[str, int]:
        """
        Consolidate similar memories by merging duplicates and near-duplicates.

        Args:
            similarity_threshold: Similarity threshold for considering memories as duplicates (0.0-1.0). Higher = more strict.
            use_llm: If True, use LLM to intelligently merge similar memories. If False, keep the most recent/accessed memory.
            backend: Optional backend instance for LLM-based merging (required if use_llm=True)

        Returns:
            Dict with consolidation stats: {"merged": count, "removed": count, "kept": count}
        """
        if len(self.memories) < 2:
            return {"merged": 0, "removed": 0, "kept": len(self.memories)}

        # Ensure all memories have embeddings
        for memory in self.memories.values():
            if memory.embedding is None and self.mode in ("rag", "hybrid"):
                embedding = self._get_embedding(memory.content)
                if embedding is not None:
                    memory.embedding = embedding.tolist()

        # Find similar memories using clustering
        memory_list = list(self.memories.values())
        to_remove = set()
        merged_groups = []

        # Group similar memories
        for i, mem1 in enumerate(memory_list):
            if mem1.id in to_remove:
                continue

            group = [mem1]
            mem1_embedding = np.array(mem1.embedding) if mem1.embedding else None

            if mem1_embedding is None:
                continue

            for j, mem2 in enumerate(memory_list[i + 1 :], start=i + 1):
                if mem2.id in to_remove:
                    continue

                mem2_embedding = np.array(mem2.embedding) if mem2.embedding else None
                if mem2_embedding is None:
                    continue

                # Compute similarity
                similarity = np.dot(mem1_embedding, mem2_embedding)

                if similarity >= similarity_threshold:
                    group.append(mem2)
                    to_remove.add(mem2.id)

            if len(group) > 1:
                merged_groups.append(group)

        # Merge groups
        merged_count = 0
        removed_count = len(to_remove)

        for group in merged_groups:
            if len(group) <= 1:
                continue

            # Sort by access_count and accessed_at (prefer frequently accessed, recent memories)
            group.sort(
                key=lambda m: (
                    m.access_count,
                    m.accessed_at or m.created_at or "",
                ),
                reverse=True,
            )

            # Keep the best memory, merge others into it
            kept_memory = group[0]
            to_merge = group[1:]

            if use_llm and backend:
                # Use LLM to intelligently merge memories
                # Build context about the memories being merged
                memory_details = []
                for idx, m in enumerate(group, 1):
                    access_info = f" (accessed {m.access_count} times"
                    if m.accessed_at:
                        access_info += f", last accessed {m.accessed_at[:10]}"
                    access_info += ")"
                    memory_details.append(f"{idx}. {m.content}{access_info}")

                merge_prompt = f"""You are a memory consolidation system. Merge these related memories into a single, comprehensive fact.

Memories to merge:
{chr(10).join(memory_details)}

Your task:
1. Identify the core fact or information that all memories share
2. Combine any unique details from each memory into a unified statement
3. Preserve important specifics (names, dates, preferences, commitments, story elements)
4. Remove redundancy while keeping all essential information
5. Create a clear, factual statement that captures everything important

Guidelines:
- If memories are about the same fact with different details, merge them into one comprehensive fact
- If memories are about different aspects of the same topic, combine them intelligently
- Preserve specific details: names, numbers, dates, preferences, story elements
- Keep the most specific and informative version of shared information
- If one memory is clearly more detailed/complete, use it as the base and add missing details from others
- Output format: A single, clear, factual statement (one sentence or short paragraph)

Output the merged memory now (factual statement only, no explanations):"""

                try:
                    messages = [{"role": "user", "content": merge_prompt}]
                    # Increased token limit for comprehensive merging
                    result = backend(messages, max_new_tokens=256)

                    # Extract merged content
                    if isinstance(result, str):
                        merged_content = result.strip()
                    elif isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict):
                            generated_text = result[0].get("generated_text", [])
                            if (
                                isinstance(generated_text, list)
                                and len(generated_text) > 0
                            ):
                                last_msg = generated_text[-1]
                                if isinstance(last_msg, dict):
                                    merged_content = last_msg.get(
                                        "content", str(last_msg)
                                    ).strip()
                                else:
                                    merged_content = str(last_msg).strip()
                            elif isinstance(generated_text, str):
                                merged_content = generated_text.strip()
                            else:
                                merged_content = str(generated_text).strip()
                        else:
                            merged_content = str(result[0]).strip()
                    else:
                        merged_content = str(result).strip()

                    # Remove <think> tags if present (but allow facts from thinking to be preserved)
                    import re

                    # Remove closed think tags
                    merged_content = re.sub(
                        r"<think>.*?</think>", "", merged_content, flags=re.DOTALL
                    )
                    # Remove unclosed think tags
                    merged_content = re.sub(
                        r"<think>.*$", "", merged_content, flags=re.DOTALL
                    )
                    # Clean up extra whitespace
                    merged_content = re.sub(r"\s+", " ", merged_content).strip()

                    if merged_content and len(merged_content) > 5:
                        kept_memory.content = merged_content
                        # Regenerate embedding for merged content
                        embedding = self._get_embedding(merged_content)
                        if embedding is not None:
                            kept_memory.embedding = embedding.tolist()
                except Exception as e:
                    print(f"Warning: LLM merge failed, using kept memory: {e}")
                    # Fall back to keeping the best memory as-is

            # Update metadata to indicate consolidation
            if "consolidated_from" not in kept_memory.metadata:
                kept_memory.metadata["consolidated_from"] = []
            kept_memory.metadata["consolidated_from"].extend([m.id for m in to_merge])
            kept_memory.metadata["consolidated_at"] = datetime.now().isoformat()

            merged_count += len(to_merge)

        # Remove merged memories
        for mem_id in to_remove:
            memory = self.memories[mem_id]
            # Remove from cache
            if memory.content in self._embedding_cache:
                del self._embedding_cache[memory.content]
            del self.memories[mem_id]

        if merged_count > 0 or removed_count > 0:
            self._save_memories()

        return {
            "merged": merged_count,
            "removed": removed_count,
            "kept": len(self.memories),
        }

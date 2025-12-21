# Memory RAG Implementation Plan

## Overview
Transform the current static memory system (all memories inserted into system prompt) into a dynamic Retrieval-Augmented Generation (RAG) system that retrieves relevant memories based on context.

## Current State
- **Storage**: Simple text file (`memory.txt`) with one memory per line
- **Usage**: All memories concatenated and inserted into system prompt at initialization
- **Limitations**: 
  - All memories sent every time (token inefficient)
  - No relevance filtering
  - Can't scale well with many memories
  - No semantic understanding

## Proposed Architecture

### Phase 1: Core Infrastructure

#### 1.1 Create `MemoryManager` class (`virgil/friend/memory.py`)
```python
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
        mode: str = "static",  # "static", "rag", or "hybrid"
        storage_file: str = "memory.txt",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_memories: int = 10,  # Max memories to retrieve
        similarity_threshold: float = 0.3,  # Minimum similarity score
        use_gpu: bool = True,
    ):
        ...
    
    def add_memory(self, content: str, metadata: Optional[dict] = None):
        """Add a new memory with optional metadata."""
        
    def remove_memory(self, content: str):
        """Remove a memory by exact match."""
        
    def get_relevant_memories(
        self, 
        query: str, 
        max_results: Optional[int] = None
    ) -> List[Memory]:
        """Retrieve relevant memories for a query."""
        
    def get_all_memories(self) -> List[str]:
        """Get all memories (for static mode or fallback)."""
```

#### 1.2 Create `Memory` dataclass
```python
@dataclass
class Memory:
    """Represents a single memory entry."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[dict] = None
    created_at: datetime = None
    accessed_at: Optional[datetime] = None
    access_count: int = 0
```

#### 1.3 Storage Format
- **Option A**: JSON file (`memories.json`) with embeddings stored
- **Option B**: Separate files: `memories.json` (metadata) + `embeddings.npy` (vectors)
- **Option C**: SQLite database (better for large-scale, but adds dependency)

**Recommendation**: Start with Option A (JSON) for simplicity, can migrate later.

### Phase 2: Embedding & Retrieval

#### 2.1 Embedding Generation
- Use `sentence-transformers` library (lightweight, fast)
- Model: `all-MiniLM-L6-v2` (80MB, good quality/speed tradeoff)
- Generate embeddings lazily (on first RAG query or when memory added)
- Cache embeddings in memory and on disk

#### 2.2 Similarity Search
- Use cosine similarity for vector comparison
- Implement efficient search:
  - For small sets (<1000): brute force (simple, fast enough)
  - For larger sets: Consider FAISS or Annoy (optional optimization)

#### 2.3 Retrieval Strategy
```python
def get_relevant_memories(self, query: str, max_results: int = 10):
    """
    Retrieve memories relevant to query.
    
    Steps:
    1. Generate embedding for query
    2. Compute similarity with all memory embeddings
    3. Filter by threshold
    4. Sort by similarity (descending)
    5. Return top N results
    """
```

### Phase 3: Integration

#### 3.1 Update `Friend` class
- Add `memory_mode` parameter to `__init__`
- Replace `self.memory` list with `self.memory_manager: MemoryManager`
- Update prompt generation to use dynamic retrieval

#### 3.2 Dynamic Prompt Generation
```python
def _get_contextual_memories(self, query: str) -> str:
    """Get relevant memories for current query."""
    if self.memory_manager.mode == "static":
        return "\n".join(self.memory_manager.get_all_memories())
    else:
        memories = self.memory_manager.get_relevant_memories(query)
        return "\n".join([m.content for m in memories])
```

#### 3.3 Update `handle_task` method
- Before calling `chat.prompt()`, retrieve relevant memories
- Inject into prompt dynamically (not at initialization)
- Format: Same as current format for compatibility

### Phase 4: Configuration & Options

#### 4.1 Command-line Options
```python
@click.option(
    "--memory-mode",
    type=click.Choice(["static", "rag", "hybrid"]),
    default="static",  # Backward compatible default
    help="Memory retrieval mode: static (all memories), rag (semantic search), hybrid (rag with static fallback)"
)
@click.option(
    "--memory-max-results",
    default=10,
    help="Maximum number of memories to retrieve in RAG mode"
)
@click.option(
    "--memory-similarity-threshold",
    default=0.3,
    help="Minimum similarity score for memory retrieval (0.0-1.0)"
)
```

#### 4.2 Hybrid Mode Logic
```python
if mode == "hybrid":
    if len(memories) < 20:  # Small set, use static
        return get_all_memories()
    else:  # Large set, use RAG
        return get_relevant_memories(query)
```

### Phase 5: Migration & Backward Compatibility

#### 5.1 Migration Script
- Read existing `memory.txt`
- Convert to new format
- Generate embeddings for existing memories
- Save as `memories.json`

#### 5.2 Backward Compatibility
- If `memory.txt` exists but `memories.json` doesn't:
  - Auto-migrate on first run
  - Keep `memory.txt` as backup
- Support both formats during transition

### Phase 6: Advanced Features (Future)

#### 6.1 Metadata Support
- Store user names, channel names, timestamps
- Filter by metadata (e.g., "memories about user X")
- Temporal relevance (recent memories weighted higher)

#### 6.2 Memory Clustering
- Group similar memories
- Deduplication
- Memory summarization

#### 6.3 Performance Optimizations
- Batch embedding generation
- Incremental updates
- Embedding cache invalidation
- Lazy loading for large memory sets

## Implementation Steps

### Step 1: Create Core Classes
1. Create `virgil/friend/memory.py`
2. Implement `Memory` dataclass
3. Implement `MemoryManager` with static mode first
4. Add unit tests

### Step 2: Add Embedding Support
1. Add `sentence-transformers` dependency
2. Implement embedding generation
3. Implement similarity search
4. Add RAG mode

### Step 3: Integration
1. Update `Friend.__init__` to use `MemoryManager`
2. Update prompt generation to be dynamic
3. Update `handle_task` to retrieve memories per query
4. Update `_handle_remember_action` and `_handle_forget_action`

### Step 4: Migration
1. Create migration script
2. Test with existing memory files
3. Update documentation

### Step 5: Testing & Refinement
1. Test with various memory sizes
2. Tune similarity thresholds
3. Performance benchmarking
4. User feedback collection

## Dependencies

### Required
- `sentence-transformers` (for embeddings)
- `numpy` (already present, for vector ops)

### Optional (for future optimization)
- `faiss-cpu` or `faiss-gpu` (for large-scale similarity search)
- `annoy` (alternative to FAISS)

## File Structure
```
virgil/friend/
  ├── memory.py          # New: MemoryManager class
  ├── friend.py          # Updated: Use MemoryManager
  ├── prompt.txt         # Updated: Remove static memory section
  └── my_prompt.txt      # Updated: Remove static memory section
```

## Prompt Changes

### Current
```
Your current memories are:
----
{memories}
----
End memories.
```

### New (Dynamic)
```
Relevant memories for this conversation:
----
{relevant_memories}  # Dynamically inserted based on query
----
End memories.
```

## Benefits

1. **Scalability**: Can handle thousands of memories efficiently
2. **Relevance**: Only relevant memories included per query
3. **Token Efficiency**: Reduces prompt size significantly
4. **Performance**: Faster with large memory sets
5. **Flexibility**: Easy to add metadata, filtering, etc.

## Risks & Mitigations

1. **Risk**: Embedding model adds dependency
   - **Mitigation**: Use lightweight model, make optional

2. **Risk**: Migration complexity
   - **Mitigation**: Auto-migration, backward compatibility

3. **Risk**: Performance overhead
   - **Mitigation**: Lazy loading, caching, optional mode

4. **Risk**: Quality of retrieval
   - **Mitigation**: Configurable thresholds, hybrid mode fallback

## Success Metrics

- Memory retrieval time < 100ms for <1000 memories
- Relevant memories retrieved > 80% accuracy (manual evaluation)
- Token usage reduction > 50% for large memory sets
- Backward compatibility maintained

# Retrieval Memory Map

This document covers only the retrieval-memory part of Devenv: the files that store, index, retrieve, inspect, and lightly control memory during request-time context recall. It intentionally excludes consolidation-specific implementation details except where they touch retrieval-facing APIs.

## End-to-End Flow

At runtime, retrieval memory works like this:

1. `core/runtime/kernel.py` records bounded working memory for the current conversation.
2. The kernel applies a cheap gateway layer first and can skip retrieval entirely for low-context prompts like acknowledgements or shell-style commands.
3. The kernel decides whether to use lexical recall, hybrid semantic retrieval, or skip retrieval for the prompt shape.
4. `core/memory/engine.py` is the main entry point and delegates retrieval to `core/memory/retrieval.py`.
5. `core/memory/retrieval.py` embeds the prompt, runs vector and FTS seed lookup, fuses them, optionally expands immediate neighbors, and rescoring them before selecting the final context.
6. `core/memory/storage.py` provides the SQLite-backed node/log/edge/state access used during expansion and bookkeeping.
7. `core/memory/vector_index.py` provides the semantic lookup layer.
8. `core/memory/working_memory.py` contributes recent in-session context to both query composition and the final prompt block.
9. The kernel persists the retrieval trace so tools like `inspect_trace` can explain what was remembered.

## Core Retrieval Files

### `core/memory/engine.py`

What it does:
- Owns the concrete `MemoryEngine` implementation.
- Wires together the embedder, vector index, SQLite store, working-memory manager, and retrieval service.
- Exposes the public retrieval API: `retrieve_context(current_prompt, top_k=5)`.

How it works:
- On startup it builds the retrieval stack and calls `_rehydrate_vector_index()` so a non-persistent vector index can be rebuilt from stored nodes.
- On startup it builds the retrieval stack only; it no longer performs startup vector-index rehydration.
- `retrieve_context()` delegates to `RetrievalService.retrieve(...)` and caches the latest `RetrievalTrace`.
- `record_working_memory()` updates the in-process working-memory buffer used during retrieval.
- `add_episodic_log()` also creates an `episodic_*` memory node and indexes it, which means past turns become searchable retrieval material.
- `_ensure_workspace_node()` creates a stable workspace parent node so episodic memories are grouped under the current repo/workspace.

### `core/memory/retrieval.py`

What it does:
- Implements the actual retrieval algorithm.
- Produces both the prompt-ready markdown context and the auditable retrieval trace.

How it works:
- `_compose_query()` combines the current prompt with up to four recent working-memory messages so vague follow-ups have better semantic grounding.
- `retrieve()` embeds that composed query, runs vector search plus SQLite FTS seed search, and returns working memory only if there are no matches.
- `_fuse_seed_matches()` merges lexical and semantic seeds with reciprocal rank fusion (RRF).
- `_expand_candidates()` turns each vector match into a wider candidate set:
  - the seed node itself
  - its immediate parent
  - its top direct siblings
  - top graph-related nodes via edges
- `_should_short_circuit_expansion()` skips graph expansion entirely when the top semantic seed is already high confidence.
- Relationship expansion is discounted with fixed factors:
  - parent: `0.92`
  - sibling: `0.84`
  - related edge: `0.8`
- Expansion is bounded:
  - parent depth: `1`
  - siblings: top `3`
  - related nodes: top `3`
- `_score_candidates()` normalizes three signals and combines them:
  - similarity: `60%`
  - access frequency: `20%`
  - recency: `20%`
- Recency now uses exponential decay instead of a simple inverse-elapsed score.
- `_compile_markdown()` emits:
  - `## Working Memory`
  - `## Retrieved Memory`
- After selection it calls `store.touch_nodes(...)` so future retrievals can use access count and recency.

### `core/memory/storage.py`

What it does:
- Implements the SQLite persistence layer used by retrieval.
- Stores associative nodes, graph edges, episodic logs, and engine state.

How it works:
- Initializes four tables:
  - `memory_nodes`
  - `node_edges`
  - `episodic_logs`
  - `engine_state`
- Retrieval-specific methods:
  - `get_node()` and `list_nodes()` fetch memory nodes.
  - `get_parent_chain()` recursively climbs the hierarchy for contextual expansion.
  - `get_sibling_nodes()` returns same-parent neighbors.
  - `get_related_nodes()` returns graph-linked nodes from `node_edges`.
  - `touch_nodes()` updates `last_accessed` and increments `access_count`.
  - `search_nodes()` provides lexical fallback over labels and summaries.
  - `search_logs()` provides lexical fallback over episodic transcripts.
  - `get_state()` / `set_state()` persist retrieval metadata like `last_retrieval_trace`.
- `insert_log()` backfills `external_context_query` and `agent_response` so later history lookup can use structured fields instead of reparsing raw JSON every time.
- It also maintains optional FTS5 virtual tables:
  - `memory_nodes_fts`
  - `episodic_logs_fts`
- Retrieval-facing helpers added for hybrid search:
  - `search_nodes_fts()`
  - `search_logs_fts()`

### `core/memory/vector_index.py`

What it does:
- Defines the vector-index contract and both production/test implementations.

How it works:
- `VectorIndex` is the protocol used by the retrieval service.
- `LanceDBVectorIndex` is the persistent semantic index:
  - stores `node_id`, `vector`, and `text_chunk`
  - supports `upsert`, `delete`, `query`, and persisted-state detection
  - computes similarity from LanceDB score/distance, or falls back to cosine similarity
- `InMemoryVectorIndex` is the lightweight test/deterministic implementation:
  - keeps vectors in a Python dict
  - scores with cosine similarity
  - never reports persisted state, which lets `MemoryEngine` test rehydration logic

### `core/memory/embeddings.py`

What it does:
- Defines how text becomes vectors for retrieval.

How it works:
- `Embedder` is the protocol expected by the engine and retrieval service.
- `SentenceTransformerEmbedder` is the production embedder:
  - lazy-loads `sentence-transformers`
  - defaults to `all-MiniLM-L6-v2`
  - uses normalized embeddings
  - assumes local/offline model files
- `HashingEmbedder` is the deterministic fallback/testing embedder:
  - hashes tokens into a fixed-size vector
  - normalizes the result
  - avoids external model dependencies

### `core/memory/working_memory.py`

What it does:
- Maintains the bounded, in-process short-term memory used during retrieval.

How it works:
- `record()` keeps only the most recent `max_messages` entries and replaces the active-state snapshot.
- `snapshot()` returns the immutable current working-memory state.
- `as_prompt_block()` formats the working-memory section that gets prepended to retrieval output.
- Retrieval now uses drift detection before reusing recent working-memory text, so unrelated recent topics can be dropped from the composed query.
- This layer is not persisted to disk; it is session-local context used to improve recall quality for the current turn.

### `core/memory/models.py`

What it does:
- Defines the data contracts used across retrieval memory.

How it works:
- Retrieval structure models:
  - `MemoryNode`
  - `NodeEdge`
  - `VectorMatch`
  - `RetrievalCandidate`
  - `RetrievalSelectedNode`
  - `RetrievalTrace`
  - `RetrievalResult`
  - `WorkingMemoryMessage`
  - `WorkingMemorySnapshot`
- These dataclasses are what let the engine, store, vector index, retrieval service, and tooling pass structured memory state around without depending on raw dicts everywhere.

### `core/memory/interface.py`

What it does:
- Declares the expected memory-engine surface used by the rest of the app.

How it works:
- Defines the methods runtime code expects, especially:
  - `retrieve_context(...)`
  - `record_working_memory(...)`
  - `get_context_trace()`
  - `forget_node(...)`
- This keeps the runtime coupled to a stable contract rather than to one specific implementation detail.

### `core/memory/__init__.py`

What it does:
- Exposes the retrieval-memory public surface for imports.

How it works:
- Re-exports `MemoryEngine` and the retrieval-related dataclasses so callers can import from `core.memory` instead of each module directly.

## Runtime Integration Files

### `core/runtime/kernel.py`

What it does:
- This is the runtime orchestrator that decides when and how memory retrieval is used for a turn.

How it works:
- `_record_working_memory(...)` compacts the live conversation and stores it before later retrieval use.
- `_retrieve_memory_context(...)` is the main runtime retrieval path:
  - exits early for low-context prompts before touching lexical or vector retrieval
  - skips retrieval for certain current-workspace inspection prompts
  - tries lexical memory first for direct recall-style questions
  - falls back to vector retrieval via `self.memory.retrieve_context(...)`
  - optionally appends external session memory afterward
- `_retrieve_lexical_memory_context(...)` searches stored logs/nodes directly through `SQLiteMemoryStore.search_logs()` and `search_nodes()`.
- `_compact_lexical_memory_context(...)` tries to keep lexical memory concise while still answering the prompt.
- `_persist_last_retrieval_trace(...)` saves the latest trace into `engine_state` as JSON.
- `_build_memory_engine(...)` chooses the default embedder strategy:
  - hashing by default
  - sentence-transformer only when `DEVENV_USE_SENTENCE_EMBEDDER=1`
- Gateway helpers now classify:
  - low-context acknowledgements
  - shell-style prompts
  - prompts too small to justify retrieval
- There are also a large number of helper functions in this file that classify prompts such as:
  - whether something is a memory-recall question
  - whether vector retrieval should be skipped
  - whether a memory-only answer is acceptable
  - how to parse and validate memory-context lines

Why this file matters:
- The memory engine itself retrieves nodes, but `kernel.py` decides whether retrieval should happen at all and how the result is shaped into the final prompt/answer path.

### `core/runtime/context_builder.py`

What it does:
- Builds external session memory context that can be appended after local retrieval memory.

How it works:
- It is not the core local memory engine, but it participates in the broader retrieval-memory story by searching indexed prior session chunks.
- The kernel calls `build_runtime_memory_context(...)` on this service after local memory retrieval if external context is still useful.
- In practical terms, this file handles cross-session recall outside the local associative/vector memory stored in `memory.db` and `vectors/`.

## Retrieval Inspection and Control Tools

### `core/tools/inspect_trace.py`

What it does:
- Exposes retrieval introspection to the agent/tool layer.

How it works:
- `last_retrieval` returns the latest `RetrievalTrace`.
- If the in-memory trace is empty, it falls back to the persisted `last_retrieval_trace` in `engine_state`.
- `node_history` returns the node payload, its edges, and whether a vector entry exists.
- Both modes now also expose retrieval sync state such as:
  - last synced node id
  - last vector delete id
  - last sync timestamp
  - whether FTS is enabled

Why it matters:
- This is the main debugging/audit tool for understanding why the system remembered something.

### `core/tools/manage_memory.py`

What it does:
- Provides manual correction controls for retrieval memory.

How it works:
- `prune` removes a node via `forget_node(..., strategy="prune")`.
- `update` rewrites a node summary through `update_associative_tree(...)`, preserving parent/category/edges when possible.
- Tool results now also include vector-sync metadata so manual memory edits are auditable.

Why it matters:
- Retrieval quality depends on memory quality, so this tool is part of the operational retrieval-memory surface even though it is not part of the ranking algorithm itself.

## Retrieval-Focused Tests

### `tests/memory/test_retrieval.py`

What it verifies:
- parent-chain expansion works
- related-node recall works
- empty indexes still return working memory
- candidate scores are normalized
- working memory helps vague follow-up recall
- vector index rehydration works across sessions

### `tests/memory/test_engine_core.py`

What it verifies:
- updating associative nodes writes both SQLite state and vector-index state
- episodic logs are serialized and indexed into retrieval memory
- updating a node refreshes the indexed summary text

### `tests/memory/test_storage.py`

What it verifies:
- node persistence
- edge replacement
- episodic log insertion
- engine state persistence

### `tests/memory/test_vector_index.py`

What it verifies:
- similarity search returns the most relevant node ordering in the in-memory implementation

### `tests/memory/test_working_memory.py`

What it verifies:
- working memory stays bounded
- the engine correctly records working-memory state

### `tests/memory/test_performance.py`

What it verifies:
- retrieval expansion stays bounded on a small graph and does not explode candidate count

### `tests/tools/test_inspect_trace.py`

What it verifies:
- retrieval traces can be inspected after a recall
- node-history inspection returns node and vector-presence details

### `tests/tools/test_manage_memory.py`

What it verifies:
- manual node updates change retrieval memory summaries
- prune deletes nodes from retrieval memory

### `tests/runtime/test_kernel.py`

What it verifies:
- runtime retrieval decisions, including:
  - when lexical retrieval is preferred
  - when vector retrieval is skipped
  - how follow-up prompts are anchored
  - how retrieval context is shaped before the model call
  - how memory traces and session-memory metadata are handled

## Retrieval Data Stores

### `memory.db`

What it does:
- SQLite database holding the persistent local memory state used by retrieval.

How it is used:
- stores nodes, edges, episodic logs, and retrieval-related engine state
- queried by `SQLiteMemoryStore`

### `vectors/`

What it does:
- Persistent vector store directory for LanceDB-backed semantic retrieval.

How it is used:
- stores vectorized node summaries
- queried by `LanceDBVectorIndex`

## Scope Notes

Files intentionally not documented in depth here:
- `core/memory/consolidation.py`
- `core/memory/extractors.py`
- `core/memory/prd_*`

Reason:
- they are about creating/updating memory from logs, not about the retrieval-memory path itself
- they only matter indirectly because retrieval consumes the nodes/logs they produce

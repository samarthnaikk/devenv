# Devenv Memory Engine

`core.memory` is the local-first memory package for Devenv. It keeps raw interaction logs, structured associative memory, a bounded in-process working-memory buffer, and a batch consolidation flow behind one Python interface.

## Responsibilities

- `Working memory`: recent messages plus active state like file paths or shell errors.
- `Episodic memory`: append-only timestamped logs of user and agent interactions.
- `Associative memory`: hierarchical nodes and graph edges stored in SQLite, with semantic lookup through LanceDB.
- `Consolidation`: manual batch processing of fresh episodic logs into structured node creates and updates.

## Public API

The main entry point is `MemoryEngine`.

```python
from core.memory import MemoryEngine

engine = MemoryEngine(db_path="memory.db", vector_dir="vectors")

engine.record_working_memory(
    messages=[{"role": "user", "content": "Fix the Django auth flow"}],
    active_state={"file": "core/memory/engine.py"},
)

engine.update_associative_tree(
    {
        "node_id": "proj_rxgpt",
        "label": "Project: RxGPT",
        "category": "project",
        "summary": "RxGPT uses React, Tailwind, and Django.",
    }
)

result = engine.retrieve_context("How do I fix the auth flow again?")
print(result.markdown_context)
```

## Storage Layout

- SQLite stores `memory_nodes`, `node_edges`, `episodic_logs`, and `engine_state`.
- LanceDB stores `node_id`, `vector`, and `text_chunk` rows for associative summaries.
- Working memory is ephemeral and remains in process only.

## Retrieval Behavior

`retrieve_context()` performs:

1. A cheap query-composition pass that can drop unrelated working-memory history when the topic drifts.
2. Prompt embedding plus lexical FTS lookup against stored nodes.
3. Hybrid seed fusion using reciprocal rank fusion (RRF).
4. Immediate-neighbor-only expansion for parents, siblings, and related edges unless a high-confidence seed short-circuits expansion.
5. Normalized scoring with similarity, frequency, and exponential recency decay.
6. Markdown context compilation plus an auditable retrieval trace.

The latest retrieval trace is available through `get_context_trace()`.

## Consolidation

`run_consolidation()` processes logs newer than the saved watermark unless an explicit `since` value is provided.

The default extractor is deterministic and metadata-driven:

- `metadata["project"]` sets the detected project in the result.
- `metadata["memory_entities"]` creates new nodes or updates matching labels.
- `metadata["memory_updates"]` appends summaries to existing node IDs.
- `associated_node_id` appends a lightweight summary of the interaction to that node.

This keeps the package fully testable without a live model while preserving a clean seam for a future model-backed extractor.

## Integration Notes

- CLI, web, or agent layers should call `record_working_memory()` before retrieval when current state changes.
- UI layers can expose `get_context_trace()` to implement a "See what I'm remembering" view.
- Explicit corrections can use `forget_node()` for prune or rewrite behavior, or `update_associative_tree()` for precise edits.
- Production use expects `lancedb` and `sentence-transformers` to be installed locally.

## Tests

Run the memory suite with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

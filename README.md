# Devenv

Devenv is a local-first coding agent project. The current implemented foundation is the **Cognitive Memory Engine (CME)** described in the PRDs under [`core/memory/prd_objective.md`](/Users/samarthnaik/Desktop/Projects/devenv/core/memory/prd_objective.md:1) and [`core/memory/prd_tech.md`](/Users/samarthnaik/Desktop/Projects/devenv/core/memory/prd_tech.md:1).

Today, this repository is centered on a working Python package, `core.memory`, plus a small tools foundation. The memory engine is built to remove the idea of isolated chat sessions and instead support continuous, auditable memory across:

- working memory for the current task window
- episodic memory for timestamped interaction history
- associative memory for structured project, component, and preference recall
- consolidation for turning raw logs into reusable memory nodes

## What Is Done

The following PRD-driven functionality is implemented in this repository today:

- A decoupled `MemoryEngine` interface in `core.memory` with injectable storage, embeddings, vector index, and consolidation extractor.
- Working memory support with a bounded recent-message buffer and active session state snapshot.
- Episodic memory logging with timestamped user/agent interactions and optional metadata.
- Associative memory storage in SQLite using hierarchical nodes plus lateral graph edges.
- Vector-backed semantic lookup for associative summaries.
- Retrieval with spreading-activation behavior:
  parent-chain expansion, sibling expansion, related-edge expansion, normalized ranking, and markdown context compilation.
- Dynamic ranking signals based on similarity, access frequency, and recency.
- Auditable retrieval traces through `get_context_trace()`, including matched nodes, expanded candidates, selected nodes, and the final injected markdown block.
- Manual memory correction through `forget_node()` with both `prune` and `rewrite` strategies.
- Consolidation flow that processes new episodic logs, creates new nodes, updates existing nodes, refreshes vectors, and stores a consolidation watermark.
- A deterministic heuristic extractor seam so consolidation is testable without a live LLM.
- Unit tests covering imports, storage, working memory, retrieval, consolidation, manual control, and vector lookup behavior.

## PRD Alignment

The current implementation covers a substantial part of the memory PRDs:

- **Working Memory Manager:** implemented
- **Episodic Memory timeline:** implemented
- **Associative tree / graph structure:** implemented with SQLite nodes and edges
- **Spreading activation retrieval:** implemented
- **Importance and decay scoring:** implemented through normalized similarity, frequency, and recency scoring
- **Auditable context trace:** implemented
- **Manual memory correction:** implemented
- **Asynchronous sleep consolidation:** partially implemented
  the consolidation service exists and is ready to be called as a background task, but the repo does not yet include an always-on inactivity scheduler or terminal-event trigger loop

## Current Architecture

### `core.memory`

Main public entry point:

```python
from core.memory import MemoryEngine

engine = MemoryEngine(db_path="memory.db", vector_dir="vectors")
```

Implemented responsibilities:

- `record_working_memory(messages, active_state)`
- `add_episodic_log(user_prompt, agent_response, node_id=None, metadata=None)`
- `update_associative_tree(node_data)`
- `retrieve_context(current_prompt, top_k=5)`
- `run_consolidation(since=None)`
- `forget_node(node_id, strategy="prune")`
- `get_context_trace()`

Storage model:

- **SQLite** stores:
  `memory_nodes`, `node_edges`, `episodic_logs`, and engine state such as the last consolidation watermark.
- **LanceDB** is the production vector store for associative summaries.
- **In-memory test doubles** exist for the vector index and embedder so the system can be tested quickly and deterministically.

### `core.tools`

There is also a small tools foundation already implemented:

- `BaseTool`
- `ToolResult`
- `ReadFileTool`

`ReadFileTool` supports content reads plus optional metadata and extension analysis in one call.

## Repository Layout

```text
core/
  memory/
    README.md
    prd_objective.md
    prd_tech.md
    interface.py
    engine.py
    retrieval.py
    consolidation.py
    storage.py
    vector_index.py
    embeddings.py
    working_memory.py
    extractors.py
    models.py
  tools/
    base.py
    read_file.py
tests/
  memory/
pyproject.toml
README.md
```

## Setup

Python 3.12+ is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

The production memory stack expects these local dependencies:

- `lancedb`
- `sentence-transformers`

The test suite primarily uses lightweight in-memory test doubles instead of the production embedding/vector stack.

## Example

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

engine.add_episodic_log(
    "We introduced a Django auth component.",
    "I'll remember the backend shape.",
    node_id="proj_rxgpt",
    metadata={
        "project": "RxGPT",
        "memory_entities": [
            {
                "node_id": "cmp_django_auth",
                "label": "Django Auth Setup",
                "category": "component",
                "summary": "Django auth relies on session cookies and middleware.",
                "parent_id": "proj_rxgpt",
            }
        ],
    },
)

engine.run_consolidation()
result = engine.retrieve_context("How do I fix my django authentication errors?")

print(result.markdown_context)
print(engine.get_context_trace())
```

## Tests

Run the memory test suite with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

The current suite covers:

- import boundaries
- working memory bounds and snapshots
- episodic log persistence
- associative node and edge storage
- vector index ranking
- hierarchical retrieval and preference recall
- retrieval trace scoring normalization
- manual prune and rewrite behavior
- consolidation creation, update, and watermark behavior

## Not Done Yet

The README should be clear about what is still future scope from the PRDs:

- no CLI, web UI, or phone companion is implemented yet
- no always-on background scheduler for inactivity-based consolidation yet
- no cross-device sync yet
- no multi-repo memory sharing yet
- no full agent orchestration loop yet
- no secure remote execution layer yet

## Development Notes

- The code is organized to keep memory logic decoupled from future UI or agent layers.
- Tests use dependency injection heavily so memory behavior can be verified without external services.
- The local-first constraint from the PRDs is preserved in the package design: raw logs, structured memory, and vector lookup are intended to live on the user machine.

./.venv/bin/python -m core.runtime.web sample-test
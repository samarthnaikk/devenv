# Devenv AI

Devenv AI is a local-first coding agent foundation for running project-aware workflows on your machine. It combines a runtime layer with a persistent Cognitive Memory Engine (CME) so interactions can build on structured, auditable memory instead of acting like isolated chat sessions.

The project currently ships as an installable Python package and includes:

- an interactive terminal runtime
- a local web runtime
- a one-shot smoke runner for single prompts
- an MCP server for exposing local tools
- a memory engine with working, episodic, and associative memory layers

## Installation

Python 3.12 or newer is required.

Install from PyPI:

```bash
pip install devenv-ai
```

Or with `uv`:

```bash
uv pip install devenv-ai
```

## Quick Start

Point Devenv AI at the folder you want it to work inside.

Devenv's OpenCode integration is server-backed by default. Install the `opencode` CLI and make sure `opencode serve` can run locally; Devenv will connect to the OpenCode HTTP server at `http://127.0.0.1:4096` by default. You can override this with:

```bash
export OPENCODE_SERVER_URL=http://127.0.0.1:4096
export OPENCODE_SERVER_USERNAME=opencode
export OPENCODE_SERVER_PASSWORD=your-password
```

Launch the local web experience:

```bash
cd /path/to/your/project
devenv-web .
```

Then open:

```text
http://127.0.0.1:4173
```

Launch the terminal experience:

```bash
cd /path/to/your/project
devenv-run .
```

Run a single prompt without entering the interactive loop:

```bash
devenv-smoke . "summarize this repository"
```

Start the local MCP tool server:

```bash
devenv-mcp --workspace .
```

## Screenshots

### Startup chunking

![Startup chunking progress](./docs/screenshots/devenv-startup-chunking.png)

### Web UI, dark theme

![Devenv web UI in dark theme](./docs/screenshots/devenv-web-dark.png)

### Web UI, light theme

![Devenv web UI in light theme](./docs/screenshots/devenv-web-light.png)

## Installed Commands

After installation, the package exposes these commands:

- `devenv-run`
- `devenv-web`
- `devenv-smoke`
- `devenv-mcp`

## What It Does Today

The current implementation is centered on the Cognitive Memory Engine and a small runtime/tooling foundation.

Implemented today:

- bounded working memory for the current task window
- episodic logging for timestamped user and agent interactions
- associative memory storage using hierarchical nodes and graph edges
- semantic retrieval over associative summaries
- spreading-activation style retrieval with parent, sibling, and related-node expansion
- auditable retrieval traces via `get_context_trace()`
- manual memory correction through `forget_node()`
- consolidation flows that can create and update memory nodes from episodic logs
- an injectable architecture for storage, embeddings, vector indexes, and extraction logic

## Memory Engine Example

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

## Architecture

The codebase is organized to keep memory logic decoupled from future user interfaces and agent orchestration layers.

Key areas:

- `core.memory`: memory interfaces, storage, retrieval, consolidation, embeddings, and models
- `core.runtime`: terminal runtime, web runtime, MCP server, and runtime orchestration
- `core.tools`: base tool abstractions and local tool implementations
- `core.ai`: OpenCode transport, routing, and model-facing contracts

### OpenCode runtime architecture

Devenv keeps control of planning, memory retrieval, verification, transcript persistence, and tool execution. OpenCode is used as the AI backend through its server/session APIs.

- Devenv talks to OpenCode through a Python HTTP client instead of scraping `opencode run` output
- OpenCode sessions are reused across a Devenv conversation and reset when a new thread starts
- Devenv tools remain the only executable tool surface; OpenCode can request them, but Devenv validates and executes them
- runtime tool execution uses an in-process transport by default to avoid extra MCP subprocess overhead; set `DEVENV_TOOL_TRANSPORT=mcp` if you explicitly want the stdio MCP hop
- the legacy CLI parser is still available only as an emergency fallback when `DEVENV_OPENCODE_USE_LEGACY_CLI=1`

Main public memory entry point:

```python
from core.memory import MemoryEngine
```

Core memory responsibilities include:

- `record_working_memory(messages, active_state)`
- `add_episodic_log(user_prompt, agent_response, node_id=None, metadata=None)`
- `update_associative_tree(node_data)`
- `retrieve_context(current_prompt, top_k=5)`
- `run_consolidation(since=None)`
- `forget_node(node_id, strategy="prune")`
- `get_context_trace()`

## Development Setup

For local development:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

The production memory stack expects local availability of:

- `lancedb`
- `sentence-transformers`

## Testing

Run the current test suite with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

The current suite covers memory imports, persistence, retrieval behavior, vector ranking, manual correction, and consolidation flows.

## Current Scope

This repository is still an early foundation, not a full end-user coding product.

Not implemented yet:

- no always-on inactivity scheduler for consolidation
- no cross-device sync
- no multi-repo memory sharing
- no full agent orchestration loop
- no secure remote execution layer

## License

This project is licensed under the MIT License. See [LICENSE](/Users/samarthnaik/Desktop/Projects/devenv/LICENSE:1).

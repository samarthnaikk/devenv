# Devenv

Devenv is a coding agent built from scratch. The goal is to create a local-first system that can inspect a codebase, reason about changes, edit files safely, run commands, and iterate on feedback in a way that is transparent and controllable.

This project is intended to be the foundation for an agent similar in spirit to tools like OpenCode or Codex, but implemented as a custom stack with explicit control over:

- planning and task execution
- repository inspection and search
- file editing and patch application
- command execution and verification
- conversation and action history
- tool orchestration

## Core Principles

- Keep the agent deterministic where possible.
- Prefer explicit state over hidden behavior.
- Make every action auditable.
- Treat the filesystem and shell as first-class tools.
- Optimize for correctness before autonomy.

## Expected Capabilities

- Read and summarize repository state
- Propose and apply code changes
- Run tests and surface failures
- Track plans and partial progress
- Support multi-step workflows across files and commands
- Work locally on macOS with Python as the primary implementation language

## Suggested Architecture

The project will likely evolve around a few core layers:

1. `agent` - orchestration, policy, task state, and tool selection
2. `tools` - shell, filesystem, search, patching, and other integrations
3. `memory` - short-term context, summaries, and persisted history
4. `models` - model/provider adapters and prompt formatting
5. `ui` - optional CLI or local interface for interacting with the agent
6. `tests` - unit and integration coverage for agent behavior

## Development Setup

This repository is currently a starting point. A typical Python workflow on macOS would look like:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

From there, add the runtime and development dependencies needed by the implementation.

## Repository Layout

The repository is intentionally minimal for now. As the implementation grows, expect to add:

- source packages under `src/`
- tests under `tests/`
- project metadata such as `pyproject.toml`
- local tooling configuration files

## Contributing

When adding features, prefer small, testable changes. If a change affects agent behavior, include tests that show the expected interaction or state transition.


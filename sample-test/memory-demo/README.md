## Memory Demo

This folder contains a small reproducible memory fixture for validating that Devenv recalls prior project context across fresh sessions.

### Files

- `demo_memory_recall.py`: seeds and exercises the demo memory database

### What it demonstrates

The demo stores memories for two separate projects:

- a calendar project with a React frontend and Python backend
- a job-management project with a Django backend and React admin UI

Then it opens a fresh `MemoryEngine` instance and asks a vague follow-up like:

`Not in the current directory, I mean the project we were working on earlier.`

The retrieval context should still surface the calendar project details because the working-memory topic and persisted episodic memory reinforce each other.

### Shared storage

This demo now uses the single global memory store at the root of this repo:

- `memory.db`
- `vectors/`

No separate database is kept under `sample-test/`.

### Seed the shared DB

```bash
./.venv/bin/python sample-test/memory-demo/demo_memory_recall.py --seed --force
```

### Run the recall demo

```bash
./.venv/bin/python sample-test/memory-demo/demo_memory_recall.py
```

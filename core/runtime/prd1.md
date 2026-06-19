# Technical Blueprint: The Devenv Kernel & TUI Runtime

## 1. Architectural Adjustments & The Sandbox Security Policy

To achieve a production-grade local-first agent, your developers must implement a strict architectural separation of powers:

### 1.1 The Isolation of AI (The Zero-Power Rule)

* **The Correction:** The `core/ai` module must remain a **pure calculator**. It receives string inputs and outputs a structural `AIResponse` containing a intent to call a tool. It has **no awareness** of whether tools actually exist, no access to the filesystem, and no capability to run code.
* **The Loop:** `core/runtime` inspects the `AIResponse`. If a tool call is requested, the runtime halts the LLM stream, validates the arguments against security parameters, executes the tool locally, and appends the raw text output back into the chat history for the next AI turn.

### 1.2 The Local Path Sandbox (Strict Directory Locking)

* **The Rule:** The agent must be restricted to a single **Workspace Source Path** (e.g., a specific project directory passed during initialization).
* **The Gatekeeper:** The runtime evaluates file paths inside a `ToolCallRequest` *before* passing them to any tool. If the AI requests a path outside the workspace (e.g., `../../etc/passwd` or an absolute root path), the runtime interceptor blocks execution instantly, returns a safety error message to the AI string buffer, and prompts the user for manual override approval via the TUI if desired.

---

## 2. Directory Layout Expansion

The `core/runtime` package will grow to support both core orchestration logic and user-facing terminal interface rendering elements:

```text
core/
  runtime/
    __init__.py          # Public exports (DevenvKernel, RunConfig)
    kernel.py            # The multi-turn execution loop & sandbox controller
    sandbox.py           # Path validation, boundary tracking, and permission gates
    tui.py               # Standard library curses/input loop for the text interface
    models.py            # Typed dataclasses for turn audits and session summaries

```

---

## 3. Data Models & API Interface Contracts

### 3.1 Public Data Models (`core/runtime/models.py`)

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class ToolExecutionStep:
    step_id: str
    tool_name: str
    arguments: Dict[str, Any]
    output: str
    success: bool
    is_sandboxed_violation: bool

@dataclass(frozen=True)
class RuntimeTurnResult:
    final_response: Optional[str]
    steps: List[ToolExecutionStep] = field(default_factory=list)
    total_usage: Dict[str, int] = field(default_factory=dict) # Aggregated token analytics

```

### 3.2 Core Runtime Orchestration Contract (`core/runtime/kernel.py`)

```python
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.memory import MemoryEngine
from core.ai import AICore
from core.tools.base import BaseTool
from .models import RuntimeTurnResult, ToolExecutionStep
from .sandbox import PathSandbox

class DevenvKernel:
    def __init__(self, workspace_path: str, db_path: str = "memory.db", vector_dir: str = "vectors"):
        # Initialize paths and absolute sandbox boundaries
        self.sandbox = PathSandbox(root_path=workspace_path)
        
        # Instantiate lower tiers completely isolated from one another
        self.memory = MemoryEngine(db_path=db_path, vector_dir=vector_dir)
        self.ai = AICore() 
        
        self.tools: Dict[str, BaseTool] = {}
        self.ephemeral_history: List[Dict[str, Any]] = []

    def register_tool(self, tool: BaseTool) -> None:
        """Saves tool locally and synchronizes its structural schema with the AI tier."""
        self.tools[tool.name] = tool
        self.ai.register_tool(tool)

    def execute_turn(self, user_prompt: str, max_consecutive_tools: int = 5) -> RuntimeTurnResult:
        """
        Coordinates the unified execution loop:
        1. Query memory.retrieve_context(user_prompt) for dynamic markdown tracking.
        2. Format state into AI execution parameters.
        3. Loop call structures to intercept, check, and safely run tools.
        4. Log complete transactional traces down into Episodic records.
        """
        pass

```

---

## 4. Detailed Component Implementation Rules

### 4.1 The Security Sandbox (`core/runtime/sandbox.py`)

Your developers must write a robust path security class that resolves symbolic links and variations cleanly to prevent path traversal hacks:

```python
class PathSandbox:
    def __init__(self, root_path: str):
        self.allowed_root = Path(root_path).expanduser().resolve()

    def is_safe(self, target_path: str) -> bool:
        try:
            # Resolve the absolute location on disk, tracking any symlinks
            resolved_target = Path(target_path).expanduser().resolve()
            # Verify if the target path is a sub-directory or child of the allowed root
            return self.allowed_root in resolved_target.parents or resolved_target == self.allowed_root
        except Exception:
            return False

```

### 4.2 The Basic Terminal User Interface (TUI) (`core/runtime/tui.py`)

To ensure everything can be run immediately without web assets, the TUI must be built using Python's standard library `input()` stream processing loops or a lightweight text canvas. It must print explicit blocks tracking the execution sequence:

```text
================================================================================
 🧵 DEVENV CORE TUI v1.0 | Workspace: /Users/samarthnaik/Desktop/Projects/demo
================================================================================
[SYSTEM]: Connected to Groq Local Pipeline. Memory Engine Online.
devenv@local_workspace:~$ Read my pyproject.toml file and describe it.

⏳ [RETRIEVING MEMORY CONTEXT]... Found 2 related concept nodes.
🤖 [AI REASONING]... AI requests tool invocation: 'read_file'
🔒 [SANDBOX CHECK]: Path 'pyproject.toml' matches workspace boundary. Safe to run.
⚙️  [EXECUTING TOOL]: read_file (features: 'all') -> Success.

🤖 [AI PROCESSING OUTPUT]...
[ASSISTANT]: Your pyproject.toml lists dependencies for lancedb and sentence-transformers...

devenv@local_workspace:~$ _

```

---

## 5. Granular 15-Commit Step-by-Step Implementation Graph

This highly itemized commit plan guides developers to assemble and test every component incrementally:

### Phase 1: Foundations & Type Specs

* **Commit 1:** Implement runtime structural data tracking types (`ToolExecutionStep`, `RuntimeTurnResult`) inside `core/runtime/models.py`.
* **Commit 2:** Write the core structural boilerplate class definitions for `DevenvKernel` and initialize its argument capture fields.

### Phase 2: Security & Path Isolation

* **Commit 3:** Implement the complete `PathSandbox` validation suite checking path traversals and symbolic link mapping arrays.
* **Commit 4:** Add robust unit tests specifically validating the sandbox with safe and unsafe parameters inside `tests/runtime/test_sandbox.py`.

### Phase 3: The Unified Orchestration Loop

* **Commit 5:** Wire up tool mapping pipelines so calling `DevenvKernel.register_tool` updates both local lists and nested `AICore` definitions.
* **Commit 6:** Write the structural core loop of `execute_turn()` to capture user entries and successfully pull Markdown blocks from `core/memory`.
* **Commit 7:** Implement the tool execution interceptor loop, ensuring that when the AI returns a `ToolCallRequest`, execution pauses before any tool runs.
* **Commit 8:** Embed the sandbox gate inside the execution path loop to flag unauthorized operations before they hit the tools tier.
* **Commit 9:** Implement the execution mapping link that fetches the target tool from the dictionary and runs `.execute(arguments)`.
* **Commit 10:** Implement the feedback logic that attaches tool output payloads directly back into the rolling chat message array for multi-turn execution.

### Phase 4: Memory Commits & Safety Margins

* **Commit 11:** Implement transaction logging that pushes complete user-agent turns into the `add_episodic_log` system tracking historical runs.
* **Commit 12:** Build the token counter aggregator that totals up multi-step turn costs across intermediate loops.

### Phase 5: Interactive TUI & Assembly Integration

* **Commit 13:** Design the TUI terminal interface execution screen loop inside `core/runtime/tui.py`, displaying clear visual feedback markers for memory lookup, reasoning steps, and sandbox checks.
* **Commit 14:** Write comprehensive integration tests inside `tests/runtime/test_kernel.py` executing mock scenarios to trace full multi-step loops safely.
* **Commit 15:** Build a top-level executable entry point (`devenv-run`) inside the project setup rules so typing a terminal command launches the entire framework instantly.

---

## 6. Verification and Integration Matrix

Your developers can immediately verify the absolute integrity of the whole system by launching the local TUI and running this precise test scenario:

1. **The Scenario:** Type: `"Read the file ../secrets.txt"` (Assuming a file exists outside the current directory workspace).
2. **Expected Verification Result:**
* The `DevenvKernel` catches the request, compiles the prompt, and passes it to `AICore`.
* `AICore` reads the command, detects it needs to read a file, and outputs a request to call `read_file` with path `../secrets.txt`.
* The `DevenvKernel` execution interceptor halts, evaluates the path using `PathSandbox`, and flags a security violation.
* The tool **never executes**. The runtime aborts or feeds an error message back to the system context stream, keeping your filesystem completely secure.
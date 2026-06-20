# Product Requirements Document (PRD)

## Project: Devenv Local MCP Tool Server Integration (`core/runtime`)

**Status:** Approved for Development | **Target Version:** 1.1

**Core Objective:** Wrap the 15 multi-mode tools developed for `core/tools` inside a compliant, zero-network-dependency local MCP Server. This server will communicate with the `DevenvKernel` (MCP Client) via an isolated standard input/output (`stdio`) process pipeline, eliminating hardcoded integration structures while enabling native Groq prompt caching.

---

## 1. System Topology & Architectural Rules

Developers **must** follow a strict multi-process boundary strategy to guarantee execution safety and modular isolation:

### 1.1 Process Separation (The STDIO Transport Contract)

* **The Server (`core/runtime/mcp_server.py`):** Operates as an independent executable sub-process. It loads your 15 tool modules and waits for structural instructions.
* **The Client (`core/runtime/kernel.py`):** Spawns and manages the server sub-process lifecycle using Python’s asynchronous subprocess utilities (`mcp.client.stdio.stdio_client`).
* **The Boundary:** The parent application and the tool deck exchange messages using standard JSON-RPC 2.0 payloads mapped strictly over the sub-process's input and output channels (`stdin`/`stdout`).

### 1.2 The Binary Logging Restriction (No Raw Prints)

* **Rule:** Tools must *never* use standard `print()` statements for telemetry or console logging. Because the `stdio` pipeline treats `stdout` as a strict protocol wire, untracked strings will corrupt the stream and break the agent.
* **The Fix:** All debugging traces, framework errors, or performance metrics must be routed directly through `sys.stderr` or python's native `logging` library.

---

## 2. Technical Contracts & Adapter Implementation

Developers must implement an engine adapter file at `core/runtime/mcp_server.py` using the official open-source `mcp[cli]` Python library.

### 2.1 The Adapter Contract Blueprint

The code must dynamically map your production-grade `input_schema()` objects and route values directly to `.execute()` targets:

```python
# System entry tracking configuration reference - core/runtime/mcp_server.py
import sys
from mcp.server.fastmcp import FastMCP
from core.tools.list_dir import ListDirectoryTool
from core.tools.edit_file import EditFileTool
# ... Import all 15 multi-mode tool modules

# 1. Initialize the FastMCP supervisor instance
mcp = FastMCP("Devenv Local Tool Deck")

# 2. Instantiate core production tool engines
list_dir_instance = ListDirectoryTool()
edit_file_instance = EditFileTool()

# 3. Dynamic Handler Mapping Configuration
@mcp.tool(name=list_dir_instance.name, description=list_dir_instance.description)
def list_dir(**kwargs) -> str:
    """Dynamic adapter wrapper for repository traversing."""
    # FastMCP uses the underlying dict keywords passed by the LLM
    result = list_dir_instance.execute(**kwargs)
    return result.output

@mcp.tool(name=edit_file_instance.name, description=edit_file_instance.description)
def edit_file(**kwargs) -> str:
    """Dynamic adapter wrapper for workspace line editing edits."""
    result = edit_file_instance.execute(**kwargs)
    return result.output

# ... Map remaining 13 tools identically using the same wrapper flow

if __name__ == "__main__":
    # Launch stdio transport listener loop
    mcp.run()

```

---

## 3. Advanced Protocol Performance Configurations

To ensure the system works smoothly with your local file setup and Groq caching, developers must enforce the following runtime parameters:

### 3.1 Synchronized Schema Introspection

When the `DevenvKernel` establishes its client connection handshake, it must invoke `session.list_tools()`. The server must compile and output your custom `input_schema()` dictionaries precisely. This guarantees that your strict structural `mode` enumerations are declared cleanly over the wire so Groq can cache the token prefix perfectly.

### 3.2 Host Execution Interception (The Runtime Sandbox Gate)

The `PathSandbox` check must remain on the **Client/Runtime side** inside `kernel.py`.
When Groq requests a tool call, the `DevenvKernel` inspects the arguments *before* dispatching the request over the standard input stream. If a directory path violation is flagged, the client blocks transmission entirely, failing the turn safely before the sub-process can touch the operating system.

---

## 4. Test Verification & Debugging Matrix

Developers must add zero-cost integration tests inside `tests/runtime/test_mcp.py` to protect the pipeline against schema or transport breaking changes:

* **Test 1: JSON-RPC Capability Introspection**
* Instantiate the `FastMCP` app instance inside a test fixture. Query its internal tool mapping indexes. Assert that all 15 custom tool names exist and that their schema properties match your original specifications.


* **Test 2: Subprocess Lifecycle Termination**
* Spin up the `DevenvKernel` client loop and shut it down. Verify using mock assertions that the underlying Python `stdio_client` context manager sends termination signals to close the server process gracefully, avoiding zombie processes.


* **Test 3: Output Serialization Safety**
* Mock a tool call sequence returning an edge-case output string (e.g., source code containing quotes and odd indentation). Verify that the protocol encoder correctly handles escaping, preventing JSON decoding errors in the client loop.



---

## 5. Developer Atomic Commit Plan

Your developers should build this out incrementally across 4 clean commits:

1. **Commit 1:** Add `mcp` into `pyproject.toml` dependencies and set up the `mcp_server.py` boilerplate.
2. **Commit 2:** Map all 15 production multi-mode tools into `@mcp.tool` decorator functions inside the server module.
3. **Commit 3:** Update `core/runtime/kernel.py` to launch the tool server as a background subprocess using `stdio_client`.
4. **Commit 4:** Implement full async integration test suites under `tests/runtime/test_mcp.py` to finalize the phase.

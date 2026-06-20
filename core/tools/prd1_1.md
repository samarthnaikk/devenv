# Product Requirements Document (PRD)

## Project: Devenv Multi-Mode Tool Suite (`core/tools`)

**Status:** Approved for Development | **Target Version:** 1.0

**Core Objective:** Implement a comprehensive collection of 15 multi-mode tools that inherit from the decoupled `BaseTool` class architecture. These tools must provide the reasoning tier (`core/ai`) with optimized, high-level semantic summaries to protect token limits, while maintaining low-level mechanical fallbacks for explicit recovery.

---

## 1. Core Engineering Constraints & Architecture Rules

Developers **must** construct every new tool component strictly inside `core/tools/` using the precise architecture constraints established by the `ReadFileTool` paradigm:

1. **Strict Type Contracts:** Every tool class must inherit from `BaseTool` and explicitly implement the `input_schema(self) -> dict[str, object]` method, returning a valid OpenAI/Groq function parameter schema configuration.
2. **Unified Response Vectors:** The `execute(self, kwargs)` method must return a cleanly packed `ToolResult` dataclass object containing:
* `success` (boolean): Flag indicating error status.
* `output` (string): Textual summary describing what happened (sent to the AI message stream).
* `data` (dictionary): High-density structured output arrays (sent to the `core/runtime` pipeline for deep processing).


3. **No Direct OS Leaks:** Tools must *never* evaluate path security parameters natively inside their modules. They assume that execution is gated by the runtime's sandbox mechanism. They should catch standard target exceptions (`FileNotFoundError`, `PermissionError`, `OSError`) gracefully and return them inside `ToolResult(success=False)`.

---

## 2. Comprehensive Tool Specifications

### 2.1 Discovery & Navigation Modules

#### 1. `ListDirectoryTool` (`list_dir`)

* **Description:** Inspects and enumerates directories.
* **Modes (`mode` parameter):**
* `"flat"`: Returns immediate child file and folder listings only.
* `"recursive"`: Recursively crawls directories down to a configured `max_depth`.
* `"topology"`: Computes a clean high-level folder blueprint map, dynamically omitting noise matching target filters (e.g., `.venv`, `__pycache__`, `.git`).


* **Schema Bounds:** Requires `mode` (enum). Accepts optional `path` (string) and `max_depth` (integer, default `3`).

#### 2. `LocateFilesTool` (`locate_files`)

* **Description:** Scans the active sandboxed repository layout to flag files matching name filters.
* **Modes (`mode` parameter):**
* `"glob"`: Runs wildcard matching checks across directories (e.g., `/test_*.py`).
* `"exact"`: Locates literal matching file names on disk.


* **Schema Bounds:** Requires `pattern` (string) and `mode` (enum).

---

### 2.2 Inspection & Intelligence Modules

#### 3. `ReadFileTool` (`read_file`)

* **Description:** *Retain existing verified production capability.* Resolves code text payloads, size arrays, extensions, and kind maps across `"content"`, `"metadata"`, `"extension"`, and `"all"` operational modes.

#### 4. `PeekLinesTool` (`peek_lines`)

* **Description:** Implements an optimized chunked text window viewer to prevent prompt buffer crowding.
* **Modes (`mode` parameter):**
* `"range"`: Extracts file text strictly between specific line numbers.
* `"head"`: Extracts the leading slice of rows from a text document.
* `"tail"`: Extracts the trailing slice of rows from a text document.


* **Schema Bounds:** Requires `path` (string), `mode` (enum). Accepts optional `start` (integer) and `end` (integer).

#### 5. `InspectSymbolsTool` (`inspect_symbols`)

* **Description:** Uses Python AST tools to build high-level codebase definition maps without processing full block text strings.
* **Modes (`mode` parameter):**
* `"outline"`: Identifies class names and standalone function signatures inside a file.
* `"signatures"`: Extracts parameter schemas, argument constraints, and type annotations.
* `"documentation"`: Isolates docstring content blocks to deduce developer design intent.


* **Schema Bounds:** Requires `path` (string) and `mode` (enum).

---

### 2.3 Query & Tracking Modules

#### 6. `SearchTextTool` (`search_text`)

* **Description:** Hunts down textual configurations inside the workspace repository.
* **Modes (`mode` parameter):**
* `"literal"`: Standard mechanical sub-string search loop (fallback behavior).
* `"regex"`: Evaluates complex regular expressions over target files.
* `"semantic"`: Integrates with vector index stores to find conceptually related text matches.


* **Schema Bounds:** Requires `query` (string) and `mode` (enum). Accepts optional `ext_filter` (string).

#### 7. `TrackSymbolTool` (`track_symbol`)

* **Description:** Maps variable relationships and code dependencies across multiple files.
* **Modes (`mode` parameter):**
* `"references"`: Finds every file and line row number where a specific symbol string is active.
* `"definitions"`: Pins the exact initialization code block location of an imported element.


* **Schema Bounds:** Requires `symbol` (string) and `mode` (enum).

---

### 2.4 File Mutation & Engineering Modules

#### 8. `WriteFileTool` (`write_file`)

* **Description:** Writes new text structures or handles major file generation tasks.
* **Modes (`mode` parameter):**
* `"fresh"`: Generates a brand-new file (must fail explicitly if path exists to avoid accidents).
* `"overwrite"`: Completely replaces the target file layout block.
* `"append"`: Places text rows directly down at the bottom margin of an active file.


* **Schema Bounds:** Requires `path` (string), `content` (string), and `mode` (enum).

#### 9. `EditFileTool` (`edit_file`)

* **Description:** Handles targeted code modifications and modular corrections.
* **Modes (`mode` parameter):**
* `"patch"`: Executes clean search-and-replace block operations on code segments.
* `"undo"`: Reverts the immediate previous modification step on this target file path.


* **Schema Bounds:** Requires `path` (string), `mode` (enum). Requires `search_block` (string) and `replace_block` (string) if running in patch mode.

#### 10. `RemoveFileTool` (`remove_file`)

* **Description:** Wipes unneeded files safely inside the workspace boundary.
* **Modes (`mode` parameter):**
* `"soft"`: Truncates target file content but retains an empty tracking placeholder.
* `"permanent"`: Erases the file completely from the local layout.


* **Schema Bounds:** Requires `path` (string) and `mode` (enum).

---

### 2.5 Testing & System Diagnostics

#### 11. `RunShellTool` (`run_shell`)

* **Description:** Direct execution of terminal utilities for low-level shell fallbacks.
* **Modes (`mode` parameter):**
* `"raw"`: Executes non-interactive commands, capturing and returning standard stdout outputs.
* `"background"`: Spawns processes asynchronously without locking active runtime loops.


* **Schema Bounds:** Requires `command` (string), `mode` (enum). Accepts optional `timeout` (integer, default `30`).

#### 12. `RunDiagnosticsTool` (`run_diagnostics`)

* **Description:** High-level testing wrapper that returns concise structural summaries rather than stdout noise.
* **Modes (`mode` parameter):**
* `"tests"`: Automatically invokes localized testing frameworks (e.g., runs `pytest`).
* `"lint"`: Evaluates syntax formatting and style layout adjustments (e.g., `flake8`).
* `"types"`: Computes strict type annotation validity reports (e.g., `mypy`).


* **Schema Bounds:** Requires `mode` (enum). Accepts optional `target_path` (string).

#### 13. `AuditChangesTool` (`audit_changes`)

* **Description:** Compiles files state track adjustments during an active turn.
* **Modes (`mode` parameter):**
* `"diff"`: Extracts local uncommitted git modifications within the workspace.
* `"status"`: Summarizes edited, untracked, or staged file listings.


* **Schema Bounds:** Requires `mode` (enum). Accepts optional `path` (string).

---

### 2.6 Memory & Context Controllers

#### 14. `ManageMemoryTool` (`manage_memory`)

* **Description:** Allows the AI to intentionally update or drop knowledge nodes inside `core.memory`.
* **Modes (`mode` parameter):**
* `"prune"`: Safely deletes a specific stale node branch out of the SQLite graph.
* `"update"`: Modifies or appends structural context updates to an existing node summary.


* **Schema Bounds:** Requires `node_id` (string), `mode` (enum). Accepts optional `text` (string).

#### 15. `InspectTraceTool` (`inspect_trace`)

* **Description:** Direct diagnostic interface for checking cognitive retrieval operations.
* **Modes (`mode` parameter):**
* `"last_retrieval"`: Traces the exact activation values from the immediate query turn.
* `"node_history"`: Dumps historical hits, decay weights, and access frequency metrics for a node.


* **Schema Bounds:** Requires `mode` (enum). Accepts optional `node_id` (string).

---

## 3. Implementation Verification & Test Suite Requirements

To lock down the behavior of these tools before they are attached to your `DevenvKernel`, developers must write explicit tests under `tests/tools/` satisfying two constraints:

* **The Static Test Target:** Tools must be tested against predefined mock directories or static string fixtures.
* **Mode Switch Coverage:** Every test must run a validation cycle verifying that changing the `mode` parameter alterations yield correctly restructured `ToolResult` outputs.
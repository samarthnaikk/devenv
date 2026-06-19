# Technical Implementation Blueprint

## Project: Devenv AI Core Orchestrator (`core/ai`) — Groq Edition

**Status:** Ready for Development | **Target Version:** 1.0

**Core Objective:** Implement a decoupled, local-first Python package (`core.ai`) that transforms python tools into OpenAI function schemas, manages system prompt caching structures, and handles turn-by-turn routing via the Groq LPU platform.

---

## 1. Technical Requirements & Optimization Rules

To ensure maximum speed and cost-efficiency when making API requests, the development team **must** adhere to the following implementation guidelines:

### 1.1 Structural Layout for Groq Prompt Caching

Groq uses prefix-matching cache hardware. If the top section of your request payload remains unchanged across multiple conversation turns, processing speeds increase significantly. The code must compile prompts in this exact order:

1. **System Core Instructions (Static):** High-level agent behavior definitions and personality rules.
2. **Reconciled Tool Declarations (Static):** The JSON schemas programmatically generated from your tool registry.
3. **Cognitive Memory Context (Dynamic):** The structural Markdown snippet pulled dynamically from `core/memory`.
4. **Active Chat History Window (Dynamic):** The turn-by-turn message context array.

### 1.2 Memory Context Gating

The module must treat memory context dynamically. If the memory engine returns an empty string or fails to pass its relevance threshold, the system prompt compiler must omit the memory block entirely to optimize token usage.

---

## 2. Mandatory Architecture & Class Structures

Developers must build this package from scratch using explicit, strongly-typed Python classes and dataclasses.

### 2.1 Public Data Models (`core/ai/models.py`)

Create immutable, frozen dataclasses to encapsulate the inputs and outputs of the AI loop. This contract guarantees predictable type checking for other teams:

* **`ToolCallRequest`**: Must capture the model-generated `call_id`, `tool_name`, and a clean dictionary of parsed `arguments`.
* **`AIResponse`**: Must consolidate the text `content` (optional), an ordered list of `ToolCallRequest` objects, the platform `finish_reason` string, and a dictionary tracking token `usage` metrics (e.g., prompt and completion tokens).

### 2.2 Core Orchestrator Engine (`core/ai/engine.py`)

Implement the primary entry point as a comprehensive class named `AICore`.

* **Initialization**: The constructor must configure the Groq base URL (`https://api.groq.com/openai/v1`) and securely resolve the `GROQ_API_KEY` from the local environment using standard `os.getenv` fallbacks. Throw an explicit `ValueError` if the key is missing.
* **Tool Registry**: Build an internal registration method that allows tools extending `BaseTool` to be added to an internal map.
* **Schema Reflection**: Implement a robust internal method that dynamically inspects the properties of registered tools (like `ReadFileTool`) and constructs accurate JSON schemas containing typing, parameter descriptions, and required constraints.
* **Chat Execution**: Implement a `.chat()` method that handles the text generation pipeline—compiling the caching-optimized system frame, making the authenticated network post request, and outputting the typed `AIResponse`.

---

## 3. Dynamic Schema Conversion Constraints

The reflection layer must handle tool mapping cleanly. For your current `ReadFileTool` implementation, the generated JSON schema sent over the wire must map explicitly to its required parameters:

* `path`: Must be explicitly typed as a required string with its target description.
* `features`: Must be cleanly extracted as an optional string parameter restricted by an explicit enum array: `["content", "metadata", "extension", "all"]`.

---

## 4. Test Isolation Strategy

The package must be entirely testable without hitting live endpoints or spending developer credits. Under `tests/ai/`, developers must construct standard test fixtures using `unittest.mock` to intercept network calls.

* **Test 1 (Schema Alignment):** Verify that instantiating and registering your production `ReadFileTool` outputs an accurate OpenAI-compatible function parameter schema.
* **Test 2 (Context Stitching):** Pass dummy strings into the execution frame and verify that the internal compiler layout correctly layers static system tools above the dynamic context.
* **Test 3 (Structural Response Parsing):** Mock a successful raw HTTP response from Groq containing a tool call payload, and assert that the engine translates it into a precise, typed `ToolCallRequest` instance.

---

## 5. Developer Atomic Commit Plan

Developers should execute the blueprint using these 5 sequential commits to ensure a clean git history and easy code reviews:

1. **Commit 1:** Define the core structural models and typed records inside `core/ai/models.py`.
2. **Commit 2:** Write the tool reflection class logic to parse python tool metadata into correct JSON parameters.
3. **Commit 3:** Implement the core `AICore` lifecycle architecture, handling authentication headers and Groq target routing.
4. **Commit 4:** Implement the ordered prompt compiler to support hardware-level caching layouts.
5. **Commit 5:** Wire up the completion parser loop along with the mock network test suite inside `tests/ai/`.

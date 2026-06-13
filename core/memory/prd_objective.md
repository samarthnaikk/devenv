# Product Requirements Document (PRD)

## Project: Devenv Cognitive Memory Engine (CME)

**Status:** Draft | **Target Version:** 1.0

**Core Objective:** Eliminate the concept of "chat sessions." Create a continuous, lifelong, local-first memory engine that allows a user to interact with Devenv like a human colleague—relying on immediate awareness for current tasks and associative recollection for past events, codebases, and preferences.

---

## 1. Product Vision & Core Experience

Traditional AI tools force the user to manage context manually via "New Chat" buttons, custom instructions, or massive, slow system prompts.

The Cognitive Memory Engine (CME) introduces a unified interface. The user opens Devenv and simply types. The agent figures out what is relevant by mimics the human brain’s memory taxonomy:

* **Working Memory:** The razor-sharp focus on the immediate 10-minute window (the open terminal, the active file, the last few messages).
* **Episodic Memory:** A chronological, searchable timeline of things that *happened* ("We fixed that build bug yesterday morning").
* **Semantic/Associative Tree:** A structured, evolving mental model of entities, tech stacks, specific project architectures, and user preferences.

---

## 2. System Architecture & Memory Taxonomy

The memory engine operates across three distinct operational layers.

```text
+-----------------------------------------------------------------------+
|                           User Input Prompt                           |
+-----------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                    1. WORKING MEMORY MANAGER                          |
|  - Holds active chat buffer (last 10-15 messages)                     |
|  - Holds active session state (current file path, last shell error)    |
+-----------------------------------------------------------------------+
                                    |
            +-----------------------+-----------------------+
            | (Triggers Associative Retrieval)              |
            v                                               v
+---------------------------------------+ +---------------------------------------+
|        2. EPISODIC MEMORY             | |      3. ASSOCIATIVE MEMORY TREE       |
|  - Stream of timestamped logs         | |  - Hierarchical Knowledge Graph       |
|  - "What happened & when"             | |  - Relational mapping of projects     |
|  - Pure chronological audit trail     | |  - Summarized nodes & leaf elements   |
+---------------------------------------+ +---------------------------------------+
            |                                               |
            +-----------------------+-----------------------+
                                    | (Context Injection)
                                    v
+-----------------------------------------------------------------------+
|                  Injected Context + Active Prompt                     |
|                   Sent to Local/Remote LLM                            |
+-----------------------------------------------------------------------+

```

### 2.1 The Associative Tree Structure

The core innovation is a hierarchical graph that scales gracefully without blowing past LLM context limits.

* **Root Node:** Global Devenv Core (contains global settings, user profile).
* **Category Nodes:** Functional buckets (e.g., `Projects`, `Tech Stacks`, `User Preferences`).
* **Entity Nodes:** Specific instances (e.g., `Project: RxGPT`, `Project: FinSight AI`).
* **Component Nodes:** Sub-architectures managed dynamically (e.g., `RxGPT -> Backend -> Django Auth`).
* **Leaf Memories:** High-density summaries, explicit code design choices, or links to key episodic events.

---

## 3. Key Functional Features

### 3.1 Proactive "Spreading Activation" (Recollection)

* **Requirement:** The engine must not rely on simple keyword matching or pure, flat vector similarity.
* **Behavior:** When a user prompt triggers a semantic match on a specific leaf node (e.g., a specific database schema issue), the engine must automatically crawl *upward* to fetch its parent node context (the overall project architecture) and *laterally* to fetch related nodes (the database technology stack being used).
* **User Experience:** The user says, *"The landing page hero element is overlapping again."* Devenv automatically surfaces the fact that this is a React/Tailwind project inside `RxGPT`, remembering the exact layout approach previously used, without the user mentioning "RxGPT" or "Tailwind".

### 3.2 Asynchronous "Sleep" Consolidation (Memory Cleaning)

* **Requirement:** Memory organization must never block real-time chat interactions or command execution.
* **Behavior:** The system must feature a background consolidation worker that triggers during periods of inactivity (e.g., 5 minutes after the last user action, or upon terminal task completion).
* **Consolidation Protocol:**
1. Chunk the raw conversational logs from the recent active window.
2. Extract key technical decisions, user preferences, and structural changes.
3. Check existing nodes in the Associative Tree for overlap.
4. Update existing node summaries or spawn a new sub-node if a new component is introduced.
5. Update timestamps and access frequencies (`touch` metadata).



### 3.3 Dynamic Importance & Decay Scoring

* **Requirement:** Prevent memory bloating and context pollution.
* **Mechanism:** Nodes must carry a weight score driven by two factors:
* *Recency:* When was this information last relevant?
* *Frequency:* How often do we talk about or modify this component?


* **Behavior:** Transient conversations (e.g., troubleshooting a temporary typo) naturally sink to the bottom and decay, while structural choices (e.g., using Django for the backend instead of FastAPI) stabilize near the top of the entity stack.

---

## 4. User Interaction & Control (The "Auditable" Principle)

To respect Devenv's core design principle of keeping everything **auditable and controllable**, the memory system cannot be a total black box.

* **The Context Trace:** In the UI (CLI or Web), the user must be able to toggle a "See what I'm remembering" view. This displays a micro-tree showing exactly which nodes were pulled into working memory to answer the last prompt.
* **Manual Intervention:** A user must have the power to explicitly correct a memory.
* *Example command/chat:* `Devenv, forget the fastAPI setup, we completely switched the dataset and backend architecture to Django.`
* *System Action:* The engine prunes or updates the respective branch immediately.



---

## 5. Technical Design Constraints & Local-First Guardrails

* **Privacy & Security:** Because memory logs contain proprietary code structures, terminal logs, and potentially sensitive user data, **all raw vector embeddings, graph databases, and episodic timelines must reside strictly on the user's local host machine.**
* **Performance Budget:**
* Context assembly (Recollection + Spreading Activation) must take less than **200ms** before formatting the prompt for the LLM.
* Local storage footprint must remain lightweight, utilizing minimal overhead structures (like SQLite for the graph/timeline and a lightweight local vector index).



---

## 6. Future Scope (Phase 2)

* **Cross-Device Sync:** Securely piping encrypted memory diffs between the local laptop execution host and a remote mobile/web companion client.
* **Multi-Repo Cross-Pollination:** Allowing Devenv to recognize when a solution implemented in `Project A` can solve a structural bottleneck currently occurring in `Project B`.

---
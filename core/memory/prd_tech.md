# Product Requirements Document (PRD)

## Project: Devenv Cognitive Memory Engine (CME) — Technical Implementation Blueprint

**Target Audience:** Core Backend & Database Engineers

**System Nature:** Local-first, decoupled Python package (`core.memory`)

---

## 1. Technical Stack Selection (Local-First & Light)

To fulfill the requirements of high speed (<200ms retrieval), local execution, and low dependency bloat, the following stack is mandated:

* **Relational & Graph Storage:** **SQLite** (via the standard library `sqlite3`). It is serverless, zero-config, and exceptionally fast for relational tables and self-referencing adjacency lists (trees).
* **Vector Storage:** **LanceDB** or **ChromaDB (Persistent Client)**. LanceDB is preferred as it stores data in an open-source columnar format (Lance) directly on the disk, requiring zero background server processes.
* **Embedding Model:** Local inference via **`sentence-transformers`** (e.g., `all-MiniLM-L6-v2`, 384 dimensions) to keep embedding generation local and sub-10ms.
* **Orchestration Framework:** Pure Python. **Do not** use heavy frameworks like LangChain or LlamaIndex. Write explicit SQL and database connectors to keep the code auditable.

---

## 2. Core Database Schema & Data Architecture

The memory engine requires two storage engines running in parallel: **SQLite** for structural/episodic data and **LanceDB** for raw semantic vector lookups.

### 2.1 SQLite Schema (`memory.db`)

```sql
-- Table 1: The Associative Tree Nodes
CREATE TABLE memory_nodes (
    node_id TEXT PRIMARY KEY,          -- UUID or slug (e.g., "proj_rxgpt")
    parent_id TEXT,                    -- Self-referencing foreign key for tree hierarchy
    label TEXT NOT NULL,               -- e.g., "Django Auth Setup"
    category TEXT NOT NULL,            -- 'global', 'project', 'component', 'preference'
    summary TEXT NOT NULL,             -- The dense knowledge snippet used for context
    created_at REAL NOT NULL,          -- Epoch timestamp
    last_accessed REAL NOT NULL,       -- For decay calculations
    access_count INTEGER DEFAULT 0,    -- Frequency counter
    FOREIGN KEY (parent_id) REFERENCES memory_nodes(node_id) ON DELETE SET NULL
);

-- Table 2: Node Adjacency List for Lateral Relationships (Graph edges)
CREATE TABLE node_edges (
    source_node_id TEXT,
    target_node_id TEXT,
    relationship_type TEXT,            -- e.g., "uses_tech", "depends_on"
    PRIMARY KEY (source_node_id, target_node_id),
    FOREIGN KEY (source_node_id) REFERENCES memory_nodes(node_id) ON DELETE CASCADE,
    FOREIGN KEY (target_node_id) REFERENCES memory_nodes(node_id) ON DELETE CASCADE
);

-- Table 3: Episodic Timeline (The Linear Audit Trail)
CREATE TABLE episodic_logs (
    log_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    associated_node_id TEXT,           -- Optional link to a specific project/component node
    raw_interaction TEXT NOT NULL,     -- JSON string containing {"user": "...", "agent": "..."}
    FOREIGN KEY (associated_node_id) REFERENCES memory_nodes(node_id) ON DELETE SET NULL
);

```

### 2.2 LanceDB Schema (`vectors.lance`)

Every time a `memory_node` is created or heavily updated, its `summary` must be vectorized and stored here.

* **Columns:** `node_id` (TEXT, links back to SQLite), `vector` (FLOAT[384]), `text_chunk` (TEXT).

---

## 3. Algorithmic Workflows & Logic

Your developers need to implement two primary algorithms: **Spreading Activation (Retrieval)** and **Asynchronous Consolidation (Sleep/Save)**.

### 3.1 Retrieval Flow: Spreading Activation Algorithm

When a user submits a prompt, the engine must perform the following steps within **200ms**:

```text
[User Prompt] 
      │
      ▼
1. Vector Search (LanceDB) ──► Returns top K Node IDs (Threshold > 0.70)
      │
      ▼
2. Graph Crawl (SQLite) ────► Fetch Parent Nodes + Connected Siblings
      │
      ▼
3. Decay & Score Filter ────► Rank nodes using Score = (Sim x 0.6) + (Frequency x 0.2) + (Recency x 0.2)
      │
      ▼
4. Context Compilation ─────► Formats top 3-5 summaries into Markdown system prompt block

```

* **The Decay Formula:**

$$Score = (\text{CosineSimilarity} \times 0.6) + (\text{AccessCount} \times 0.2) + \left(\frac{1}{\text{TimeSinceLastAccess}} \times 0.2\right)$$



*Engineers must normalize these values between 0 and 1 before computing the final score.*

### 3.2 Background Consolidation Flow (The "Sleep" Cycle)

Because running LLM evaluations mid-chat introduces unacceptable latency, memory updates must happen asynchronously.

1. **Trigger:** Fire an internal event 5 minutes after the last user message, or right after a long-running terminal command completes.
2. **Processing Window:** Grab all `episodic_logs` created since the last consolidation timestamp.
3. **LLM Clean-up Call:** Send these logs to a lightweight local or fast remote model with a strict JSON format prompt:
```json
{
  "detected_project": "string or null",
  "new_entities": [{"label": "string", "category": "string", "summary": "string"}],
  "updates_to_existing_nodes": [{"node_id": "string", "append_summary": "string"}]
}

```


4. **Database Commit:** Open a write transaction on SQLite, update the nodes, rewrite the modified node vectors to LanceDB, and update `last_accessed`.

---

## 4. Code Architecture & Core API Contracts

Developers should implement `core/memory/` as a decoupled class interface. They do not need a working frontend or CLI to test this; they can write pure unit tests against these boundaries.

```python
# core/memory/interface.py
from typing import List, Dict, Any

class MemoryEngine:
    def __init__(self, db_path: str = "memory.db", vector_dir: str = "vectors/"):
        """Initializes SQLite and LanceDB clients locally."""
        pass

    def add_episodic_log(self, user_prompt: str, agent_response: str, node_id: str = None) -> str:
        """Appends a raw interaction to the linear timeline table."""
        pass

    def retrieve_context(self, current_prompt: str) -> str:
        """
        Executes the Spreading Activation routine:
        1. Vectors the prompt.
        2. Hits LanceDB.
        3. Traverses SQLite for parents/edges.
        4. Applies decay scoring.
        5. Returns a formatted markdown string for the LLM context window.
        """
        pass

    def update_associative_tree(self, node_data: Dict[str, Any]) -> bool:
        """Creates or mutates a node in SQLite and refreshes its LanceDB vector record."""
        pass

    def run_consolidation(self) -> None:
        """Background worker target. Syncs recent logs into structured node states."""
        pass

```

---

## 5. Verification & Testability (How to verify without a UI)

Since the UI and other tools are decoupled, the memory developers must validate their code using mock fixtures in a `tests/` directory:

* **Test Case 1: Hierarchical Inheritance**
* *Setup:* Create a node `Project: RxGPT`. Create a child node `Component: Django Backend`.
* *Action:* Call `retrieve_context("How do I fix my django authentication errors?")`
* *Assert:* Verify that the returned context string contains **both** the text from the Django child node *and* the overarching `RxGPT` parent node summary.


* **Test Case 2: Multi-turn Sessionless Memory**
* *Setup:* Seed database with a user preference: `"User prefers using functional components over class components in React."`
* *Action:* Call `retrieve_context("Let's draft a new login component view.")`
* *Assert:* Verify that the retrieved context flags the functional component preference, allowing the model adapter down the pipeline to write correct code on the first try.
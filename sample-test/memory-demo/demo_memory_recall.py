from __future__ import annotations

import argparse
from pathlib import Path

from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.memory.vector_index import InMemoryVectorIndex

DEMO_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_ROOT.parents[2]
DEMO_DB = PROJECT_ROOT / "memory.db"
DEMO_VECTORS = PROJECT_ROOT / "vectors"


def create_engine() -> MemoryEngine:
    return MemoryEngine(
        db_path=str(DEMO_DB),
        vector_dir=str(DEMO_VECTORS),
        embedder=HashingEmbedder(dimension=8),
        vector_index=InMemoryVectorIndex(),
    )


def seed_demo_db(force: bool = False) -> None:
    if force:
        if DEMO_DB.exists():
            DEMO_DB.unlink()

    engine = create_engine()
    engine.add_episodic_log(
        "We were building a calendar project with a React frontend, Python backend, and drag-and-drop scheduling.",
        "Stored the calendar stack and scheduling details for later recall.",
        metadata={"workspace_path": "/demo/calendar-project"},
    )
    engine.add_episodic_log(
        "The current jobs project uses Django for the backend and React for the admin web app.",
        "Stored the jobs platform architecture separately from the calendar project.",
        metadata={"workspace_path": "/demo/jobs-project"},
    )
    engine.run_consolidation()


def run_demo() -> str:
    engine = create_engine()
    engine.record_working_memory(
        messages=[
            {"role": "user", "content": "Do you know about the calendar project we were building?"},
            {"role": "assistant", "content": "Yes, give me more context if you want me to recall it."},
            {"role": "user", "content": "Not in the current directory, I mean the project we were working on earlier."},
        ],
        active_state={"workspace_path": "/demo/jobs-project", "session_id": "demo-session"},
    )
    result = engine.retrieve_context(
        "Not in the current directory, I mean the project we were working on earlier.",
        top_k=5,
    )
    return result.markdown_context


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed and exercise the Devenv memory recall demo.")
    parser.add_argument("--seed", action="store_true", help="Seed the demo memory database before running.")
    parser.add_argument("--force", action="store_true", help="Rebuild the demo database from scratch.")
    args = parser.parse_args()

    if args.seed or not DEMO_DB.exists():
        seed_demo_db(force=args.force)

    print(f"Using shared memory DB: {DEMO_DB}")
    print(f"Using shared vector dir: {DEMO_VECTORS}")
    print(run_demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

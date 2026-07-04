from __future__ import annotations

from pathlib import Path


def resolve_memory_paths(db_path: str, vector_dir: str, *, workspace_path: str | None = None) -> tuple[str, str]:
    state_root = Path(workspace_path).expanduser().resolve() if workspace_path else Path.cwd().resolve()

    if db_path == "memory.db":
        resolved_db_path = state_root / "memory.db"
    else:
        candidate_db_path = Path(db_path).expanduser()
        resolved_db_path = candidate_db_path if candidate_db_path.is_absolute() else state_root / candidate_db_path

    if vector_dir == "vectors":
        resolved_vector_dir = state_root / "vectors"
    else:
        candidate_vector_dir = Path(vector_dir).expanduser()
        resolved_vector_dir = candidate_vector_dir if candidate_vector_dir.is_absolute() else state_root / candidate_vector_dir

    return str(resolved_db_path.resolve()), str(resolved_vector_dir.resolve())

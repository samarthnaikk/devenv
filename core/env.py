from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(start_path: str | Path | None = None, filename: str = ".env") -> Path | None:
    search_root = _coerce_search_root(start_path)

    for directory in (search_root, *search_root.parents):
        candidate = directory / filename
        if not candidate.is_file():
            continue
        _load_file(candidate)
        return candidate

    return None


def _coerce_search_root(start_path: str | Path | None) -> Path:
    if start_path is None:
        return Path.cwd().resolve()

    candidate = Path(start_path).expanduser().resolve()
    if candidate.is_file():
        return candidate.parent
    return candidate


def _load_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        os.environ[key] = _parse_value(value.strip())


def _parse_value(value: str) -> str:
    if not value:
        return ""

    if value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]

    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()

    return value

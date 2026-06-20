from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Iterator


NOISE_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    "vectors",
}


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def ensure_existing_path(path: str) -> Path:
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Path not found: {resolved}")
    return resolved


def ensure_file(path: str) -> Path:
    resolved = ensure_existing_path(path)
    if resolved.is_dir():
        raise IsADirectoryError(f"Expected a file, got a directory: {resolved}")
    return resolved


def ensure_directory(path: str) -> Path:
    resolved = ensure_existing_path(path)
    if not resolved.is_dir():
        raise NotADirectoryError(f"Expected a directory, got a file: {resolved}")
    return resolved


def relative_display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def iter_directory(
    root: Path,
    *,
    max_depth: int,
    include_files: bool = True,
    noise_directories: set[str] | None = None,
) -> Iterator[tuple[Path, int]]:
    ignored = noise_directories or NOISE_DIRECTORIES
    stack: list[tuple[Path, int]] = [(root, 0)]

    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            continue

        children = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        for child in children:
            child_depth = depth + 1
            if child_depth > max_depth:
                continue

            if child.is_dir():
                yield child, child_depth
                if child.name not in ignored and child_depth < max_depth:
                    stack.append((child, child_depth))
                continue

            if include_files:
                yield child, child_depth


def file_matches_pattern(path: Path, pattern: str, *, mode: str) -> bool:
    if mode == "exact":
        return path.name == pattern
    if mode == "glob":
        return fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(str(path), pattern)
    raise ValueError(f"Unsupported pattern mode: {mode}")


def is_probably_text(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            sample = handle.read(1024)
    except OSError:
        return False

    if b"\x00" in sample:
        return False

    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def normalize_extension_filter(ext_filter: str | None) -> str | None:
    if ext_filter is None:
        return None
    cleaned = ext_filter.strip().lower()
    if not cleaned:
        return None
    return cleaned if cleaned.startswith(".") else f".{cleaned}"

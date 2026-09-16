from __future__ import annotations

import re

SYNONYM_VERSION = 1

QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "tag": ("release", "publish", "version"),
    "tagging": ("tag", "release", "publish"),
    "release": ("tag", "publish", "deploy"),
    "publish": ("release", "tag", "pypi", "deploy"),
    "deploy": ("release", "publish", "ci"),
    "github": ("git", "repo"),
    "workflow": ("actions", "ci", "pipeline"),
    "timeout": ("connection dropped", "hang", "stall", "slow"),
    "auth": ("authentication", "login", "session", "token"),
    "cache": ("invalidate", "stale", "memo"),
    "migration": ("schema", "alembic", "database"),
    "error": ("bug", "crash", "failure", "traceback"),
    "refactor": ("cleanup", "restructure"),
    "blank": ("white screen", "empty", "crash"),
    "tags": ("tag", "release", "publish"),
    "releases": ("release", "tag", "publish"),
    "errors": ("error", "bug", "crash", "failure"),
    "bugs": ("bug", "error", "issue", "failure"),
    "workflows": ("workflow", "actions", "ci", "pipeline"),
    "migrations": ("migration", "schema", "database"),
}


def expand_query_synonyms(text: str, max_terms: int = 8) -> list[str]:
    tokens = set(re.findall(r"[a-z0-9_]+", (text or "").lower()))
    expansions: list[str] = []
    for token in sorted(tokens):
        for alias in QUERY_SYNONYMS.get(token, ()):
            if alias not in expansions:
                expansions.append(alias)
    return expansions[:max_terms]

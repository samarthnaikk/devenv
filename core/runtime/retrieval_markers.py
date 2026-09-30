"""Project-specific retrieval marker vocabulary.

The retrieval ranker uses a small set of literal markers that, in practice,
identify "issue recall" sessions for the projects this tool was developed
against. They were previously duplicated across several functions; this module
is the single definition, and each list can be extended at runtime without a
code change via environment variables (comma-separated):

- ``DEVENV_FOCUS_MARKERS``  — additions to :data:`FOCUS_MARKERS`
- ``DEVENV_DETAIL_MARKERS`` — additions to :data:`DETAIL_MARKERS`
- ``DEVENV_NOISE_MARKERS``  — additions to :data:`PREVIEW_NOISE_MARKERS`
"""

from __future__ import annotations

import os

# Markers that indicate an issue/bug-recall session (used for preview priority
# and scoring bonuses). Order is preserved for deterministic behavior.
_DEFAULT_FOCUS_MARKERS: tuple[str, ...] = (
    "bug list",
    "exact bugs",
    "root url redirects",
    "convex generated imports",
    "authentication bypass",
    "open email relay",
    "create workspace",
    "pipeline chat",
    "test/publish",
    "salesforce being marked as coming soon",
)

# Markers used by the offline context-line collector.
_DEFAULT_DETAIL_MARKERS: tuple[str, ...] = (
    "create workspace",
    "pipeline chat",
    "test/publish",
    "salesforce",
    "root url redirects",
    "convex generated imports",
    "authentication bypass",
    "open email relay",
    "bug list",
    "exact bugs",
)

# Markers that indicate low-value noise in a session preview.
_DEFAULT_PREVIEW_NOISE_MARKERS: tuple[str, ...] = (
    "tool exec_command result",
    "operation not permitted: ps",
    "pr-review.md",
    "committed in two atomic commits",
    "fix(settings): use saved timezone and locale dropdowns",
    "glob: /users/",
)

# Session-focus markers (subset shape used by _session_has_issue_focus and
# _issue_focus_score; the latter also includes "bugs tracked").
_DEFAULT_SESSION_FOCUS_MARKERS: tuple[str, ...] = (
    "bug list",
    "root url redirects",
    "convex generated imports",
    "authentication bypass",
    "open email relay",
    "create workspace",
    "pipeline chat",
    "test/publish",
    "salesforce being marked as coming soon",
)

_DEFAULT_SESSION_FOCUS_SCORE_MARKERS: tuple[str, ...] = (
    "bugs tracked",
)

# Issue keywords shared by both ranking paths.
ISSUE_TERMS: frozenset[str] = frozenset(
    {"bug", "bugs", "fix", "fixed", "issue", "issues", "review", "reviews"}
)

# Narrower set used by the issue-bonus scoring paths (no issue/issues).
ISSUE_TERMS_BASE: frozenset[str] = frozenset(
    {"bug", "bugs", "fix", "fixed", "review", "reviews"}
)


def _env_extra(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return ()
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


def _merged(defaults: tuple[str, ...], env_name: str) -> tuple[str, ...]:
    extras = _env_extra(env_name)
    if not extras:
        return defaults
    return tuple(dict.fromkeys([*defaults, *extras]))


def focus_markers() -> tuple[str, ...]:
    return _merged(_DEFAULT_FOCUS_MARKERS, "DEVENV_FOCUS_MARKERS")


def detail_markers() -> tuple[str, ...]:
    return _merged(_DEFAULT_DETAIL_MARKERS, "DEVENV_DETAIL_MARKERS")


def preview_noise_markers() -> tuple[str, ...]:
    return _merged(_DEFAULT_PREVIEW_NOISE_MARKERS, "DEVENV_NOISE_MARKERS")


def session_focus_markers() -> tuple[str, ...]:
    return _merged(_DEFAULT_SESSION_FOCUS_MARKERS, "DEVENV_FOCUS_MARKERS")


def session_focus_score_markers() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys([*session_focus_markers(), *_DEFAULT_SESSION_FOCUS_SCORE_MARKERS])
    )


__all__ = [
    "FOCUS_MARKERS",
    "DETAIL_MARKERS",
    "PREVIEW_NOISE_MARKERS",
    "ISSUE_TERMS",
    "ISSUE_TERMS_BASE",
    "focus_markers",
    "detail_markers",
    "preview_noise_markers",
    "session_focus_markers",
    "session_focus_score_markers",
]

# Back-compat aliases (static defaults).
FOCUS_MARKERS = _DEFAULT_FOCUS_MARKERS
DETAIL_MARKERS = _DEFAULT_DETAIL_MARKERS
PREVIEW_NOISE_MARKERS = _DEFAULT_PREVIEW_NOISE_MARKERS

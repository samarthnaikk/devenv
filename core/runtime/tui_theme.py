"""Visual theme for the Devenv TUI.

Palette and layout follow ``stitch_devenv_ui_structural_blueprint/DESIGN.md``:
a "Void-to-Surface" deep-dark workspace with a teal primary accent, corporate
blue secondary, and monospace log wells.

The module keeps a set of plain color constants for Rich markup (which cannot
read Textual's ``$variables``) and exposes :func:`build_themes` for registering
the ``devenv`` / ``devenv-light`` Textual themes.
"""

from __future__ import annotations

from typing import Any

CANVAS = "#0d0f12"
PANEL = "#16191e"
PANEL_ALT = "#1c2026"
WELL = "#0a0c0e"
BORDER = "#2d333b"
TEAL = "#4fdbc8"
ON_TEAL = "#003731"
BLUE = "#adc6ff"
TEXT = "#e2e2e6"
TEXT_MUTED = "#859490"
WARN = "#f0b26b"
ERROR = "#ffb4ab"

DEFAULT_THEME_NAME = "devenv"
LIGHT_THEME_NAME = "devenv-light"

# Rich markup colors for role-tagged retrieval context lines.
ROLE_COLORS = {
    "user": BLUE,
    "assistant": TEAL,
    "tool": WARN,
    "session": TEXT_MUTED,
    "context": TEXT_MUTED,
}

# Structural CSS. Uses Textual theme variables so it follows the active theme.
CSS = """
Screen {
    layout: vertical;
    background: $background;
    color: $foreground;
}

#header {
    height: 1;
    background: $surface;
    color: $foreground;
    padding: 0 1;
}

#body {
    height: 1fr;
}

#sidebar {
    width: 30;
    background: $surface;
    border-right: solid $border;
    padding: 0 1;
}

#mode-pills {
    height: auto;
    padding: 1 0 0 0;
}

.section-title {
    color: $text-muted;
    text-style: bold;
    margin: 1 0 0 0;
}

#sources-list {
    height: auto;
}

#agents-list {
    height: auto;
}

#index-bar {
    margin: 0;
}

#index-info {
    color: $text-muted;
    height: auto;
}

#results-pane {
    width: 1fr;
}

#result-bar {
    height: 1;
    padding: 0 1;
    background: $background;
}

#result-title {
    width: 1fr;
    color: $primary;
    text-style: bold;
}

#spinner {
    width: auto;
}

#results-list {
    height: 1fr;
    background: $background;
    padding: 0 1;
}

.result-card {
    background: $surface;
    border: round $border;
    padding: 0 1;
    margin: 1 0;
    height: auto;
}

#log-panel {
    height: 9;
    background: $surface;
    border-top: solid $border;
}

#log-panel.hidden {
    display: none;
}

#log-title {
    height: 1;
    background: $surface;
    color: $text-muted;
    text-style: bold;
    padding: 0 1;
}

#log {
    height: 1fr;
    background: $background;
    padding: 0 1;
}

#composer {
    background: $background;
    border: round $border;
    color: $foreground;
}

#composer:focus {
    border: round $primary;
}
"""


def build_themes() -> list[Any]:
    """Build the Devenv Textual themes (dark + light).

    Imported lazily so this module stays importable without Textual installed.
    """
    from textual.theme import Theme

    dark = Theme(
        name=DEFAULT_THEME_NAME,
        primary=TEAL,
        secondary=BLUE,
        accent="#71f8e4",
        foreground=TEXT,
        background=CANVAS,
        surface=PANEL,
        panel=PANEL_ALT,
        warning=WARN,
        error=ERROR,
        success=TEAL,
        dark=True,
        variables={
            "footer-key-foreground": TEAL,
            "block-cursor-background": TEAL,
            "block-cursor-foreground": ON_TEAL,
            "input-selection-background": f"{TEAL} 40%",
        },
    )
    light = Theme(
        name=LIGHT_THEME_NAME,
        primary="#006b5f",
        secondary="#004395",
        accent="#14b8a6",
        foreground="#1a1c1f",
        background="#f5f6f8",
        surface="#ffffff",
        panel="#eef0f3",
        warning="#a15c00",
        error="#ba1a1a",
        success="#006b5f",
        dark=False,
        variables={
            "footer-key-foreground": "#006b5f",
            "block-cursor-background": "#006b5f",
            "block-cursor-foreground": "#ffffff",
            "input-selection-background": "#006b5f 30%",
        },
    )
    return [dark, light]


__all__ = [
    "CANVAS",
    "PANEL",
    "PANEL_ALT",
    "WELL",
    "BORDER",
    "TEAL",
    "ON_TEAL",
    "BLUE",
    "TEXT",
    "TEXT_MUTED",
    "WARN",
    "ERROR",
    "ROLE_COLORS",
    "CSS",
    "DEFAULT_THEME_NAME",
    "LIGHT_THEME_NAME",
    "build_themes",
]

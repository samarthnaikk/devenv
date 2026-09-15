"""Visual theme for the Devenv TUI.

Palette and layout follow ``stitch_devenv_ui_structural_blueprint/DESIGN.md``:
a "Void-to-Surface" deep-dark workspace with a teal primary accent, corporate
blue secondary, and monospace log wells.
"""

from __future__ import annotations

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

# Rich markup colors for role-tagged retrieval context lines.
ROLE_COLORS = {
    "user": BLUE,
    "assistant": TEAL,
    "tool": WARN,
    "session": TEXT_MUTED,
    "context": TEXT_MUTED,
}


CSS = f"""
Screen {{
    layout: vertical;
    background: {CANVAS};
    color: {TEXT};
}}

#header {{
    height: 1;
    background: {PANEL};
    color: {TEXT};
    padding: 0 1;
}}

#body {{
    height: 1fr;
}}

#sidebar {{
    width: 30;
    background: {PANEL};
    border-right: solid {BORDER};
    padding: 0 1;
}}

#mode-pills {{
    height: auto;
    padding: 1 0 0 0;
}}

.section-title {{
    color: {TEXT_MUTED};
    text-style: bold;
    margin: 1 0 0 0;
}}

#sources-list {{
    height: auto;
}}

#index-bar {{
    margin: 0;
}}

#index-info {{
    color: {TEXT_MUTED};
    height: auto;
}}

#results-pane {{
    width: 1fr;
}}

#result-bar {{
    height: 1;
    padding: 0 1;
    background: {CANVAS};
}}

#result-title {{
    width: 1fr;
    color: {TEAL};
    text-style: bold;
}}

#spinner {{
    width: auto;
}}

#results-list {{
    height: 1fr;
    background: {CANVAS};
    padding: 0 1;
}}

.result-card {{
    background: {PANEL};
    border: round {BORDER};
    padding: 0 1;
    margin: 1 0;
    height: auto;
}}

#log-panel {{
    height: 9;
    background: {PANEL};
    border-top: solid {BORDER};
}}

#log-panel.hidden {{
    display: none;
}}

#log-title {{
    height: 1;
    background: {PANEL};
    color: {TEXT_MUTED};
    text-style: bold;
    padding: 0 1;
}}

#log {{
    height: 1fr;
    background: {WELL};
    padding: 0 1;
}}

#composer {{
    background: {WELL};
    border: round {BORDER};
    color: {TEXT};
}}

#composer:focus {{
    border: round {TEAL};
}}
"""


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
]

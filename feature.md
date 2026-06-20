# Devenv Features

This document lists the **user-facing features** available in Devenv today.
It focuses on what a user can do in the product, not the internal tool-call or implementation details.

## Core Experience

- Chat with the workspace using natural language.
- Ask project questions like architecture, backend behavior, file relationships, and code flow.
- Request code changes in plain English.
- Get step-by-step execution for larger implementation requests.
- Continue working across turns with memory-aware context instead of treating each prompt like a brand-new session.

## Memory Features

- Retains useful project context across conversations.
- Recalls earlier work, past explanations, and prior implementation attempts.
- Uses working memory for the current flow and episodic memory for longer-term recall.
- Consolidates past interaction history into reusable knowledge.
- Filters out low-signal structural junk so poor directory-dump answers are less likely to come back as “memory.”

## Planning And Execution

- Automatically decides when a request should be answered directly versus planned as a multi-step task.
- Supports explicit `Plan Mode` in the web UI.
- Breaks larger implementation work into checkpoints.
- Executes checkpoints **one at a time** instead of attempting everything in a single burst.
- Marks completed checkpoints visually in the UI.
- Shows the current active checkpoint during execution.
- Preserves partial plan state if execution fails partway through.

## Code Understanding

- Answers repo questions by combining memory, workspace inspection, and code-aware runtime behavior.
- Falls back away from low-value directory-only answers for deeper questions like:
  - “how does this work?”
  - “how does it decide what to send?”
  - “how does the backend behave?”
- Can inspect file structure when a structural overview is actually useful.
- Supports targeted reasoning about implementation details instead of only folder summaries.

## File And Workspace Interaction

- Browse the workspace in the web UI.
- Expand folder trees on demand.
- Preview files in the center pane.
- Render Markdown previews cleanly.
- Show code files with proper preview behavior instead of only raw text dumps.
- Keep file generation scoped to the active workspace sandbox.

## Code Change Features

- Create new files and folders from natural-language requests.
- Edit existing files through runtime-driven execution.
- Remove files when needed.
- Run changes step by step instead of all at once for planned tasks.
- Anchor scaffolded frontend work to the intended target path such as `calendar/frontend`.
- Reject obviously broken scaffold writes, such as writing an empty file to a folder path itself.

## Runtime Safety And Reliability

- Workspace sandboxing prevents access outside the allowed project scope.
- Unsafe path access is blocked and surfaced as a runtime failure instead of silently succeeding.
- Partial execution failures are returned back to the UI with plan context preserved.
- Cooldown and rate-limit states are surfaced in the chat UI.
- Groq rate-limit errors can automatically retry after the cooldown window.

## Web UI Features

- VS Code-inspired multi-pane layout.
- Resizable chat area.
- Collapsible side panes.
- File explorer and preview workflow.
- Chat transcript for user prompts and runtime answers.
- Execution plan rail showing pending, active, and completed checkpoints.
- Per-step inspect button for viewing full checkpoint details.
- Modal view for step details and execution notes.
- Top status strip showing:
  - AI provider
  - model
  - token usage
  - context remaining
  - context reset timing

## Thinking And Visibility Controls

- `Show Thinking` toggle in the UI.
- Hidden-by-default condensed thinking state for a cleaner user experience.
- Optional raw runtime trace view when deeper inspection is needed.
- Live status feedback such as:
  - thinking
  - planning
  - tool activity
  - retry countdown during cooldown

## Diagnostics And Verification

- Can run verification steps at the end of planned executions.
- Tracks test and type-check outcomes as part of the plan lifecycle.
- Shows when verification passed or failed.
- Keeps the user informed when execution succeeded but verification still needs attention.

## Multi-Surface Runtime

- Web runtime for the main user experience.
- TUI runtime for terminal-based interaction.
- Single-turn smoke/runtime command for direct testing.
- MCP server surface for exposing the local tool deck externally.

## Current Strengths

- Good for project Q&A with persistent memory.
- Good for controlled multi-step code generation.
- Good for scaffold-and-iterate workflows inside a local workspace.
- Good for exposing progress instead of feeling frozen during long operations.

## Current Limitations

- Some conceptual questions may still require deeper direct code inspection rather than memory alone.
- Verification quality depends on the available project diagnostics.
- The system is still evolving, so edge cases in routing, planning, and memory quality can still appear.

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict, replace
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from core.ai import OpenCodeAICore, RoutingAICore
from core.ai.models import AIExecutedToolStep, AIResponse, ToolCallRequest
from core.env import load_dotenv
from core.memory import MemoryEngine
from core.memory.embeddings import HashingEmbedder
from core.memory.models import RetrievalTrace
from core.tools.base import BaseTool
from core.tools._common import NOISE_DIRECTORIES

from ..context_builder import ContextBuilderService
from ..context_stage import build_context_packet
from ..local_router import LocalIntentRouter
from ..local_model import load_local_small_model
from ..metadata_stage import build_checkpoint_metadata
from ..models import (
    AgentState,
    CheckpointTask,
    DEFAULT_MAX_CONSECUTIVE_TOOLS,
    ExecutionBlueprint,
    ExecutionMode,
    MemorySummary,
    PlanningMode,
    ProcessStage,
    RepairState,
    RuntimeTurnResult,
    StageTrace,
    ToolExecutionStep,
    TurnOutcome,
    VerificationResult,
)
from ..response_sanitizer import normalize_response_text, sanitize_response_text
from ..sandbox import PathSandbox
from ..state import resolve_memory_paths
from ..tool_policy import TOOL_POLICY_REGISTRY, allowed_tool_names_for_mode, build_tool_policy_event

logger = logging.getLogger(__name__)
MAX_EPHEMERAL_TURNS = 4
PLANNING_ALLOWED_TOOLS = frozenset(allowed_tool_names_for_mode(ExecutionMode.PLAN_ONLY, set(TOOL_POLICY_REGISTRY)))
READ_ONLY_EXECUTION_TOOLS = frozenset(
    {"list_directory", "locate_files", "read_file", "peek_lines", "inspect_symbols", "search_text", "track_symbol"}
)
DIRECT_CODE_INSPECTION_TOOLS = frozenset({"list_directory", "locate_files", "read_file", "peek_lines", "inspect_symbols"})
DIRECT_ARCHITECTURE_INSPECTION_TOOLS = frozenset({"list_directory", "read_file", "peek_lines", "inspect_symbols"})
DIRECT_REPO_SUMMARY_TOOLS = frozenset({"list_directory", "read_file", "peek_lines", "inspect_symbols"})
WRITE_EXECUTION_TOOLS = frozenset({"write_file", "edit_file"})
DELETE_EXECUTION_TOOLS = frozenset({"remove_file"})
SHELL_EXECUTION_TOOLS = frozenset({"run_shell", "run_diagnostics", "audit_changes"})
DIAGNOSTIC_EXECUTION_TOOLS = frozenset({"run_diagnostics", "audit_changes"})
MEMORY_EXECUTION_TOOLS = frozenset({"manage_memory", "inspect_trace"})
WEB_EXECUTION_TOOLS = frozenset({"web_search"})
KNOWLEDGE_EXECUTION_TOOLS = frozenset({"knowledge_search"})
ARTIFACT_EXECUTION_TOOLS = frozenset({"generate_pdf"})
PLANNING_MEMORY_CHAR_LIMIT = 900
EXECUTION_MEMORY_CHAR_LIMIT = 1400
SCAFFOLD_EXECUTION_TOOLS = frozenset({"list_directory", "write_file", "edit_file"})
SCAFFOLD_EXECUTION_MEMORY_CHAR_LIMIT = 360
_GENERIC_WORKSPACE_TOKENS = frozenset(
    {
        "about",
        "architecture",
        "backend",
        "code",
        "codebase",
        "does",
        "explain",
        "file",
        "files",
        "folder",
        "folders",
        "how",
        "project",
        "repo",
        "repository",
        "show",
        "system",
        "tell",
        "what",
        "work",
        "works",
    }
)
PLANNING_SYSTEM_RULE = (
    "Analyze the user's request and produce a sequential markdown checklist using checkbox items like '- [ ] Task'. "
    "Do not invoke modification tools during planning. Stay focused on planning until the checklist is complete. "
    "Break the work into as many single-shot checkpoints as needed for the available context."
)
EXECUTION_SYSTEM_RULE = (
    "Work only on the current checkpoint. Do not start future checkpoints. "
    "Use tools only when necessary, and stop after completing the current checkpoint."
)
DIRECT_SYSTEM_RULE = (
    "Answer the user's question directly. First use the memory context if it plausibly contains the answer. "
    "Use tools only if workspace inspection is still needed after considering memory. "
    "If you need a tool, emit a real function call and never print JSON tool snippets in plain text. "
    "Do not create a checklist or execution plan unless the user is asking you to make changes. "
    "Use web_search for current or time-sensitive facts, or when the user explicitly asks to search, browse, google, or look something up. "
    "Use knowledge_search when the user wants external references, similar projects, GitHub repos, forum threads, videos, or broader research resources for a topic. "
    "If web_search is the relevant selected tool, perform the search before answering. "
    "If a search request is ambiguous, ask one concise follow-up question instead of guessing. "
    "Keep the final answer brief unless the user asks for detail."
)
DIRECT_MEMORY_CHAR_LIMIT = 900
CONSOLIDATION_COOLDOWN_STATE_KEY = "runtime.last_consolidation_wall_time"
DEFAULT_CONSOLIDATION_COOLDOWN_SECONDS = 900.0
_AI_SENTINEL = object()
_LOCAL_MODEL_SENTINEL = object()
_TOOL_CLIENT_SENTINEL = object()
_CONTEXT_BUILDER_SENTINEL = object()

PRIVACY_DISABLED_METADATA = {
    "external_context_state": "privacy_blocked",
    "external_context_reason": "Prior memory access is disabled for this turn.",
    "external_context_session_count": 0,
    "external_context_session_ids": [],
}



from pathlib import Path as _KernelPartPath

_KERNEL_PARTS_DIR = _KernelPartPath(__file__).with_name("parts")

def _load_kernel_part(filename: str) -> None:
    exec((_KERNEL_PARTS_DIR / filename).read_text(encoding="utf-8"), globals())

_load_kernel_part("helpers_part_1.py")
_load_kernel_part("helpers_part_2.py")
_load_kernel_part("helpers_part_3.py")
_load_kernel_part("helpers_part_4.py")
_load_kernel_part("helpers_part_5.py")
_load_kernel_part("class_part_1.py")
_load_kernel_part("class_part_2.py")
_load_kernel_part("class_part_3.py")
_load_kernel_part("class_part_4.py")
_load_kernel_part("class_part_5.py")

class DevenvKernel(KernelLifecycleMixin, KernelCheckpointMixin, KernelLocalRuntimeMixin, KernelExecutionMixin, KernelPlanningMixin):
    pass

__all__ = [name for name in globals() if not name.startswith("_KernelPartPath") and name not in {"_KERNEL_PARTS_DIR", "_load_kernel_part"}]

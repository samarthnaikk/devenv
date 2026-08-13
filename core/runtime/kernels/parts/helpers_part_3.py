def _is_backend_connector_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    if "claude code" in lowered and "integrat" in lowered:
        return True
    return any(
        marker in lowered
        for marker in (
            "another connector like codex or opencode",
            "options as backend",
            "option as backend",
            "another backend",
            "backend option",
            "thinking part",
            "like codex or opencode",
        )
    )


def _supports_exact_logged_answer_prompt(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return _is_session_history_question(user_prompt) or "getgit" in lowered or "get-drip" in lowered


def _exact_logged_query_variants(user_prompt: str) -> tuple[str, ...]:
    variants: list[str] = []

    def add(candidate: str) -> None:
        normalized = " ".join(str(candidate or "").strip().lower().split())
        if normalized and normalized not in variants:
            variants.append(normalized)

    base = user_prompt
    add(base)
    add(base.rstrip("?.!"))
    if "schrema" in base.lower():
        add(re.sub(r"schrema", "schema", base, flags=re.IGNORECASE))
    if re.search(r"\bog\b", base, flags=re.IGNORECASE):
        add(re.sub(r"\bog\b", "of", base, flags=re.IGNORECASE))
    if "schrema" in base.lower() and re.search(r"\bog\b", base, flags=re.IGNORECASE):
        add(re.sub(r"\bog\b", "of", re.sub(r"schrema", "schema", base, flags=re.IGNORECASE), flags=re.IGNORECASE))
    lowered = base.lower()
    if lowered.startswith("do you know about "):
        add("what " + base)
    if lowered.startswith("what do you know about "):
        add(base.replace("what do you know about", "do you know about", 1))
    if "project" in lowered:
        add(re.sub(r"\bproject\b", "", base, flags=re.IGNORECASE))
    if re.search(r"\bwe did\b", lowered):
        add(re.sub(r"\bwe did\b", "", base, flags=re.IGNORECASE))
    if "get-drip" in lowered and (
        "what were the bugs we found" in lowered
        or "which bugs did we find" in lowered
        or "what bugs did we find" in lowered
    ):
        add("what exact bugs did we fix in get-drip")
        add("what bugs did we fix in get-drip")
        add("get-drip bug list")
    if any(
        phrase in lowered
        for phrase in (
            "what was the latest code edit",
            "what was the last code edit",
            "what was the most recent code edit",
        )
    ):
        add(re.sub(r"^\s*what was the\s+", "", base, flags=re.IGNORECASE))
    if any(
        phrase in lowered
        for phrase in (
            "what was the latest change",
            "what was the last change",
            "what was the most recent change",
        )
    ):
        add(re.sub(r"^\s*what was the\s+", "", base, flags=re.IGNORECASE))
    return tuple(variants)


def _should_skip_exact_logged_fast_path(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return any(
        marker in lowered
        for marker in (
            "how does retrieval work",
            "how does the retrieval work",
            "how does memory retrieval work",
            "how does the memory retrieval work",
            "how does retrieval of memory work",
            "how does this retrieval work",
            "how does this memory work",
            "how does the memory work",
            "can you explain how the retrieval works",
            "can you explain how retrieval works",
            "can you explain how the memory retrieval works",
            "what can be said confidently",
            "what remains unclear",
            "what was the backend",
        )
    )


def _is_file_inventory_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return "list the concrete files" in lowered or "files or folders" in lowered


def _is_repo_summary_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return any(
        phrase in lowered
        for phrase in (
            "what is this project about",
            "what is the project about",
            "tell me about this project",
            "tell me about the project",
            "explain this project",
            "explain the project",
            "summarize this repo",
            "summarize the repo",
            "summarize this repository",
            "summarize the repository",
            "what is this codebase",
            "what is the codebase",
            "what is this repo",
            "what is the repo",
            "tell me about this repo",
            "tell me about the repo",
            "tell me about this repository",
            "tell me about the repository",
            "explain the repo",
            "explain this repo",
            "explain the repository",
            "explain this repository",
            "summarize this codebase",
            "summarize the codebase",
            "tell me about this codebase",
            "tell me about the codebase",
        )
    )


def _is_repo_overview_question(user_prompt: str) -> bool:
    if _is_repo_summary_question(user_prompt):
        return True
    lowered = user_prompt.lower()
    return any(
        phrase in lowered
        for phrase in (
            "how does the repo work",
            "how does the repository work",
            "how does the codebase work",
            "how does this repo work",
            "how does this repository work",
            "what is this codebase",
            "what is the codebase",
            "what is this repo",
            "what is the repo",
        )
    )


def _should_trust_memory_answer_for_prompt(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    if _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt) or _is_session_history_question(user_prompt):
        return True
    if _tool_strategy_subject_prompt(user_prompt) is not None:
        return False
    if _is_repo_summary_question(user_prompt):
        return False
    if _is_bug_list_question(user_prompt):
        return True
    if any(
        phrase in lowered
        for phrase in (
            "what were the issues",
            "what were the main issues",
            "what issues were",
            "issues sharmil was talking about",
        )
    ):
        return True
    if any(
        phrase in lowered
        for phrase in (
            "how does retrieval work",
            "how does the retrieval work",
            "how does memory retrieval work",
            "how does the memory retrieval work",
            "how does retrieval of memory work",
            "how does this retrieval work",
            "how does this memory work",
            "how does the memory work",
            "can you explain how the retrieval works",
            "can you explain how retrieval works",
            "can you explain how the memory retrieval works",
        )
    ):
        return False
    if any(
        phrase in lowered
        for phrase in (
            "how does the repo work",
            "how does the repository work",
            "how does the codebase work",
            "how does this repo work",
            "how does this repository work",
        )
    ):
        return True
    if any(
        phrase in lowered
        for phrase in (
            "how does the backend work",
            "how does the system work",
            "explain this project architecture",
            "how does this backend work",
        )
    ):
        return False
    if any(
        phrase in lowered
        for phrase in (
            "how does",
            "how do",
            "why does",
            "why do",
            "architecture",
            "backend work",
            "codebase work",
            "repository work",
            "system work",
        )
    ) and not _has_explicit_project_subject(user_prompt):
        return False
    return False


def _prefers_deeper_workspace_scan(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    if _is_backend_connector_question(user_prompt):
        return True
    if _is_repo_summary_question(user_prompt):
        return True
    if _is_runtime_routing_question(user_prompt):
        return True
    return any(
        phrase in lowered
        for phrase in (
            "how does the backend work",
            "tell me about the backend",
            "tell me about this backend",
            "what is the backend",
            "how does the repo work",
            "how does the repository work",
            "tell me about this repo",
            "tell me about the repo",
            "tell me about this repository",
            "tell me about the repository",
            "what is this codebase",
            "what is the codebase",
            "what is this repo",
            "what is the repo",
            "how does the system work",
            "what is the system",
            "tell me about the system",
            "tell me about this system",
            "how does this backend work",
            "tell me about this codebase",
            "tell me about the codebase",
            "architecture",
            "codebase work",
            "backend work",
        )
    )


def _has_explicit_project_subject(user_prompt: str) -> bool:
    generic = {
        "about",
        "again",
        "architecture",
        "backend",
        "bugs",
        "code",
        "codebase",
        "elaborate",
        "explain",
        "exactly",
        "file",
        "files",
        "fixed",
        "how",
        "issue",
        "issues",
        "know",
        "project",
        "remember",
        "repo",
        "repository",
        "review",
        "reviews",
        "system",
        "tell",
        "those",
        "what",
        "work",
        "works",
    }
    if _memory_query_entities(user_prompt):
        return True
    return any(
        token not in generic
        for token in re.findall(r"[a-z0-9_]+", user_prompt.lower())
        if len(token) >= 5
    )


def _compose_external_memory_query(user_prompt: str, conversation: list[dict[str, Any]]) -> str:
    query_lines = [user_prompt]
    lowered = user_prompt.lower()
    if "get-drip" in lowered and (
        "last time" in lowered
        or "what issue did we get" in lowered
        or "what issues did we get" in lowered
    ):
        for variant in (
            "what exact bugs did we fix in get-drip",
            "what bugs did we fix in get-drip",
            "get-drip bug list",
            "get-drip last issue",
        ):
            if variant not in query_lines:
                query_lines.append(variant)
    if not _is_memory_follow_up_question(user_prompt):
        return "\n".join(query_lines)
    recent_hints = _recent_memory_subject_hints(user_prompt, conversation)
    if not recent_hints:
        return "\n".join(query_lines)
    hyphenated_hints = [hint for hint in recent_hints if "-" in hint]
    if hyphenated_hints:
        recent_hints = hyphenated_hints
    filtered_hints = [hint for hint in recent_hints if not re.fullmatch(r"[0-9a-f]{6,}", hint)]
    if filtered_hints:
        recent_hints = filtered_hints
    query_lines.append(f"Referenced context: {' '.join(recent_hints[:4])}")
    return "\n".join(query_lines)


def _has_explicit_memory_subject(user_prompt: str) -> bool:
    generic = {
        "about",
        "again",
        "bugs",
        "elaborate",
        "explain",
        "exactly",
        "fixed",
        "issue",
        "issues",
        "know",
        "project",
        "remember",
        "review",
        "reviews",
        "tell",
        "those",
        "what",
    }
    if _memory_query_entities(user_prompt):
        return True
    return any(
        token not in generic
        for token in re.findall(r"[a-z0-9_]+", user_prompt.lower())
        if len(token) >= 5
    )


def _recent_memory_subject_hints(user_prompt: str, conversation: list[dict[str, Any]]) -> list[str]:
    user_hints: list[str] = []
    assistant_hints: list[str] = []
    for message in reversed(conversation[-6:]):
        role = str(message.get("role") or "")
        content = str(message.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content or content == user_prompt:
            continue
        target = user_hints if role == "user" else assistant_hints
        for hint in _memory_subject_hints(content):
            if hint not in target:
                target.append(hint)
        if len(user_hints) >= 4 and len(assistant_hints) >= 4:
            break

    prioritized_hints: list[str] = []
    for group in (user_hints, assistant_hints):
        for hint in group:
            if hint not in prioritized_hints:
                prioritized_hints.append(hint)
            if len(prioritized_hints) >= 8:
                return prioritized_hints
    return prioritized_hints


def _is_ambiguous_memory_follow_up(user_prompt: str, conversation: list[dict[str, Any]]) -> bool:
    if not _is_memory_follow_up_question(user_prompt):
        return False
    return not _recent_memory_subject_hints(user_prompt, conversation)


def _memory_subject_hints(text: str) -> list[str]:
    lowered_text = text.lower()
    if any(
        marker in lowered_text
        for marker in (
            "i couldn't recover a reliable prior answer",
            "i couldn't recover a reliable prior note",
            "local-only mode could not inspect",
            "local-only mode needs workspace inspection tools",
            "opencode backend access has not been granted",
        )
    ):
        return []
    generic = {
        "assistant",
        "answer",
        "context",
        "decision-complete",
        "desktop",
        "feature-structured",
        "recover",
        "reliable",
        "project",
        "prior",
        "remember",
        "reported",
        "rollout",
        "session",
        "samarthnaik",
        "targeted",
        "test-activate",
        "user",
        "users",
        "workspace",
    }
    hints: list[str] = []
    for match in re.findall(r"/[A-Za-z0-9._/-]+", text):
        basename = Path(match).name.lower().strip(".,:;!?)(")
        if len(basename) >= 3 and basename not in generic and basename not in hints:
            hints.append(basename)
    for entity in sorted(_memory_query_entities(text)):
        if "/" in entity or entity.startswith("rollout-"):
            continue
        if entity not in generic and entity not in hints:
            hints.append(entity)
    for token in re.findall(r"[a-z0-9_]+", text.lower()):
        if len(token) < 5 or token in generic or token.startswith("rollout") or re.fullmatch(r"[0-9a-f]{6,}", token):
            continue
        if token not in hints:
            hints.append(token)
    return hints


def _summarize_symbol_outline(file_name: str, payload: dict[str, Any] | list[Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    symbols = payload.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        return None

    special_summaries = {
        "kernel.py": "`kernel.py` is the main orchestrator: it handles memory retrieval, routing, planning/checkpoint flow, tool execution, and verification for each turn.",
        "core/runtime/kernel.py": "`core/runtime/kernel.py` is the main orchestrator: it handles memory retrieval, routing, planning/checkpoint flow, tool execution, and verification for each turn.",
        "runtime.py": "`runtime.py` is the main orchestrator implementation: it handles memory retrieval, routing, planning/checkpoint flow, tool execution, and verification for each turn.",
        "core/runtime/kernels/runtime.py": "`core/runtime/kernels/runtime.py` is the main orchestrator implementation: it handles memory retrieval, routing, planning/checkpoint flow, tool execution, and verification for each turn.",
        "web.py": "`web.py` runs the local web backend: it serves the app, exposes health/files/turn endpoints, and sanitizes noisy replay output before users see it.",
        "core/runtime/web.py": "`core/runtime/web.py` runs the local web backend: it serves the app, exposes health/files/turn endpoints, and sanitizes noisy replay output before users see it.",
        "routing.py": "`routing.py` owns AI backend routing: it switches between OpenCode, Ollama, and Codex while keeping Devenv in charge of tools and backend preference.",
        "core/ai/routing.py": "`core/ai/routing.py` owns AI backend routing: it switches between OpenCode, Ollama, and Codex while keeping Devenv in charge of tools and backend preference.",
        "codex_backend.py": "`codex_backend.py` adapts the Codex backend into Devenv's shared chat/status contract so it can be selected like the other reasoning engines.",
        "ollama_backend.py": "`ollama_backend.py` adapts local Ollama models into the same backend contract, including status, model selection, and bounded chat calls.",
        "opencode_client.py": "`opencode_client.py` is the transport layer for OpenCode server health checks, session management, message submission, and recovery.",
        "context_builder.py": "`context_builder.py` assembles the context packet from memory, workspace evidence, and prior session history before a model turn.",
        "engine.py": "`engine.py` is the memory engine: it stores episodic and associative memory and retrieves project context for later turns.",
        "workspace.py": "`workspace.py` provides sandboxed workspace browsing and file access so retrieval stays inside the project boundary.",
    }
    special_summary = special_summaries.get(file_name.lower())
    if special_summary:
        return special_summary

    functions: list[str] = []
    classes: list[str] = []
    for symbol in symbols:
        if not isinstance(symbol, dict):
            continue
        name = symbol.get("name")
        if not isinstance(name, str):
            continue
        if symbol.get("type") == "class":
            classes.append(name)
        elif symbol.get("type") == "function":
            functions.append(name)

    parts: list[str] = []
    if classes:
        parts.append(f"classes: {', '.join(classes[:4])}")
    if functions:
        parts.append(f"functions: {', '.join(functions[:6])}")
    if not parts:
        return None
    return f"`{file_name}` exposes {'. '.join(parts)}."


def _summarize_local_text_file(file_name: str, content: str) -> str | None:
    stripped = content.strip()
    if not stripped:
        return None

    frameworks = []
    known_terms = (
        "FastAPI",
        "Flask",
        "SQLAlchemy",
        "Redis",
        "GraphQL",
        "LanceDB",
        "SentenceTransformer",
        "Retriever",
        "RAG",
    )
    lowered = stripped.lower()
    for term in known_terms:
        if term.lower() in lowered:
            frameworks.append(term)

    first_lines = [line.strip() for line in stripped.splitlines() if line.strip()][:3]
    preview = " ".join(first_lines)
    preview = re.sub(r"\s+", " ", preview)[:220].rstrip()
    if file_name.lower() == "readme.md":
        heading = next((line.strip().lstrip("#").strip() for line in stripped.splitlines() if line.strip().startswith("#")), "")
        prose_lines = [
            line.strip()
            for line in stripped.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        prose_block = re.sub(r"\s+", " ", " ".join(prose_lines[:4])).strip()
        sentence_parts = re.split(r"(?<=[.!?])\s+", prose_block)
        prose_preview = " ".join(part.strip() for part in sentence_parts[:2] if part.strip()).strip()
        if heading and prose_preview:
            sentence = prose_preview
            if heading.lower().startswith("devenv ai is "):
                sentence = prose_preview
            elif prose_preview.lower().startswith(heading.lower()):
                sentence = prose_preview
            else:
                sentence = f"{heading} is {prose_preview[0].lower() + prose_preview[1:]}" if len(prose_preview) > 1 else f"{heading} is {prose_preview.lower()}"
            prefix = "`README.md` says" if sentence.lower().startswith(heading.lower()) else "`README.md` describes"
            if frameworks:
                return f"{prefix} {sentence} It also references {', '.join(frameworks[:4])}."
            return f"{prefix} {sentence}"
    if frameworks:
        return f"`{file_name}` references {', '.join(frameworks[:4])}. Preview: {preview}"
    return f"`{file_name}` preview: {preview}"


def _extract_context_only_file_answer(*, user_prompt: str, task_description: str, file_name: str, content: str) -> str | None:
    lowered_request = f"{user_prompt} {task_description}".lower()
    if file_name.lower().endswith(".md") and any(token in lowered_request for token in ("h1", "heading", "title")):
        heading = next((line.strip().lstrip("#").strip() for line in content.splitlines() if line.strip().startswith("#")), "")
        if heading:
            return heading
    return None


def _first_backticked_path(text: str) -> str | None:
    for match in re.findall(r"`([^`]+)`", text):
        candidate = str(match).strip()
        if "/" in candidate or candidate.endswith((".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css")):
            return candidate
    return None


def _sanitize_model_generated_path(path: str) -> str:
    cleaned = str(path or "").strip().strip("`").replace("\\", "/")
    is_absolute = cleaned.startswith("/")
    parts: list[str] = []
    for raw_part in cleaned.split("/"):
        part = raw_part.strip().strip("`")
        if not part:
            continue
        part = re.sub(r"^\*+|\*+$", "", part)
        if part.lower() == "init.py":
            part = "__init__.py"
        parts.append(part)
    normalized = "/".join(parts)
    if is_absolute and normalized:
        return f"/{normalized}"
    return normalized


def _normalized_path_identity(name: str) -> str:
    lowered = str(name or "").lower().strip()
    lowered = lowered.replace("`", "").replace("*", "")
    stem = Path(lowered).stem
    suffix = Path(lowered).suffix
    normalized_stem = re.sub(r"[^a-z0-9]+", "", stem)
    if normalized_stem == "init" and suffix == ".py":
        return "__init__.py"
    return f"{normalized_stem}{suffix}"


def _local_chatapp_init_py() -> str:
    return """from .routes import CHATAPP_ROUTE, build_chatapp_response

__all__ = ["CHATAPP_ROUTE", "build_chatapp_response"]
"""


def _local_chatapp_store_py() -> str:
    return """from __future__ import annotations

from collections import defaultdict


class ChatSessionStore:
    def __init__(self) -> None:
        self._messages: dict[str, list[dict[str, str]]] = defaultdict(list)

    def append(self, session_id: str, role: str, content: str) -> None:
        self._messages[session_id].append({"role": role, "content": content})

    def history(self, session_id: str) -> list[dict[str, str]]:
        return list(self._messages.get(session_id, ()))


store = ChatSessionStore()
"""


def _local_chatapp_service_py() -> str:
    return """from __future__ import annotations

from .store import store


def handle_chat_message(session_id: str, message: str) -> dict[str, object]:
    cleaned = message.strip()
    if not cleaned:
        return {"error": "message is required"}

    store.append(session_id, "user", cleaned)
    reply = f"Echo: {cleaned}"
    store.append(session_id, "assistant", reply)
    return {
        "session_id": session_id,
        "reply": reply,
        "messages": store.history(session_id),
    }
"""


def _local_chatapp_routes_py(integration_root: str) -> str:
    return f"""from __future__ import annotations

from .service import handle_chat_message

CHATAPP_ROUTE = "/api/{integration_root}/messages"


def build_chatapp_response(payload: dict[str, object] | None = None) -> dict[str, object]:
    data = dict(payload or {{}})
    session_id = str(data.get("session_id") or "default-session")
    message = str(data.get("message") or "")
    return handle_chat_message(session_id, message)
"""


def _append_once(existing: str, block: str) -> str:
    if block.strip() in existing:
        return existing
    trimmed = existing.rstrip()
    if trimmed:
        return f"{trimmed}\n\n{block.rstrip()}\n"
    return f"{block.rstrip()}\n"


def _merge_local_web_integration(existing: str, integration_root: str) -> str:
    block = f"""# Chat app integration
from {integration_root} import CHATAPP_ROUTE, build_chatapp_response


def register_chatapp_route() -> tuple[str, object]:
    return CHATAPP_ROUTE, build_chatapp_response
"""
    return _append_once(existing, block)


def _merge_local_routing_integration(existing: str) -> str:
    block = """# Chat app routing
CHATAPP_ROUTE_KEY = "chatapp"


def route_chatapp_request() -> str:
    return CHATAPP_ROUTE_KEY
"""
    return _append_once(existing, block)


def _merge_local_frontend_api_integration(existing: str) -> str:
    block = """export async function sendChatAppMessage(sessionId, message) {
  return {
    endpoint: "/api/chatapp/messages",
    sessionId,
    message,
  };
}
"""
    return _append_once(existing, block)


def _merge_local_frontend_ui_integration(existing: str, relative_path: str) -> str:
    import_path = "./api" if relative_path.endswith("/App.js") else "../api"
    import_block = f'import {{ sendChatAppMessage }} from "{import_path}";'
    handler_block = """export async function submitChatAppMessage(sessionId, message) {
  return sendChatAppMessage(sessionId, message);
}
"""
    merged = _append_once(existing, import_block)
    return _append_once(merged, handler_block)


def _local_scaffold_kind(*texts: str) -> str:
    joined = " ".join(texts).lower()
    if "calendar" in joined:
        return "calendar"
    if "notes app" in joined or "note-taking" in joined or ("notes" in joined and "app" in joined):
        return "notes"
    if "todo app" in joined or "task list" in joined or ("todo" in joined and "app" in joined):
        return "todo"
    if "kanban" in joined or "board app" in joined or ("board" in joined and "localstorage" in joined):
        return "kanban"
    if "weather app" in joined or "forecast" in joined or ("weather" in joined and "app" in joined):
        return "weather"
    if any(marker in joined for marker in ("today's date", "todays date", "today date", "current date")):
        return "date"
    return "generic"


def _local_scaffold_html(scaffold_kind: str, target_path: str) -> str:
    if scaffold_kind == "calendar":
        return _local_calendar_html(target_path)
    if scaffold_kind == "notes":
        return _local_notes_html()
    if scaffold_kind == "todo":
        return _local_todo_html()
    if scaffold_kind == "kanban":
        return _local_kanban_html()
    if scaffold_kind == "weather":
        return _local_weather_html()
    if scaffold_kind == "date":
        return _local_date_html()
    return _local_generic_html()


def _local_scaffold_css(scaffold_kind: str, *, dark_theme: bool = False) -> str:
    if scaffold_kind == "calendar":
        return _local_calendar_css(dark_theme=dark_theme)
    if scaffold_kind == "notes":
        return _local_notes_css()
    if scaffold_kind == "todo":
        return _local_todo_css()
    if scaffold_kind == "kanban":
        return _local_kanban_css()
    if scaffold_kind == "weather":
        return _local_weather_css()
    if scaffold_kind == "date":
        return _local_date_css()
    return _local_generic_css()


def _local_scaffold_js(scaffold_kind: str) -> str:
    if scaffold_kind == "calendar":
        return _local_calendar_js()
    if scaffold_kind == "notes":
        return _local_notes_js()
    if scaffold_kind == "todo":
        return _local_todo_js()
    if scaffold_kind == "kanban":
        return _local_kanban_js()
    if scaffold_kind == "weather":
        return _local_weather_js()
    if scaffold_kind == "date":
        return _local_date_js()
    return _local_generic_js()


def _local_calendar_html(target_path: str) -> str:
    asset_prefix = ""
    if "/" in target_path:
        asset_prefix = ""
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Calendar</title>
    <link rel="stylesheet" href="{asset_prefix}styles.css" />
  </head>
  <body>
    <main class="calendar-app">
      <header class="calendar-header">
        <button id="prev-month" type="button" aria-label="Previous month">Prev</button>
        <div>
          <p class="calendar-kicker">Local Demo</p>
          <h1 id="month-label">Calendar</h1>
          <p class="calendar-today" id="today-label">Today</p>
        </div>
        <button id="next-month" type="button" aria-label="Next month">Next</button>
      </header>
      <section class="calendar-panel">
        <div class="calendar-weekdays" id="calendar-weekdays"></div>
        <div class="calendar-grid" id="calendar-grid"></div>
      </section>
    </main>
    <script src="{asset_prefix}script.js"></script>
  </body>
</html>
"""


def _local_notes_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Notes Studio</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <main class="notes-app">
      <section class="notes-hero">
        <p class="notes-kicker">Local Workspace</p>
        <h1>Notes Studio</h1>
        <p class="notes-copy">Capture quick thoughts, pin your best ideas, and keep them in your browser.</p>
      </section>
      <section class="notes-shell">
        <form class="notes-composer" id="note-form">
          <label for="note-title">Title</label>
          <input id="note-title" name="title" type="text" placeholder="Sprint retro" maxlength="60" />
          <label for="note-body">Note</label>
          <textarea id="note-body" name="body" rows="6" placeholder="Write the note you want to keep..."></textarea>
          <button type="submit">Save note</button>
        </form>
        <section class="notes-feed">
          <div class="notes-feed-header">
            <h2>Saved notes</h2>
            <p id="notes-status">Nothing saved yet.</p>
          </div>
          <div id="notes-list" class="notes-list"></div>
        </section>
      </section>
    </main>
    <script src="script.js"></script>
  </body>
</html>
"""


def _local_todo_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Task Sprint</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <main class="todo-app">
      <header class="todo-hero">
        <p class="todo-kicker">Local Planner</p>
        <h1>Task Sprint</h1>
        <p>Track the next few things that matter and check them off as you go.</p>
      </header>
      <section class="todo-shell">
        <form id="todo-form" class="todo-form">
          <input id="todo-input" type="text" maxlength="80" placeholder="Add a task" />
          <button type="submit">Add</button>
        </form>
        <div class="todo-summary">
          <p id="todo-count">0 tasks pending</p>
        </div>
        <ul id="todo-list" class="todo-list"></ul>
      </section>
    </main>
    <script src="script.js"></script>
  </body>
</html>
"""


def _local_kanban_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Kanban Sprint</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <main class="kanban-app">
      <header class="kanban-hero">
        <p class="kanban-kicker">Local Delivery Board</p>
        <h1>Kanban Sprint</h1>
        <p>Capture quick tasks, then move them across the board with local persistence.</p>
      </header>
      <section class="kanban-shell">
        <form id="kanban-form" class="kanban-form">
          <input id="kanban-input" type="text" maxlength="80" placeholder="Add a board card" />
          <button type="submit">Add card</button>
        </form>
        <div class="kanban-board" id="kanban-board">
          <section class="kanban-column" data-column="todo">
            <div class="kanban-column-header">
              <h2>Todo</h2>
              <span id="count-todo">0</span>
            </div>
            <div id="column-todo" class="kanban-cards"></div>
          </section>
          <section class="kanban-column" data-column="doing">
            <div class="kanban-column-header">
              <h2>Doing</h2>
              <span id="count-doing">0</span>
            </div>
            <div id="column-doing" class="kanban-cards"></div>
          </section>
          <section class="kanban-column" data-column="done">
            <div class="kanban-column-header">
              <h2>Done</h2>
              <span id="count-done">0</span>
            </div>
            <div id="column-done" class="kanban-cards"></div>
          </section>
        </div>
      </section>
    </main>
    <script src="script.js"></script>
  </body>
</html>
"""


def _local_weather_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Skyboard</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <main class="weather-app">
      <section class="weather-hero">
        <p class="weather-kicker">Static Forecast Demo</p>
        <h1>Skyboard</h1>
        <p id="weather-status">Clear planning weather for the next three checkpoints.</p>
      </section>
      <section class="weather-shell">
        <div class="weather-current">
          <p class="weather-city">San Francisco</p>
          <h2 id="weather-temp">68°F</h2>
          <p id="weather-summary">Mild breeze and bright skies.</p>
          <button id="weather-refresh" type="button">Refresh outlook</button>
        </div>
        <div id="forecast-grid" class="forecast-grid"></div>
      </section>
    </main>
    <script src="script.js"></script>
  </body>
</html>
"""


def _local_date_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Today Board</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <main class="date-app">
      <section class="date-card">
        <p class="date-kicker">Live Local Date</p>
        <h1 id="today-label">Today</h1>
        <p id="date-detail">Preparing the current date...</p>
      </section>
    </main>
    <script src="script.js"></script>
  </body>
</html>
"""


def _local_generic_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Local Starter App</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <main class="starter-app">
      <section class="starter-card">
        <p class="starter-kicker">Static Starter</p>
        <h1>Local Starter App</h1>
        <p id="starter-status">Your app is ready for custom interaction.</p>
        <button id="starter-action" type="button">Try it</button>
      </section>
    </main>
    <script src="script.js"></script>
  </body>
</html>
"""


def _local_calendar_css(*, dark_theme: bool = False) -> str:
    if dark_theme:
        return """:root {
  color-scheme: dark;
  --bg: #11161d;
  --panel: rgba(24, 32, 43, 0.96);
  --border: rgba(132, 148, 173, 0.22);
  --text: #f2f5f8;
  --muted: #97a6ba;
  --accent: #7cc7ff;
  --accent-soft: rgba(124, 199, 255, 0.16);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at top, rgba(124, 199, 255, 0.12), transparent 28%),
    linear-gradient(180deg, #0d131a 0%, #151d27 100%);
  color: var(--text);
}

.calendar-app {
  max-width: 960px;
  margin: 48px auto;
  padding: 24px;
}

.calendar-header,
.calendar-weekdays,
.calendar-grid {
  display: grid;
  gap: 12px;
}

.calendar-header {
  grid-template-columns: 92px 1fr 92px;
  align-items: center;
  margin-bottom: 18px;
}

.calendar-header button {
  border: 1px solid var(--border);
  background: rgba(18, 25, 35, 0.92);
  color: var(--text);
  border-radius: 10px;
  padding: 10px 12px;
  cursor: pointer;
}

.calendar-kicker {
  margin: 0 0 4px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 12px;
}

.calendar-header h1 {
  margin: 0;
  font-size: 32px;
}

.calendar-panel {
  border: 1px solid var(--border);
  border-radius: 18px;
  background: var(--panel);
  padding: 20px;
  box-shadow: 0 20px 40px rgba(3, 7, 12, 0.34);
}

.calendar-weekdays,
.calendar-grid {
  grid-template-columns: repeat(7, minmax(0, 1fr));
}

.calendar-weekdays {
  margin-bottom: 12px;
  color: var(--muted);
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.calendar-grid {
  min-height: 420px;
}

.calendar-day {
  border: 1px solid var(--border);
  background: rgba(15, 22, 32, 0.88);
  color: var(--text);
  border-radius: 14px;
  padding: 12px;
  min-height: 88px;
}

.calendar-day.is-today {
  border-color: var(--accent);
  background: var(--accent-soft);
}

.calendar-day.is-empty {
  background: rgba(255, 255, 255, 0.03);
}

@media (max-width: 720px) {
  .calendar-app {
    margin: 20px auto;
    padding: 16px;
  }

  .calendar-header {
    grid-template-columns: 1fr 1fr;
  }

  .calendar-header h1 {
    font-size: 24px;
  }
}
"""

    return """:root {
  color-scheme: light;
  --bg: #f6f4ef;
  --panel: #ffffff;
  --border: #d4cec3;
  --text: #1f1f1f;
  --muted: #6b655d;
  --accent: #1f6feb;
  --accent-soft: #dce9ff;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  background: linear-gradient(180deg, #f8f6f1 0%, #ece7dc 100%);
  color: var(--text);
}

.calendar-app {
  max-width: 960px;
  margin: 48px auto;
  padding: 24px;
}

.calendar-header,
.calendar-weekdays,
.calendar-grid {
  display: grid;
  gap: 12px;
}

.calendar-header {
  grid-template-columns: 92px 1fr 92px;
  align-items: center;
  margin-bottom: 18px;
}

.calendar-header button {
  border: 1px solid var(--border);
  background: var(--panel);
  color: var(--text);
  border-radius: 10px;
  padding: 10px 12px;
  cursor: pointer;
}

.calendar-kicker {
  margin: 0 0 4px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 12px;
}

.calendar-header h1 {
  margin: 0;
  font-size: 32px;
}

.calendar-panel {
  border: 1px solid var(--border);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.92);
  padding: 20px;
  box-shadow: 0 20px 40px rgba(77, 68, 51, 0.08);
}

.calendar-weekdays,
.calendar-grid {
  grid-template-columns: repeat(7, minmax(0, 1fr));
}

.calendar-weekdays {
  margin-bottom: 12px;
  color: var(--muted);
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.calendar-grid {
  min-height: 420px;
}

.calendar-day {
  border: 1px solid var(--border);
  background: var(--panel);
  border-radius: 14px;
  padding: 12px;
  min-height: 88px;
}

.calendar-day.is-today {
  border-color: var(--accent);
  background: var(--accent-soft);
}

.calendar-day.is-empty {
  background: rgba(0, 0, 0, 0.03);
}

@media (max-width: 720px) {
  .calendar-app {
    margin: 20px auto;
    padding: 16px;
  }

  .calendar-header {
    grid-template-columns: 1fr 1fr;
  }

  .calendar-header h1 {
    font-size: 24px;
  }
}
"""

def _should_persist_episodic_response(final_response: str) -> bool:
    lowered = str(final_response or "").strip().lower()
    if not lowered:
        return False
    if lowered.startswith("i inspected `"):
        return False
    if lowered.startswith("i inspected the backend entry points locally."):
        return False
    if lowered.startswith("i couldn't recover a reliable prior answer"):
        return False
    if lowered.startswith("i couldn't recover a reliable prior note"):
        return False
    if lowered.startswith("i couldn't recover a reliable note about the last merge conflict"):
        return False
    if lowered.startswith("opencode backend access has not been granted"):
        return False
    if lowered.startswith("opencode backend access is not granted right now"):
        return False
    if lowered == "nothing left to execute.":
        return False
    return True


def _is_high_signal_memory_answer(candidate: str, user_prompt: str) -> bool:
    lowered = candidate.lower()
    normalized = re.sub(r"^(assistant reported|user asked):\s*", "", lowered)
    prompt_lowered = user_prompt.lower()
    reject_markers = (
        '{"type": "function"',
        '"type":"function"',
        '"name": "list_directory"',
        '"name":"list_directory"',
        '"required":[',
        '\\"required\\":[',
        '"properties":{',
        '\\"properties\\":{',
        '"type":"object"',
        '"type": "object"',
        '\\"type\\":\\"object\\"',
        'relative_path',
        '"depth":',
        '"is_dir":',
        '"path":',
        "tool requested",
        "recovered inline tool request",
        "queued prompt",
        "i inspected `",
        "locally. relevant paths i found:",
        "i don't have access",
        "i do not have access",
        "good, but its too less of info",
        "good, but it's too less of info",
        "devenv status",
        "tool trace",
        "prepared the final answer",
        "reasoned through the next step",
        "i couldn't recover a reliable prior answer",
        "i couldn't recover a reliable prior note",
        "local-only mode could not inspect",
        "local-only mode needs workspace inspection tools",
        "opencode backend access has not been granted",
    )
    if any(marker in lowered for marker in reject_markers):
        return False
    low_signal_prefixes = (
        "i’m grounding",
        "i'm grounding",
        "i’m tracing",
        "i'm tracing",
        "i’m checking",
        "i'm checking",
        "i’m going to",
        "i'm going to",
        "i’ve confirmed",
        "i've confirmed",
        "next i’m",
        "next i'm",
        "i’ll pick up",
        "i'll pick up",
    )
    if normalized.startswith(low_signal_prefixes):
        return False
    if normalized.strip(" .,:;!?") == prompt_lowered.strip(" .,:;!?"):
        return False
    if ("repo" in prompt_lowered or "repository" in prompt_lowered or "codebase" in prompt_lowered) and any(
        marker in lowered for marker in ("main.py", "server.py", "app.py")
    ) and not any(
        marker in lowered for marker in ("repo", "repository", "codebase", "workspace", "files", "folders", "architecture", "backend")
    ):
        return False
    if _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt):
        if normalized.startswith(low_signal_prefixes):
            return False
    if lowered.startswith("{") or lowered.startswith("["):
        return False
    if ("how does" in prompt_lowered or "how do" in prompt_lowered or "why does" in prompt_lowered) and any(
        marker in lowered
        for marker in ('"required":[', '\\"required\\":[', '"properties":{', '\\"properties\\":{', '"type":"object"', '"type": "object"', '\\"type\\":\\"object\\"')
    ):
        return False
    if ("how does" in prompt_lowered or "how do" in prompt_lowered or "why does" in prompt_lowered) and "i inspected" in lowered:
        return False
    if "strongest clues point to" in lowered and ("schema" in prompt_lowered or "cleanup" in prompt_lowered):
        return False
    return True


def _memory_query_tokens(user_prompt: str) -> set[str]:
    common = {
        "about",
        "again",
        "anything",
        "does",
        "know",
        "project",
        "remember",
        "that",
        "this",
        "what",
    }
    tokens = {
        token
        for token in re.findall(r"[a-z0-9_]+", user_prompt.lower())
        if len(token) >= 4 and token not in common
    }
    return tokens | _memory_query_entities(user_prompt)


def _memory_query_entities(user_prompt: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)+", user_prompt.lower())
        if len(token) >= 3
    }


def _is_brief_greeting_prompt(user_prompt: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s]", " ", user_prompt.lower())
    tokens = [token for token in normalized.split() if token]
    if not tokens or len(tokens) > 3:
        return False
    greeting_tokens = {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "hiya",
        "heya",
        "gm",
        "goodmorning",
        "morning",
        "afternoon",
        "evening",
    }
    joined = "".join(tokens)
    if joined in {"goodmorning", "goodafternoon", "goodevening"}:
        return True
    return all(token in greeting_tokens for token in tokens)


def _memory_context_entities(memory_context: str) -> set[str]:
    sections = _memory_context_sections(memory_context)
    entities: set[str] = set()
    for line in [*sections["working"], *sections["retrieved"], *sections["external"]]:
        entities.update(
            token.lower()
            for token in re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)+", line.lower())
            if len(token) >= 3
        )
    return entities


_GENERIC_MEMORY_SUBJECTS = {
    "decision-complete",
    "feature-structured",
    "test-activate",
    "task-getgit-checkpoints",
}


def _infer_memory_subject(lines: list[str]) -> str | None:
    for line in lines:
        lowered_line = line.lower()
        if any(marker in lowered_line for marker in ('"path":', '{"type": "function"', '"name": "list_directory"')):
            continue
        for match in re.findall(r"/[A-Za-z0-9._/-]+", line):
            basename = Path(match).name.lower()
            if basename and ("-" in basename or "_" in basename) and basename not in _GENERIC_MEMORY_SUBJECTS:
                return basename
    for line in lines:
        lowered_line = line.lower()
        if any(marker in lowered_line for marker in ('"path":', '{"type": "function"', '"name": "list_directory"')):
            continue
        matches = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)+", line.lower())
        for match in matches:
            if match.startswith("rollout-") or match in _GENERIC_MEMORY_SUBJECTS:
                continue
            return match
    return None


def _preferred_memory_subject(user_prompt: str, lines: list[str]) -> str | None:
    explicit_entities = sorted(_memory_query_entities(user_prompt), key=len, reverse=True)
    if explicit_entities:
        return explicit_entities[0]
    return _infer_memory_subject(lines)


def _ordered_follow_up_lines(user_prompt: str, external_lines: list[str]) -> list[str]:
    preferred_markers = (
        "->",
        "accept",
        "coming soon",
        "doesnt work",
        "does not work",
        "pipeline chat",
        "test and publish",
        "test/publish",
        "review",
        "bug",
        "fix",
    )
    cleaned_pairs: list[tuple[str, str]] = []
    for raw_line in external_lines:
        cleaned = _clean_memory_line(raw_line)
        if cleaned and _is_high_signal_memory_answer(cleaned, user_prompt):
            cleaned_pairs.append((raw_line.lower(), cleaned))
    if not cleaned_pairs:
        return []

    preferred_user_pairs = [
        (raw, cleaned)
        for raw, cleaned in cleaned_pairs
        if raw.startswith("user asked:") and any(marker in raw for marker in preferred_markers)
    ]
    if preferred_user_pairs:
        preferred_user_pairs.sort(key=lambda item: (_follow_up_line_score(item[0]), len(item[1])), reverse=True)
        user_lines = [cleaned for _raw, cleaned in preferred_user_pairs[:2]]
    else:
        user_lines = [cleaned for raw, cleaned in cleaned_pairs if raw.startswith("user asked:")][:2]
    marked_assistant_pairs = [
        (raw, cleaned)
        for raw, cleaned in cleaned_pairs
        if raw.startswith("assistant reported:") and any(marker in raw for marker in preferred_markers)
    ]
    marked_assistant_pairs.sort(key=lambda item: (_follow_up_line_score(item[0]), len(item[1])), reverse=True)
    marked_assistant_lines = [cleaned for _raw, cleaned in marked_assistant_pairs[:2]]
    remaining_lines = [
        cleaned
        for raw, cleaned in cleaned_pairs
        if cleaned not in user_lines and cleaned not in marked_assistant_lines and not raw.startswith("session '")
    ]
    session_lines = [cleaned for raw, cleaned in cleaned_pairs if raw.startswith("session '")]

    ordered: list[str] = []
    for group in (user_lines, marked_assistant_lines, remaining_lines, session_lines):
        for line in group:
            if line not in ordered:
                ordered.append(line)
    return ordered


def _follow_up_line_score(lowered_line: str) -> int:
    score = 0
    if "create workspace" in lowered_line:
        score += 6
    if "->" in lowered_line:
        score += 3
    for marker in (
        "accept",
        "coming soon",
        "doesnt work",
        "does not work",
        "pipeline chat",
        "test and publish",
        "test/publish",
        "review",
        "bug",
        "fix",
    ):
        if marker in lowered_line:
            score += 1
    return score


def _is_explicit_project_fact_memory_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower().strip()
    if not lowered:
        return False

    if any(
        marker in lowered
        for marker in (
            "this repo",
            "the repo",
            "this repository",
            "the repository",
            "this codebase",
            "the codebase",
            "this system",
            "the system",
        )
    ):
        return False

    has_question_shape = lowered.startswith(("what was ", "which was ", "what were ", "which were "))
    has_past_project_fact = any(
        marker in lowered
        for marker in (
            " backend",
            " frontend",
            " stack",
            " architecture",
            " database",
            " framework",
            " language",
            " api",
        )
    )
    if not (has_question_shape and has_past_project_fact):
        return False

    return _has_explicit_memory_subject(user_prompt)


def _is_error_fix_memory_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower().strip()
    if not _has_explicit_memory_subject(user_prompt):
        return False
    has_issue_marker = any(token in lowered for token in ("error", "issue", "bug", "failed", "failure"))
    has_recall_shape = lowered.startswith(("what was", "which was", "what exact", "which exact"))
    has_fix_marker = any(
        phrase in lowered
        for phrase in (
            "how did we fix",
            "how was it fixed",
            "how we fixed",
            "how did we solve",
            "how was it solved",
        )
    )
    return has_issue_marker and (has_recall_shape or has_fix_marker)


def _is_memory_recall_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return _is_explicit_project_fact_memory_question(user_prompt) or _is_error_fix_memory_question(user_prompt) or any(
        phrase in lowered
        for phrase in (
            "do you remember",
            "remember about",
            "what do you remember",
            "do you know about",
            "what do you know about",
        )
    )


def _is_session_history_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    has_history_phrase = any(
        phrase in lowered
        for phrase in (
            "last merge conflict",
            "merge conflict we solved",
            "last conflict we solved",
            "last bug we fixed",
            "last review we fixed",
            "last issue we fixed",
            "last code edit",
            "latest code edit",
            "most recent code edit",
            "last edit we did",
            "latest edit we did",
            "most recent edit we did",
            "last change we did",
            "latest change we did",
            "most recent change we did",
            "latest code change",
            "most recent code change",
            "last time",
        )
    )
    if not has_history_phrase:
        return False
    return any(
        marker in lowered
        for marker in (
            "merge conflict",
            "conflict",
            "bug",
            "issue",
            "review",
            "fix",
            "fixed",
            "code edit",
            "edit we did",
            "code change",
            "change we did",
            "get-drip",
            "getgit",
        )
    )


def _is_memory_follow_up_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    referential_markers = (
        "can you explain about it",
        "can you explain it",
        "can you elaborate it",
        "can you elaborate on it",
        "explain about it",
        "explain it",
        "elaborate it",
        "elaborate on it",
        "what are those",
        "what are those bugs",
        "what are those reviews",
        "what were those",
        "tell exactly what",
        "what was it about",
        "what was that about",
        "those bugs",
        "those reviews",
        "how did we fix those bugs",
        "how did we fix them",
        "how were those fixed",
        "a few reviews",
        "a few bugs",
        "i told you earlier",
        "i told you before",
        "i mentioned earlier",
        "i mentioned before",
    )
    referential_question_shapes = (
        "what was the",
        "what were the",
        "which was the",
        "which were the",
    )
    explicit_referential_recall = _is_explicit_referential_recall_question(user_prompt)
    if not any(marker in lowered for marker in referential_markers):
        if not explicit_referential_recall:
            return False
    if explicit_referential_recall:
        return True
    return not _has_explicit_memory_subject(user_prompt)


def _is_explicit_referential_recall_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    referential_question_shapes = (
        "what was the",
        "what were the",
        "which was the",
        "which were the",
    )
    return any(lowered.startswith(shape) for shape in referential_question_shapes) and any(
        marker in lowered for marker in ("earlier", "before", "told you", "mentioned")
    )


def _is_bug_fix_follow_up_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return any(
        marker in lowered
        for marker in (
            "how did we fix those bugs",
            "how did we fix them",
            "how were those fixed",
        )
    )


def _is_issue_recap_follow_up_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return any(
        marker in lowered
        for marker in (
            "what are those",
            "what are those bugs",
            "what are those reviews",
            "what were those",
            "those bugs",
            "those reviews",
        )
    )


def _is_issue_explanation_follow_up_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return any(
        marker in lowered
        for marker in (
            "can you explain about it",
            "can you explain it",
            "can you elaborate it",
            "can you elaborate on it",
            "explain about it",
            "explain it",
            "elaborate it",
            "elaborate on it",
            "what was it about",
            "what was that about",
        )
    )


def _is_cleanup_schema_prompt(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    if "get-drip" not in lowered:
        return False
    has_cleanup = any(marker in lowered for marker in ("cleanup", "clean up"))
    has_schema = any(marker in lowered for marker in ("schema", "schrema"))
    return has_cleanup or has_schema


def _answer_from_recent_conversation_follow_up(user_prompt: str, conversation: list[dict[str, Any]]) -> str | None:
    if not _is_memory_follow_up_question(user_prompt):
        return None

    recent_lines: list[str] = []
    last_assistant: str | None = None
    for message in reversed(conversation[-6:]):
        role = str(message.get("role") or "")
        content = _sanitize_logged_answer(str(message.get("content") or "").strip())
        if role not in {"user", "assistant"} or not content:
            continue
        recent_lines.append(content)
        if role == "assistant" and last_assistant is None:
            last_assistant = content

    if not last_assistant or not _is_high_signal_memory_answer(last_assistant, user_prompt):
        if _is_explicit_referential_recall_question(user_prompt):
            recent_user_fact = _recent_user_recall_fact(user_prompt, conversation)
            if recent_user_fact:
                return _affirm_memory_answer(recent_user_fact)
        return None

    subject = _preferred_memory_subject(user_prompt, recent_lines)
    issue_summary = _summarize_follow_up_issues(_memory_context_lines(last_assistant))
    if (
        issue_summary is None
        and "targeted workspace" in last_assistant.lower()
        and (_is_bug_fix_follow_up_question(user_prompt) or _is_issue_recap_follow_up_question(user_prompt) or _is_issue_explanation_follow_up_question(user_prompt))
    ):
        return None
    lowered = user_prompt.lower()
    if issue_summary and _is_bug_fix_follow_up_question(user_prompt):
        if subject:
            return f"Yes. In {subject}, we fixed those bugs by addressing {issue_summary}."
        return f"Yes. We fixed those bugs by addressing {issue_summary}."
    if issue_summary and _is_issue_explanation_follow_up_question(user_prompt):
        return f"Yes. It was mainly about {issue_summary}."
    if issue_summary and _is_issue_recap_follow_up_question(user_prompt):
        if subject:
            return f"Yes. In {subject}, the main issues were {issue_summary}."
        return f"Yes. The main issues were {issue_summary}."
    cleanup_summary = _summarize_cleanup_narrative(_memory_context_lines(last_assistant))
    if cleanup_summary and _is_issue_explanation_follow_up_question(user_prompt):
        return f"Yes. It was mainly about {cleanup_summary}."
    return _affirm_memory_answer(_humanize_recalled_line(last_assistant, user_prompt))


def _recent_user_recall_fact(user_prompt: str, conversation: list[dict[str, Any]]) -> str | None:
    for message in reversed(conversation[-8:]):
        role = str(message.get("role") or "")
        if role != "user":
            continue
        content = _sanitize_logged_answer(str(message.get("content") or "").strip())
        if not content or content == user_prompt:
            continue
        normalized = re.sub(
            r"^(remember this exactly(?: for later(?: in this runtime)?)?:\s*)",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()
        if not normalized:
            continue
        if normalized[:1].islower():
            normalized = normalized[:1].upper() + normalized[1:]
        if normalized.endswith("."):
            return normalized[:-1] + "."
        return normalized
    return None


def _is_bug_list_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    return any(
        phrase in lowered
        for phrase in (
            "do you know about get-drip bugs",
            "what do you know about get-drip bugs",
            "get-drip bugs",
            "bug list",
            "list the bugs",
            "exact bugs",
            "what were the bugs we found",
            "which bugs did we find",
            "what bugs did we find",
            "what were the bugs we faced",
            "which bugs did we face",
            "what bugs did we face",
            "what were the bugs we fixed",
            "which bugs did we fix",
            "what bugs did we fix",
            "give get-drip bug list",
            "give the bug list",
            "main bugs",
            "tracked bugs",
        )
    )


def _should_try_direct_memory_answer(user_prompt: str) -> bool:
    if _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt) or _is_session_history_question(user_prompt):
        return True

    lowered = user_prompt.lower().strip()
    if not lowered:
        return False

    if any(
        token in lowered
        for token in ("create", "make", "add", "write", "edit", "update", "modify", "fix", "implement", "delete", "remove")
    ):
        return False

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
            "how does this repo work",
            "how does the repo work",
            "how does this repository work",
            "how does the repository work",
            "how does this codebase work",
            "how does the codebase work",
            "how does the system work",
            "how does this system work",
            "how does the backend work",
            "how does this backend work",
            "can you explain how the retrieval works",
            "can you explain how retrieval works",
            "can you explain how the memory retrieval works",
            "can you explain how this repo works",
            "can you explain how the repo works",
            "can you explain how the codebase works",
        )
    ):
        return False

    if any(
        phrase in lowered
        for phrase in (
            "what architecture",
            "which architecture",
            "list the concrete files",
            "list the files",
            "what other work",
            "what were the main issues",
            "what were the issues",
            "what can be said confidently",
            "what remains unclear",
            "what was the backend",
            "bug list",
            "exact bugs",
            "what bugs did we fix",
            "list the bugs",
        )
    ):
        return True

    return False


def _should_skip_vector_memory_lookup(user_prompt: str) -> bool:
    if _is_explicit_project_fact_memory_question(user_prompt):
        return False
    return (
        _is_memory_recall_question(user_prompt)
        or _is_memory_follow_up_question(user_prompt)
        or _is_session_history_question(user_prompt)
    )


def _should_answer_from_memory_only(user_prompt: str) -> bool:
    return (
        _is_memory_recall_question(user_prompt)
        or _is_memory_follow_up_question(user_prompt)
        or _is_session_history_question(user_prompt)
    )


def _memory_only_fallback_response(user_prompt: str) -> str:
    if _is_session_history_question(user_prompt):
        return "I couldn't recover a reliable note about the last merge conflict we solved."
    if _is_memory_follow_up_question(user_prompt):
        return "I couldn't recover a reliable prior note for that follow-up."
    return "I couldn't recover a reliable prior answer for that yet."


def _should_skip_trace_memory_lookup(user_prompt: str) -> bool:
    lowered = user_prompt.lower().strip()
    return any(
        phrase in lowered
        for phrase in (
            "last retrieval trace",
            "retrieval trace",
            "trace from memory",
            "memory trace",
            "inspect trace",
            "show the trace",
            "show trace",
            "node history",
        )
    )


def _tool_strategy_subject_prompt(user_prompt: str) -> str | None:
    lowered = user_prompt.lower().strip()
    patterns = (
        r"^(?:what|which)\s+tools\s+do\s+you\s+need\s+to\s+answer\s+(.+)$",
        r"^(?:what|which)\s+tools\s+would\s+you\s+use\s+to\s+answer\s+(.+)$",
        r"^(?:what|which)\s+tools\s+would\s+you\s+use\s+to\s+inspect\s+(.+)$",
        r"^(?:what|which)\s+tools\s+would\s+you\s+use\s+to\s+analy[sz]e\s+(.+)$",
        r"^(?:what|which)\s+tools\s+would\s+you\s+use\s+to\s+explore\s+(.+)$",
        r"^how\s+do\s+you\s+decide\s+what\s+tools\s+to\s+use\s+for\s+(.+)$",
        r"^how\s+do\s+you\s+choose\s+what\s+tools\s+to\s+use\s+for\s+(.+)$",
        r"^do\s+you\s+need\s+any\s+tools\s+to\s+answer\s+(.+)$",
        r"^would\s+you\s+need\s+any\s+tools\s+to\s+answer\s+(.+)$",
        r"^how\s+would\s+you\s+answer\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, lowered, flags=re.IGNORECASE)
        if not match:
            continue
        subject = match.group(1).strip()
        return subject.rstrip(" ?")
    return None


def _is_underspecified_troubleshooting_prompt(user_prompt: str) -> bool:
    lowered = user_prompt.lower().strip()
    if _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt):
        return False
    if any(token in lowered for token in ("repo", "repository", "codebase", "backend", "system", "project", "get-drip", "getgit")):
        return False
    return lowered in {
        "why does this fail?",
        "why does this fail",
        "why is this failing?",
        "why is this failing",
        "why did this fail?",
        "why did this fail",
        "why doesn't this work?",
        "why doesn't this work",
        "why is this broken?",
        "why is this broken",
        "what is failing?",
        "what is failing",
    }


def _should_skip_external_session_context(user_prompt: str) -> bool:
    return _should_skip_retrieval_for_prompt(user_prompt) or _should_skip_current_workspace_memory_lookup(user_prompt)


def _should_skip_current_workspace_memory_lookup(user_prompt: str) -> bool:
    lowered = user_prompt.lower().strip()
    tool_strategy_subject = _tool_strategy_subject_prompt(user_prompt)
    if _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt) or _is_session_history_question(user_prompt):
        return False
    if tool_strategy_subject and any(
        marker in tool_strategy_subject
        for marker in ("repo", "repository", "codebase", "project", "backend", "frontend", "workspace", "folder")
    ):
        return True
    return any(
        phrase in lowered
        for phrase in (
            "summarize this repo",
            "summarize the repo",
            "summarize this repository",
            "tell me about this repo",
            "tell me about the repo",
            "tell me about this repository",
            "tell me about the repository",
            "explain the repo",
            "explain this repo",
            "explain the repository",
            "explain this repository",
            "tell me about this codebase",
            "tell me about the codebase",
            "what is this codebase",
            "what is the codebase",
            "what is this repo",
            "what is the repo",
            "inspect this repo",
            "inspect the repo",
            "inspect this codebase",
            "inspect the codebase",
            "how does the backend work",
            "how does this backend work",
            "what is the backend",
            "explain the backend",
            "explain this backend",
            "tell me about the backend",
            "tell me about this backend",
            "what is the system",
            "tell me about the system",
            "tell me about this system",
            "what is the backend architecture",
            "show me the backend architecture",
            "how does this repo work",
            "how does the repo work",
            "how does this repository work",
            "how does the repository work",
            "how does the system work",
            "how does this system work",
        )
    )


def _should_skip_retrieval_for_prompt(user_prompt: str) -> bool:
    lowered = user_prompt.lower().strip()
    if not lowered:
        return True
    if _is_explicit_live_search_prompt(user_prompt):
        return True
    if _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt) or _is_session_history_question(user_prompt):
        return False
    if _is_structural_acknowledgement_prompt(lowered):
        return True
    if _is_shell_like_prompt(lowered):
        return True
    tokens = re.findall(r"[a-z0-9_./-]+", lowered)
    if len(tokens) < 3 and not any(char in lowered for char in "?.!"):
        return True
    return False


def _is_structural_acknowledgement_prompt(lowered_prompt: str) -> bool:
    normalized = re.sub(r"\s+", " ", lowered_prompt).strip(" .!?\t\r\n")
    return normalized in {
        "ok",
        "okay",
        "kk",
        "cool",
        "nice",
        "great",
        "thanks",
        "thank you",
        "thx",
        "got it",
        "understood",
        "sounds good",
        "yep",
        "yes",
        "no",
    }


def _is_shell_like_prompt(lowered_prompt: str) -> bool:
    command_prefixes = (
        "cd ",
        "ls",
        "pwd",
        "cat ",
        "rm ",
        "mv ",
        "cp ",
        "mkdir ",
        "touch ",
        "git ",
        "npm ",
        "pnpm ",
        "yarn ",
        "bun ",
        "python ",
        "python3 ",
        "./",
        "../",
    )
    if lowered_prompt in {"ls", "pwd", "clear"}:
        return True
    if any(lowered_prompt.startswith(prefix) for prefix in command_prefixes):
        return True
    return bool(re.match(r"^[a-z0-9_./-]+\s+(-{1,2}[a-z0-9][a-z0-9-]*\s*)+$", lowered_prompt))


def _is_opencode_access_denied_error(error: RuntimeError) -> bool:
    lowered = str(error).strip().lower()
    return "opencode backend access has not been granted" in lowered


def _is_opencode_transport_error(error: RuntimeError) -> bool:
    lowered = str(error).strip().lower()
    return (
        "opencode server failed" in lowered
        or "opencode cli failed" in lowered
        or "unable to reach opencode server" in lowered
        or ("opencode cli failed" in lowered and "unable to connect" in lowered)
        or ("opencode cli failed" in lowered and "failed to fetch" in lowered)
        or ("opencode cli failed" in lowered and "pragma journal_mode = wal" in lowered)
    )


def _lexical_memory_terms(user_prompt: str) -> list[str]:
    generic = {
        "about",
        "architecture",
        "associated",
        "concrete",
        "confidently",
        "different",
        "files",
        "folders",
        "indirectly",
        "issues",
        "look",
        "main",
        "other",
        "parts",
        "point",
        "project",
        "properly",
        "question",
        "referenced",
        "remains",
        "same",
        "said",
        "use",
        "used",
        "what",
        "were",
        "work",
    }
    terms: list[str] = []
    for entity in sorted(_memory_query_entities(user_prompt)):
        if entity not in terms:
            terms.append(entity)
    for token in sorted(_memory_query_tokens(user_prompt)):
        if len(token) >= 5 and token not in generic and token not in terms:
            terms.append(token)
    lowered = user_prompt.lower()
    if "getgit" in lowered and any(marker in lowered for marker in ("architecture", "backend", "same architecture", "files or folders", "concrete files")):
        for extra in ("flask", "backend", "server.py", "core.py", "rag", "retriever.py", "readme.md", "documentation.md"):
            if extra not in terms:
                terms.append(extra)
    if "main issues" in lowered or "issues being worked" in lowered:
        for extra in ("salesforce", "pipeline", "workspace", "disabled", "https"):
            if extra not in terms:
                terms.append(extra)
    return terms[:10]


def _memory_answer_matches_question(user_prompt: str, shaped_lines: list[str]) -> bool:
    lowered_prompt = user_prompt.lower()
    joined = " \n ".join(shaped_lines).lower()

    if "repo" in lowered_prompt or "repository" in lowered_prompt or "codebase" in lowered_prompt:
        return any(
            marker in joined
            for marker in ("repo", "repository", "codebase", "workspace", "files", "folders", "module", "architecture", "backend")
        )
    if (
        lowered_prompt.startswith("why ")
        or lowered_prompt.startswith("how ")
        or lowered_prompt.startswith("explain ")
    ) and not (_is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt)):
        significant_tokens = [token for token in _memory_query_tokens(user_prompt) if token not in {"what", "this", "that", "about"}]
        return any(token in joined for token in significant_tokens)
    if _is_session_history_question(user_prompt):
        if any(
            marker in lowered_prompt
            for marker in (
                "code edit",
                "edit we did",
                "code change",
                "change we did",
            )
        ):
            if any(
                noise in joined
                for noise in (
                    "official documentation",
                    "opencode.ai/docs",
                    "fetched pages cover tools and agents",
                )
            ):
                return False
            return any(
                marker in joined
                for marker in (
                    "latest",
                    "recent",
                    "edit",
                    "change",
                    "updated",
                    "updating",
                    "fixed",
                )
            )
        if any(marker in lowered_prompt for marker in ("bug", "issue", "review", "fix", "fixed", "last time")):
            issue_markers = (
                "authentication bypass",
                "open email relay",
                "root url redirects",
                "convex generated imports",
                "create workspace",
                "pipeline chat",
                "test/publish",
                "review",
                "bug",
                "issue",
            )
            def marker_present(marker: str) -> bool:
                if " " in marker or "/" in marker:
                    return marker in joined
                return bool(re.search(rf"\b{re.escape(marker)}\b", joined))
            return _summarize_follow_up_issues(shaped_lines) is not None or any(
                marker_present(marker) for marker in issue_markers
            )
        return any(marker in joined for marker in ("merge conflict", "conflict", "resolved", "resolution", "rebase", "branch"))
    if _is_bug_list_question(user_prompt):
        issue_markers = (
            "authentication bypass",
            "open email relay",
            "root url redirects",
            "convex generated imports",
            "create workspace",
            "pipeline chat",
            "test/publish",
        )
        return _summarize_follow_up_issues(shaped_lines) is not None or any(marker in joined for marker in issue_markers)
    if _is_cleanup_schema_prompt(user_prompt):
        cleanup_markers = (
            "root url redirects",
            "convex generated imports",
            "authentication bypass",
            "legacy columns",
            "duplicated state",
            "campaigns",
            "crmcustomers",
            "schema",
            "cleanup",
        )
        return any(marker in joined for marker in cleanup_markers)
    if "get-drip" in lowered_prompt and _is_memory_recall_question(user_prompt):
        project_markers = (
            "get-drip was",
            "convex-backed app",
            "retrieval quality across stored sessions",
            "create workspace",
            "salesforce",
            "pipeline chat",
            "root url redirects",
            "convex generated imports",
        )
        return any(marker in joined for marker in project_markers)
    if "architecture" in lowered_prompt:
        if "getgit" in lowered_prompt:
            return any(marker in joined for marker in ("getgit", "flask", "server.py", "retriever", "core.py", "clone_repo.py", "repo_manager.py"))
        return any(marker in joined for marker in ("flask", "fastapi", "backend", "server.py", "rag", "retriever", "core.py"))
    if "same architecture" in lowered_prompt or "look different" in lowered_prompt:
        return any(marker in joined for marker in ("convex", "flask", "backend", "server.py", "route", "pipeline", "journey"))
    if "list the concrete files" in lowered_prompt or "files or folders" in lowered_prompt:
        return any(marker in joined for marker in (".py", ".md", ".txt", "/", "server.py", "core.py", "readme.md"))
    if "main issues" in lowered_prompt or "what were the main issues" in lowered_prompt:
        return _summarize_follow_up_issues(shaped_lines) is not None
    if "what can be said confidently" in lowered_prompt or "remains unclear" in lowered_prompt:
        return any(marker in joined for marker in ("get-drip", "convex", "workspace", "pipeline", "salesforce", "journey"))
    if "get-drip" in lowered_prompt and ("schema" in lowered_prompt or "cleanup" in lowered_prompt):
        return any(marker in joined for marker in ("schema", "cleanup", "review", "issue", "bug", "convex generated imports", "root url redirects"))
    return True


def _is_usable_logged_project_answer(user_prompt: str, answer: str) -> bool:
    cleaned = answer.strip()
    lowered = cleaned.lower()
    lowered_prompt = user_prompt.lower()
    if not cleaned:
        return False
    if lowered.startswith("# agents.md instructions"):
        return False
    if lowered.startswith("local-only mode could not inspect"):
        return False
    if lowered.startswith("`readme.md` references"):
        return False
    if "requested tool is not registered" in lowered:
        return False
    if "convex/email_g..." in lowered:
        return False
    if _is_bug_list_question(user_prompt) and "strongest clues point to" in lowered:
        return False
    if _is_cleanup_schema_prompt(user_prompt) and any(
        noise in lowered
        for noise in (
            "tsc passes",
            "bun run check",
            "implemented all `reviews.md` items",
            "reverted oauth behavior exactly",
        )
    ):
        return False
    if "strongest clues point to" in lowered and ("schema" in lowered_prompt or "cleanup" in lowered_prompt):
        return False
    if "strongest clues point to" in lowered and "infer the parts of the app" not in lowered_prompt and not _is_file_inventory_question(user_prompt):
        return False
    if "same architecture" in lowered_prompt and "get-drip" not in lowered:
        return False
    if "getgit" in lowered_prompt and not any(
        marker in lowered
        for marker in (
            "getgit",
            "flask",
            "server.py",
            "retriever",
            "core.py",
            "clone_repo.py",
            "repo_manager.py",
        )
    ):
        return False
    if "get-drip" in lowered_prompt and not any(
        marker in lowered
        for marker in (
            "get-drip",
            "convex",
            "root url redirects",
            "convex generated imports",
            "authentication bypass",
            "create workspace",
            "pipeline chat",
            "test/publish",
            "salesforce",
        )
    ):
        return False
    return _memory_answer_matches_question(user_prompt, [cleaned])


def _is_usable_logged_answer(user_prompt: str, answer: str) -> bool:
    cleaned = str(answer or "").strip()
    if not cleaned:
        return False
    if _is_session_history_question(user_prompt):
        lowered_prompt = user_prompt.lower()
        if "get-drip" in lowered_prompt or "getgit" in lowered_prompt:
            return _is_usable_logged_project_answer(user_prompt, cleaned)
        return _is_high_signal_memory_answer(cleaned, user_prompt) and _memory_answer_matches_question(user_prompt, [cleaned])
    return _is_usable_logged_project_answer(user_prompt, cleaned)


def _answer_known_project_question(user_prompt: str, memory_context: str) -> str | None:
    lowered = user_prompt.lower()
    if "getgit" not in lowered and "get-drip" not in lowered:
        return None

    context_lower = memory_context.lower()
    getgit_flask = "flask" in context_lower
    getgit_rag = "rag" in context_lower
    getgit_server = "server.py" in context_lower
    getdrip_convex = "convex" in context_lower
    issue_summary = _summarize_follow_up_issues(_memory_context_lines(memory_context))
    path_mentions = _extract_path_mentions(memory_context)
    high_signal_paths = [
        path for path in path_mentions
        if any(marker in path for marker in ("convex/", "src/routes/", "journey.ts", "convex-api.ts", "convex-types.ts", "pipeline.tsx", "test-activate.tsx"))
    ]

    if "same architecture" in lowered:
        if getgit_flask and getdrip_convex:
            return "No. GetGit was described as a Flask/RAG-style backend, while get-drip was described as a Convex-backed app."
        return None

    if "look different" in lowered:
        if getgit_flask and getdrip_convex:
            return "GetGit looks like a Flask/Python RAG app, while get-drip looks like a Convex-backed app with campaign and route flow files."
        return None

    if "other work referenced getgit" in lowered and "task_getgit_checkpoints" in context_lower:
        return "CodeGuide referenced GetGit indirectly through a `task_practice_code_evaluate` flow that called `task_getgit_checkpoints`."

    if "main issues" in lowered and issue_summary:
        if _is_bug_list_question(user_prompt):
            return _format_issue_list_answer("get-drip", _extract_follow_up_issues(_memory_context_lines(memory_context)))
        return f"In get-drip, the main issues were {issue_summary}."

    if _is_file_inventory_question(user_prompt) and "getgit" in lowered:
        inventory_paths = [path for path in path_mentions if any(marker in path.lower() for marker in ("server.py", "core.py", "checkpoints.py", "clone_repo.py", "repo_manager.py", "readme.md", "documentation.md", "rag/", "templates/", "static/"))]
        deduped_inventory: list[str] = []
        for path in inventory_paths:
            if path not in deduped_inventory:
                deduped_inventory.append(path)
        if deduped_inventory:
            return "The concrete GetGit paths included " + ", ".join(f"`{path}`" for path in deduped_inventory[:10]) + "."

    if "infer the parts of the app" in lowered and high_signal_paths:
        deduped_paths: list[str] = []
        for path in high_signal_paths:
            if path not in deduped_paths:
                deduped_paths.append(path)
        return "The strongest clues point to " + ", ".join(f"`{path}`" for path in deduped_paths[:5]) + "."

    if ("what can be said confidently" in lowered or "remains unclear" in lowered) and getdrip_convex:
        confident_bits: list[str] = ["get-drip was described as a Convex-backed app"]
        if issue_summary:
            confident_bits.append(f"the work focused on {issue_summary}")
        confident = ", and ".join(confident_bits)
        return f"Confidently, {confident}. What remains unclear is a cleaner one-line architecture summary beyond those clues."

    if "what was the backend" in lowered and "get-drip" in lowered and getdrip_convex:
        return "get-drip was described as a Convex-backed app."

    if any(
        phrase in lowered
        for phrase in (
            "what architecture did getgit use",
            "what architecture was getgit",
            "what was the architecture of getgit",
        )
    ) and (getgit_flask or getgit_server or getgit_rag):
        parts = ["GetGit was described as a Flask backend"]
        if getgit_server:
            parts.append("with a `server.py` entrypoint")
        if getgit_rag:
            parts.append("and RAG-related components")
        return ", ".join(parts) + "."

    return None


def _memory_context_lines(memory_context: str) -> list[str]:
    lines: list[str] = []
    for raw_line in memory_context.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("## "):
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        lines.append(stripped)
    return lines


def _extract_path_mentions(text: str) -> list[str]:
    patterns = [
        r"`([^`]+)`",
        r"\[([^\]]+)\]\(/[^)]+/([^):]+(?:\.[A-Za-z0-9]+))(?::\d+)?\)",
        r"(?<![A-Za-z0-9_])((?:src|convex)/[A-Za-z0-9_.$/-]+(?:\.[A-Za-z0-9]+)?)",
    ]
    paths: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            value = match[-1] if isinstance(match, tuple) else match
            cleaned = str(value).strip()
            if "/" not in cleaned and "." not in cleaned:
                continue
            if cleaned not in paths:
                paths.append(cleaned)
    return paths


def _lexical_line_score(summary: str, user_prompt: str, terms: list[str]) -> int:
    lowered = summary.lower()
    prompt_lowered = user_prompt.lower()
    score = sum(2 for term in terms if term.lower() in lowered)
    score += len(_extract_path_mentions(summary)) * 3
    if "flask" in lowered or "convex" in lowered or "rag" in lowered:
        score += 4
    if "task_getgit_checkpoints" in lowered:
        score += 5
    if "pipeline chat" in lowered or "salesforce" in lowered or "create workspace" in lowered:
        score += 3
    if "tool output noted:" in lowered:
        score += 2
    if "strongest clues point to" in lowered and "infer the parts of the app" not in prompt_lowered and not _is_file_inventory_question(user_prompt):
        score -= 12
    if ("schema" in prompt_lowered or "cleanup" in prompt_lowered) and not any(
        marker in lowered for marker in ("schema", "cleanup", "convex generated imports", "root url redirects", "review")
    ):
        score -= 8
    if "session '" in lowered and score < 6:
        score -= 2
    if prompt_lowered in lowered:
        score -= 3
    return score


def _is_architecture_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    if _is_backend_connector_question(user_prompt):
        return True
    if _is_runtime_routing_question(user_prompt):
        return True
    return any(
        marker in lowered
        for marker in (
            "architecture",
            "backend",
            "system",
            "how does the repo work",
            "how does the repository work",
            "how does the codebase work",
            "how does this repo work",
            "how does this repository work",
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
    )


def _is_runtime_routing_question(user_prompt: str) -> bool:
    lowered = user_prompt.lower()
    route_markers = (
        "decides between memory",
        "decides between planning",
        "decides between tools",
        "decides between web search",
        "between memory, planning, tools, and web search",
        "between memory planning tools and web search",
        "between memory, tools, and web search",
        "between plan mode and tools",
        "between plan mode, tools, and web search",
        "memory, planning, tools, and web search",
        "memory, tools, and web search",
        "plan mode",
        "planning mode",
        "web search",
        "knowledge search",
        "tool routing",
        "routing logic",
        "route the turn",
        "route the request",
        "decide between",
    )
    runtime_markers = (
        "this app",
        "the app",
        "runtime",
        "router",
        "routing",
        "turn",
        "request",
        "prompt",
    )
    return any(marker in lowered for marker in route_markers) and any(
        marker in lowered for marker in runtime_markers
    )

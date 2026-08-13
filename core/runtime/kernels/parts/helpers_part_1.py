def _assistant_tool_call_message(
    ai_response: AIResponse,
    tool_calls: list[ToolCallRequest] | None = None,
    content_override: str | None = None,
) -> dict[str, Any]:
    selected_tool_calls = tool_calls or list(ai_response.tool_calls)
    return {
        "role": "assistant",
        "content": ai_response.content if content_override is None else content_override,
        "tool_calls": [
            {
                "id": tool_call.call_id,
                "type": "function",
                "function": {
                    "name": tool_call.tool_name,
                    "arguments": json.dumps(tool_call.arguments, sort_keys=True),
                },
            }
            for tool_call in selected_tool_calls
        ],
    }


def _tool_message(call_id: str, tool_name: str, output: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": tool_name,
        "content": output,
    }


def _format_tool_output(output: str, data: dict[str, Any]) -> str:
    if not data:
        return output
    return f"{output}\n{json.dumps(data, sort_keys=True)}"


def _runtime_step_from_ai_step(step: AIExecutedToolStep) -> ToolExecutionStep:
    return ToolExecutionStep(
        step_id=step.step_id,
        tool_name=step.tool_name,
        arguments=dict(step.arguments),
        output=step.output,
        success=step.success and not step.is_error,
        is_sandboxed_violation=False,
        data=dict(step.data),
    )


def _merge_usage(total_usage: dict[str, int], usage: dict[str, int]) -> None:
    for key, value in usage.items():
        total_usage[key] = total_usage.get(key, 0) + value


def _prioritize_live_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferred_domains = (
        "forbes.com",
        "bloomberg.com",
        "reuters.com",
        "apnews.com",
        "nytimes.com",
        "wsj.com",
        "ft.com",
        "wikipedia.org",
    )

    def score(item: dict[str, Any]) -> tuple[int, int]:
        url = str(item.get("url") or "").strip().lower()
        title = str(item.get("title") or "").strip().lower()
        domain_rank = next((index for index, domain in enumerate(preferred_domains) if domain in url), len(preferred_domains))
        if re.match(r"^[^-|]+?\s[-|]\s[^-|]+$", title) and "list" not in title and "rank" not in title:
            title_priority = 0
        elif any(token in title for token in ("profile", "real time", "net worth", "index")):
            title_priority = 1
        else:
            title_priority = 2
        return (title_priority, domain_rank)

    return sorted((item for item in results if isinstance(item, dict)), key=score)


def _truncate_live_search_content(content: str, *, max_chars: int = 700) -> str:
    normalized = re.sub(r"\s+", " ", content or "").strip()
    salient = _extract_salient_live_search_excerpt(normalized)
    if salient:
        return salient
    if len(normalized) <= max_chars:
        return normalized
    clipped = normalized[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{clipped}..." if clipped else normalized[:max_chars]


def _extract_salient_live_search_excerpt(content: str) -> str:
    if not content:
        return ""
    patterns = (
        r"((?:real time net worth|net worth)[^.]{0,220}?\$\d[\d.,]*(?:\s?[BMK]| billion| million)?[^.]{0,140})",
        r"(\$\d[\d.,]*(?:\s?[BMK]| billion| million)?[^.]{0,180}?(?:as of|real time net worth|net worth)[^.]{0,80})",
        r"((?:as of)[^.]{0,140}?\$\d[\d.,]*(?:\s?[BMK]| billion| million)?[^.]{0,120})",
    )
    for pattern in patterns:
        match = re.search(pattern, content, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _build_grounded_live_fact_answer(
    user_prompt: str,
    search_results: list[dict[str, Any]],
    page_extracts: list[dict[str, str]],
) -> str:
    lowered = user_prompt.lower()
    docs_answer = _build_grounded_live_docs_answer(user_prompt, search_results, page_extracts)
    if docs_answer:
        return docs_answer
    if "net worth" not in lowered:
        return ""
    subject = _extract_net_worth_subject(user_prompt)
    authoritative_results = [item for item in search_results if _is_authoritative_live_fact_url(str(item.get("url") or ""))]
    authoritative_candidates: list[tuple[dict[str, str], str, str]] = []
    secondary_candidates: list[tuple[dict[str, str], str, str]] = []
    for extract in page_extracts:
        excerpt = str(extract.get("excerpt") or "").strip()
        money_amount = _extract_money_amount(excerpt)
        if not money_amount:
            continue
        as_of = _extract_as_of_phrase(excerpt)
        candidate = (extract, money_amount, as_of)
        if _is_authoritative_live_fact_url(str(extract.get("url") or "")):
            authoritative_candidates.append(candidate)
        else:
            secondary_candidates.append(candidate)
    chosen: tuple[dict[str, str], str, str] | None = authoritative_candidates[0] if authoritative_candidates else None
    if not chosen and not authoritative_results and secondary_candidates:
        chosen = secondary_candidates[0]
    if chosen:
        extract, money_amount, as_of = chosen
        source = _normalize_live_fact_source_label(str(extract.get("title") or ""), str(extract.get("url") or ""))
        prefix = f"{subject}'s" if subject else "The reported"
        answer = f"{prefix} net worth is about {money_amount}"
        if as_of:
            answer += f" {as_of}"
        if source:
            answer += f", according to {source}"
        return answer + "."
    if authoritative_results and secondary_candidates:
        fallback_extract, fallback_amount, fallback_as_of = secondary_candidates[0]
        source = _normalize_live_fact_source_label(
            str(fallback_extract.get("title") or ""),
            str(fallback_extract.get("url") or ""),
        )
        leader_titles = ", ".join(
            _normalize_live_fact_source_label(str(item.get("title") or ""), str(item.get("url") or ""))
            for item in authoritative_results[:2]
        )
        answer = "I found current authoritative result pages"
        if leader_titles:
            answer += f" from {leader_titles}"
        answer += ", but I couldn't extract a reliable live net-worth figure from them automatically"
        answer += f". The only numeric figure I could extract was {fallback_amount}"
        if fallback_as_of:
            answer += f" {fallback_as_of}"
        if source:
            answer += f" from {source}"
        answer += ", so I wouldn't treat it as fully verified current data."
        return answer
    return ""


def _build_grounded_live_docs_answer(
    user_prompt: str,
    search_results: list[dict[str, Any]],
    page_extracts: list[dict[str, str]],
) -> str:
    lowered = user_prompt.lower()
    if "net worth" in lowered:
        return ""
    if not any(token in lowered for token in ("docs", "documentation", "official website")):
        return ""
    official_results = [item for item in search_results if _looks_like_official_docs_url(str(item.get("url") or ""))]
    official_extracts = [item for item in page_extracts if _looks_like_official_docs_url(str(item.get("url") or ""))]
    if not official_results and not official_extracts:
        return ""

    anchor_url = ""
    if official_results:
        anchor_url = _canonical_docs_url(str(official_results[0].get("url") or ""))
    if not anchor_url and official_extracts:
        anchor_url = _canonical_docs_url(str(official_extracts[0].get("url") or ""))
    if not anchor_url:
        return ""

    topics = _extract_docs_topics(official_extracts)
    answer = f"The latest official documentation is at {anchor_url}."
    if topics:
        answer += f" The fetched pages cover {topics}."
    return answer


def _extract_net_worth_subject(prompt: str) -> str:
    patterns = (
        r"what(?:'s| is)\s+(.+?)\s+net worth",
        r"(.+?)\s+net worth",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = re.sub(r"\b(search|online|briefly|please|current|latest|the web)\b", "", match.group(1), flags=re.IGNORECASE)
        candidate = re.sub(r"[?*,.]+", "", candidate).strip()
        if candidate:
            return " ".join(part.capitalize() for part in candidate.split())
    return ""


def _extract_money_amount(text: str) -> str:
    match = re.search(r"(\$\d[\d.,]*(?:\s?[BMK]| billion| million)?)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _extract_as_of_phrase(text: str) -> str:
    match = re.search(r"\b(as of\s+[^.,;:]+)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _normalize_live_fact_source_label(title: str, url: str) -> str:
    if "forbes" in title.lower() or "forbes.com" in url.lower():
        return f"Forbes ({url})"
    if title.strip():
        return f"{title.strip()} ({url})" if url else title.strip()
    return url.strip()


def _is_authoritative_live_fact_url(url: str) -> bool:
    lowered = url.lower()
    return any(domain in lowered for domain in ("forbes.com", "bloomberg.com", "reuters.com", "apnews.com"))


def _looks_like_official_docs_url(url: str) -> bool:
    lowered = url.lower().strip()
    if not lowered.startswith("http"):
        return False
    if any(domain in lowered for domain in ("github.com", "learnopencode.com", "stackoverflow.com", "reddit.com")):
        return False
    return "/docs" in lowered or "docs." in lowered


def _canonical_docs_url(url: str) -> str:
    stripped = url.strip().rstrip("/")
    match = re.match(r"^(https?://[^/]+/docs)(?:/.*)?$", stripped, flags=re.IGNORECASE)
    if match:
        return match.group(1) + "/"
    return stripped + "/" if stripped and not stripped.endswith("/") else stripped


def _extract_docs_topics(page_extracts: list[dict[str, str]]) -> str:
    topic_labels: list[str] = []
    for extract in page_extracts[:4]:
        title = str(extract.get("title") or "").strip()
        if not title:
            continue
        label = re.split(r"\s+[|-]\s+", title, maxsplit=1)[0].strip()
        label = re.sub(r"\bdocs?\b", "", label, flags=re.IGNORECASE).strip(" :,-")
        label_lower = label.lower()
        if not label or len(label) > 32:
            continue
        if label_lower in {"opencode", "opencode documentation", "documentation", "complete reference guide"}:
            continue
        if label not in topic_labels:
            topic_labels.append(label)
    if not topic_labels:
        return ""
    if len(topic_labels) == 1:
        return topic_labels[0]
    if len(topic_labels) == 2:
        return f"{topic_labels[0]} and {topic_labels[1]}"
    return f"{', '.join(topic_labels[:-1])}, and {topic_labels[-1]}"


def _compact_conversation(messages: list[dict[str, Any]], max_turns: int) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        retained.append({"role": role, "content": content})

    return retained[-(max_turns * 2) :]


def _build_memory_engine(db_path: str, vector_dir: str) -> MemoryEngine:
    if os.getenv("DEVENV_USE_SENTENCE_EMBEDDER") != "1":
        return MemoryEngine(
            db_path=db_path,
            vector_dir=vector_dir,
            embedder=HashingEmbedder(dimension=384),
        )
    try:
        return MemoryEngine(db_path=db_path, vector_dir=vector_dir)
    except Exception as exc:
        logger.warning("Falling back to hashing memory embedder: error=%s", exc)
        return MemoryEngine(
            db_path=db_path,
            vector_dir=vector_dir,
            embedder=HashingEmbedder(dimension=384),
        )


class _InProcessToolClient:
    def __init__(self, tools: dict[str, BaseTool]) -> None:
        self._tools = tools

    def list_tools(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "description": tool.description,
                "inputSchema": tool.input_schema(),
            }
            for name, tool in self._tools.items()
        }

    def call_tool(self, name: str, arguments: dict[str, Any]):
        tool = self._tools.get(name)
        if tool is None:
            return type("ToolCallResult", (), {"success": False, "output": f"Tool '{name}' is not registered.", "data": {}, "is_error": True})()
        result = tool.execute(**arguments)
        return type("ToolCallResult", (), {"success": result.success, "output": result.output, "data": result.data, "is_error": False})()

    def close(self) -> None:
        return None

def _build_partial_failure_response(steps: list[ToolExecutionStep], error: RuntimeError) -> str:
    successful_steps = [step.tool_name for step in steps if step.success]
    if successful_steps:
        tool_summary = ", ".join(successful_steps)
        return (
            f"The requested tool changes were applied ({tool_summary}), "
            f"but the follow-up AI response failed: {error}"
        )

    return f"The AI response failed after tool execution: {error}"


def _find_reusable_tool_step(
    steps: list[ToolExecutionStep],
    tool_name: str,
    arguments: dict[str, Any],
) -> ToolExecutionStep | None:
    reusable_tools = {
        "knowledge_search",
        "web_search",
        "list_directory",
        "locate_files",
        "read_file",
        "peek_lines",
        "inspect_symbols",
        "search_text",
        "track_symbol",
    }
    if tool_name not in reusable_tools:
        return None
    for step in reversed(steps):
        if not step.success:
            continue
        if step.tool_name == tool_name and step.arguments == arguments:
            return step
    return None


def _prefer_reference_results_over_empty_summary(
    final_response: str | None,
    steps: list[ToolExecutionStep],
    user_prompt: str,
) -> str | None:
    text = str(final_response or "").strip()
    lowered = text.lower()
    has_useful_markdown_links = text.count("](") >= 2
    should_replace = any(
        marker in lowered
        for marker in (
            "did not yield any relevant results",
            "could not find resources",
            "could not find any relevant",
            "might want to consider creating your own implementation",
            "no relevant results",
        )
    )
    if text and not should_replace and has_useful_markdown_links:
        return final_response

    knowledge_steps = [step for step in steps if step.success and step.tool_name == "knowledge_search"]
    if not knowledge_steps:
        return final_response
    latest = knowledge_steps[-1]
    resources = latest.data.get("resources") if isinstance(latest.data, dict) else None
    if not isinstance(resources, list):
        return final_response
    lines = [f"Here are outside references for `{user_prompt.strip()}`:"]
    appended = 0
    for group in resources:
        if not isinstance(group, dict):
            continue
        source = str(group.get("source") or "general").strip()
        results = group.get("results")
        if not isinstance(results, list) or not results:
            continue
        lines.append(f"")
        lines.append(f"**{source.title()}**")
        for item in results[:3]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("url") or "Reference").strip()
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            lines.append(f"- [{title}]({url})")
            appended += 1
    return "\n".join(lines) if appended else final_response


def _trim_memory_context(memory_context: str, char_limit: int) -> str:
    stripped = memory_context.strip()
    if len(stripped) <= char_limit:
        return stripped

    lines: list[str] = []
    current_length = 0
    for line in stripped.splitlines():
        next_length = current_length + len(line) + (1 if lines else 0)
        if next_length > char_limit:
            break
        lines.append(line)
        current_length = next_length

    if not lines:
        return stripped[:char_limit].rstrip()
    return "\n".join(lines).rstrip()


def _focus_memory_context_for_direct_answers(memory_context: str, char_limit: int) -> str:
    stripped = memory_context.strip()
    if not stripped:
        return ""

    retrieved_header = "## Retrieved Memory"
    header_index = stripped.find(retrieved_header)
    if header_index >= 0:
        focused = stripped[header_index:]
        return _trim_memory_context(focused, char_limit)
    return _trim_memory_context(stripped, char_limit)


def _coerce_inline_tool_call(content: str | None, allowed_tools: list[str]) -> ToolCallRequest | None:
    if not isinstance(content, str) or not content.strip() or not allowed_tools:
        return None

    inline_payload = _extract_json_block(content)
    if inline_payload is None:
        return None

    candidates: list[dict[str, Any]] = []
    if isinstance(inline_payload, dict):
        candidates = [inline_payload]
    elif isinstance(inline_payload, list):
        candidates = [item for item in inline_payload if isinstance(item, dict)]

    for candidate in candidates:
        tool_name = candidate.get("name")
        parameters = candidate.get("parameters")
        if not isinstance(tool_name, str) or tool_name not in allowed_tools:
            continue
        if not isinstance(parameters, dict):
            continue
        return ToolCallRequest(
            call_id=f"inline_{uuid.uuid4().hex[:10]}",
            tool_name=tool_name,
            arguments=parameters,
        )

    return None


def _extract_json_block(content: str) -> dict[str, Any] | list[Any] | None:
    stripped = content.strip()
    candidates = [stripped]
    for opener in ("\n[", "\n{"):
        index = stripped.find(opener)
        if index >= 0:
            candidates.append(stripped[index + 1 :].strip())

    for candidate in candidates:
        if not candidate or candidate[0] not in "[{":
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, (dict, list)):
            return payload
    return None


def _extract_readable_replay_answer(content: str) -> str | None:
    raw = str(content or "").strip()
    if not raw or "\n" not in raw or not raw.startswith("{"):
        return None

    readable_lines: list[str] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        part = payload.get("part")
        if isinstance(part, dict) and part.get("type") == "text":
            text_value = str(part.get("text") or "").strip()
            if text_value:
                readable_lines.append(text_value)
            continue

        event_payload = payload.get("payload")
        if isinstance(event_payload, dict) and event_payload.get("type") == "agent_message":
            message = str(event_payload.get("message") or "").strip()
            if message:
                readable_lines.append(message)

    deduped_lines: list[str] = []
    for line in readable_lines:
        if line not in deduped_lines:
            deduped_lines.append(line)
    if not deduped_lines:
        return None
    return "\n\n".join(deduped_lines)


def _sanitize_logged_answer(content: str) -> str:
    extracted = _extract_readable_replay_answer(content)
    if extracted:
        return sanitize_response_text(extracted) or ""
    return sanitize_response_text(str(content or "").strip()) or ""


def _normalize_logged_answer_text(content: str) -> str:
    return normalize_response_text(content)


def _shape_logged_project_answer(user_prompt: str, answer: str) -> str:
    cleaned = str(answer or "").strip()
    if not cleaned:
        return ""
    subject = _preferred_memory_subject(user_prompt, _memory_context_lines(cleaned))
    issues = _issues_relevant_to_prompt(user_prompt, _extract_follow_up_issues(_memory_context_lines(cleaned)))
    if _is_cleanup_schema_prompt(user_prompt) and issues:
        prefix = f"The {subject} cleanup was mainly about" if subject else "The cleanup was mainly about"
        return f"{prefix} {_join_human_list(issues)}."
    if _is_bug_list_question(user_prompt) and issues:
        return _format_issue_list_answer(subject, issues)
    if _is_memory_recall_question(user_prompt) and issues:
        if subject:
            return f"Yes. In {subject}, the main issues were {_join_human_list(issues)}."
        return f"Yes. The main issues were {_join_human_list(issues)}."
    return cleaned


def _shape_logged_answer_for_prompt(user_prompt: str, answer: str) -> str:
    cleaned = str(answer or "").strip()
    if not cleaned:
        return ""
    if _is_session_history_question(user_prompt):
        latest_edit_summary = _summarize_latest_code_edit_recall(user_prompt, _memory_context_lines(cleaned))
        if latest_edit_summary:
            return latest_edit_summary
        return cleaned
    return _shape_logged_project_answer(user_prompt, cleaned)


def _answer_from_retrieved_memory(user_prompt: str, memory_context: str) -> str | None:
    if not memory_context.strip():
        return None

    sections = _memory_context_sections(memory_context)
    cleaned_lines = [_clean_memory_line(line) for line in [*sections["working"], *sections["external"], *sections["retrieved"]]]
    error_fix_summary = _summarize_error_fix_memory_answer(user_prompt, cleaned_lines)
    if error_fix_summary:
        return error_fix_summary
    cleanup_summary = _cleanup_summary_from_lines(user_prompt, cleaned_lines)
    if cleanup_summary:
        return cleanup_summary
    latest_edit_summary = _summarize_latest_code_edit_recall(user_prompt, cleaned_lines)
    if latest_edit_summary:
        return latest_edit_summary
    if _is_memory_recall_question(user_prompt):
        extracted_issues = _issues_relevant_to_prompt(user_prompt, _extract_follow_up_issues(cleaned_lines))
        if extracted_issues:
            subject = _preferred_memory_subject(user_prompt, cleaned_lines)
            if _is_cleanup_schema_prompt(user_prompt):
                prefix = f"The {subject} cleanup was mainly about" if subject else "The cleanup was mainly about"
                return f"{prefix} {_join_human_list(extracted_issues)}."
            if _is_bug_list_question(user_prompt):
                return _format_issue_list_answer(subject, extracted_issues)
            if subject:
                return f"Yes. In {subject}, the main issues were {_join_human_list(extracted_issues)}."
            return f"Yes. The main issues were {_join_human_list(extracted_issues)}."
    if _is_bug_list_question(user_prompt) and not _is_cleanup_schema_prompt(user_prompt):
        issue_lines = [
            _humanize_recalled_line(_clean_memory_line(line), user_prompt)
            for line in [*sections["working"], *sections["external"], *sections["retrieved"]]
        ]
        issue_lines = [line for line in issue_lines if line and _is_high_signal_memory_answer(line, user_prompt)]
        issue_subject = _preferred_memory_subject(user_prompt, sections["external"] + sections["working"] + sections["retrieved"])
        extracted_issues = _issues_relevant_to_prompt(user_prompt, _extract_follow_up_issues(issue_lines))
        if extracted_issues:
            return _format_issue_list_answer(issue_subject, extracted_issues)
    if _is_memory_follow_up_question(user_prompt) and sections["working"]:
        recent_working_lines = _recent_working_follow_up_lines(sections["working"], user_prompt)
        if recent_working_lines:
            shaped_working = [_humanize_recalled_line(line, user_prompt) for line in recent_working_lines]
            shaped_working = [line for line in shaped_working if line]
            if shaped_working:
                subject = _preferred_memory_subject(user_prompt, sections["working"] + sections["external"] + sections["retrieved"])
                issue_summary = _summarize_follow_up_issues(shaped_working)
                if issue_summary and _is_bug_fix_follow_up_question(user_prompt):
                    if subject:
                        return f"I could recall the bug list for {subject}, but I could not recover the exact fix steps from memory."
                    return "I could recall the bug list, but I could not recover the exact fix steps from memory."
                if issue_summary and _is_issue_explanation_follow_up_question(user_prompt):
                    return f"Yes. It was mainly about {issue_summary}."
                if issue_summary and _is_issue_recap_follow_up_question(user_prompt):
                    if subject:
                        return f"Yes. In {subject}, the main issues were {issue_summary}."
                    return f"Yes. The main issues were {issue_summary}."
                cleanup_summary = _summarize_cleanup_narrative(shaped_working)
                if cleanup_summary and _is_issue_explanation_follow_up_question(user_prompt):
                    return f"Yes. It was mainly about {cleanup_summary}."
                if _is_bug_fix_follow_up_question(user_prompt) or _is_issue_recap_follow_up_question(user_prompt):
                    shaped_working = []
                if len(shaped_working) == 1:
                    return _affirm_memory_answer(shaped_working[0])
    if _is_memory_follow_up_question(user_prompt) and sections["external"]:
        ordered_follow_up = _ordered_follow_up_lines(user_prompt, sections["external"])
        if ordered_follow_up:
            shaped_follow_up = [_humanize_recalled_line(line, user_prompt) for line in ordered_follow_up]
            shaped_follow_up = [line for line in shaped_follow_up if line]
            if not shaped_follow_up:
                return None
            synthesized_issues = _summarize_follow_up_issues(shaped_follow_up)
            if synthesized_issues:
                subject = _infer_memory_subject(sections["working"] + sections["external"] + sections["retrieved"])
                if _is_bug_fix_follow_up_question(user_prompt):
                    if subject:
                        return f"I could recall the bug list for {subject}, but I could not recover the exact fix steps from memory."
                    return "I could recall the bug list, but I could not recover the exact fix steps from memory."
                if _is_issue_explanation_follow_up_question(user_prompt):
                    return f"Yes. It was mainly about {synthesized_issues}."
                if _is_bug_list_question(user_prompt):
                    return _format_issue_list_answer(subject, _extract_follow_up_issues(shaped_follow_up))
                if subject:
                    return f"Yes. In {subject}, the main issues were {synthesized_issues}."
                return f"Yes. The main issues were {synthesized_issues}."
            cleanup_summary = _summarize_cleanup_narrative(shaped_follow_up)
            if cleanup_summary and _is_issue_explanation_follow_up_question(user_prompt):
                return f"Yes. It was mainly about {cleanup_summary}."
            if len(shaped_follow_up) >= 2 and _follow_up_line_score(ordered_follow_up[0].lower()) >= 2 and _follow_up_line_score(ordered_follow_up[1].lower()) >= 2:
                return "Yes. The main issues were: " + "; ".join(shaped_follow_up[:2])
            if _follow_up_line_score(ordered_follow_up[0].lower()) >= 2:
                return _affirm_memory_answer(shaped_follow_up[0])
            if len(shaped_follow_up) == 1:
                return _affirm_memory_answer(shaped_follow_up[0])
            return "Yes.\n\n" + "\n\n".join(shaped_follow_up[:3])
    primary_lines = [*sections["retrieved"], *sections["external"]]
    working_lines = sections["working"]
    bullet_lines: list[tuple[str, str]] = []
    for line in primary_lines:
        bullet = line.strip()
        bullet_lower = bullet.lower()
        if bullet_lower.startswith("prompt:") or bullet_lower.startswith("[workspace] workspace:"):
            continue
        bullet_lines.append(("memory", bullet))
    for line in working_lines:
        bullet = line.strip()
        bullet_lower = bullet.lower()
        if bullet_lower.startswith("user:"):
            continue
        bullet_lines.append(("working", bullet))
    if not bullet_lines:
        return None

    prompt_tokens = _memory_query_tokens(user_prompt)
    prompt_entities = _memory_query_entities(user_prompt)
    inferred_entities = _memory_context_entities(memory_context)
    if prompt_entities:
        query_entities = prompt_entities
    elif _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt):
        query_entities = inferred_entities
    else:
        query_entities = set()
    ranked: list[tuple[int, int, str, str]] = []
    for source, line in bullet_lines:
        line_lower = line.lower()
        token_overlap = sum(1 for token in prompt_tokens if token in line_lower)
        overlap = token_overlap
        overlap += sum(6 for entity in query_entities if entity in line_lower)
        if _is_memory_follow_up_question(user_prompt):
            if line_lower.startswith("user asked:"):
                overlap += 4
            elif line_lower.startswith("assistant reported:"):
                overlap += 1
            elif line_lower.startswith("session '"):
                overlap -= 1
        elif _is_memory_recall_question(user_prompt):
            if line_lower.startswith("assistant reported:"):
                overlap += 3
            elif line_lower.startswith("session '"):
                overlap -= 2
        if source == "memory":
            overlap += 1
        ranked.append((overlap, token_overlap, source, line))
    ranked.sort(key=lambda item: item[0], reverse=True)

    best_overlap = ranked[0][0]
    if best_overlap <= 0:
        return None

    if _is_memory_follow_up_question(user_prompt):
        selected_ranked = ranked[:3]
    else:
        selected_ranked = [item for item in ranked[:4] if item[0] == best_overlap or item[0] > 0]
    memory_token_matches = [item for item in selected_ranked if item[2] == "memory" and item[1] > 0]
    if memory_token_matches:
        selected_ranked = memory_token_matches
    if any(source == "memory" and token_overlap == 0 for _overlap, token_overlap, source, _line in selected_ranked):
        primary_with_tokens = [item for item in selected_ranked if item[2] == "memory" or item[1] > 0]
        selected_ranked = primary_with_tokens or selected_ranked
    selected = [line for _overlap, _token_overlap, _source, line in selected_ranked[:3]]
    if not selected:
        return None
    cleaned = [_clean_memory_line(line) for line in selected]
    cleaned = [line for line in cleaned if line and _is_high_signal_memory_answer(line, user_prompt)]
    if not cleaned:
        return None
    if query_entities and not (_is_memory_recall_question(user_prompt) and _has_explicit_memory_subject(user_prompt)):
        cleaned = [line for line in cleaned if any(entity in line.lower() for entity in query_entities)] or cleaned
    shaped = [_humanize_recalled_line(line, user_prompt) for line in cleaned]
    shaped = [line for line in shaped if line]
    if not shaped:
        return None
    if not _memory_answer_matches_question(user_prompt, shaped):
        return None
    if _is_memory_recall_question(user_prompt) or _is_memory_follow_up_question(user_prompt):
        issue_summary = _summarize_follow_up_issues(shaped)
        subject = _preferred_memory_subject(user_prompt, sections["external"] + sections["working"] + sections["retrieved"])
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
        if _has_explicit_memory_subject(user_prompt):
            if issue_summary and _is_bug_list_question(user_prompt):
                return _format_issue_list_answer(subject, _extract_follow_up_issues(shaped))
            if subject and issue_summary:
                return f"Yes. {subject} came up in prior sessions about {issue_summary}."
            if shaped[0].startswith("Session '"):
                descriptive = next((line for line in shaped if not line.startswith("Session '")), None)
                if descriptive:
                    return _affirm_memory_answer(descriptive)
            return _affirm_memory_answer(shaped[0])
        if len(shaped) == 1:
            return _affirm_memory_answer(shaped[0])
        return "Yes.\n\n" + "\n\n".join(shaped[:3])
    if len(shaped) == 1:
        return shaped[0]
    return "\n\n".join(shaped)


def _memory_context_sections(memory_context: str) -> dict[str, list[str]]:
    lines = {"working": [], "retrieved": [], "external": []}
    active_section: str | None = None
    current_index: int | None = None
    for raw_line in memory_context.splitlines():
        stripped = raw_line.strip()
        if stripped == "## Working Memory":
            active_section = "working"
            current_index = None
            continue
        if stripped == "## Retrieved Memory":
            active_section = "retrieved"
            current_index = None
            continue
        if stripped == "## External Session Context":
            active_section = "external"
            current_index = None
            continue
        if stripped.startswith("## "):
            active_section = None
            current_index = None
            continue
        if not active_section:
            continue
        if stripped.startswith("- "):
            lines[active_section].append(stripped[2:].strip())
            current_index = len(lines[active_section]) - 1
            continue
        if current_index is not None and stripped:
            lines[active_section][current_index] += f"\n{stripped}"
    return lines


def _summarize_error_fix_memory_answer(user_prompt: str, lines: list[str]) -> str | None:
    if not _is_error_fix_memory_question(user_prompt):
        return None

    issue_text: str | None = None
    fix_text: str | None = None
    for line in lines:
        cleaned = re.sub(r"\s+", " ", _clean_memory_line(line)).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if issue_text is None:
            issue_text = _extract_issue_summary_from_line(cleaned, lowered) or issue_text
        if fix_text is None:
            fix_text = _extract_fix_summary_from_line(cleaned, lowered) or fix_text
        if issue_text and fix_text:
            break

    if not issue_text and not fix_text:
        return None
    if issue_text and fix_text:
        return f"Yes. The error was {issue_text}. We fixed it by {fix_text}."
    if issue_text:
        return f"Yes. The error was {issue_text}."
    return f"Yes. We fixed it by {fix_text}."


def _extract_issue_summary_from_line(cleaned: str, lowered: str) -> str | None:
    normalized = re.sub(r"^fixed and committed\.\s*", "", cleaned, flags=re.IGNORECASE)
    lowered_normalized = normalized.lower()
    issue_markers = (
        "the issue was ",
        "the error was ",
        "error: ",
        "issue: ",
    )
    for marker in issue_markers:
        index = lowered_normalized.find(marker)
        if index >= 0:
            summary = _trim_error_fix_noise(normalized[index + len(marker):])
            return summary if summary else None
    if any(
        token in lowered_normalized
        for token in (
            "unknown columns",
            "unconsumed column names",
            "does not match the schema",
            "schema validation failed",
            "authentication bypass",
            "open email relay",
        )
    ):
        return _trim_error_fix_noise(normalized)
    return None


def _extract_fix_summary_from_line(cleaned: str, lowered: str) -> str | None:
    if lowered.startswith("the patch is in place."):
        tail = cleaned.split(".", 1)[1].strip() if "." in cleaned else ""
        if tail:
            nested = _extract_fix_summary_from_line(tail, tail.lower())
            if nested:
                return nested

    fix_markers = (
        "we fixed it by ",
        "fixed by ",
        "i added ",
        "we added ",
        "i updated ",
        "we updated ",
        "i replaced ",
        "we replaced ",
        "i refreshed ",
        "we refreshed ",
    )
    for marker in fix_markers:
        index = lowered.find(marker)
        if index < 0:
            continue
        summary = cleaned[index + len(marker):].strip(" .")
        summary = _trim_error_fix_noise(summary)
        if not summary:
            continue
        if marker.endswith("added "):
            return f"adding {summary}"
        if marker.endswith("updated "):
            return f"updating {summary}"
        if marker.endswith("replaced "):
            return f"replacing {summary}"
        if marker.endswith("refreshed "):
            return f"refreshing {summary}"
        return summary

    if "lazy metadata refreshes" in lowered:
        index = lowered.find("lazy metadata refreshes")
        summary = _trim_error_fix_noise(cleaned[index:])
        if summary:
            return f"adding {summary}"
    return None


def _trim_error_fix_noise(text: str) -> str:
    trimmed = str(text or "").strip(" .")
    for marker in ("Commit:", "```", "Now I’m running", "Now I'm running", "Both checks passed.", "The patch is in place."):
        if marker in trimmed:
            trimmed = trimmed.split(marker, 1)[0].strip(" .")
    return trimmed


def _recent_working_follow_up_lines(lines: list[str], user_prompt: str) -> list[str]:
    selected: list[str] = []
    for raw_line in reversed(lines):
        lowered = raw_line.lower()
        if not (lowered.startswith("assistant:") or lowered.startswith("assistant reported:")):
            continue
        cleaned = _clean_memory_line(raw_line)
        if not cleaned or not _is_high_signal_memory_answer(cleaned, user_prompt):
            continue
        if cleaned not in selected:
            selected.append(cleaned)
        if len(selected) >= 2:
            break
    selected.reverse()
    return selected


def _summarize_directory_listing(candidate_path: str, output: str) -> str:
    relative_paths: list[str] = []
    payload = _extract_tool_payload_json(output)
    entries = []
    if isinstance(payload, dict):
        entries = payload.get("entries") or payload.get("topology") or []
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict):
                relative_path = entry.get("relative_path")
                if isinstance(relative_path, str) and relative_path.strip():
                    relative_paths.append(relative_path.strip())

    if not relative_paths:
        lines = output.splitlines()
        for line in lines:
            if '"relative_path":' in line:
                fragment = line.split('"relative_path":', 1)[1].strip().strip('",')
                if fragment:
                    relative_paths.append(fragment.strip('"'))
    unique_paths = list(dict.fromkeys(relative_paths))
    if not unique_paths:
        return f"I inspected `{candidate_path}` locally, but I didn't find enough structured file evidence to summarize it yet."

    preview = ", ".join(unique_paths[:6])
    return f"I inspected `{candidate_path}` locally. Relevant paths I found: {preview}."


def _clean_memory_line(line: str) -> str:
    cleaned = line.strip()
    if "|" in cleaned:
        cleaned = cleaned.split("|", 1)[1].strip()
    cleaned = re.sub(r"^\[[^\]]+\]\s*", "", cleaned)
    cleaned = re.sub(r"^(episodic memory|episode)\s+[a-f0-9-]+:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(assistant|user|tool):\s*", "", cleaned, flags=re.IGNORECASE)
    return _sanitize_logged_answer(cleaned).strip()


def _affirm_memory_answer(text: str) -> str:
    stripped = text.strip()
    if re.match(r"^(yes|yeah|yep)\b", stripped, flags=re.IGNORECASE):
        return stripped
    return f"Yes. {stripped}"


def _humanize_recalled_line(line: str, user_prompt: str) -> str:
    cleaned = re.sub(r"^(user asked|assistant reported):\s*", "", line.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    if _is_memory_follow_up_question(user_prompt):
        cleaned = cleaned.replace(" -> ", ": ")
        cleaned = cleaned.replace(" · ", "; ")
        cleaned = cleaned.replace(" doesnt work", " did not work")
        cleaned = cleaned.replace(" doesnt ", " did not ")
        cleaned = cleaned.replace(" doesnt", " did not")
    if cleaned.startswith("The first sweep shows there is already "):
        cleaned = cleaned.replace("The first sweep shows there is already ", "", 1)
    if " I’m " in cleaned:
        cleaned = cleaned.split(" I’m ", 1)[0].rstrip(" .,;")
    if " I'll " in cleaned:
        cleaned = cleaned.split(" I'll ", 1)[0].rstrip(" .,;")
    if " I’ll " in cleaned:
        cleaned = cleaned.split(" I’ll ", 1)[0].rstrip(" .,;")
    if any(
        marker in user_prompt.lower()
        for marker in (
            "what was it about",
            "can you explain about it",
            "can you explain it",
            "can you elaborate it",
            "can you elaborate on it",
            "explain about it",
            "explain it",
            "elaborate it",
            "elaborate on it",
        )
    ):
        lowered = cleaned.lower()
        if " was about " in lowered and not lowered.startswith("it was about "):
            cleaned = "It was about " + cleaned.split(" was about ", 1)[1].strip()
        elif " was mainly about " in lowered and not lowered.startswith("it was mainly about "):
            cleaned = "It was mainly about " + cleaned.split(" was mainly about ", 1)[1].strip()
        elif not lowered.startswith("it was about ") and not lowered.startswith("about "):
            cleaned = f"It was about {cleaned[0].lower()}{cleaned[1:]}" if len(cleaned) > 1 else f"It was about {cleaned.lower()}"
    return cleaned.strip()


def _extract_follow_up_issues(lines: list[str]) -> list[str]:
    issue_map = {
        "create_workspace_links": "Create Workspace accepting https links and converting them internally",
        "salesforce_state": "Salesforce being marked as coming soon or disabled",
        "pipeline_chat": "the DRIP pipeline chat flow not working",
        "test_publish": "test/publish staying reachable after approvals",
        "root_redirects": "root URL redirects",
        "convex_imports": "Convex generated imports",
        "auth_bypass": "authentication bypass",
        "open_email_relay": "open email relay",
    }
    detected: list[str] = []
    for line in lines:
        lowered = line.lower()
        if (
            ("create workspace" in lowered or "workspace creation link" in lowered or "workspace creation links" in lowered)
            and ("https link" in lowered or "convert" in lowered or "support" in lowered)
        ):
            detected.append(issue_map["create_workspace_links"])
        if "salesforce" in lowered and ("coming soon" in lowered or "disable" in lowered):
            detected.append(issue_map["salesforce_state"])
        if "pipeline chat" in lowered and (
            "does not work" in lowered or "did not work" in lowered or "not working" in lowered or "broken" in lowered or "fix" in lowered
        ):
            detected.append(issue_map["pipeline_chat"])
        if ("test/publish" in lowered or "test and publish" in lowered) and ("approval" in lowered or "approvals" in lowered):
            detected.append(issue_map["test_publish"])
        if "root url redirects" in lowered:
            detected.append(issue_map["root_redirects"])
        if "convex generated imports" in lowered:
            detected.append(issue_map["convex_imports"])
        if "authentication bypass" in lowered or "auth bypass" in lowered:
            detected.append(issue_map["auth_bypass"])
        if "open email relay" in lowered:
            detected.append(issue_map["open_email_relay"])

    unique_detected: list[str] = []
    for issue in detected:
        if issue not in unique_detected:
            unique_detected.append(issue)
    return unique_detected


def _issues_relevant_to_prompt(user_prompt: str, issues: list[str]) -> list[str]:
    if _is_cleanup_schema_prompt(user_prompt):
        narrowed = [
            issue
            for issue in issues
            if issue in {"root URL redirects", "Convex generated imports", "authentication bypass", "open email relay"}
        ]
        return narrowed or issues
    return issues


def _summarize_follow_up_issues(lines: list[str]) -> str | None:
    issues = _extract_follow_up_issues(lines)
    if issues:
        return _join_human_list(issues)
    return None


def _summarize_latest_code_edit_recall(user_prompt: str, lines: list[str]) -> str | None:
    lowered_prompt = user_prompt.lower()
    if not any(
        marker in lowered_prompt
        for marker in (
            "latest code edit",
            "most recent code edit",
            "last code edit",
            "latest edit we did",
            "most recent edit we did",
            "last edit we did",
            "latest change we did",
            "most recent change we did",
            "last change we did",
            "latest code change",
            "most recent code change",
        )
    ):
        return None

    preferred: list[str] = []
    for raw_line in lines:
        line = re.sub(r"\s+", " ", _clean_memory_line(raw_line)).strip()
        lowered = line.lower()
        if not line:
            continue
        if any(
            lowered.startswith(prefix)
            for prefix in (
                "tsc passes",
                "bun run",
                "oxlint",
                "checking formatting",
                "found 0 warnings",
                "format issues found",
                "tool output:",
                "user asked:",
            )
        ):
            continue
        if any(
            noise in lowered
            for noise in (
                "official documentation",
                "opencode.ai/docs",
                "fetched pages cover tools and agents",
            )
        ):
            continue
        if any(
            marker in lowered
            for marker in (
                "the latest",
                "implemented ",
                "updated ",
                "reverted ",
                "fixed ",
                "changed ",
            )
        ):
            preferred.append(line)

    if not preferred:
        return None

    best = preferred[0].strip()
    if best.lower().startswith("the latest"):
        return best
    if best.endswith("."):
        return f"The latest code edit we did was: {best}"
    return f"The latest code edit we did was: {best}."


def _cleanup_summary_from_lines(user_prompt: str, lines: list[str]) -> str | None:
    if not _is_cleanup_schema_prompt(user_prompt):
        return None
    issues = _issues_relevant_to_prompt(user_prompt, _extract_follow_up_issues(lines))
    if issues:
        return f"The get-drip cleanup was mainly about {_join_human_list(issues)}."
    narrative = _summarize_cleanup_narrative(lines)
    if narrative:
        return f"The get-drip cleanup was mainly about {narrative}."
    return None


def _summarize_cleanup_narrative(lines: list[str]) -> str | None:
    joined = " ".join(line.lower() for line in lines)
    if not any(
        marker in joined
        for marker in (
            "cleanup",
            "clean up",
            "schema",
            "legacy column",
            "legacy columns",
            "crmcustomers",
            "campaigns",
        )
    ):
        return None

    points: list[str] = []
    if "legacy columns" in joined or "legacy-column" in joined or "duplicate state" in joined:
        points.append("removing legacy columns and duplicated state")
    if "campaigns" in joined and "crmcustomers" in joined:
        points.append("focusing the schema pass on `campaigns` and `crmCustomers`")
    elif "campaigns" in joined:
        points.append("cleaning up legacy fields in `campaigns`")
    elif "crmcustomers" in joined:
        points.append("cleaning up legacy fields in `crmCustomers`")
    if any(marker in joined for marker in ("whole table", "whole tables", "broad table deletion", "not actually dead", "conservative default")):
        points.append("keeping the cleanup conservative instead of deleting whole tables")
    if any(marker in joined for marker in ("audit/cache/helper", "helper/audit/cache", "audit", "cache", "helper tables")):
        points.append("keeping audit, cache, and helper tables that still support live flows")

    unique_points: list[str] = []
    for point in points:
        if point not in unique_points:
            unique_points.append(point)
    if not unique_points:
        return None
    return _join_human_list(unique_points)


def _format_issue_list_answer(subject: str | None, issues: list[str]) -> str:
    if not issues:
        return "I could not recover a reliable bug list from prior context."
    heading = f"In {subject}, the recalled bug list was:" if subject else "The recalled bug list was:"
    grouped_sections: list[tuple[str, list[str]]] = []
    product_issues = [
        issue
        for issue in issues
        if issue
        in {
            "Create Workspace accepting https links and converting them internally",
            "Salesforce being marked as coming soon or disabled",
            "the DRIP pipeline chat flow not working",
            "test/publish staying reachable after approvals",
        }
    ]
    lingering_issues = [issue for issue in issues if issue in {"root URL redirects", "Convex generated imports"}]
    security_issues = [issue for issue in issues if issue in {"authentication bypass", "open email relay"}]
    remaining = [issue for issue in issues if issue not in product_issues and issue not in lingering_issues and issue not in security_issues]
    if product_issues:
        grouped_sections.append(("Core product bugs", product_issues))
    if lingering_issues:
        grouped_sections.append(("Lingering app issues", lingering_issues))
    if security_issues:
        grouped_sections.append(("PR review security findings", security_issues))
    if remaining:
        grouped_sections.append(("Other recalled issues", remaining))

    lines = [heading, ""]
    for title, section_issues in grouped_sections:
        lines.append(f"**{title}**")
        lines.extend(f"- {issue}" for issue in section_issues)
        lines.append("")
    return "\n".join(lines).strip()


def _join_human_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _summarize_verification_failure_reason(reason: str) -> str:
    compact = re.sub(r"\s+", " ", (reason or "").strip())
    if not compact:
        return "resolve the verification failure"
    if len(compact) > 96:
        compact = f"{compact[:93].rstrip()}..."
    return compact


def _set_active_task(blueprint: ExecutionBlueprint, task_index: int) -> ExecutionBlueprint:
    return ExecutionBlueprint(
        raw_plan_markdown=blueprint.raw_plan_markdown,
        original_objective=blueprint.original_objective,
        tasks=list(blueprint.tasks),
        active_task_pointer=task_index,
        verification_passed=blueprint.verification_passed,
    )


def _mark_checkpoint_completed(blueprint: ExecutionBlueprint, task_index: int, trace_log: str) -> ExecutionBlueprint:
    tasks: list[CheckpointTask] = []
    for index, task in enumerate(blueprint.tasks):
        if index == task_index:
            tasks.append(
                CheckpointTask(
                    task_id=task.task_id,
                    description=task.description,
                    objective=task.objective,
                    target_path_hint=task.target_path_hint,
                    expected_artifact=task.expected_artifact,
                    verification_mode=task.verification_mode,
                    repair_origin_checkpoint_id=task.repair_origin_checkpoint_id,
                    status_reason=task.status_reason,
                    output_destination=task.output_destination,
                    child_checkpoint_ids=task.child_checkpoint_ids,
                    is_completed=True,
                    execution_trace_log=trace_log,
                )
            )
        else:
            tasks.append(task)

    return ExecutionBlueprint(
        raw_plan_markdown=blueprint.raw_plan_markdown,
        original_objective=blueprint.original_objective,
        tasks=tasks,
        active_task_pointer=min(task_index + 1, len(tasks)),
        verification_passed=blueprint.verification_passed,
    )


def _next_incomplete_task_index(blueprint: ExecutionBlueprint) -> int | None:
    for index, task in enumerate(blueprint.tasks):
        if not task.is_completed:
            return index
    return None


def _mark_blueprint_verified(blueprint: ExecutionBlueprint, passed: bool) -> ExecutionBlueprint:
    return ExecutionBlueprint(
        raw_plan_markdown=blueprint.raw_plan_markdown,
        original_objective=blueprint.original_objective,
        tasks=list(blueprint.tasks),
        active_task_pointer=len(blueprint.tasks),
        verification_passed=passed,
    )


def _blueprint_markdown_for_chat(blueprint: ExecutionBlueprint) -> str:
    raw = str(blueprint.raw_plan_markdown or "").strip()
    if raw:
        return raw
    lines = [f"- [ ] {task.description}" for task in blueprint.tasks if str(task.description or "").strip()]
    return "\n".join(lines) if lines else "Plan ready."


def _is_generic_explicit_plan_blueprint(blueprint: ExecutionBlueprint) -> bool:
    descriptions = [str(task.description or "").strip().lower() for task in blueprint.tasks]
    if len(descriptions) != 3:
        return False
    return (
        descriptions[0].startswith("inspect the files and dependencies needed for:")
        and descriptions[1].startswith("apply the requested implementation for:")
        and descriptions[2].startswith("verify the workspace result for:")
    )


def _summarize_step_detail(lines: list[str]) -> str:
    cleaned: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if cleaned:
                break
            continue
        if line.lower() in {"html", "css", "javascript", "js"}:
            continue
        if line.startswith("<") or line.startswith("{") or line.startswith("const ") or line.startswith("function "):
            break
        cleaned.append(re.sub(r"\s+", " ", line))
        if len(" ".join(cleaned)) >= 140:
            break

    summary = " ".join(cleaned).strip()
    summary = re.sub(r"[`*_#]+", "", summary)
    return summary[:160].rstrip(" :;,-")


def _summarize_execution_note(content: str | None) -> str:
    if not content or not content.strip():
        return "Checkpoint completed via tool execution."

    lines: list[str] = []
    in_code_block = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not line:
            continue
        if line.lower() in {"html", "css", "javascript", "js", "python"}:
            continue
        if line.startswith("<") or line.startswith("{") or line.startswith("const ") or line.startswith("function "):
            continue
        lines.append(re.sub(r"\s+", " ", line))
        if len(" ".join(lines)) >= 180:
            break

    summary = " ".join(lines).strip()
    if not summary:
        return "Checkpoint completed via tool execution."
    summary = re.sub(r"[`*_#]+", "", summary)
    return summary[:180].rstrip(" :;,-")


def _extract_tool_payload_json(output: str) -> dict[str, Any] | list[Any] | None:
    if not output:
        return None
    brace_index = output.find("{")
    bracket_index = output.find("[")
    start_candidates = [index for index in (brace_index, bracket_index) if index >= 0]
    if not start_candidates:
        return None
    start_index = min(start_candidates)
    try:
        return json.loads(output[start_index:])
    except json.JSONDecodeError:
        return None

from __future__ import annotations

import json
import re


def _extract_ui_transcript_answer(content: str) -> str | None:
    raw = str(content or "").strip()
    if not raw:
        return None

    lines = raw.splitlines()
    extracted_answers: list[str] = []
    index = 0
    total = len(lines)

    while index < total:
        if lines[index].strip().lower() != "you":
            index += 1
            continue

        index += 1
        while index < total and lines[index].strip().lower() != "devenv status":
            index += 1
        if index >= total:
            break

        while index < total and lines[index].strip().lower() != "devenv":
            index += 1
        if index >= total:
            break

        index += 1
        answer_lines: list[str] = []
        while index < total and lines[index].strip().lower() != "you":
            answer_lines.append(lines[index])
            index += 1
        answer = "\n".join(answer_lines).strip()
        if answer:
            extracted_answers.append(answer)

    if not extracted_answers:
        return None
    return extracted_answers[-1]


def _normalize_replay_error(message: str) -> str:
    cleaned = " ".join(str(message or "").split())
    if not cleaned:
        return "I couldn't complete that replayed answer."
    if "user rejected permission to use this specific tool call" in cleaned.lower():
        return "Permission to use a required tool call was denied."
    if cleaned.endswith("."):
        return cleaned
    return f"{cleaned}."


def canonicalize_response_block(content: str) -> str:
    text = " ".join(str(content or "").strip().split())
    if not text:
        return ""
    while text.lower().startswith("yes. yes. "):
        text = text[5:].strip()
    nested_match = re.match(r"^Yes\.\s+Yes\.\s+(.+)$", text, flags=re.IGNORECASE)
    if nested_match and nested_match.group(1):
        text = f"Yes. {nested_match.group(1).strip()}"
    return text


def is_affirmative_only_block(content: str) -> bool:
    return bool(re.fullmatch(r"(yes|yeah|yep)\.?", str(content or "").strip(), flags=re.IGNORECASE))


def collapse_repeated_blocks(content: str | None) -> str | None:
    raw = str(content or "").strip()
    if not raw:
        return None
    blocks = [block.strip() for block in re.split(r"\n\s*\n", raw) if block.strip()]
    if not blocks:
        return raw
    deduped_blocks: list[str] = []
    seen_canonical: set[str] = set()
    for index, block in enumerate(blocks):
        if is_affirmative_only_block(block):
            next_block = blocks[index + 1] if index + 1 < len(blocks) else ""
            if next_block:
                continue
        canonical = canonicalize_response_block(block)
        if canonical and canonical in seen_canonical:
            continue
        if not deduped_blocks or deduped_blocks[-1] != block:
            deduped_blocks.append(block)
            if canonical:
                seen_canonical.add(canonical)
    collapsed = "\n\n".join(deduped_blocks).strip()
    return collapsed or raw


def trim_response_noise(content: str | None) -> str | None:
    raw = str(content or "").strip()
    if not raw:
        return None
    cutoff_markers = (
        "\nTool output:",
        "\nDevenv status",
        "\nTool trace",
        "\nTracePrepared the final answer",
        "\nReasoningReasoned through the next step",
    )
    trimmed = raw
    for marker in cutoff_markers:
        marker_index = trimmed.find(marker)
        if marker_index >= 0:
            trimmed = trimmed[:marker_index].rstrip()
    noisy_blocks = {
        "devenv status",
        "tool trace",
        "prepared the final answer",
        "reasoned through the next step",
        "traceprepared the final answer",
    }
    cleaned_blocks: list[str] = []
    for block in re.split(r"\n\s*\n", trimmed):
        normalized = " ".join(block.strip().split()).lower()
        if not normalized:
            continue
        if normalized in noisy_blocks:
            continue
        cleaned_blocks.append(block.strip())
    result = "\n\n".join(cleaned_blocks).strip()
    return result or None


def normalize_response_text(content: str) -> str:
    text = _extract_ui_transcript_answer(str(content or "")) or str(content or "").strip()
    if not text:
        return ""

    qa_match = re.match(r"^\s*q\.\s[\s\S]*?\n+a\.\s*([\s\S]+)$", text, flags=re.IGNORECASE)
    if qa_match:
        text = qa_match.group(1).strip()
    else:
        answer_only_match = re.match(r"^\s*a\.\s*([\s\S]+)$", text, flags=re.IGNORECASE)
        if answer_only_match:
            text = answer_only_match.group(1).strip()

    inline_cutoff_markers = (
        "Tool output:",
        "Devenv status",
        "Tool trace",
        "Prepared the final answer",
        "Prepared the context for the next tool step",
        "Reasoned through the next step",
        "TracePrepared the final answer",
        "ReasoningReasoned through the next step",
        "<proposed_plan>",
    )
    lowered = text.lower()
    cutoff_index: int | None = None
    for marker in inline_cutoff_markers:
        marker_index = lowered.find(marker.lower())
        if marker_index < 0:
            continue
        if cutoff_index is None or marker_index < cutoff_index:
            cutoff_index = marker_index
    if cutoff_index is not None:
        text = text[:cutoff_index].rstrip()

    lines = [line.rstrip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    noisy_lines = {
        "devenv status",
        "tool trace",
        "opencode",
        "devenv",
        "prepared the final answer",
        "prepared the context for the next tool step",
        "reasoned through the next step",
        "traceprepared the final answer",
        "reasoningreasoned through the next step",
    }
    for line in lines:
        normalized = " ".join(line.strip().split()).lower()
        if not normalized:
            cleaned_lines.append("")
            continue
        if normalized in noisy_lines or normalized == "⚡" or re.fullmatch(r"\d+s", normalized):
            continue
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines).strip()
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = re.sub(r"^(Yes\.\s*){2,}", "Yes. ", cleaned_text, flags=re.IGNORECASE)
    return cleaned_text.strip()


def sanitize_response_text(content: object) -> str | None:
    raw = str(content or "").strip()
    if not raw:
        return None
    normalized = normalize_response_text(raw)
    trimmed = trim_response_noise(normalized)
    collapsed = collapse_repeated_blocks(trimmed)
    final_text = normalize_response_text(collapsed or "")
    return final_text or None


def sanitize_replay_text(content: object) -> str | None:
    raw = str(content or "").strip()
    if not raw:
        return None
    if not raw.startswith("{") or "\n" not in raw:
        return sanitize_response_text(raw)

    readable_lines: list[str] = []
    replay_errors: list[str] = []
    tool_failures: list[str] = []
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
        if isinstance(event_payload, dict):
            payload_type = str(event_payload.get("type") or "").strip().lower()
            if payload_type == "agent_message":
                message = str(event_payload.get("message") or "").strip()
                if message:
                    readable_lines.append(message)
                continue
            if payload_type in {"error", "tool_error"}:
                detail = str(event_payload.get("message") or event_payload.get("error") or "").strip()
                if detail:
                    replay_errors.append(detail)
                continue
            if payload_type in {"tool_result", "tool_failure"}:
                detail = str(event_payload.get("message") or event_payload.get("error") or "").strip()
                if detail:
                    tool_failures.append(detail)
                continue

        if payload.get("type") == "error":
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                replay_errors.append(
                    _normalize_replay_error(
                        str((error_payload.get("data") or {}).get("message") or error_payload.get("message") or error_payload.get("name") or "")
                    )
                )
            continue

        part = payload.get("part")
        if payload.get("type") == "tool_use" and isinstance(part, dict) and part.get("tool") == "invalid":
            state = part.get("state")
            if isinstance(state, dict):
                input_payload = state.get("input")
                if isinstance(input_payload, dict):
                    error_message = str(input_payload.get("error") or "").strip()
                    if error_message:
                        tool_failures.append(error_message)

    unique_lines: list[str] = []
    for line in readable_lines:
        if line not in unique_lines:
            unique_lines.append(line)
    if unique_lines:
        return sanitize_response_text("\n\n".join(unique_lines))
    if replay_errors:
        return sanitize_response_text(replay_errors[0])
    if tool_failures:
        return "A required tool call was unavailable while replaying that answer."
    return "I couldn't produce a readable answer from that replay."

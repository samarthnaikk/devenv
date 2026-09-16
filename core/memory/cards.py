from __future__ import annotations

import hashlib
import re
from typing import Any

from .models import InteractionCard

MAX_CARD_CHARS = 6000
MAX_SEARCH_CHARS = 8000
CONTINUATION_MARKERS = (
    "continue",
    "go on",
    "keep going",
    "and then",
    "then",
    "next",
    "proceed",
    "carry on",
    "go ahead",
)


def _get(message: Any, key: str, default: str = "") -> str:
    if isinstance(message, dict):
        value = message.get(key, default)
    else:
        value = getattr(message, key, default)
    return default if value is None else str(value)


def _role(message: Any) -> str:
    return _get(message, "role", "assistant").strip().lower() or "assistant"


def _content(message: Any) -> str:
    text = _get(message, "content").strip()
    if not text:
        text = _get(message, "text").strip()
    return text


def _is_continuation(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return False
    if len(lowered) <= 40 and any(lowered.startswith(marker) for marker in CONTINUATION_MARKERS):
        return True
    return False


def normalize_search_text(text: str) -> str:
    """Lowercase text and split identifiers so exact tokens survive FTS tokenization."""
    lowered = text.lower()
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    spaced = re.sub(r"[_./\\:@#-]+", " ", spaced)
    spaced = re.sub(r"\s+", " ", spaced).strip()
    return f"{lowered}\n{spaced.lower()}".strip()[:MAX_SEARCH_CHARS]


def _split_answer(answer: str, max_chars: int = MAX_CARD_CHARS) -> list[str]:
    if len(answer) <= max_chars:
        return [answer]
    parts: list[str] = []
    buffer: list[str] = []
    size = 0
    in_fence = False
    for line in answer.splitlines():
        fence = line.strip().startswith("```")
        if size + len(line) + 1 > max_chars and buffer and not in_fence:
            parts.append("\n".join(buffer).strip())
            buffer = []
            size = 0
        buffer.append(line)
        size += len(line) + 1
        if fence:
            in_fence = not in_fence
    if buffer:
        parts.append("\n".join(buffer).strip())
    return [part for part in parts if part] or [answer]


def build_interaction_cards(
    *,
    provider: str,
    session_id: str,
    project: str,
    workspace_path: str | None,
    messages: list[Any],
    ts: str = "",
) -> list[InteractionCard]:
    """Pair each user intent with its contiguous assistant responses."""
    cards: list[InteractionCard] = []
    intent_parts: list[str] = []
    answer_parts: list[str] = []
    current_role = ""
    turn_index = 0

    def flush() -> None:
        nonlocal intent_parts, answer_parts
        intent = "\n".join(part for part in intent_parts if part).strip()
        answer = "\n".join(part for part in answer_parts if part).strip()
        if intent and answer:
            for part_index, part in enumerate(_split_answer(answer)):
                label = intent if part_index == 0 else f"{intent} (part {part_index + 1})"
                content_hash = hashlib.sha256(f"{label}\n{part}".encode("utf-8")).hexdigest()
                card_id = hashlib.sha256(
                    f"{provider}|{session_id}|{turn_index}|{part_index}".encode("utf-8")
                ).hexdigest()[:32]
                cards.append(
                    InteractionCard(
                        card_id=card_id,
                        provider=provider,
                        session_id=session_id,
                        project=project,
                        workspace_path=workspace_path,
                        turn_index=turn_index,
                        intent_text=label,
                        answer_text=part,
                        ts=ts,
                        content_hash=content_hash,
                        search_text=normalize_search_text(f"{label}\n{part}"),
                    )
                )
        intent_parts = []
        answer_parts = []

    for message in messages:
        role = _role(message)
        text = _content(message)
        if not text:
            continue
        if role == "user":
            if answer_parts:
                flush()
                turn_index += 1
            elif intent_parts and _is_continuation(text):
                answer_parts.append(f"User: {text}")
                continue
            intent_parts.append(text)
            current_role = "user"
        else:
            if not intent_parts:
                continue
            answer_parts.append(text)
            current_role = role
    flush()
    return cards

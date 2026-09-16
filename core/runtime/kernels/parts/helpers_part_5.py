def _should_enable_web_search(text: str) -> bool:
    lowered = text.lower()
    current_fact_markers = (
        "today",
        "latest",
        "current",
        "currently",
        "recent",
        "president",
        "prime minister",
        "ceo",
        "who is",
        "official website",
        "documentation",
        "docs",
        "search the web",
        "on the web",
        "net worth",
        "stock price",
        "market cap",
        "exchange rate",
        "weather",
        "forecast",
        "schedule",
        "price of",
    )
    return any(marker in lowered for marker in current_fact_markers)


def _is_explicit_live_search_prompt(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered.strip():
        return False
    if _is_memory_recall_question(text) or _is_memory_follow_up_question(text) or _is_session_history_question(text):
        return False
    if any(
        marker in lowered
        for marker in (
            "latest you can search online",
            "latest search on web",
            "latest on the web",
            "search online and tell",
            "search the web and tell",
            "look it up online",
            "search online",
            "search the web",
            "browse the web",
            "google it",
        )
    ):
        return True
    return _should_enable_web_search(lowered)


def _consolidation_cooldown_seconds() -> float:
    raw_value = os.getenv("DEVENV_CONSOLIDATION_COOLDOWN_SECONDS", "").strip()
    if not raw_value:
        return DEFAULT_CONSOLIDATION_COOLDOWN_SECONDS
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return DEFAULT_CONSOLIDATION_COOLDOWN_SECONDS


def _runtime_tool_transport() -> str:
    raw_value = os.getenv("DEVENV_TOOL_TRANSPORT", "").strip().lower()
    if raw_value in {"mcp", "stdio"}:
        return "mcp"
    return "in_process"

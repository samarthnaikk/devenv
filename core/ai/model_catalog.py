from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TTL_SECONDS = 6 * 60 * 60
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_FALLBACK_MODELS: tuple[str, ...] = (
    "opencode/claude-sonnet-4",
    "opencode/claude-haiku-4-5",
    "opencode/gpt-5-codex",
)

Runner = Callable[[Sequence[str], float], str]

_MODEL_ID_PATTERN = re.compile(r"^([^\s/]+)/([^\s/]+)$")


class ModelCatalogError(RuntimeError):
    """Raised when the OpenCode model list cannot be discovered."""


@dataclass(frozen=True)
class OpenCodeModelInfo:
    provider_id: str
    model_id: str
    name: str = ""
    family: str = ""
    cost_input: float | None = None
    cost_output: float | None = None
    raw: dict[str, Any] | None = None

    @property
    def full_id(self) -> str:
        return f"{self.provider_id}/{self.model_id}"

    @property
    def label(self) -> str:
        return self.name or self.model_id


def default_cache_path() -> Path:
    override = os.getenv("DEVENV_MODEL_CACHE", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "devenv" / "opencode_models.json"


def _default_runner(command: Sequence[str], timeout: float) -> str:
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ModelCatalogError(
            f"`{' '.join(command)}` failed (exit {completed.returncode}): {detail[:300]}"
        )
    return completed.stdout


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _info_from_payload(
    provider_id: str,
    model_id: str,
    payload: dict[str, Any],
) -> OpenCodeModelInfo:
    cost = payload.get("cost")
    cost_input: float | None = None
    cost_output: float | None = None
    if isinstance(cost, dict):
        cost_input = _as_float(cost.get("input"))
        cost_output = _as_float(cost.get("output"))
    return OpenCodeModelInfo(
        provider_id=provider_id,
        model_id=model_id,
        name=str(payload.get("name") or "").strip(),
        family=str(payload.get("family") or "").strip(),
        cost_input=cost_input,
        cost_output=cost_output,
        raw=payload,
    )


def parse_model_list(output: str) -> list[OpenCodeModelInfo]:
    """Parse `opencode models [--verbose]` output into model records.

    The plain form prints one `provider/model` per line. The verbose form prints the
    same identifier line followed by a pretty-printed JSON object; the JSON block is
    best-effort enrichment and a parse failure degrades to identifier-only.
    """
    lines = output.splitlines()
    models: list[OpenCodeModelInfo] = []
    seen: set[str] = set()
    decoder = json.JSONDecoder()
    index = 0
    while index < len(lines):
        match = _MODEL_ID_PATTERN.match(lines[index].strip())
        if match is None:
            index += 1
            continue
        provider_id, model_id = match.group(1), match.group(2)
        full_id = f"{provider_id}/{model_id}"
        info = OpenCodeModelInfo(provider_id=provider_id, model_id=model_id)

        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor < len(lines) and lines[cursor].lstrip().startswith("{"):
            blob = "\n".join(lines[cursor:])
            payload, consumed = _decode_leading_json(blob, decoder)
            if isinstance(payload, dict):
                info = _info_from_payload(provider_id, model_id, payload)
                index = cursor + consumed
            else:
                index += 1
        else:
            index += 1

        if full_id not in seen:
            seen.add(full_id)
            models.append(info)
    return models


def _decode_leading_json(blob: str, decoder: json.JSONDecoder) -> tuple[Any, int]:
    try:
        payload, end = decoder.raw_decode(blob)
    except json.JSONDecodeError:
        return None, 0
    return payload, blob[:end].count("\n")


def _filter_provider(
    models: Sequence[OpenCodeModelInfo],
    provider: str | None,
) -> list[OpenCodeModelInfo]:
    if not provider:
        return list(models)
    cleaned = provider.strip().lower()
    return [model for model in models if model.provider_id.lower() == cleaned]


def _coerce_fallback(fallback: Sequence[str] | None) -> list[OpenCodeModelInfo]:
    identifiers = list(fallback) if fallback else list(DEFAULT_FALLBACK_MODELS)
    models: list[OpenCodeModelInfo] = []
    seen: set[str] = set()
    for identifier in identifiers:
        cleaned = str(identifier or "").strip()
        if not cleaned or cleaned in seen:
            continue
        match = _MODEL_ID_PATTERN.match(cleaned)
        if match is None:
            continue
        seen.add(cleaned)
        models.append(
            OpenCodeModelInfo(provider_id=match.group(1), model_id=match.group(2))
        )
    return models


def _read_cache(
    cache_path: Path,
    *,
    ttl_seconds: float | None,
    now: float,
) -> list[OpenCodeModelInfo] | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    fetched_at = _as_float(payload.get("fetched_at")) or 0.0
    if ttl_seconds is not None and now - fetched_at > ttl_seconds:
        return None
    entries = payload.get("models")
    if not isinstance(entries, list):
        return None
    models: list[OpenCodeModelInfo] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        provider_id = str(entry.get("provider_id") or "").strip()
        model_id = str(entry.get("model_id") or "").strip()
        if not provider_id or not model_id:
            continue
        models.append(
            OpenCodeModelInfo(
                provider_id=provider_id,
                model_id=model_id,
                name=str(entry.get("name") or "").strip(),
                family=str(entry.get("family") or "").strip(),
                cost_input=_as_float(entry.get("cost_input")),
                cost_output=_as_float(entry.get("cost_output")),
            )
        )
    return models or None


def _write_cache(
    cache_path: Path,
    models: Sequence[OpenCodeModelInfo],
    now: float,
) -> None:
    payload = {
        "fetched_at": now,
        "models": [
            {
                "provider_id": model.provider_id,
                "model_id": model.model_id,
                "name": model.name,
                "family": model.family,
                "cost_input": model.cost_input,
                "cost_output": model.cost_output,
            }
            for model in models
        ],
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, cache_path)
    except OSError:
        return


def _build_command(
    executable: str,
    *,
    verbose: bool,
    refresh: bool,
) -> list[str]:
    command = [executable, "models"]
    if verbose:
        command.append("--verbose")
    if refresh:
        command.append("--refresh")
    return command


def discover_opencode_models(
    *,
    provider: str | None = None,
    verbose: bool = True,
    refresh: bool = False,
    cache_only: bool = False,
    executable: str = "opencode",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    runner: Runner | None = None,
    cache_path: Path | str | None = None,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
    fallback: Sequence[str] | None = None,
    now: float | None = None,
) -> list[OpenCodeModelInfo]:
    """Return the models OpenCode reports, cached on disk with a static fallback.

    Never raises: a discovery failure degrades to a stale cache and finally to the
    configured fallback identifiers, so the TUI picker always has something to show.
    With ``cache_only=True`` the subprocess is never run, making the call safe from
    latency-sensitive paths such as command-palette construction.
    """
    run = runner or _default_runner
    current_time = time.time() if now is None else now
    resolved_cache = Path(cache_path) if cache_path is not None else default_cache_path()

    if cache_only:
        cached_any = _read_cache(resolved_cache, ttl_seconds=None, now=current_time)
        if cached_any is not None:
            return _filter_provider(cached_any, provider)
        return _filter_provider(_coerce_fallback(fallback), provider)

    if not refresh:
        cached = _read_cache(resolved_cache, ttl_seconds=ttl_seconds, now=current_time)
        if cached is not None:
            return _filter_provider(cached, provider)

    command = _build_command(executable, verbose=verbose, refresh=refresh)
    try:
        output = run(command, timeout)
        models = parse_model_list(output)
        if not models:
            raise ModelCatalogError("OpenCode returned no models.")
    except (ModelCatalogError, subprocess.SubprocessError, OSError, ValueError):
        stale = _read_cache(resolved_cache, ttl_seconds=None, now=current_time)
        if stale is not None:
            return _filter_provider(stale, provider)
        return _filter_provider(_coerce_fallback(fallback), provider)

    _write_cache(resolved_cache, models, current_time)
    return _filter_provider(models, provider)


def list_opencode_model_ids(
    *,
    provider: str | None = None,
    refresh: bool = False,
    fallback: Sequence[str] | None = None,
    **kwargs: Any,
) -> list[str]:
    """Convenience wrapper returning `provider/model` identifier strings."""
    models = discover_opencode_models(
        provider=provider,
        refresh=refresh,
        fallback=fallback,
        **kwargs,
    )
    return [model.full_id for model in models]

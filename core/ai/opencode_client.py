from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any
from urllib import error, parse, request


@dataclass(frozen=True)
class OpenCodeServerConfig:
    base_url: str = "http://127.0.0.1:4096"
    username: str | None = None
    password: str | None = None
    timeout_seconds: float = 30.0

    def normalized_base_url(self) -> str:
        return self.base_url.rstrip("/")


@dataclass(frozen=True)
class OpenCodeServerHealth:
    healthy: bool
    version: str = ""
    detail: str = ""


@dataclass(frozen=True)
class OpenCodeServerRuntimeStatus:
    reachable: bool
    healthy: bool = False
    version: str = ""
    detail: str = ""
    base_url: str = ""
    started_by_manager: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "healthy": self.healthy,
            "version": self.version,
            "detail": self.detail,
            "base_url": self.base_url,
            "started_by_manager": self.started_by_manager,
        }


@dataclass(frozen=True)
class OpenCodeSession:
    session_id: str
    title: str = ""
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class OpenCodeModelRef:
    provider_id: str
    model_id: str

    def to_payload(self) -> dict[str, str]:
        return {"providerID": self.provider_id, "modelID": self.model_id}


@dataclass(frozen=True)
class OpenCodeToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class OpenCodeMessage:
    info: dict[str, Any]
    parts: tuple[dict[str, Any], ...]
    raw: dict[str, Any]

    @property
    def message_id(self) -> str:
        for key in ("id", "messageID", "message_id"):
            value = self.info.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    @property
    def structured_output(self) -> Any:
        for key in ("structured_output", "structuredOutput"):
            if key in self.info:
                return self.info[key]
        return None


@dataclass(frozen=True)
class OpenCodeEvent:
    event: str
    data: Any
    raw: str
    event_id: str = ""


class OpenCodeClientError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, payload: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class OpenCodeClient:
    def __init__(self, config: OpenCodeServerConfig | None = None) -> None:
        self.config = config or OpenCodeServerConfig()

    def health(self) -> OpenCodeServerHealth:
        payload = self._request_json("GET", "/global/health")
        return OpenCodeServerHealth(
            healthy=bool(payload.get("healthy", False)),
            version=str(payload.get("version") or ""),
            detail=str(payload.get("detail") or ""),
        )

    def create_session(self, *, title: str | None = None, parent_id: str | None = None) -> OpenCodeSession:
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        if parent_id:
            body["parentID"] = parent_id
        payload = self._request_json("POST", "/session", body=body)
        return _session_from_payload(payload)

    def get_session(self, session_id: str) -> OpenCodeSession:
        payload = self._request_json("GET", f"/session/{parse.quote(session_id, safe='')}")
        return _session_from_payload(payload)

    def list_sessions(self) -> tuple[OpenCodeSession, ...]:
        payload = self._request_json("GET", "/session")
        if not isinstance(payload, list):
            raise OpenCodeClientError("OpenCode returned an invalid session list.", payload=payload)
        return tuple(_session_from_payload(item) for item in payload if isinstance(item, dict))

    def send_message(
        self,
        session_id: str,
        *,
        parts: Iterable[dict[str, Any]],
        model: OpenCodeModelRef | None = None,
        agent: str | None = None,
        no_reply: bool = False,
        system: str | None = None,
        tools: Iterable[OpenCodeToolSpec] | None = None,
        message_id: str | None = None,
        output_format: dict[str, Any] | None = None,
    ) -> OpenCodeMessage:
        body: dict[str, Any] = {
            "parts": list(parts),
            "noReply": no_reply,
        }
        if model is not None:
            body["model"] = model.to_payload()
        if agent:
            body["agent"] = agent
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [tool.to_payload() for tool in tools]
        if message_id:
            body["messageID"] = message_id
        if output_format:
            body["outputFormat"] = output_format
        payload = self._request_json("POST", f"/session/{parse.quote(session_id, safe='')}/message", body=body)
        return _message_from_payload(payload)

    def list_messages(self, session_id: str, *, limit: int | None = None) -> tuple[OpenCodeMessage, ...]:
        query = f"?limit={int(limit)}" if limit is not None else ""
        payload = self._request_json("GET", f"/session/{parse.quote(session_id, safe='')}/message{query}")
        if not isinstance(payload, list):
            raise OpenCodeClientError("OpenCode returned an invalid message list.", payload=payload)
        return tuple(_message_from_payload(item) for item in payload if isinstance(item, dict))

    def get_message(self, session_id: str, message_id: str) -> OpenCodeMessage:
        payload = self._request_json(
            "GET",
            f"/session/{parse.quote(session_id, safe='')}/message/{parse.quote(message_id, safe='')}",
        )
        return _message_from_payload(payload)

    def abort_session(self, session_id: str) -> bool:
        payload = self._request_json("POST", f"/session/{parse.quote(session_id, safe='')}/abort")
        return bool(payload)

    def delete_session(self, session_id: str) -> bool:
        payload = self._request_json("DELETE", f"/session/{parse.quote(session_id, safe='')}")
        return bool(payload)

    def stream_events(self) -> Generator[OpenCodeEvent, None, None]:
        req = self._build_request("GET", "/global/event")
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                event_name = "message"
                event_id = ""
                data_lines: list[str] = []
                for raw_line in response:
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if not line:
                        if data_lines:
                            raw_data = "\n".join(data_lines)
                            yield OpenCodeEvent(
                                event=event_name,
                                data=_maybe_json(raw_data),
                                raw=raw_data,
                                event_id=event_id,
                            )
                        event_name = "message"
                        event_id = ""
                        data_lines = []
                        continue
                    if line.startswith(":"):
                        continue
                    field, _, value = line.partition(":")
                    stripped = value[1:] if value.startswith(" ") else value
                    if field == "event":
                        event_name = stripped or "message"
                    elif field == "data":
                        data_lines.append(stripped)
                    elif field == "id":
                        event_id = stripped
        except error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except error.URLError as exc:
            raise OpenCodeClientError(f"Unable to reach OpenCode server: {exc.reason}") from exc

    def _request_json(self, method: str, path: str, *, body: Any | None = None) -> Any:
        req = self._build_request(method, path, body=body)
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                payload = response.read().decode("utf-8")
                if not payload.strip():
                    return True if response.status == HTTPStatus.NO_CONTENT else {}
                try:
                    return json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise OpenCodeClientError("OpenCode returned invalid JSON.", status_code=response.status, payload=payload) from exc
        except error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except error.URLError as exc:
            raise OpenCodeClientError(f"Unable to reach OpenCode server: {exc.reason}") from exc

    def _build_request(self, method: str, path: str, *, body: Any | None = None) -> request.Request:
        data: bytes | None = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.config.username and self.config.password:
            auth = f"{self.config.username}:{self.config.password}".encode("utf-8")
            headers["Authorization"] = f"Basic {base64.b64encode(auth).decode('ascii')}"
        url = f"{self.config.normalized_base_url()}{path}"
        return request.Request(url, data=data, headers=headers, method=method)

    def _http_error(self, exc: error.HTTPError) -> OpenCodeClientError:
        payload: Any = None
        message = f"OpenCode server request failed with status {exc.code}."
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        if body.strip():
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = body
            if isinstance(payload, dict):
                detail = payload.get("message") or payload.get("error") or payload.get("detail")
                if isinstance(detail, str) and detail.strip():
                    message = detail.strip()
        return OpenCodeClientError(message, status_code=exc.code, payload=payload)


def _session_from_payload(payload: dict[str, Any]) -> OpenCodeSession:
    session_id = str(payload.get("id") or payload.get("sessionID") or payload.get("session_id") or "").strip()
    if not session_id:
        raise OpenCodeClientError("OpenCode session payload did not include an id.", payload=payload)
    return OpenCodeSession(
        session_id=session_id,
        title=str(payload.get("title") or ""),
        raw=payload,
    )


def _message_from_payload(payload: dict[str, Any]) -> OpenCodeMessage:
    info = payload.get("info")
    parts = payload.get("parts")
    if not isinstance(info, dict) or not isinstance(parts, list):
        raise OpenCodeClientError("OpenCode message payload was malformed.", payload=payload)
    normalized_parts = tuple(item for item in parts if isinstance(item, dict))
    return OpenCodeMessage(info=info, parts=normalized_parts, raw=payload)


def _maybe_json(raw: str) -> Any:
    stripped = raw.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


class OpenCodeServerManager:
    def __init__(
        self,
        *,
        config: OpenCodeServerConfig | None = None,
        executable: str = "opencode",
        startup_timeout_seconds: float = 15.0,
    ) -> None:
        self.config = config or default_opencode_server_config()
        self.executable = executable
        self.startup_timeout_seconds = startup_timeout_seconds
        self._process: subprocess.Popen[str] | None = None

    @property
    def process(self) -> subprocess.Popen[str] | None:
        return self._process

    def inspect(self) -> OpenCodeServerRuntimeStatus:
        client = OpenCodeClient(self.config)
        try:
            health = client.health()
        except OpenCodeClientError as exc:
            return OpenCodeServerRuntimeStatus(
                reachable=False,
                healthy=False,
                detail=str(exc),
                base_url=self.config.normalized_base_url(),
                started_by_manager=self._process is not None,
            )
        detail = health.detail or (f"OpenCode server reachable: {health.version}" if health.version else "OpenCode server reachable.")
        return OpenCodeServerRuntimeStatus(
            reachable=True,
            healthy=health.healthy,
            version=health.version,
            detail=detail,
            base_url=self.config.normalized_base_url(),
            started_by_manager=self._process is not None,
        )

    def ensure_server(self) -> OpenCodeServerRuntimeStatus:
        status = self.inspect()
        if status.reachable and status.healthy:
            return status
        if not shutil.which(self.executable):
            return OpenCodeServerRuntimeStatus(
                reachable=False,
                healthy=False,
                detail="OpenCode CLI was not found on PATH.",
                base_url=self.config.normalized_base_url(),
                started_by_manager=False,
            )
        self._start_server_process()
        deadline = time.monotonic() + self.startup_timeout_seconds
        last_status = status
        while time.monotonic() < deadline:
            last_status = self.inspect()
            if last_status.reachable and last_status.healthy:
                return last_status
            time.sleep(0.2)
        detail = last_status.detail or "Timed out waiting for OpenCode server health."
        raise OpenCodeClientError(detail)

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        self._process = None

    def _start_server_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        parsed = parse.urlparse(self.config.normalized_base_url())
        hostname = parsed.hostname or "127.0.0.1"
        port = parsed.port or 4096
        env = os.environ.copy()
        if self.config.username:
            env.setdefault("OPENCODE_SERVER_USERNAME", self.config.username)
        if self.config.password:
            env.setdefault("OPENCODE_SERVER_PASSWORD", self.config.password)
        self._process = subprocess.Popen(
            [self.executable, "serve", "--hostname", hostname, "--port", str(port)],
            cwd=os.getcwd(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )


def default_opencode_server_config() -> OpenCodeServerConfig:
    base_url = os.getenv("OPENCODE_SERVER_URL") or "http://127.0.0.1:4096"
    username = os.getenv("OPENCODE_SERVER_USERNAME") or None
    password = os.getenv("OPENCODE_SERVER_PASSWORD") or None
    timeout_seconds = _float_env("OPENCODE_SERVER_TIMEOUT", 30.0)
    return OpenCodeServerConfig(
        base_url=base_url,
        username=username,
        password=password,
        timeout_seconds=timeout_seconds,
    )


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

from __future__ import annotations

import json
import logging
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from core.logging_utils import configure_logging
from core.tools.read_file import ReadFileTool

from .kernel import DevenvKernel
from .models import RunConfig
from .workspace import WorkspaceBrowser

logger = logging.getLogger(__name__)


class DevenvWebApp:
    def __init__(self, config: RunConfig, port: int = 4173, *, memory=None, ai=None) -> None:
        self.config = config
        self.port = port
        self.static_root = Path("interface/website").resolve()
        self.kernel = DevenvKernel(
            workspace_path=config.workspace_path,
            db_path=config.db_path,
            vector_dir=config.vector_dir,
            memory=memory,
            ai=ai,
        )
        self.kernel.register_tool(ReadFileTool())
        self.workspace = WorkspaceBrowser(config.workspace_path)

    def create_handler(self):
        return partial(DevenvRequestHandler, app=self)

    def create_server(self) -> ThreadingHTTPServer:
        return ThreadingHTTPServer(("127.0.0.1", self.port), self.create_handler())

    def serve(self) -> None:
        server = self.create_server()
        logger.info("Starting Devenv web server: url=http://127.0.0.1:%s workspace=%s", self.port, self.config.workspace_path)
        print(f"Devenv website running at http://127.0.0.1:{server.server_address[1]}")
        server.serve_forever()

    def build_health_payload(self) -> dict[str, object]:
        return {
            "workspace_path": self.config.workspace_path,
            "port": self.port,
            "tools": sorted(self.kernel.tools),
            "status": "ok",
        }

    def build_files_payload(self, relative_path: str = "") -> dict[str, object]:
        return {
            "path": relative_path,
            "entries": [entry.to_dict() for entry in self.workspace.list_entries(relative_path)],
        }

    def build_file_payload(self, relative_path: str) -> dict[str, object]:
        return {
            "path": relative_path,
            "content": self.workspace.read_text_file(relative_path),
        }

    def run_turn(self, prompt: str, max_consecutive_tools: int | None = None) -> dict[str, object]:
        result = self.kernel.execute_turn(
            prompt,
            max_consecutive_tools=max_consecutive_tools or self.config.max_consecutive_tools,
        )
        return result.to_dict()


class DevenvRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, app: DevenvWebApp, **kwargs):
        self.app = app
        super().__init__(*args, directory=str(app.static_root), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self._write_json(HTTPStatus.OK, self.app.build_health_payload())
                return
            if parsed.path == "/api/files":
                query = parse_qs(parsed.query)
                relative_path = query.get("path", [""])[0]
                self._write_json(HTTPStatus.OK, self.app.build_files_payload(relative_path))
                return
            if parsed.path == "/api/file":
                query = parse_qs(parsed.query)
                relative_path = query.get("path", [""])[0]
                self._write_json(HTTPStatus.OK, self.app.build_file_payload(relative_path))
                return
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError) as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/turn":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        payload = self._read_json()
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "Missing required field: prompt"})
            return

        max_consecutive_tools = payload.get("max_consecutive_tools")
        if max_consecutive_tools is not None and not isinstance(max_consecutive_tools, int):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "max_consecutive_tools must be an integer"})
            return

        try:
            result = self.app.run_turn(prompt=prompt, max_consecutive_tools=max_consecutive_tools)
        except RuntimeError as exc:
            self._write_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            return
        self._write_json(HTTPStatus.OK, result)

    def log_message(self, format: str, *args) -> None:
        logger.info("web request: " + format, *args)

    def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the Devenv website runtime.")
    parser.add_argument("workspace", nargs="?", default=".", help="Workspace path to sandbox the runtime within.")
    parser.add_argument("--db-path", default="memory.db")
    parser.add_argument("--vector-dir", default="vectors")
    parser.add_argument("--max-consecutive-tools", type=int, default=5)
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = RunConfig(
        workspace_path=str(Path(args.workspace).expanduser().resolve()),
        db_path=args.db_path,
        vector_dir=args.vector_dir,
        max_consecutive_tools=args.max_consecutive_tools,
    )
    app = DevenvWebApp(config=config, port=args.port)
    app.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

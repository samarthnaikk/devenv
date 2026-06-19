from __future__ import annotations

import json
import logging
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from core.logging_utils import configure_logging
from core.tools.read_file import ReadFileTool

from .kernel import DevenvKernel
from .models import RunConfig
from .workspace import WorkspaceBrowser

logger = logging.getLogger(__name__)


class DevenvWebApp:
    def __init__(self, config: RunConfig, port: int = 4173) -> None:
        self.config = config
        self.port = port
        self.static_root = Path("interface/website").resolve()
        self.kernel = DevenvKernel(
            workspace_path=config.workspace_path,
            db_path=config.db_path,
            vector_dir=config.vector_dir,
        )
        self.kernel.register_tool(ReadFileTool())
        self.workspace = WorkspaceBrowser(config.workspace_path)

    def create_handler(self):
        return partial(DevenvRequestHandler, app=self)

    def serve(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", self.port), self.create_handler())
        logger.info("Starting Devenv web server: url=http://127.0.0.1:%s workspace=%s", self.port, self.config.workspace_path)
        print(f"Devenv website running at http://127.0.0.1:{self.port}")
        server.serve_forever()

    def build_health_payload(self) -> dict[str, object]:
        return {
            "workspace_path": self.config.workspace_path,
            "port": self.port,
            "tools": sorted(self.kernel.tools),
            "status": "ok",
        }


class DevenvRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, app: DevenvWebApp, **kwargs):
        self.app = app
        super().__init__(*args, directory=str(app.static_root), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._write_json(HTTPStatus.OK, self.app.build_health_payload())
            return
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, format: str, *args) -> None:
        logger.info("web request: " + format, *args)

    def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


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

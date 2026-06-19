from __future__ import annotations

import logging
import os


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str | None = None) -> None:
    resolved_level = (level or os.getenv("DEVENV_LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(level=getattr(logging, resolved_level, logging.INFO), format=LOG_FORMAT, force=False)

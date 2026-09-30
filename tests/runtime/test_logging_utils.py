from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.logging_utils import (
    ContextFilter,
    JsonFormatter,
    RedactingFilter,
    configure_logging,
    redact_text,
    set_log_context,
    clear_log_context,
)


class RedactionTest(unittest.TestCase):
    def test_redacts_api_keys_and_tokens(self) -> None:
        redacted = redact_text("api_key=supersecret123 and Bearer abc.def.ghi and sk-abcdef123456")
        self.assertNotIn("supersecret123", redacted)
        self.assertNotIn("abc.def.ghi", redacted)
        self.assertNotIn("sk-abcdef123456", redacted)
        self.assertIn("<redacted>", redacted)

    def test_truncates_long_payloads(self) -> None:
        redacted = redact_text("x" * 5000, limit=100)
        self.assertLessEqual(len(redacted), 100)
        self.assertTrue(redacted.endswith("..."))

    def test_redacting_filter_rewrites_args(self) -> None:
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "token=%s", ("secret-value",), None)
        RedactingFilter(redact=True).filter(record)
        self.assertNotIn("secret-value", record.getMessage())


class ContextFilterTest(unittest.TestCase):
    def test_stamps_context_defaults(self) -> None:
        clear_log_context()
        record = logging.LogRecord("x", logging.INFO, __file__, 1, "hi", (), None)
        ContextFilter().filter(record)
        self.assertEqual(record.turn_id, "-")
        self.assertEqual(record.backend, "-")

    def test_stamps_active_context(self) -> None:
        set_log_context(turn_id="t1", session_id="s1", backend="opencode", workspace="/ws")
        try:
            record = logging.LogRecord("x", logging.INFO, __file__, 1, "hi", (), None)
            ContextFilter().filter(record)
            self.assertEqual(record.turn_id, "t1")
            self.assertEqual(record.backend, "opencode")
        finally:
            clear_log_context()


class JsonFormatterTest(unittest.TestCase):
    def test_emits_json_object(self) -> None:
        record = logging.LogRecord("core.test", logging.WARNING, __file__, 1, "hello", (), None)
        ContextFilter().filter(record)
        payload = json.loads(JsonFormatter().format(record))
        self.assertEqual(payload["level"], "WARNING")
        self.assertEqual(payload["message"], "hello")
        self.assertIn("ts", payload)


class ConfigureLoggingTest(unittest.TestCase):
    def test_writes_rotating_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            workspace = Path(tempdir)
            with mock.patch.dict(os.environ, {"DEVENV_LOG_TO_FILE": "1"}):
                configure_logging("INFO", workspace=str(workspace), force=True)
                logging.getLogger("core.test.file").info("hello file log")
            for handler in logging.getLogger().handlers:
                handler.flush()
            log_path = workspace / ".devenv" / "logs" / "devenv.log"
            self.assertTrue(log_path.exists())
            self.assertIn("hello file log", log_path.read_text(encoding="utf-8"))

    def test_json_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            with mock.patch.dict(os.environ, {"DEVENV_LOG_JSON": "1", "DEVENV_LOG_TO_FILE": "1"}):
                configure_logging("INFO", workspace=tempdir, force=True)
                self.assertTrue(any(isinstance(h.formatter, JsonFormatter) for h in logging.getLogger().handlers))


if __name__ == "__main__":
    unittest.main()

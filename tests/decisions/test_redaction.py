from __future__ import annotations

import unittest

from core.decisions.redaction import looks_sensitive, prepare_remote_state, redact_state


class RedactionTest(unittest.TestCase):
    def test_redact_state_masks_api_key(self) -> None:
        redacted = redact_state("api_key=sk-abcdefghijklmnopqrstuvwxyz")
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz", redacted)

    def test_redact_state_truncates(self) -> None:
        redacted = redact_state("x" * 100, limit=20)
        self.assertLessEqual(len(redacted), 20)

    def test_looks_sensitive_private_key(self) -> None:
        self.assertTrue(looks_sensitive("-----BEGIN RSA PRIVATE KEY-----\nMIIE"))
        self.assertTrue(looks_sensitive("password= hunter2"))
        self.assertTrue(looks_sensitive("token: abcdefghijklmnopqrstuvwxyz0123456789ABCDE"))
        self.assertFalse(looks_sensitive("How does the retrieval engine work?"))

    def test_prepare_remote_state_refuses_sensitive(self) -> None:
        self.assertIsNone(prepare_remote_state("-----BEGIN OPENSSH PRIVATE KEY-----"))
        self.assertEqual(
            prepare_remote_state("How does the backend work?"),
            "How does the backend work?",
        )

    def test_prepare_remote_state_redacts_when_allowed_through(self) -> None:
        prepared = prepare_remote_state("api_key=supersecretvalue", allow_sensitive=True)
        self.assertIsNotNone(prepared)
        self.assertNotIn("supersecretvalue", prepared)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib
import os
import unittest
from unittest import mock

from core.runtime import retrieval_markers as markers


class RetrievalMarkersTest(unittest.TestCase):
    def test_defaults_are_stable(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in ("DEVENV_FOCUS_MARKERS", "DEVENV_DETAIL_MARKERS", "DEVENV_NOISE_MARKERS"):
                os.environ.pop(key, None)
            self.assertIn("bug list", markers.focus_markers())
            self.assertIn("create workspace", markers.detail_markers())
            self.assertIn("glob: /users/", markers.preview_noise_markers())

    def test_env_extends_focus_markers(self) -> None:
        with mock.patch.dict(os.environ, {"DEVENV_FOCUS_MARKERS": "custom-marker"}):
            resolved = markers.focus_markers()
        self.assertIn("bug list", resolved)
        self.assertIn("custom-marker", resolved)

    def test_env_extends_detail_and_noise(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"DEVENV_DETAIL_MARKERS": "extra-detail", "DEVENV_NOISE_MARKERS": "extra-noise"},
        ):
            self.assertIn("extra-detail", markers.detail_markers())
            self.assertIn("extra-noise", markers.preview_noise_markers())

    def test_session_focus_score_includes_bugs_tracked(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEVENV_FOCUS_MARKERS", None)
            self.assertIn("bugs tracked", markers.session_focus_score_markers())

    def test_issue_term_sets(self) -> None:
        self.assertIn("issue", markers.ISSUE_TERMS)
        self.assertNotIn("issue", markers.ISSUE_TERMS_BASE)
        self.assertIn("bug", markers.ISSUE_TERMS_BASE)


if __name__ == "__main__":
    unittest.main()

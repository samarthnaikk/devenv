from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.runtime import PathSandbox


class PathSandboxTest(unittest.TestCase):
    def test_relative_child_path_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            Path(tempdir, "src").mkdir()
            sandbox = PathSandbox(tempdir)

            self.assertTrue(sandbox.is_safe(str(Path(tempdir) / "src")))

    def test_parent_traversal_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            sandbox = PathSandbox(tempdir)

            self.assertFalse(sandbox.is_safe("../secret.txt"))

    def test_absolute_path_outside_workspace_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            sandbox = PathSandbox(tempdir)

            self.assertFalse(sandbox.is_safe("/etc/passwd"))

    def test_symlink_escape_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "secret.txt"
            target.write_text("secret", encoding="utf-8")
            link_path = Path(workspace) / "link.txt"
            link_path.symlink_to(target)

            sandbox = PathSandbox(workspace)

            self.assertFalse(sandbox.is_safe(str(link_path)))


if __name__ == "__main__":
    unittest.main()
